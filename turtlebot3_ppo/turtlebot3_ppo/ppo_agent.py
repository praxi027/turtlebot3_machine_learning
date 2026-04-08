#!/usr/bin/env python3
#
# PPO (Proximal Policy Optimization) agent for TurtleBot3 navigation.
# Continuous 2D action space: [linear_vel, angular_vel].
# Separate actor/critic networks (64x64 Tanh), Gaussian policy, GAE,
# LR linear decay, observation and return normalization.

import datetime
import os
import time
from collections import deque

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from turtlebot3_msgs.srv import Ppo


class Actor(nn.Module):
    """
    Policy network (separate from critic).

    Net: state -> 64 -> tanh -> 64 -> tanh -> action_mean
    Outputs mean of Gaussian; log_std is a clamped learnable parameter.

    Actions are squashed via tanh:
      linear_vel  = 0.11 + 0.11 * tanh(raw[0])  -> [0.0, 0.22] m/s
      angular_vel = angular_vel_max * tanh(raw[1])
    """

    def __init__(self, state_size, action_size, angular_vel_max=2.84):
        super().__init__()
        self.angular_vel_max = angular_vel_max

        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.mean = nn.Linear(64, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=numpy.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)

    def _clamped_std(self):
        return torch.clamp(self.log_std, -2.0, 0.5).exp()

    def forward(self, state):
        return self.mean(self.net(state))

    def get_action(self, state):
        mean = self.forward(state)
        std = self._clamped_std().expand_as(mean)
        dist = Normal(mean, std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        action = self._squash(action_raw)
        return action, action_raw, log_prob

    def get_deterministic_action(self, state):
        mean = self.forward(state)
        return self._squash(mean)

    def evaluate(self, state, action_raw):
        mean = self.forward(state)
        std = self._clamped_std().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def _squash(self, action_raw):
        linear = 0.11 + 0.11 * torch.tanh(action_raw[..., 0:1])   # [0.0, 0.22]
        angular = self.angular_vel_max * torch.tanh(action_raw[..., 1:2])
        return torch.cat([linear, angular], dim=-1)


class Critic(nn.Module):
    """Value network (separate from actor). Net: state -> 64 -> tanh -> 64 -> tanh -> V(s)."""

    def __init__(self, state_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.value = nn.Linear(64, 1)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=numpy.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, state):
        return self.value(self.net(state)).squeeze(-1)


class RunningMeanStd:
    """Welford's online algorithm for tracking mean/variance (matches SB3/OpenAI baselines)."""

    def __init__(self, shape):
        self.mean = numpy.zeros(shape, dtype=numpy.float64)
        self.var = numpy.ones(shape, dtype=numpy.float64)
        self.count = 1e-4  # avoid div-by-zero on first use

    def update(self, x):
        """Update statistics with a batch of observations. x shape: (batch, *shape) or (*shape,)."""
        batch = numpy.asarray(x, dtype=numpy.float64)
        if batch.ndim == len(self.mean.shape):
            batch = batch[numpy.newaxis]  # single sample → batch of 1
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return (x - self.mean.astype(numpy.float32)) / (
            numpy.sqrt(self.var.astype(numpy.float32)) + 1e-8)

    def state_dict(self):
        return {'mean': self.mean.copy(), 'var': self.var.copy(), 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


class RolloutBuffer:
    """On-policy rollout buffer storing N steps of experience."""

    def __init__(self):
        self.states = []
        self.actions = []       # squashed actions sent to env
        self.actions_raw = []   # pre-tanh actions (needed for log_prob recompute)
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.actions_raw.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, state, action, action_raw, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.actions_raw.append(action_raw)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def __len__(self):
        return len(self.rewards)


class PPOAgent(Node):

    def __init__(self):
        super().__init__('ppo_agent')
        self.declare_parameter('max_training_episodes', 1000)
        self.declare_parameter('model_file', '')
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('verbose', True)
        self.declare_parameter('experiment_name', '')
        self.declare_parameter('run_id', '')
        self.declare_parameter('logging', True)
        self.declare_parameter('tensorboard_dir', '')
        self.declare_parameter('checkpoint_dir', '')

        # PPO hyperparameters (configurable via ROS2 parameters)
        self.declare_parameter('gamma', 0.99)
        self.declare_parameter('gae_lambda', 0.95)
        self.declare_parameter('clip_epsilon', 0.2)
        self.declare_parameter('learning_rate', 3e-4)
        self.declare_parameter('rollout_steps', 2048)
        self.declare_parameter('n_epochs', 10)
        self.declare_parameter('minibatch_size', 64)
        self.declare_parameter('value_coeff', 0.5)
        self.declare_parameter('entropy_coeff', 0.01)
        self.declare_parameter('max_grad_norm', 0.5)
        self.declare_parameter('save_interval', 100)
        self.declare_parameter('eval_interval', 100)
        self.declare_parameter('eval_episodes', 5)
        self.declare_parameter('eval_deterministic', True)
        self.declare_parameter('save_best_eval', True)
        self.declare_parameter('angular_vel_max', 2.84)
        self.declare_parameter('action_smoothing', 0.0)
        self.declare_parameter('lidar_state_size', 48)

        self.max_training_episodes = self.get_parameter(
            'max_training_episodes'
        ).get_parameter_value().integer_value
        model_file = self.get_parameter('model_file').get_parameter_value().string_value
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        experiment_name = self.get_parameter('experiment_name').get_parameter_value().string_value
        run_id = self.get_parameter('run_id').get_parameter_value().string_value
        self.logging = self.get_parameter('logging').get_parameter_value().bool_value
        tensorboard_dir = self.get_parameter('tensorboard_dir').get_parameter_value().string_value
        checkpoint_dir = self.get_parameter('checkpoint_dir').get_parameter_value().string_value

        self.gamma = self.get_parameter('gamma').get_parameter_value().double_value
        self.gae_lambda = self.get_parameter('gae_lambda').get_parameter_value().double_value
        self.clip_epsilon = self.get_parameter('clip_epsilon').get_parameter_value().double_value
        self.learning_rate = self.get_parameter('learning_rate').get_parameter_value().double_value
        self.rollout_steps = self.get_parameter('rollout_steps').get_parameter_value().integer_value
        self.n_epochs = self.get_parameter('n_epochs').get_parameter_value().integer_value
        self.minibatch_size = self.get_parameter('minibatch_size').get_parameter_value().integer_value
        self.value_coeff = self.get_parameter('value_coeff').get_parameter_value().double_value
        self.entropy_coeff = self.get_parameter('entropy_coeff').get_parameter_value().double_value
        self.max_grad_norm = self.get_parameter('max_grad_norm').get_parameter_value().double_value
        self.save_interval = self.get_parameter('save_interval').get_parameter_value().integer_value
        self.eval_interval = max(
            0, self.get_parameter('eval_interval').get_parameter_value().integer_value
        )
        self.eval_episodes = max(
            1, self.get_parameter('eval_episodes').get_parameter_value().integer_value
        )
        self.eval_deterministic = self.get_parameter(
            'eval_deterministic'
        ).get_parameter_value().bool_value
        self.save_best_eval = self.get_parameter(
            'save_best_eval'
        ).get_parameter_value().bool_value
        self.angular_vel_max = self.get_parameter('angular_vel_max').get_parameter_value().double_value
        self.action_smoothing = self.get_parameter('action_smoothing').get_parameter_value().double_value
        self.lidar_state_size = self.get_parameter('lidar_state_size').get_parameter_value().integer_value

        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        self.get_logger().info(f'Using device: {self.device}')

        self.state_size = None
        self.action_size = 2  # [linear_vel, angular_vel]

        # Resolve run directory from launch script args or fall back to manual layout
        if checkpoint_dir:
            self.model_dir_path = checkpoint_dir
            self.run_dir = os.path.dirname(self.model_dir_path)
        elif tensorboard_dir:
            self.run_dir = os.path.dirname(tensorboard_dir)
            self.model_dir_path = os.path.join(self.run_dir, 'checkpoints')
        else:
            run_id = run_id or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_dir = os.path.join(
                os.path.expanduser('~'), 'turtlebot3_runs', 'manual',
                experiment_name or 'default', run_id,
            )
            self.model_dir_path = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.model_dir_path, exist_ok=True)

        # ROS clients and publishers
        self.rl_agent_interface_client = self.create_client(Ppo, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Ppo, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/ppo_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, '/ppo_result', 10)

        initial_state = self.initialize_environment_state()
        self.state_size = initial_state.shape[1]
        self.get_logger().info(f'Inferred state size from environment: {self.state_size}')

        # Separate actor and critic networks
        self.initial_lr = self.learning_rate
        self.actor = Actor(
            self.state_size, self.action_size, angular_vel_max=self.angular_vel_max
        ).to(self.device)
        self.critic = Critic(self.state_size).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate, eps=1e-5
        )

        # Observation and return normalization (a la SB3 VecNormalize)
        self.obs_rms = RunningMeanStd(shape=(self.state_size,))
        self.ret_rms = RunningMeanStd(shape=())
        self.ret_running = 0.0  # discounted return accumulator for reward normalization

        self.load_episode = 0
        if model_file:
            model_path = os.path.join(self.model_dir_path, model_file)
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.actor.load_state_dict(checkpoint['actor_state'])
            self.critic.load_state_dict(checkpoint['critic_state'])
            if 'obs_rms' in checkpoint:
                self.obs_rms.load_state_dict(checkpoint['obs_rms'])
            if 'ret_rms' in checkpoint:
                self.ret_rms.load_state_dict(checkpoint['ret_rms'])
            self.load_episode = checkpoint.get('trained_episodes', 0)
            self.get_logger().info(f'Loaded model from {model_path} (episode {self.load_episode})')

        # TensorBoard
        if self.logging:
            if tensorboard_dir:
                log_dir = tensorboard_dir
            else:
                log_dir = os.path.join(self.run_dir, 'tensorboard')
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir, flush_secs=30)
            self._save_config(log_dir)

        self.best_eval_score = None
        self.best_eval_metrics = None
        self.last_eval_episode = None

        # Rollout buffer
        self.buffer = RolloutBuffer()

        self.process(initial_state)

    # ------------------------------------------------------------------
    # ROS environment interface
    # ------------------------------------------------------------------

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'make_environment service not available, waiting ...'
            )
        future = self.make_environment_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error('make_environment call failed')

    def _reshape_env_state(self, state_values):
        state = numpy.asarray(state_values, dtype=numpy.float32)
        if state.ndim == 0:
            state = numpy.asarray([], dtype=numpy.float32)
        if state.size == 0:
            width = self.state_size if self.state_size is not None else 0
            return numpy.zeros((1, width), dtype=numpy.float32)
        if self.state_size is not None and state.size != self.state_size:
            raise RuntimeError(
                f'Environment state size mismatch: expected {self.state_size}, got {state.size}'
            )
        return state.reshape(1, state.size)

    def initialize_environment_state(self):
        self.env_make()
        time.sleep(1.0)
        min_state_size = 4 + self.lidar_state_size
        for attempt in range(20):
            state = self.reset_environment()
            if state.shape[1] >= min_state_size:
                time.sleep(1.0)
                return state
            self.get_logger().warn(
                f'Initialization state too small ({state.shape[1]} < {min_state_size}), '
                f'waiting for lidar data (attempt {attempt + 1}/20)'
            )
            time.sleep(0.25)
        raise RuntimeError(
            f'Environment returned incomplete state during initialization '
            f'(expected at least {min_state_size}, got {state.shape[1]})'
        )

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'reset_environment service not available, waiting ...'
            )
        future = self.reset_environment_client.call_async(Ppo.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None:
            state = self._reshape_env_state(result.state)
        else:
            self.get_logger().error('reset_environment call failed')
            width = self.state_size if self.state_size is not None else 0
            state = numpy.zeros((1, width), dtype=numpy.float32)
        return state

    def step(self, action):
        """Send continuous action to environment, return (next_state, reward, done)."""
        req = Ppo.Request()
        req.action = [float(action[0, 0]), float(action[0, 1])]

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('rl_agent_interface service not available, waiting ...')

        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is not None:
            next_state = self._reshape_env_state(result.state)
            reward = float(result.reward)
            done = bool(result.done)
            zone_failed = bool(result.zone_failed)
        else:
            self.get_logger().error('rl_agent_interface call failed')
            next_state = numpy.zeros((1, self.state_size), dtype=numpy.float32)
            reward = 0.0
            done = False
            zone_failed = False

        return next_state, reward, done, zone_failed

    def classify_outcome(self, reward, next_state, zone_failed=False):
        goal_dist = float(next_state[0][0]) if next_state.size else float('inf')
        if reward == 100.0:
            outcome = 'success'
        elif reward == -50.0:
            lidar_start = 4
            lidar_end = min(lidar_start + self.lidar_state_size, next_state.shape[1])
            lidar_window = next_state[0][lidar_start:lidar_end]
            min_lidar = float(numpy.min(lidar_window)) if lidar_window.size else float('inf')
            if min_lidar < 0.20:
                outcome = 'collision'
            elif zone_failed:
                outcome = 'zone'
            else:
                outcome = 'timeout'
        else:
            outcome = 'timeout'
        return outcome, goal_dist

    def run_evaluation_episode(self):
        state = self.reset_environment()
        prev_action = None
        ep_reward = 0.0
        ep_steps = 0

        while True:
            norm_state = self.obs_rms.normalize(state)
            with torch.no_grad():
                state_t = torch.FloatTensor(norm_state).to(self.device)
                if self.eval_deterministic:
                    action = self.actor.get_deterministic_action(state_t)
                else:
                    action, _, _ = self.actor.get_action(state_t)

            action_np = action.cpu().numpy()
            if self.action_smoothing > 0.0 and prev_action is not None:
                alpha = self.action_smoothing
                action_np = (1.0 - alpha) * action_np + alpha * prev_action
            prev_action = action_np.copy()

            next_state, reward, done, zone_failed = self.step(action_np)
            ep_reward += reward
            ep_steps += 1
            state = next_state

            if done:
                outcome, goal_dist = self.classify_outcome(reward, next_state, zone_failed)
                return {
                    'reward': ep_reward,
                    'length': ep_steps,
                    'goal_distance_at_end': goal_dist,
                    'success': 1.0 if outcome == 'success' else 0.0,
                    'collision': 1.0 if outcome == 'collision' else 0.0,
                    'zone_fail': 1.0 if outcome == 'zone' else 0.0,
                    'timeout': 1.0 if outcome == 'timeout' else 0.0,
                }

            time.sleep(0.01)

    def evaluate_policy(self, episode):
        self.get_logger().info(
            f'Starting evaluation at training episode {episode} '
            f'({self.eval_episodes} episode(s), deterministic={self.eval_deterministic})'
        )

        actor_was_training = self.actor.training
        critic_was_training = self.critic.training
        self.actor.eval()
        self.critic.eval()

        episode_metrics = [self.run_evaluation_episode() for _ in range(self.eval_episodes)]

        if actor_was_training:
            self.actor.train()
        if critic_was_training:
            self.critic.train()

        summary = {
            'success_rate': sum(m['success'] for m in episode_metrics) / len(episode_metrics),
            'collision_rate': sum(m['collision'] for m in episode_metrics) / len(episode_metrics),
            'zone_fail_rate': sum(m['zone_fail'] for m in episode_metrics) / len(episode_metrics),
            'timeout_rate': sum(m['timeout'] for m in episode_metrics) / len(episode_metrics),
            'episode_reward_mean': sum(m['reward'] for m in episode_metrics) / len(episode_metrics),
            'episode_length_mean': sum(m['length'] for m in episode_metrics) / len(episode_metrics),
            'goal_distance_at_end_mean': (
                sum(m['goal_distance_at_end'] for m in episode_metrics) / len(episode_metrics)
            ),
        }

        if self.logging:
            self.writer.add_scalar('ppo_eval/success_rate', summary['success_rate'], episode)
            self.writer.add_scalar('ppo_eval/collision_rate', summary['collision_rate'], episode)
            self.writer.add_scalar('ppo_eval/zone_fail_rate', summary['zone_fail_rate'], episode)
            self.writer.add_scalar('ppo_eval/timeout_rate', summary['timeout_rate'], episode)
            self.writer.add_scalar(
                'ppo_eval/episode_reward_mean', summary['episode_reward_mean'], episode
            )
            self.writer.add_scalar(
                'ppo_eval/episode_length_mean', summary['episode_length_mean'], episode
            )
            self.writer.add_scalar(
                'ppo_eval/goal_distance_at_end_mean',
                summary['goal_distance_at_end_mean'],
                episode
            )

        score = (
            summary['success_rate'],
            -summary['zone_fail_rate'],
            -summary['collision_rate'],
            -summary['timeout_rate'],
            summary['episode_reward_mean'],
        )
        is_best = self.best_eval_score is None or score > self.best_eval_score
        if is_best:
            self.best_eval_score = score
            self.best_eval_metrics = dict(summary)
            if self.save_best_eval:
                self._save_named_model(
                    'best_eval.pt',
                    episode,
                    extra_state={'best_eval_metrics': summary}
                )

        self.last_eval_episode = episode
        self.get_logger().info(
            'Evaluation | episode: %d | success: %.1f%% | collision: %.1f%% | '
            'zone_fail: %.1f%% | timeout: %.1f%% | mean_reward: %.2f%s' % (
                episode,
                100.0 * summary['success_rate'],
                100.0 * summary['collision_rate'],
                100.0 * summary['zone_fail_rate'],
                100.0 * summary['timeout_rate'],
                summary['episode_reward_mean'],
                ' | new best' if is_best else ''
            )
        )

        state = self.reset_environment()
        time.sleep(1.0)
        return state

    # ------------------------------------------------------------------
    # PPO core: GAE + update
    # ------------------------------------------------------------------

    def compute_gae(self, last_value):
        """Compute GAE advantages and lambda-returns over the current buffer."""
        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones
        n = len(rewards)

        advantages = [0.0] * n
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            next_done = float(dones[t])
            delta = rewards[t] + self.gamma * next_val * (1.0 - next_done) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * gae
            advantages[t] = gae

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def ppo_update(self, advantages, returns):
        """Run PPO update for n_epochs over the rollout buffer."""
        n = len(self.buffer.states)

        states_np = numpy.array(self.buffer.states).squeeze(axis=1)
        actions_raw_np = numpy.array(self.buffer.actions_raw).squeeze(1)  # (n, 2)
        old_log_probs_np = numpy.array(self.buffer.log_probs)             # (n,)
        advantages_np = numpy.array(advantages, dtype=numpy.float32)
        returns_np = numpy.array(returns, dtype=numpy.float32)

        # Normalise advantages
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

        states_t = torch.FloatTensor(states_np).to(self.device)
        actions_raw_t = torch.FloatTensor(actions_raw_np).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs_np).to(self.device)
        advantages_t = torch.FloatTensor(advantages_np).to(self.device)
        returns_t = torch.FloatTensor(returns_np).to(self.device)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        indices = numpy.arange(n)
        for _ in range(self.n_epochs):
            numpy.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start:start + self.minibatch_size]
                if len(mb_idx) < 2:
                    continue

                mb_states = states_t[mb_idx]
                mb_actions_raw = actions_raw_t[mb_idx]
                mb_old_log_probs = old_log_probs_t[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                new_log_probs, entropy = self.actor.evaluate(
                    mb_states, mb_actions_raw
                )
                new_values = self.critic(mb_states)

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) \
                    * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(new_values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl = ((ratio - 1.0) - torch.log(ratio)).mean().item()
                total_clip_fraction += clip_frac
                total_approx_kl += approx_kl
                num_updates += 1

        if num_updates > 0:
            return {
                'policy_loss': total_policy_loss / num_updates,
                'value_loss': total_value_loss / num_updates,
                'entropy': total_entropy / num_updates,
                'clip_fraction': total_clip_fraction / num_updates,
                'approx_kl': total_approx_kl / num_updates,
            }
        return {
            'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
            'clip_fraction': 0.0, 'approx_kl': 0.0,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def process(self, initial_state=None):
        episode = self.load_episode
        global_step = 0
        ep_reward = 0.0
        ep_steps = 0
        rollout_num = 0

        # Rolling trackers for episode-level metrics
        recent_rewards = deque(maxlen=100)
        recent_outcomes = deque(maxlen=100)  # 'success', 'collision', 'zone', 'timeout'
        state = initial_state if initial_state is not None else self.initialize_environment_state()
        prev_action = None

        while episode < self.max_training_episodes:
            # === ROLLOUT COLLECTION ===
            self.buffer.clear()

            for _ in range(self.rollout_steps):
                # Normalize observation
                self.obs_rms.update(state)
                norm_state = self.obs_rms.normalize(state)

                self.actor.eval()
                self.critic.eval()
                with torch.no_grad():
                    state_t = torch.FloatTensor(norm_state).to(self.device)
                    action, action_raw, log_prob = self.actor.get_action(state_t)
                    value = self.critic(state_t)
                self.actor.train()
                self.critic.train()

                action_np = action.cpu().numpy()    # (1, 2) squashed action
                action_raw_np = action_raw.cpu().numpy()  # (1, 2) pre-tanh
                log_prob_np = log_prob.cpu().item()
                value_np = value.cpu().item()

                # Action smoothing (EMA, disabled by default with alpha=0.0)
                if self.action_smoothing > 0.0 and prev_action is not None:
                    alpha = self.action_smoothing
                    action_np = (1.0 - alpha) * action_np + alpha * prev_action
                prev_action = action_np.copy()

                next_state, reward, done, zone_failed = self.step(action_np)

                # Normalize reward by running return std
                self.ret_running = self.ret_running * self.gamma + reward
                self.ret_rms.update(numpy.array([self.ret_running]))
                norm_reward = reward / (numpy.sqrt(self.ret_rms.var) + 1e-8)

                # Publish live action info
                msg = Float32MultiArray()
                msg.data = [float(action_np[0, 0]), float(action_np[0, 1]),
                            float(ep_reward), float(reward)]
                self.action_pub.publish(msg)

                self.buffer.add(
                    norm_state, action_np, action_raw_np,
                    log_prob_np, norm_reward, done, value_np
                )

                ep_reward += reward
                ep_steps += 1
                global_step += 1
                state = next_state

                if done:
                    episode += 1

                    outcome, goal_dist = self.classify_outcome(reward, next_state, zone_failed)

                    recent_rewards.append(ep_reward)
                    recent_outcomes.append(outcome)

                    print(
                        'Episode:', episode,
                        '| reward:', round(ep_reward, 2),
                        '| steps:', ep_steps,
                        '| outcome:', outcome
                    )

                    if self.logging:
                        self.writer.add_scalar('ppo/episode_reward', ep_reward, episode)
                        self.writer.add_scalar('ppo/episode_length', ep_steps, episode)
                        self.writer.add_scalar('ppo/goal_distance_at_end', goal_dist, episode)
                        n = len(recent_outcomes)
                        self.writer.add_scalar(
                            'ppo/success_rate',
                            sum(1 for o in recent_outcomes if o == 'success') / n,
                            episode)
                        self.writer.add_scalar(
                            'ppo/collision_rate',
                            sum(1 for o in recent_outcomes if o == 'collision') / n,
                            episode)
                        self.writer.add_scalar(
                            'ppo/zone_fail_rate',
                            sum(1 for o in recent_outcomes if o == 'zone') / n,
                            episode)
                        self.writer.add_scalar(
                            'ppo/timeout_rate',
                            sum(1 for o in recent_outcomes if o == 'timeout') / n,
                            episode)
                        self.writer.add_scalar(
                            'ppo/episode_reward_mean100',
                            sum(recent_rewards) / len(recent_rewards),
                            episode)

                    ep_reward = 0.0
                    ep_steps = 0
                    prev_action = None
                    self.ret_running = 0.0  # reset return accumulator
                    state = self.reset_environment()

                    if episode % self.save_interval == 0:
                        self._save_model(episode)

                    if self.eval_interval > 0 and episode % self.eval_interval == 0:
                        state = self.evaluate_policy(episode)

                    if episode >= self.max_training_episodes:
                        break

                time.sleep(0.01)

            # Bootstrap last value for GAE (use normalized state)
            norm_state = self.obs_rms.normalize(state)
            self.critic.eval()
            with torch.no_grad():
                last_state_t = torch.FloatTensor(norm_state).to(self.device)
                last_value = self.critic(last_state_t).cpu().item()
            self.critic.train()

            # === PPO UPDATE ===
            advantages, returns = self.compute_gae(last_value)

            # Explained variance: how well the value function predicts returns
            values_np = numpy.array(self.buffer.values)
            returns_np = numpy.array(returns, dtype=numpy.float32)
            var_returns = numpy.var(returns_np)
            if var_returns > 1e-8:
                explained_var = 1.0 - numpy.var(returns_np - values_np) / var_returns
            else:
                explained_var = 0.0

            rollout_start_time = time.time()
            metrics = self.ppo_update(advantages, returns)
            update_time = time.time() - rollout_start_time
            rollout_num += 1

            policy_loss = metrics['policy_loss']
            value_loss = metrics['value_loss']
            entropy = metrics['entropy']

            print(
                'Rollout:', rollout_num,
                '| policy_loss:', round(policy_loss, 4),
                '| value_loss:', round(value_loss, 4),
                '| entropy:', round(entropy, 4),
                '| clip_frac:', round(metrics['clip_fraction'], 4),
                '| expl_var:', round(explained_var, 4)
            )

            # Publish rollout metrics
            msg = Float32MultiArray()
            msg.data = [float(ep_reward), float(policy_loss), float(value_loss), float(entropy)]
            self.result_pub.publish(msg)

            # Linear LR decay
            frac = 1.0 - (episode / self.max_training_episodes)
            self.learning_rate = self.initial_lr * max(frac, 0.0)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

            if self.logging:
                self.writer.add_scalar('ppo/policy_loss', policy_loss, rollout_num)
                self.writer.add_scalar('ppo/value_loss', value_loss, rollout_num)
                self.writer.add_scalar('ppo/entropy', entropy, rollout_num)
                self.writer.add_scalar('ppo/clip_fraction', metrics['clip_fraction'], rollout_num)
                self.writer.add_scalar('ppo/approx_kl', metrics['approx_kl'], rollout_num)
                self.writer.add_scalar('ppo/explained_variance', explained_var, rollout_num)
                self.writer.add_scalar('ppo/learning_rate', self.learning_rate, rollout_num)
                with torch.no_grad():
                    std_mean = self.actor._clamped_std().mean().item()
                    log_std_mean = self.actor.log_std.mean().item()
                self.writer.add_scalar('ppo/action_std', std_mean, rollout_num)
                self.writer.add_scalar('ppo/log_std_raw', log_std_mean, rollout_num)

        if (
            self.eval_interval > 0
            and episode > 0
            and self.last_eval_episode != episode
        ):
            self.evaluate_policy(episode)

    def _save_config(self, log_dir):
        """Write hyperparameters to config.txt inside the TensorBoard log dir."""
        config_path = os.path.join(log_dir, 'config.txt')
        with open(config_path, 'w') as f:
            f.write(f'Date: {datetime.datetime.now():%Y-%m-%d %H:%M}\n')
            f.write(f'Max episodes: {self.max_training_episodes}\n')
            f.write(f'Device: {self.device}\n\n')
            f.write('=== PPO Hyperparameters ===\n')
            for name in ['gamma', 'gae_lambda', 'clip_epsilon', 'learning_rate',
                         'rollout_steps', 'n_epochs', 'minibatch_size',
                         'value_coeff', 'entropy_coeff', 'max_grad_norm',
                         'eval_interval', 'eval_episodes', 'eval_deterministic',
                         'save_best_eval',
                         'angular_vel_max', 'lidar_state_size']:
                f.write(f'{name} = {getattr(self, name)}\n')
            with torch.no_grad():
                log_std = self.actor.log_std.tolist()
            f.write(f'log_std_init = {log_std}\n')
            f.write(f'\n=== Network ===\n')
            f.write(f'state_size = {self.state_size}\n')
            f.write(f'action_size = {self.action_size}\n')
            f.write(f'Actor:\n{self.actor}\n')
            f.write(f'Critic:\n{self.critic}\n')
        self.get_logger().info(f'Config saved: {config_path}')

    def _save_model(self, episode):
        idx = 1
        while True:
            model_path = os.path.join(self.model_dir_path, f'model{idx}.pt')
            if not os.path.exists(model_path):
                break
            idx += 1
        torch.save({
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'obs_rms': self.obs_rms.state_dict(),
            'ret_rms': self.ret_rms.state_dict(),
            'trained_episodes': episode,
        }, model_path)
        self.get_logger().info(f'Model saved: {model_path} (episode {episode})')

    def _save_named_model(self, filename, episode, extra_state=None):
        model_path = os.path.join(self.model_dir_path, filename)
        state = {
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'obs_rms': self.obs_rms.state_dict(),
            'ret_rms': self.ret_rms.state_dict(),
            'trained_episodes': episode,
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, model_path)
        self.get_logger().info(f'Model saved: {model_path} (episode {episode})')


def main(args=None):
    rclpy.init(args=args)

    ppo_agent = PPOAgent()
    rclpy.spin(ppo_agent)

    ppo_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
