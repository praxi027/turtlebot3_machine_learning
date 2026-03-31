#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# PPO (Proximal Policy Optimization) agent for TurtleBot3 navigation.
# Continuous 2D action space: [linear_vel, angular_vel].
# Actor-Critic network with shared backbone, Gaussian policy, GAE.

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


current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')


class ActorCritic(nn.Module):
    """
    Shared-backbone Actor-Critic network.

    Backbone: 50 -> 512 -> 256 -> 128 (mirrors DQN architecture)
    Actor head: 128 -> 2 (mean for linear_vel, angular_vel)
    Critic head: 128 -> 1 (state value V(s))

    Actions are sampled from a Gaussian distribution and squashed via tanh:
      linear_vel  = 0.11 + 0.11 * tanh(raw[0])  -> [0.0, 0.22] m/s
      angular_vel = 1.5  * tanh(raw[1])          -> [-1.5, 1.5] rad/s
    """

    def __init__(self, state_size=50, action_size=2, angular_vel_max=2.84):
        super().__init__()
        self.angular_vel_max = angular_vel_max

        self.backbone = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(128, action_size)
        # Learnable log_std shared across all states (common PPO practice)
        self.log_std = nn.Parameter(torch.zeros(action_size))
        self.critic = nn.Linear(128, 1)

        # Orthogonal init for stability
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=numpy.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.actor_mean(x)
        value = self.critic(x).squeeze(-1)
        return mean, value

    def get_action_and_value(self, state):
        """Sample action from policy, return (action, action_raw, log_prob, value)."""
        mean, value = self.forward(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action_raw = dist.sample()
        log_prob = dist.log_prob(action_raw).sum(dim=-1)

        action = self._squash(action_raw)
        return action, action_raw, log_prob, value

    def evaluate(self, state, action_raw):
        """Recompute log_prob and entropy for given (state, action_raw) pairs."""
        mean, value = self.forward(state)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action_raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, value

    def get_value(self, state):
        _, value = self.forward(state)
        return value

    def _squash(self, action_raw):
        """Map unbounded raw actions to [linear_min, linear_max] x [-angular_max, angular_max]."""
        linear = 0.11 + 0.11 * torch.tanh(action_raw[..., 0:1])   # [0.0, 0.22]
        angular = self.angular_vel_max * torch.tanh(action_raw[..., 1:2])
        return torch.cat([linear, angular], dim=-1)


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
        self.declare_parameter('angular_vel_max', 2.84)
        self.declare_parameter('action_smoothing', 0.0)

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
        self.angular_vel_max = self.get_parameter('angular_vel_max').get_parameter_value().double_value
        self.action_smoothing = self.get_parameter('action_smoothing').get_parameter_value().double_value

        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        self.get_logger().info(f'Using device: {self.device}')

        # Dimensions
        self.state_size = 50
        self.action_size = 2  # [linear_vel, angular_vel]

        # Network and optimiser
        self.model = ActorCritic(
            self.state_size, self.action_size, angular_vel_max=self.angular_vel_max
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, eps=1e-5)

        run_experiment = experiment_name if experiment_name else 'default'
        if checkpoint_dir or tensorboard_dir:
            if checkpoint_dir:
                self.model_dir_path = checkpoint_dir
                self.run_dir = os.path.dirname(self.model_dir_path)
            else:
                self.run_dir = os.path.dirname(tensorboard_dir)
                self.model_dir_path = os.path.join(self.run_dir, 'checkpoints')
        else:
            if not run_id:
                run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.run_dir = os.path.join(
                os.path.expanduser('~'),
                'turtlebot3_runs',
                'manual',
                run_experiment,
                run_id,
            )
            self.model_dir_path = os.path.join(self.run_dir, 'checkpoints')
        os.makedirs(self.model_dir_path, exist_ok=True)

        self.load_episode = 0
        if model_file:
            model_path = os.path.join(self.model_dir_path, model_file)
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
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

        # ROS clients and publishers
        self.rl_agent_interface_client = self.create_client(Ppo, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Ppo, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/ppo_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, '/ppo_result', 10)

        # Rollout buffer
        self.buffer = RolloutBuffer()

        self.process()

    # ------------------------------------------------------------------
    # ROS environment interface
    # ------------------------------------------------------------------

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'make_environment service not available, waiting ...'
            )
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'reset_environment service not available, waiting ...'
            )
        future = self.reset_environment_client.call_async(Ppo.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None:
            state = numpy.reshape(numpy.asarray(result.state), [1, self.state_size])
        else:
            self.get_logger().error('reset_environment call failed')
            state = numpy.zeros([1, self.state_size])
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
            next_state = numpy.reshape(numpy.asarray(result.state), [1, self.state_size])
            reward = float(result.reward)
            done = bool(result.done)
        else:
            self.get_logger().error('rl_agent_interface call failed')
            next_state = numpy.zeros([1, self.state_size])
            reward = 0.0
            done = False

        return next_state, reward, done

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

        states_np = numpy.array(self.buffer.states).squeeze(axis=1)       # (n, 50)
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

                new_log_probs, entropy, new_values = self.model.evaluate(
                    mb_states, mb_actions_raw
                )

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
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode = self.load_episode
        global_step = 0
        ep_reward = 0.0
        ep_steps = 0
        rollout_num = 0

        # Rolling trackers for episode-level metrics
        recent_rewards = deque(maxlen=100)
        recent_outcomes = deque(maxlen=100)  # 'success', 'collision', 'timeout'

        state = self.reset_environment()
        time.sleep(1.0)
        prev_action = None  # for action smoothing EMA

        while episode < self.max_training_episodes:
            # === ROLLOUT COLLECTION ===
            self.buffer.clear()

            for _ in range(self.rollout_steps):
                self.model.eval()
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(self.device)
                    action, action_raw, log_prob, value = self.model.get_action_and_value(state_t)
                self.model.train()

                action_np = action.cpu().numpy()    # (1, 2) squashed action
                action_raw_np = action_raw.cpu().numpy()  # (1, 2) pre-tanh
                log_prob_np = log_prob.cpu().item()
                value_np = value.cpu().item()

                # Action smoothing: EMA blends with previous action
                if self.action_smoothing > 0.0 and prev_action is not None:
                    alpha = self.action_smoothing
                    action_np = (1.0 - alpha) * action_np + alpha * prev_action
                prev_action = action_np.copy()

                next_state, reward, done = self.step(action_np)

                # Publish live action info
                msg = Float32MultiArray()
                msg.data = [float(action_np[0, 0]), float(action_np[0, 1]),
                            float(ep_reward), float(reward)]
                self.action_pub.publish(msg)

                self.buffer.add(
                    state, action_np, action_raw_np,
                    log_prob_np, reward, done, value_np
                )

                ep_reward += reward
                ep_steps += 1
                global_step += 1
                state = next_state

                if done:
                    episode += 1

                    # Classify outcome using reward (the environment already
                    # determined success/failure; state may reflect the next goal).
                    goal_dist = float(next_state[0][0])
                    if reward == 100.0:
                        outcome = 'success'
                    elif reward == -50.0:
                        min_lidar = float(numpy.min(next_state[0][2:]))
                        outcome = 'collision' if min_lidar < 0.20 else 'timeout'
                    else:
                        outcome = 'timeout'

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
                    state = self.reset_environment()

                    if episode % self.save_interval == 0:
                        self._save_model(episode)

                    if episode >= self.max_training_episodes:
                        break

                time.sleep(0.01)

            # Bootstrap last value for GAE
            self.model.eval()
            with torch.no_grad():
                last_state_t = torch.FloatTensor(state).to(self.device)
                last_value = self.model.get_value(last_state_t).cpu().item()
            self.model.train()

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

            if self.logging:
                self.writer.add_scalar('ppo/policy_loss', policy_loss, rollout_num)
                self.writer.add_scalar('ppo/value_loss', value_loss, rollout_num)
                self.writer.add_scalar('ppo/entropy', entropy, rollout_num)
                self.writer.add_scalar('ppo/clip_fraction', metrics['clip_fraction'], rollout_num)
                self.writer.add_scalar('ppo/approx_kl', metrics['approx_kl'], rollout_num)
                self.writer.add_scalar('ppo/explained_variance', explained_var, rollout_num)
                self.writer.add_scalar('ppo/learning_rate', self.learning_rate, rollout_num)
                with torch.no_grad():
                    std_mean = self.model.log_std.exp().mean().item()
                self.writer.add_scalar('ppo/action_std', std_mean, rollout_num)

    def _save_config(self, log_dir):
        """Write hyperparameters to config.txt inside the TensorBoard log dir."""
        config_path = os.path.join(log_dir, 'config.txt')
        with open(config_path, 'w') as f:
            f.write(f'Date: {current_time}\n')
            f.write(f'Max episodes: {self.max_training_episodes}\n')
            f.write(f'Device: {self.device}\n\n')
            f.write('=== PPO Hyperparameters ===\n')
            for name in ['gamma', 'gae_lambda', 'clip_epsilon', 'learning_rate',
                         'rollout_steps', 'n_epochs', 'minibatch_size',
                         'value_coeff', 'entropy_coeff', 'max_grad_norm',
                         'angular_vel_max']:
                f.write(f'{name} = {getattr(self, name)}\n')
            with torch.no_grad():
                log_std = self.model.log_std.tolist()
            f.write(f'log_std_init = {log_std}\n')
            f.write(f'\n=== Network ===\n')
            f.write(f'state_size = {self.state_size}\n')
            f.write(f'action_size = {self.action_size}\n')
            f.write(f'{self.model}\n')
        self.get_logger().info(f'Config saved: {config_path}')

    def _save_model(self, episode):
        idx = 1
        while True:
            model_path = os.path.join(self.model_dir_path, f'model{idx}.pt')
            if not os.path.exists(model_path):
                break
            idx += 1
        torch.save({
            'model_state': self.model.state_dict(),
            'trained_episodes': episode,
        }, model_path)
        self.get_logger().info(f'Model saved: {model_path} (episode {episode})')


def main(args=None):
    rclpy.init(args=args)

    ppo_agent = PPOAgent()
    rclpy.spin(ppo_agent)

    ppo_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
