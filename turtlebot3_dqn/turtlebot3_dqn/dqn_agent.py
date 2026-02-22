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
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee, Hyungyu Kim

import collections
import datetime
import math
import os
import random
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from turtlebot3_msgs.srv import Dqn


LOGGING = True
current_time = datetime.datetime.now().strftime('[%mm%dd-%H:%M]')


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent(Node):

    def __init__(self):
        super().__init__('dqn_agent')
        self.declare_parameter('epsilon_decay', 6000)
        self.declare_parameter('max_training_episodes', 1000)
        self.declare_parameter('model_file', '')
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('verbose', True)
        self.max_training_episodes = self.get_parameter(
            'max_training_episodes'
        ).get_parameter_value().integer_value
        model_file = self.get_parameter('model_file').get_parameter_value().string_value
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        self.get_logger().info(f'Using device: {self.device}')

        self.train_mode = True
        self.state_size = 26
        self.action_size = 5

        self.done = False
        self.succeed = False
        self.fail = False

        self.discount_factor = 0.99
        self.learning_rate = 0.0007
        self.epsilon = 1.0
        self.step_counter = 0
        self.epsilon_decay = self.get_parameter(
            'epsilon_decay'
        ).get_parameter_value().integer_value
        self.epsilon_min = 0.05
        self.batch_size = 128

        self.replay_buffer_size = 500000
        self.replay_memory = [None] * self.replay_buffer_size
        self.priorities = numpy.zeros(self.replay_buffer_size, dtype=numpy.float32)
        self.replay_pos = 0
        self.replay_count = 0
        self.per_alpha = 0.6
        self.per_beta = 0.4
        self.per_beta_increment = 0.0005
        self.per_epsilon = 1e-6
        self.min_replay_memory_size = 5000

        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.target_model = QNetwork(self.state_size, self.action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.use_pretrained_model = bool(model_file)
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        model_path = os.path.join(self.model_dir_path, model_file)

        if self.use_pretrained_model:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.step_counter = checkpoint.get('step_counter', self.step_counter)
            self.load_episode = checkpoint.get('trained_episodes', self.load_episode)
            if self.load_episode >= self.max_training_episodes:
                self.get_logger().error('Loaded model episode exceeds max training episodes.')
                raise ValueError('Loaded model episode exceeds max training episodes.')

        self.update_target_after = 5000
        self.target_update_after_counter = 0
        self.update_target_model()

        if LOGGING:
            tensorboard_file_name = current_time + ' dqn_reward'
            home_dir = os.path.expanduser('~')
            dqn_reward_log_dir = os.path.join(
                home_dir, 'turtlebot3_dqn_logs', 'gradient_tape', tensorboard_file_name
            )
            self.writer = SummaryWriter(dqn_reward_log_dir)

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)

        episode_num = self.load_episode

        for episode in range(self.load_episode + 1, self.max_training_episodes + 1):
            state = self.reset_environment()
            episode_num += 1
            local_step = 0
            score = 0
            sum_max_q = 0.0

            time.sleep(1.0)

            while True:
                local_step += 1

                self.model.eval()
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).to(self.device)
                    q_values = self.model(state_t).cpu().numpy()
                self.model.train()
                sum_max_q += float(numpy.max(q_values))

                action = int(self.get_action(state))
                next_state, reward, done = self.step(action)
                score += reward

                msg = Float32MultiArray()
                msg.data = [float(action), float(score), float(reward)]
                self.action_pub.publish(msg)

                if self.train_mode:
                    self.append_sample((state, action, reward, next_state, done))
                    self.train_model(done)

                state = next_state

                if done:
                    avg_max_q = sum_max_q / local_step if local_step > 0 else 0.0

                    msg = Float32MultiArray()
                    msg.data = [float(score), float(avg_max_q)]
                    self.result_pub.publish(msg)

                    if LOGGING:
                        self.writer.add_scalar('dqn_reward', score, episode_num)

                    print(
                        'Episode:', episode,
                        'score:', score,
                        'memory length:', self.replay_count,
                        'epsilon:', self.epsilon)

                    param_dictionary = {
                        'epsilon': self.epsilon,
                        'step_counter': self.step_counter,
                        'trained_episodes': episode,
                    }
                    break

                time.sleep(0.001)

            if self.train_mode:
                if episode % 100 == 0:
                    idx = 1
                    while True:
                        model_path = os.path.join(
                            self.model_dir_path,
                            f'model{idx}.pt'
                        )
                        if not os.path.exists(model_path):
                            break
                        idx += 1
                    torch.save({
                        'model_state': self.model.state_dict(),
                        **param_dictionary,
                    }, model_path)

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )
        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )
        future = self.reset_environment_client.call_async(Dqn.Request())
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state = future.result().state
            state = numpy.reshape(numpy.asarray(state), [1, self.state_size])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))
        return state

    def get_action(self, state):
        if self.train_mode:
            self.step_counter += 1
            self.epsilon = self.epsilon_min + (1.0 - self.epsilon_min) * math.exp(
                -1.0 * self.step_counter / self.epsilon_decay)
            lucky = random.random()
            if lucky > (1 - self.epsilon):
                return random.randint(0, self.action_size - 1)

        self.model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.device)
            q_values = self.model(state_t).cpu().numpy()
        self.model.train()
        return int(numpy.argmax(q_values))

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            next_state = future.result().state
            next_state = numpy.reshape(numpy.asarray(next_state), [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return next_state, reward, done

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_update_after_counter = 0
        print('*Target model updated*')

    def append_sample(self, transition):
        max_p = self.priorities[:self.replay_count].max() if self.replay_count > 0 else 1.0
        self.replay_memory[self.replay_pos] = transition
        self.priorities[self.replay_pos] = max_p
        self.replay_pos = (self.replay_pos + 1) % self.replay_buffer_size
        self.replay_count = min(self.replay_count + 1, self.replay_buffer_size)

    def train_model(self, terminal):
        if self.replay_count < self.min_replay_memory_size:
            return

        n = self.replay_count
        raw = self.priorities[:n] ** self.per_alpha
        probs = raw / raw.sum()
        indices = numpy.random.choice(n, self.batch_size, replace=False, p=probs)
        data_in_mini_batch = [self.replay_memory[i] for i in indices]

        is_weights = (n * probs[indices]) ** (-self.per_beta)
        is_weights = (is_weights / is_weights.max()).astype(numpy.float32)
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        current_states = numpy.array([t[0] for t in data_in_mini_batch]).squeeze()
        next_states = numpy.array([t[3] for t in data_in_mini_batch]).squeeze()

        self.model.eval()
        with torch.no_grad():
            current_states_t = torch.FloatTensor(current_states).to(self.device)
            next_states_t = torch.FloatTensor(next_states).to(self.device)
            current_qvalues_list = self.model(current_states_t).cpu().numpy()
            next_qvalues_list = self.target_model(next_states_t).cpu().numpy()
            next_qvalues_online = self.model(next_states_t).cpu().numpy()
        self.model.train()

        x_train = []
        y_train = []
        td_errors = []

        for index, (current_state, action, reward, _, done) in enumerate(data_in_mini_batch):
            current_q_values = current_qvalues_list[index]
            old_q = current_q_values[action]

            if not done:
                best_next_action = numpy.argmax(next_qvalues_online[index])
                future_reward = next_qvalues_list[index][best_next_action]
                desired_q = reward + self.discount_factor * future_reward
            else:
                desired_q = reward

            td_errors.append(abs(desired_q - old_q) + self.per_epsilon)
            current_q_values[action] = desired_q
            x_train.append(current_state)
            y_train.append(current_q_values)

        x_t = torch.FloatTensor(
            numpy.reshape(numpy.array(x_train), [self.batch_size, self.state_size])
        ).to(self.device)
        y_t = torch.FloatTensor(
            numpy.reshape(numpy.array(y_train), [self.batch_size, self.action_size])
        ).to(self.device)
        w_t = torch.FloatTensor(is_weights).to(self.device)

        self.optimizer.zero_grad()
        pred = self.model(x_t)
        loss = (w_t * ((pred - y_t) ** 2).mean(dim=1)).mean()
        loss.backward()
        self.optimizer.step()

        for idx, td_err in zip(indices, td_errors):
            self.priorities[idx] = td_err

        self.target_update_after_counter += 1
        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_model()


def main(args=None):
    rclpy.init(args=args)

    dqn_agent = DQNAgent()
    rclpy.spin(dqn_agent)

    dqn_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
