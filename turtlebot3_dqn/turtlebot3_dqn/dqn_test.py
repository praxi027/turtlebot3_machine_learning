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

import os
import sys
import time

import numpy
import rclpy
from rclpy.node import Node
import torch

from turtlebot3_msgs.srv import Dqn

from turtlebot3_dqn.dqn_agent import QNetwork


class DQNTest(Node):

    def __init__(self):
        super().__init__('dqn_test')
        self.declare_parameter('model_file', '')
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('verbose', True)
        model_file = self.get_parameter('model_file').get_parameter_value().string_value
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value

        self.device = torch.device(
            'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        )
        self.get_logger().info(f'Using device: {self.device}')

        self.state_size = 26
        self.action_size = 5

        if not model_file:
            self.get_logger().error('model_file parameter is required')
            sys.exit(1)

        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model',
            model_file
        )

        self.model = QNetwork(self.state_size, self.action_size).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')

        self.run_test()

    def get_action(self, state):
        state_t = torch.FloatTensor(numpy.asarray(state).reshape(1, -1)).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_t).cpu().numpy()
        return int(numpy.argmax(q_values[0]))

    def run_test(self):
        while True:
            done = False
            init = True
            score = 0
            local_step = 0
            next_state = []

            time.sleep(1.0)

            while not done:
                local_step += 1
                action = 2 if local_step == 1 else self.get_action(next_state)

                req = Dqn.Request()
                req.action = action
                req.init = init

                while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().warn(
                        'rl_agent interface service not available, waiting again...')

                future = self.rl_agent_interface_client.call_async(req)
                rclpy.spin_until_future_complete(self, future)

                if future.done() and future.result() is not None:
                    next_state = future.result().state
                    reward = future.result().reward
                    done = future.result().done
                    score += reward
                    init = False
                else:
                    self.get_logger().error(f'Service call failure: {future.exception()}')

                time.sleep(0.01)


def main(args=None):
    rclpy.init(args=args if args else sys.argv)
    node = DQNTest()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
