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
# Adapted from dqn_environment.py for PPO with continuous actions.
# Key changes:
#   - Uses Ppo.srv instead of Dqn.srv
#   - Action is a float32[2] array: [linear_vel, angular_vel]
#   - Agent has full control over linear velocity (no adaptive safety scalar)
#   - Goal services renamed to ppo_task_succeed / ppo_task_failed / ppo_initialize_env
#   - Result topic is /ppo_result, action topic is /ppo_action

import math
import os
import time

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Goal
from turtlebot3_msgs.srv import Ppo


ROS_DISTRO = os.environ.get('ROS_DISTRO')


def wait_for_future(future):
    """Wait for an async service call to complete (MultiThreadedExecutor-safe)."""
    while not future.done():
        time.sleep(0.01)
    return future.result()


class RLEnvironment(Node):

    def __init__(self):
        super().__init__('rl_environment')

        # Reward parameters (configurable via ROS2 parameters)
        self.declare_parameter('reward_progress_scale', 5.0)
        self.declare_parameter('reward_yaw_scale', 0.5)
        self.declare_parameter('reward_obstacle_scale', -5.0)
        self.declare_parameter('reward_obstacle_safe_dist', 0.6)
        self.declare_parameter('reward_obstacle_danger_dist', 0.15)
        self.declare_parameter('reward_success', 100.0)
        self.declare_parameter('reward_fail', -50.0)
        self.declare_parameter('penalty_zones', [''])

        # Environment parameters
        self.declare_parameter('max_step', 800)
        self.declare_parameter('goal_threshold', 0.20)
        self.declare_parameter('collision_threshold', 0.15)
        self.declare_parameter('angular_vel_max', 2.84)
        self.declare_parameter('lyapunov_scale', 0.0)

        self.reward_progress_scale = self.get_parameter('reward_progress_scale').get_parameter_value().double_value
        self.reward_yaw_scale = self.get_parameter('reward_yaw_scale').get_parameter_value().double_value
        self.reward_obstacle_scale = self.get_parameter('reward_obstacle_scale').get_parameter_value().double_value
        self.reward_obstacle_safe_dist = self.get_parameter('reward_obstacle_safe_dist').get_parameter_value().double_value
        self.reward_obstacle_danger_dist = self.get_parameter('reward_obstacle_danger_dist').get_parameter_value().double_value
        self.reward_success = self.get_parameter('reward_success').get_parameter_value().double_value
        self.reward_fail = self.get_parameter('reward_fail').get_parameter_value().double_value
        penalty_zone_entries = self.get_parameter(
            'penalty_zones'
        ).get_parameter_value().string_array_value
        self.max_step = self.get_parameter('max_step').get_parameter_value().integer_value
        self.goal_threshold = self.get_parameter('goal_threshold').get_parameter_value().double_value
        self.collision_threshold = self.get_parameter('collision_threshold').get_parameter_value().double_value
        self.angular_vel_max = self.get_parameter('angular_vel_max').get_parameter_value().double_value
        self.lyapunov_scale = self.get_parameter('lyapunov_scale').get_parameter_value().double_value
        self.penalty_zones = self.parse_penalty_zones(penalty_zone_entries)

        self.goal_pose_x = 0.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.done = False
        self.fail = False
        self.succeed = False

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.5
        self.scan_ranges = []
        self.front_ranges = []
        self.front_angles = []
        self.min_obstacle_distance = 10.0

        self.zone_steps_in_episode = 0
        self.zone_entered_in_episode = False

        self.local_step = 0
        self.stop_cmd_vel_timer = None

        # Fixed control period in sim time (seconds).
        # At 10 Hz: 800 steps = 80 s of sim time regardless of real_time_factor.
        self.control_period = 0.1
        self.sim_time = 0.0

        qos = QoSProfile(depth=10)

        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        # Sensor and clock subs in ReentrantCallbackGroup so they can fire
        # while the service callback is waiting for the control period.
        self.sensor_cb_group = ReentrantCallbackGroup()

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_sub_callback,
            qos,
            callback_group=self.sensor_cb_group
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_sub_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_cb_group
        )
        self.clock_sub = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            qos_profile_sensor_data,
            callback_group=self.sensor_cb_group
        )

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal,
            'ppo_task_succeed',
            callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal,
            'ppo_task_failed',
            callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal,
            'ppo_initialize_env',
            callback_group=self.clients_callback_group
        )

        self.rl_agent_interface_service = self.create_service(
            Ppo,
            'rl_agent_interface',
            self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty,
            'make_environment',
            self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Ppo,
            'reset_environment',
            self.reset_environment_callback
        )

    def make_environment_callback(self, request, response):
        self.get_logger().info('Make environment called')
        while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'service for initialize the environment is not available, waiting ...'
            )
        future = self.initialize_environment_client.call_async(Goal.Request())
        response_goal = wait_for_future(future)
        if not response_goal.success:
            self.get_logger().error('initialize environment request failed')
        else:
            self.goal_pose_x = response_goal.pose_x
            self.goal_pose_y = response_goal.pose_y
            self.get_logger().info(
                'goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y)
            )

        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        self.zone_steps_in_episode = 0
        self.zone_entered_in_episode = False
        response.state = state

        return response

    def call_task_succeed(self):
        while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task succeed is not available, waiting ...')
        future = self.task_succeed_client.call_async(Goal.Request())
        result = wait_for_future(future)
        if result is not None:
            self.goal_pose_x = result.pose_x
            self.goal_pose_y = result.pose_y
            self.get_logger().info('service for task succeed finished')
        else:
            self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('service for task failed is not available, waiting ...')
        future = self.task_failed_client.call_async(Goal.Request())
        result = wait_for_future(future)
        if result is not None:
            self.goal_pose_x = result.pose_x
            self.goal_pose_y = result.pose_y
            self.get_logger().info('service for task failed finished')
        else:
            self.get_logger().error('task failed service call failed')

    def clock_callback(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def scan_sub_callback(self, scan):
        self.scan_ranges = []
        self.front_ranges = []
        self.front_angles = []

        num_of_lidar_rays = len(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = scan.ranges[i]

            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)

            if (0 <= angle <= math.pi / 2) or (3 * math.pi / 2 <= angle <= 2 * math.pi):
                self.front_ranges.append(distance)
                self.front_angles.append(angle)

        self.min_obstacle_distance = min(self.scan_ranges) if self.scan_ranges else 10.0

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state(self):
        state = []
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        for var in self.scan_ranges:
            state.append(float(var))
        self.local_step += 1

        if self.goal_distance < self.goal_threshold:
            self.get_logger().info('Goal Reached')
            self.succeed = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_succeed()

        if self.min_obstacle_distance < self.collision_threshold:
            self.get_logger().info('Collision happened')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        if self.local_step == self.max_step:
            self.get_logger().info('Time out!')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        return state

    def calculate_reward(self):
        distance_delta = self.prev_goal_distance - self.goal_distance
        progress_reward = self.reward_progress_scale * distance_delta

        yaw_reward = self.reward_yaw_scale * (1.0 - 2.0 * abs(self.goal_angle) / math.pi)

        d = self.min_obstacle_distance
        denom = self.reward_obstacle_safe_dist - self.reward_obstacle_danger_dist
        if d < self.reward_obstacle_safe_dist and denom > 0.0:
            ratio = (self.reward_obstacle_safe_dist - d) / denom
            obstacle_reward = self.reward_obstacle_scale * ratio ** 2
        else:
            obstacle_reward = 0.0

        if self.lyapunov_scale > 0.0:
            lyapunov_reward = self.lyapunov_scale * (
                0.99 * (-self.goal_distance) - (-self.prev_goal_distance))
        else:
            lyapunov_reward = 0.0

        zone_reward = self.calculate_penalty_zone_reward()

        reward = progress_reward + yaw_reward + obstacle_reward + lyapunov_reward + zone_reward
        print(
            'progress: %.3f  yaw: %.3f  obstacle: %.3f  lyapunov: %.3f  zone: %.3f' % (
                progress_reward, yaw_reward, obstacle_reward, lyapunov_reward, zone_reward
            )
        )

        if self.succeed:
            reward = self.reward_success
        elif self.fail:
            reward = self.reward_fail

        return reward

    def rl_agent_interface_callback(self, request, response):
        # Continuous actions: request.action = [linear_vel, angular_vel]
        linear_vel = float(numpy.clip(request.action[0], 0.0, 0.22))
        angular_vel = float(numpy.clip(request.action[1], -self.angular_vel_max, self.angular_vel_max))

        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = linear_vel
            msg.angular.z = angular_vel
        else:
            msg = TwistStamped()
            msg.twist.linear.x = linear_vel
            msg.twist.angular.z = angular_vel

        self.cmd_vel_pub.publish(msg)
        if self.stop_cmd_vel_timer is None:
            self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)

        # Wait for control_period of sim time so the robot actually moves.
        # Works at any real_time_factor (including 0 = as fast as possible).
        target_sim_time = self.sim_time + self.control_period
        while self.sim_time < target_sim_time:
            time.sleep(0.001)

        response.state = self.calculate_state()
        response.reward = self.calculate_reward()
        self.prev_goal_distance = self.goal_distance
        response.done = self.done
        response.zone_steps = self.zone_steps_in_episode
        response.zone_entered = self.zone_entered_in_episode

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False

        return response

    def timer_callback(self):
        self.get_logger().info('Stop called')
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def parse_penalty_zones(self, entries):
        zones = []
        for index, entry in enumerate(entries):
            entry = entry.strip()
            if not entry:
                continue

            tokens = [token.strip() for token in entry.split(',')]
            if tokens[0] == 'circle':
                if len(tokens) != 5:
                    raise ValueError(
                        'penalty_zones[%d] circle format is "circle,x,y,radius,penalty"' % index
                    )
                center_x = float(tokens[1])
                center_y = float(tokens[2])
                radius = float(tokens[3])
                penalty = float(tokens[4])
                zones.append({
                    'type': 'circle',
                    'center_x': center_x,
                    'center_y': center_y,
                    'radius': radius,
                    'penalty': penalty,
                })
                continue

            if tokens[0] == 'box':
                values = [float(token) for token in tokens[1:]]
            else:
                values = [float(token) for token in tokens]

            if len(values) != 5:
                raise ValueError(
                    'penalty_zones[%d] must be "x_min,y_min,x_max,y_max,penalty" or '
                    '"circle,x,y,radius,penalty"' % index
                )

            x_min, y_min, x_max, y_max, penalty = values
            zones.append({
                'type': 'box',
                'x_min': min(x_min, x_max),
                'x_max': max(x_min, x_max),
                'y_min': min(y_min, y_max),
                'y_max': max(y_min, y_max),
                'penalty': penalty,
            })

        if zones:
            self.get_logger().info('Loaded %d penalty zone(s)' % len(zones))
        return zones

    def calculate_penalty_zone_reward(self):
        zone_reward = 0.0
        in_zone = False
        for zone in self.penalty_zones:
            if zone['type'] == 'circle':
                dx = self.robot_pose_x - zone['center_x']
                dy = self.robot_pose_y - zone['center_y']
                if dx * dx + dy * dy <= zone['radius'] * zone['radius']:
                    zone_reward += zone['penalty']
                    in_zone = True
            else:
                inside_x = zone['x_min'] <= self.robot_pose_x <= zone['x_max']
                inside_y = zone['y_min'] <= self.robot_pose_y <= zone['y_max']
                if inside_x and inside_y:
                    zone_reward += zone['penalty']
                    in_zone = True

        if in_zone:
            self.zone_steps_in_episode += 1
            self.zone_entered_in_episode = True

        return zone_reward

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    rl_environment = RLEnvironment()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(rl_environment)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        rl_environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
