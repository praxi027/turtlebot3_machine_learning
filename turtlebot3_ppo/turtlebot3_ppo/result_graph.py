#!/usr/bin/env python3
#
# Live training visualisation for PPO.
# Subscribes to /ppo_result (Float32MultiArray: [ep_reward, policy_loss, value_loss, entropy])
# and displays 3 live plots: episode reward, policy loss, value loss.

import signal
import sys
import threading

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
import pyqtgraph
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class GraphSubscriber(Node):

    def __init__(self, window):
        super().__init__('ppo_graph')
        self.window = window
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/ppo_result',
            self.data_callback,
            10
        )

    def data_callback(self, msg):
        self.window.receive_data(msg)


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()

        self.setWindowTitle('PPO Training Results')
        self.setGeometry(50, 50, 650, 750)

        self.ep = []
        self.rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.count = 1

        self.plot()

        self.ros_subscriber = GraphSubscriber(self)
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.ros_subscriber,), daemon=True
        )
        self.ros_thread.start()

    def receive_data(self, msg):
        # msg.data = [ep_reward, policy_loss, value_loss, entropy]
        if len(msg.data) < 3:
            return
        self.ep.append(self.count)
        self.rewards.append(msg.data[0])
        self.policy_losses.append(msg.data[1])
        self.value_losses.append(msg.data[2])
        self.count += 1

    def plot(self):
        self.rewardsPlt = pyqtgraph.PlotWidget(self, title='Episode Reward')
        self.rewardsPlt.setGeometry(0, 10, 650, 220)

        self.policyLossPlt = pyqtgraph.PlotWidget(self, title='Policy Loss')
        self.policyLossPlt.setGeometry(0, 250, 650, 220)

        self.valueLossPlt = pyqtgraph.PlotWidget(self, title='Value Loss')
        self.valueLossPlt.setGeometry(0, 490, 650, 220)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(200)

        self.show()

    def update(self):
        self.rewardsPlt.showGrid(x=True, y=True)
        self.policyLossPlt.showGrid(x=True, y=True)
        self.valueLossPlt.showGrid(x=True, y=True)

        self.rewardsPlt.plot(self.ep, self.rewards, pen=(255, 0, 0), clear=True)
        self.policyLossPlt.plot(self.ep, self.policy_losses, pen=(0, 200, 255), clear=True)
        self.valueLossPlt.plot(self.ep, self.value_losses, pen=(0, 255, 0), clear=True)

    def closeEvent(self, event):
        if self.ros_subscriber is not None:
            self.ros_subscriber.destroy_node()
        rclpy.shutdown()
        event.accept()


def main():
    rclpy.init()
    app = QApplication(sys.argv)
    win = Window()

    def shutdown_handler(sig, frame):
        print('shutdown')
        if win.ros_subscriber is not None:
            win.ros_subscriber.destroy_node()
        rclpy.shutdown()
        app.quit()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
