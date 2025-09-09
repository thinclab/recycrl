#!/usr/bin/env python3

"""
This script publishes Roll and Pitch values of the Oak-D S2 Camera for alignment
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from math import atan2, sqrt
import numpy as np


def main():
    # Initialize rclpy and the Node, then spin until "Ctrl+C"
    rclpy.init()
    imu_filter = ImuFilter()
    rclpy.spin(imu_filter)


class ImuFilter(Node):
    def __init__(self, alpha=0.99):
        # Register Node
        super().__init__("imu_filter")

        # Define global variables
        self.roll_vals = []
        self.pitch_vals = []

        # Create subscriber and timer
        self.imu_sub = self.create_subscription(Imu, "/oak/imu/data", self.sub_callback, 5)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def sub_callback(self, msg):
        # Extract the linear accelerations from the passed Imu msg
        lin_acc_x = msg.linear_acceleration.x
        lin_acc_y = msg.linear_acceleration.y
        lin_acc_z = msg.linear_acceleration.z

        # Get the roll and pitch based on the linear acceleration values
        roll_acc = atan2(lin_acc_y, lin_acc_z)
        pitch_acc = atan2(-lin_acc_x, sqrt(lin_acc_y**2 + lin_acc_z**2))

        # Add to list of roll and pitch values
        self.roll_vals.append(roll_acc)
        self.pitch_vals.append(pitch_acc)

    def timer_callback(self):
        # Find the average roll and pitch values over the time period
        roll_avg = sum(self.roll_vals) / len(self.roll_vals)
        pitch_avg = sum(self.pitch_vals) / len(self.pitch_vals)

        # Reset lists
        self.roll_vals = []
        self.pitch_vals = []

        # Print the values to the terminal
        print("Roll: " + str(round(roll_avg, 5)) + "     Pitch: " + str(round(pitch_avg, 5)))


if __name__ == "__main__":
    main()
