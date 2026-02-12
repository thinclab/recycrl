#!/usr/bin/env python3

"""
This node publishes Roll and Pitch values of the Oak-D S2 Camera for alignment
"""

import rclpy
from time import sleep
from rclpy.node import Node
from math import atan2, sqrt
from sensor_msgs.msg import Imu


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

        # Check that the camera is publishing, and notify the user if it is not
        while not self.get_publishers_info_by_topic("/oak/imu/data"):
            print("IMU topic currently unavailable, make sure Oak-D S2 is active")
            sleep(5.0)

        # After exiting the "while" loop above, notify the user
        print("Beginning Roll and Pitch Calculations")

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
        # Check that the lists aren't empty
        if len(self.roll_vals) and len(self.pitch_vals) > 0:
            # Find the average roll and pitch values over the time period
            roll_avg = sum(self.roll_vals) / len(self.roll_vals)
            pitch_avg = sum(self.pitch_vals) / len(self.pitch_vals)

            # Print the values to the terminal
            print("Roll: " + str(round(roll_avg, 5)) + "     Pitch: " + str(round(pitch_avg, 5)))

        # Reset lists
        self.roll_vals = []
        self.pitch_vals = []


if __name__ == "__main__":
    main()
