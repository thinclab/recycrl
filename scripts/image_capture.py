#!/usr/bin/env python3

"""
This node simplifies the process of collecting Image data for training the YOLO model
"""

import os
import cv2
import rclpy
from time import sleep
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from argparse import ArgumentParser


def main():
    # Define arguments
    description = "Script to simplify the image collection process"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-data_dir", dest="data_dir", default="~/Workspace_Images", help="Directory where the captured images will be saved"
    )

    # Parse and assign arguments
    args = parser.parse_args()
    data_dir = args.data_dir

    # Expand the user to handle "~"
    data_dir = os.path.expanduser(data_dir)

    # Initialize rclpy and the ImageCapture Node
    rclpy.init()
    image_cap = ImageCapture(data_dir)

    # Try to start the loop
    try:
        while True:
            # Blocking call that asks the user to rearrange the state for a new image
            image_cap.reset_state()

            # Take a picture of the state after the user has rearranged the items, and save it locally
            image_cap.capture_image()

    # If there is an exception with the loop, notify the user
    except Exception as e:
        print("\nYolo Client Failed: %r" % (e,))

    # If there is a Keyboard Interrupt, gracefully shut down the Node
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


class ImageCapture(Node):
    def __init__(self, data_dir):
        # Register the ROS2 Node
        super().__init__("capture_image")

        # Define the Cv Bridge
        self.bridge = CvBridge()

        # Define global variables
        self.data_dir = data_dir
        self.rgb = None
        self.image_name = None

        # If the data directory doesn't exist
        if not os.path.exists(data_dir):
            # Create the directory
            os.makedirs(data_dir)

        # Create a subscriber for RGB Images
        self.sub = self.create_subscription(Image, "/oak/rgb/image_raw", self.sub_callback, 1)

    def sub_callback(self, msg):
        # Assign the message to the global variable
        self.rgb = msg

    def capture_image(self):
        # Check that the camera is publishing, and notify the user if it is not
        while not self.get_publishers_info_by_topic("/oak/rgb/image_raw"):
            self.get_logger().error("Camera is inactive")
            sleep(5.0)

        # Reset the RGB image so that spin gets the most recent
        self.rgb = None

        # Spin until the messages are received
        while not self.rgb:
            rclpy.spin_once(self)

        # Convert the message to a cv image
        cv_image = self.bridge.imgmsg_to_cv2(self.rgb, desired_encoding="bgr8")

        # Get the list of previously saved images
        images = os.listdir(self.data_dir)

        # If images exist
        if images:
            # Define variable to be assigned locally
            largest_image_number = 0

            # Go through each image
            for image in images:
                # Extract the number from the image name
                image_number = int(image[len("image_") : -len(".png")])

                # If the current image number is greater than the largest, assign this to be the new largest
                if image_number > largest_image_number:
                    largest_image_number = image_number

            # Define the image name
            self.image_name = f"image_{largest_image_number + 1}"

        # If there are no images, create the first one
        else:
            self.image_name = "image_0"

        # Save the image
        cv2.imwrite(f"{self.data_dir}/{self.image_name}.png", cv_image)

        # Print the image number
        self.get_logger().info(f"Captured {self.image_name}")

    def reset_state(self):
        # Block the program until the user has to notified that the workspace has been reset
        self.get_logger().info("Finished resetting workspace? ('Enter')")
        answer = input()

        # If they do not click "Enter", make sure the user is sure
        while answer != "":
            self.get_logger().warn("You typed '" + str(answer) + "', click 'Enter' to capture the current workspace and continue")
            answer = input()


if __name__ == "__main__":
    main()
