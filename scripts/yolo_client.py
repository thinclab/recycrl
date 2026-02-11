#!/usr/bin/env python3

"""
Client to test functionality of YOLO service
"""

import rclpy
from math import degrees
from rclpy.node import Node
from functools import partial
from recycrl.srv import Recyclables
from argparse import ArgumentParser


class YOLOClient(Node):
    def __init__(self, once):
        # Register the ROS2 Node
        super().__init__("yolo_client")

        # Define a global variable from the passed one
        self.once = once

        # Create a client to the YOLO Service
        self.client = self.create_client(Recyclables, "/get_recyclables")

        # Wait until the service is ready
        while not self.client.wait_for_service(4.0):
            self.get_logger().info("Waiting for YOLO service ...")

        # If we want to call the service only once
        if once:
            self.send_request()

        # If we do not want to call the service only once
        if not once:
            # Create a timer to repeatedly call the service
            self.timer = self.create_timer(2.0, self.send_request)

    def send_request(self):
        # Define the request, call the service, and add a callback for when the request completes
        request = Recyclables.Request()
        future = self.client.call_async(request)
        future.add_done_callback(partial(self.service_callback))

    def service_callback(self, future):
        # Define list to convert integer representation of type to string
        type = ["Bottle", "Can"]

        try:
            # Retrieve the result from the response
            response = future.result()

            # If one or more objects was detected, print the results
            if len(response.positions) > 0:
                for i, position in enumerate(response.positions):
                    print(f"Object #{i + 1} - {type[response.types[i]]}\n-------------------")
                    print(
                        f"Position:     X: {round(position.x, 3)}  " f"Y: {round(position.y, 3)}  " f"Z: {round(position.z, 3)}"
                    )
                    print(
                        f"Orientation:  Roll: 0.0  "
                        f"Pitch: {round(degrees(response.pitches[i]), 3)}  "
                        f"Yaw: {round(degrees(response.yaws[i]), 3)}"
                    )
                    print("")

            # If no objects were detected, notify the user
            else:
                print("No objects detected")

        # If there is a failure in the service, notify the user
        except Exception as e:
            self.get_logger().warn("YOLO Service failed: %r" % (e,))

        # If we only want to send one request
        if self.once:
            # Shutdown the node after the request has been processed
            rclpy.shutdown()


def main(args=None):
    # Define arguments
    description = "Client Node to test the YOLO Service Node"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--once",
        dest="once",
        action="store_true",
        help="Only send one request; if unspecified, it will send a request repeatedly",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    once = args.once

    # Initialize rclpy and define the YOLO Client Node
    rclpy.init()
    node = YOLOClient(once)

    # Try to spin the Node
    try:
        rclpy.spin(node)

    # If there is an exception with spinning the Node, notify the user
    except Exception as e:
        print("\nYolo Client Failed: %r" % (e,))

    # If there is a Keyboard Interrupt, gracefully shut down the Node
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
