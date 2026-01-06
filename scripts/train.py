#!/usr/bin/env python3

"""
This script gives the RL training sequence for the local grasping policy
"""

import os
import rclpy
from TD3.TD3 import TD3
from argparse import ArgumentParser
from recycrl.scripts.recycRL import RecycRL


def main():
    # Define arguments
    description = "Script to that goes through the training process"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-buffer_path",
        dest="buffer_path",
        default="~/Replay_Buffer.npy",
        help="Path to save the replay buffer with demonstrations",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    buffer_path = args.buffer_path

    # Expand the user to handle "~"
    buffer_path = os.path.expanduser(buffer_path)

    # Initialize rclpy
    rclpy.init()

    # Try the following
    try:
        # Initialize the RecycRL Node
        train = RecycRL(buffer_path)

        # Start the loop, ends after time out reached
        while not train.timed_out:
            # Move the robot to the hidden state so that the workspace can be seen clearly
            train.go_to(train.hidden)

            # Blocking call that asks user to rearrange the items in the workspace
            train.set_workspace_state()

            # Get the poses of the items after the user has rearranged the items
            workspace_state = train.get_workspace_state()

            # Get the robot state to move to for item grabbing from the Neural Network
            robot_state = train.get_action(workspace_state)
            # robot_state = TD3.select_action(workspace_state) (JK)

            # Go to the robot pose
            train.go_to(robot_state)

            # After the robot has moved to neural net output position, grab and lift the gripper
            train.grab_and_lift()

            # Get the reward of the action
            reward = train.get_reward()

            # Match the image with the provided pose and resulting reward and save it
            train.save_data(workspace_state, robot_state, reward)

    # If there is an exception with the loop, notify the user
    except Exception as e:
        print("\nTrain script failed: %r" % (e,))

    # If there is a Keyboard Interrupt, gracefully shut down the Node
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
