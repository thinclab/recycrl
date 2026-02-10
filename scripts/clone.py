#!/usr/bin/env python3

"""
This node provides the ability to easily collect expert demos for the same state quicker than 'imitate.py'. Instead,
of grabbing, getting the reward, and obtaining the next state, this script assumes the following: the state remains the same,
there is one item in the workspace, the grasp will be successful and therefore the reward will be 1 and the next state will be an
empty workspace. This allows multiple demos for same state to be collected quickly
"""

import os
import rclpy
import numpy as np
from recycRL import RecycRL
from TD3.utils import ReplayBuffer
from argparse import ArgumentParser
from rclpy.logging import get_logger


def main():
    # Define arguments
    description = "Node to simplify the expert data collection process for the same state"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-buffer_path",
        dest="buffer_path",
        default="~/RD3/Expert_Buffer",
        help="Path to load and save the expert replay buffer with demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="4", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument(
        "-clone_amount", dest="clone_amount", default="5", help="Number of demonstrations to collect for the same state"
    )

    # Parse and assign arguments
    args = parser.parse_args()
    buffer_path = args.buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    clone_amount = int(args.clone_amount)

    # Expand the user to handle "~"
    buffer_path = os.path.expanduser(buffer_path)

    # If the expert replay buffer does not exist, initialize a new ReplayBuffer() Class
    if not os.path.exists(buffer_path + ".npy"):
        os.makedirs(os.path.dirname(buffer_path), exist_ok=True)
        buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    # If the replay buffer exists, load the Class
    else:
        buffer = np.load(buffer_path + ".npy", allow_pickle=True).item()

    # Logging for buffer size
    logger = get_logger("clone")
    logger.info("Buffer ready with size of " + str(buffer.size))

    # Initialize rclpy
    rclpy.init()

    # Try the following
    try:
        # Initialize the Clone Node
        clone = RecycRL()

        # Set 'run' to True initially to start the loop
        run = True

        # Start the loop
        while run:
            # Move the robot to the hidden state so that the workspace can be seen clearly
            clone.go_to(clone.bin)

            # Open the gripper to so that it is ready to grab an item
            clone.open_gripper()

            # Get the state of the workspace (this will remain constant)
            network_state, actual_state = clone.get_workspace_state()

            # Move the robot to home
            clone.go_to(clone.home)

            for _ in range(clone_amount):
                # Deactivate the KUKA robot's external control to allow operator to hand guide robot
                clone.deactivate_external_control()

                # Wait for the operator to move the robot to the desired pose
                clone.set_robot_state()

                # Once the operator has provided a pose, reactivate the robot's external control
                clone.activate_external_control()

                # Get the current state of the robot for the action now that external control is active
                action = clone.get_robot_state(actual_state)

                # Save the state, action to the buffer; fill in 'next state', 'reward', and 'done' values for successful action
                clone.add_to_buffer(buffer, network_state, action, 1, False)

                # Save the buffer
                clone.save_buffer(buffer, buffer_path)

                # Check if the loop should continue
                run = clone.loop_check()

            # Move the robot to home
            clone.go_to(clone.home)

        # Save the buffer after exiting the loop
        clone.save_buffer(buffer, buffer_path)

    # If there is an exception with the loop, notify the user
    except Exception as e:
        print("\nClone script failed: %r" % (e,))

    # If there is a Keyboard Interrupt, gracefully shut down the Node
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
