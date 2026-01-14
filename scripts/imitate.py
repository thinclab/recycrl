#!/usr/bin/env python3

"""
This node provides the ability to easily reset the workspace state and a provide a hand-guided robot pose
for the corresponding state. The transition and reward is recorded and all data is saved to the replay buffer
"""

import os
import rclpy
from recycRL import RecycRL
from argparse import ArgumentParser


def main():
    # Define arguments
    description = "Node to simplify the data collection process for the replay buffer"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-buffer_path",
        dest="buffer_path",
        default="~/RD3/Replay_Buffer.npy",
        help="Path to save the replay buffer with demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="8", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="7", help="Dimension size of action")

    # Parse and assign arguments
    args = parser.parse_args()
    buffer_path = args.buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)

    # Expand the user to handle "~"
    buffer_path = os.path.expanduser(buffer_path)

    # Initialize rclpy
    rclpy.init()

    # Try the following
    try:
        # Initialize the Imitate Node
        imitate = RecycRL(buffer_path, state_dim, action_dim)

        # Move the robot to the hidden state so that the workspace can be seen clearly
        imitate.go_to(imitate.bin)

        # Open the gripper to so that it is ready to grab an item
        imitate.open_gripper()

        # Get the initial poses of the items in the workspace
        state = imitate.get_workspace_state()

        # Start the loop
        while True:
            # If the workspace is empty
            if state[-1] == 0:
                # Wait for the user to arrange items in the workspace
                imitate.set_workspace_state()

                # Get the poses of the items after the user has rearranged the items
                state = imitate.get_workspace_state()

            # Move the robot to home
            imitate.go_to(imitate.home)

            # Deactivate the KUKA robot's external control to allow operator to hand guide robot
            imitate.deactivate_external_control()

            # Wait for the operator to move the robot to the desired pose
            imitate.set_robot_state()

            # Once the operator has provided a pose, reactivate the robot's external control
            imitate.activate_external_control()

            # Get the current state of the robot for the action now that external control is active
            action = imitate.get_robot_state()

            # After the robot state has been recorded, close the gripper, lift, and go to the bin
            imitate.grab_and_go_to_bin()

            # Get the poses of the items after executing the action
            next_state = imitate.get_workspace_state()

            # Get the reward of the action
            reward, done = imitate.get_reward(state[-1], next_state[-1])

            # After getting the reward, open the gripper
            imitate.open_gripper()

            # Save the state, action, transition, and reward to the replay buffer
            imitate.save_to_buffer(state, action, next_state, reward, done)

            # Assign the next state to the current state for the next iteration, more efficient
            state = next_state

    # If there is an exception with the loop, notify the user
    except Exception as e:
        print("\nImitate script failed: %r" % (e,))

    # If there is a Keyboard Interrupt, gracefully shut down the Node
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
