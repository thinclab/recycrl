#!/usr/bin/env python3

"""
This script provides the ability to easily reset the workspace state and a provide a hand-guided robot pose
for the corresponding state. The transition and reward is recorded and all data is saved to the replay buffer
"""

import os
import numpy as np
from argparse import ArgumentParser
from recycrl.scripts.recycRL import RecycRL


def main():
    # Define arguments
    description = "Script to simplify the data collection process for the replay buffer"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-buffer_path",
        dest="buffer_path",
        default="~/Replay_Buffer.npy",
        help="Path to save the replay buffer with demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="8", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")

    # Parse and assign arguments
    args = parser.parse_args()
    buffer_path = args.buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)

    # Expand the user to handle "~"
    buffer_path = os.path.expanduser(buffer_path)

    # Try the following
    try:
        # Initialize the Imitate Node
        imitate = RecycRL(buffer_path, state_dim, action_dim)

        # Move the robot to the hidden state so that the workspace can be seen clearly
        imitate.go_to(imitate.hidden)

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

            # After the robot state has been recorded, grab and lift the gripper
            imitate.grab_and_lift()

            # Get the poses of the items after executing the action
            next_state = imitate.get_workspace_state()

            # Get the reward of the action
            reward, done = imitate.get_reward(state[-1], next_state[-1])

            # Save the state, action, transition, and reward to the replay buffer
            imitate.save_to_buffer(state, action, next_state, reward, done)

            # Assign the next state to the current state for the next iteration, more efficient
            state = next_state

            # Every 10 new samples added to the buffer
            if imitate.buffer.size % 10 == 0:
                # Save the replay buffer
                np.save(buffer_path, imitate.buffer)

    # If there is an exception with the loop, notify the user
    except Exception as e:
        # Save the replay buffer
        np.save(buffer_path, imitate.buffer)

        print("\nImitate script failed: %r" % (e,))

    # If there is a Keyboard Interrupt, gracefully shut down the Node
    except KeyboardInterrupt:
        # Save the replay buffer
        np.save(buffer_path, imitate.buffer)

        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
