#!/usr/bin/env python3

"""
This node provides a loop to collect random data for the online replay buffer
"""

import os
import rclpy
import numpy as np
from recycRL import RecycRL
from ast import literal_eval
from TD3.utils import ReplayBuffer
from argparse import ArgumentParser
from rclpy.logging import get_logger


def main():
    # Define arguments
    description = "Node to collect random data for the online replay buffer"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-online_buffer_path",
        dest="online_buffer_path",
        default="~/RD3/Online_Buffer",
        help="Path to load and save the replay buffer with online policy demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="4", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument(
        "-min_action",
        dest="min_action",
        default="[-0.1, -0.1, -0.01, -0.7853981634, -0.7853981634, -0.7853981634]",
        help="Minimum action values vector for x, y, and z position outputs",
    )
    parser.add_argument(
        "-max_action",
        dest="max_action",
        default="[0.1, 0.1, 0.1, 0.7853981634, 0.7853981634, 0.7853981634]",
        help="Maximum action values vector for x, y, and z position outputs",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    online_buffer_path = args.online_buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)

    # Expand the user to handle "~"
    online_buffer_path = os.path.expanduser(online_buffer_path)

    # Get the logger
    logger = get_logger("collect")

    # If the online replay buffer does not exist, initialize a new ReplayBuffer() Class
    if not os.path.exists(online_buffer_path + ".npy"):
        os.makedirs(os.path.dirname(online_buffer_path), exist_ok=True)
        online_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    # If the online replay buffer exists, load the Class
    else:
        online_buffer = np.load(online_buffer_path + ".npy", allow_pickle=True).item()

    # Logging for buffer sizes
    logger.info("Online buffer ready with size of " + str(online_buffer.size))

    # Initialize rclpy
    rclpy.init()

    # Try the following
    try:
        # Initialize the RecycRL Node
        collect = RecycRL()

        # Move the robot to the bin position so that the workspace can be seen clearly
        collect.go_to(collect.bin)

        # Open the gripper to so that it is ready to grab an item
        collect.open_gripper()

        # Get the current state of the workspace
        network_state, actual_state = collect.get_workspace_state()

        # Set 'run' to True initially to start the loop
        run = True

        # Start the loop
        while run:
            # If the workspace is empty
            if actual_state[-1] == 0:
                # Wait for the user to arrange items in the workspace
                collect.set_workspace_state()

                # Get the poses of the items after the user has rearranged the items
                network_state, actual_state = collect.get_workspace_state()

            # Move the robot to home
            collect.go_to(collect.home)

            # Select a random action to take
            action = collect.select_random_action(min_action, max_action)

            # Execute the action
            executed, action = collect.execute_action(actual_state, action, penalize=False, train=True)

            # If the action was not executed
            if not executed:
                # There is no change in state due to failed action, so next state is same as current state
                next_network_state, next_actual_state = network_state, actual_state

                # Reward of failed action is 0
                reward = 0
                collect.get_logger().warn(f"Reward: 0")

            # If the action was executed
            elif executed:
                # After the robot has moved to the position, close the gripper, lift, and go to the bin
                collect.grab_and_go_to_bin()

                # Get the poses of the items after the action has been executed
                next_network_state, next_actual_state = collect.get_workspace_state()

                # Get the reward of the action
                reward = collect.get_reward(actual_state, next_actual_state)

            # After getting the reward, open the gripper
            collect.open_gripper()

            # Add the state, action, transition, and reward to the replay buffer
            collect.add_to_buffer(online_buffer, network_state, action, reward, check=False)

            # Assign the next state to the current state for the next iteration, more efficient
            network_state, actual_state = next_network_state, next_actual_state

            # Save the online buffer
            collect.save_buffer(online_buffer, online_buffer_path)

            # Check if the loop should continue
            run = collect.loop_check()

        # Save the online buffer after exiting the loop
        collect.save_buffer(online_buffer, online_buffer_path)

    # If there is an exception with the loop
    except Exception as e:
        print("\nCollect script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
