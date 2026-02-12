#!/usr/bin/env python3

"""
This node loads the trained policy, executes the policy, and adds to the online buffer
"""

import os
import rclpy
import numpy as np
from utils import Utility
from ast import literal_eval
from argparse import ArgumentParser
from rclpy.logging import get_logger
from recycRL import RecycRL
from replay_buffer import ReplayBuffer


def main():
    # Define arguments
    description = ""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-rl_path",
        dest="rl_path",
        default="~/RecycRL/recycrl",
        help="Path to save and load the RL model or actor",
    )
    parser.add_argument(
        "-online_buffer_path",
        dest="online_buffer_path",
        default="~/RecycRL/Online_Buffer",
        help="Path to load and save the replay buffer with online policy demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="4", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument(
        "-min_action",
        dest="min_action",
        default="[-0.10, -0.10, -0.01, -0.7853981634, -0.7853981634, -0.7853981634]",
        help="Minimum action values vector for x, y, and z position outputs",
    )
    parser.add_argument(
        "-max_action",
        dest="max_action",
        default="[0.10, 0.10, 0.10, 0.7853981634, 0.7853981634, 0.7853981634]",
        help="Maximum action values vector for x, y, and z position outputs",
    )
    parser.add_argument(
        "-expl_noise",
        dest="expl_noise",
        default="[0.02, 0.02, 0.02, 0.10, 0.10, 0.10]",
        help="Exploration noise standard deviation",
    )
    parser.add_argument(
        "-noise_clip",
        dest="noise_clip",
        default="[0.04, 0.04, 0.04, 0.20, 0.20, 0.20]",
        help="Maximum noise value",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    rl_path = args.rl_path
    online_buffer_path = args.online_buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)
    expl_noise = literal_eval(args.expl_noise)
    noise_clip = literal_eval(args.noise_clip)

    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)
    online_buffer_path = os.path.expanduser(online_buffer_path)

    # Define loggers
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

    # Initialize the RecycRL Class
    rl = RecycRL(state_dim, action_dim, min_action, max_action, expl_noise, noise_clip)

    # If the RL model has been saved previously
    if os.path.exists(f"{rl_path}_actor"):
        # Load the RL model
        rl.load(rl_path)

    elif not os.path.exists(f"{rl_path}_actor"):
        os.makedirs(os.path.dirname(rl_path), exist_ok=True)

    # Try the following
    try:
        # Initialize the Utility Node
        collect = Utility()

        # Move the robot to the bin position so that the workspace can be seen clearly
        collect.go_to(collect.bin)

        # Open the gripper to so that it is ready to grab an item
        collect.open_gripper()

        # Get the initial poses of the items in the workspace
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

            # Pass the state through the actor network to get the action, add noise for exploration
            action = rl.select_action(network_state, add_noise=True)

            # Go to the robot pose defined by the action
            executed, action = collect.execute_action(actual_state, action, penalize=False, train=True)

            # If the robot successfully moved to the desired pose
            if executed:
                # After the robot has moved output position, close the gripper, lift, and go to the bin
                collect.grab_and_go_to_bin()

                # Get the poses of the items after the action has been executed
                next_network_state, next_actual_state = collect.get_workspace_state()

                # Get the reward of the action
                reward = collect.get_reward(actual_state, next_actual_state)

            # If the robot could not reach the desired pose
            elif not executed:
                # There is no change in state due to failed action, so next state is same as current state
                next_network_state, next_actual_state = network_state, actual_state

                # Reward of failed action is 0
                reward = 0
                collect.get_logger().warn("Reward: 0")

            # Open the gripper to so that it is ready to grab an item
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
