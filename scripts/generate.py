#!/usr/bin/env python3

"""
This node samples from the expert buffer and generates a number of random trajectories to fill the online buffer efficiently
"""

# generate.py: Create a new script to generate random trajectories for the online buffer efficiently

import os
import rclpy
import numpy as np
from math import degrees
from recycRL import RecycRL
from ast import literal_eval
from TD3.utils import ReplayBuffer
from argparse import ArgumentParser
from rclpy.logging import get_logger
from tf_transformations import euler_from_quaternion


def main():
    # Define arguments
    description = "Node to simplify the random data collection process for the online replay buffer"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-expert_buffer_path",
        dest="expert_buffer_path",
        default="~/RD3/Expert_Buffer",
        help="Path to load and save the replay buffer with expert demonstrations",
    )
    parser.add_argument(
        "-online_buffer_path",
        dest="online_buffer_path",
        default="~/RD3/Online_Buffer",
        help="Path to load and save the replay buffer with online policy demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="9", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument(
        "-min_action",
        dest="min_action",
        default="[0.2, -0.35, 0.81, -0.7853981634, -0.7853981634, -1.570796327]",
        help="Minimum action values vector for x, y, and z position outputs",
    )
    parser.add_argument(
        "-max_action",
        dest="max_action",
        default="[0.55, 0.35, 0.93, 0.7853981634, 0.7853981634, 1.570796327]",
        help="Maximum action values vector for x, y, and z position outputs",
    )
    parser.add_argument("-generation_amount", dest="generation_amount", default="5", help="Number of generated samples per state")

    # Parse and assign arguments
    args = parser.parse_args()
    expert_buffer_path = args.expert_buffer_path
    online_buffer_path = args.online_buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)
    generation_amount = int(args.generation_amount)

    # Expand the user to handle "~"
    expert_buffer_path = os.path.expanduser(expert_buffer_path)
    online_buffer_path = os.path.expanduser(online_buffer_path)

    # Get the logger
    logger = get_logger("generate")

    # If the expert replay buffer does not exist, notify the user
    if not os.path.exists(expert_buffer_path + ".npy"):
        logger.warn("Expert replay buffer does not exist, provide correct path")
        return

    # If the expert replay buffer exists, load the Class
    else:
        expert_buffer = np.load(expert_buffer_path + ".npy", allow_pickle=True).item()

    # If the online replay buffer does not exist, initialize a new ReplayBuffer() Class
    if not os.path.exists(online_buffer_path + ".npy"):
        os.makedirs(os.path.dirname(online_buffer_path), exist_ok=True)
        online_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    # If the online replay buffer exists, load the Class
    else:
        online_buffer = np.load(online_buffer_path + ".npy", allow_pickle=True).item()

    # Logging for buffer sizes
    logger.info("Expert buffer ready with size of " + str(expert_buffer.size))
    logger.info("Online buffer ready with size of " + str(online_buffer.size))

    # Initialize rclpy
    rclpy.init()

    # Try the following
    try:
        # Initialize the RecycRL Node
        generate = RecycRL()

        # Move the robot to the bin position so that the workspace can be seen clearly
        generate.go_to(generate.bin)

        # Open the gripper to so that it is ready to grab an item
        generate.open_gripper()

        # Set 'run' to True initially to start the loop
        run = True

        # Start the loop
        while run:
            # Sample one state from the expert buffer and convert it to a numpy array
            state, _, _, _, _ = expert_buffer.sample(1)
            state = state.cpu().detach().numpy().flatten()

            # Convert the quaternion to pitch and yaw angles and define a list for printing the sampled state
            _, pitch, yaw = euler_from_quaternion([state[3], state[4], state[5], state[6]])
            print_state = [state[0], state[1], state[2], round(degrees(pitch), 3), round(degrees(yaw), 3), state[7], state[8]]

            # Print the state to recreate
            generate.get_logger().warn(f"Sampled state: {print_state}\n")

            # Loop for the number of times specified by 'generation_amount'
            for i in range(0, generation_amount):
                # Select a random action to take
                action = generate.select_random_action(min_action, max_action)

                # Log the action to execute
                generate.get_logger().warn("Action " + str(list(action)))

                # Convert the action from its 3-angle representation into a quaternion
                converted_action = generate.convert_action(action)

                # Check that the action has a possibility of succeeding
                valid, _, _ = generate.distance_check(state, converted_action)

                # If the action is not valid
                if not valid:
                    # There is no change in state due to failed action, so next state is same as current state
                    next_state = state

                # If the action is valid
                elif valid:
                    # Print the state to recreate
                    generate.get_logger().warn(f"Recreate the following state: {print_state}")

                    # Get the actual state of the workspace and retry until it matches the sampled state
                    state = generate.get_workspace_state()

                    # Move the robot to home
                    generate.go_to(generate.home)

                    # Go to the robot pose defined by the action
                    valid, approached, executed, radius, length = generate.execute_action(state, action)

                    # After the robot has moved to the position, close the gripper, lift, and go to the bin
                    generate.grab_and_go_to_bin()

                    # Get the poses of the items after the action has been executed
                    next_state = generate.get_workspace_state()

                # Get the reward of the action
                reward, done = generate.get_reward(state, action, next_state)

                # After getting the reward, open the gripper
                generate.open_gripper()

                # Add the state, action, transition, and reward to the replay buffer
                generate.add_to_buffer(online_buffer, state, action, next_state, reward, done, False)

                # Assign the next state to the current state for the next iteration, more efficient
                state = next_state

            # Save the online buffer
            generate.save_buffer(online_buffer, online_buffer_path)

            # Check if the loop should continue
            run = generate.loop_check()

        # Save the online buffer after exiting the loop
        generate.save_buffer(online_buffer, online_buffer_path)

    # If there is an exception with the loop
    except Exception as e:
        print("\nGenerate script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
