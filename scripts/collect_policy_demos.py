#!/usr/bin/env python3

"""
This node loads the trained policy, executes the policy, and adds to the online buffer
"""

import os
import rclpy
import torch
import random
import numpy as np
from time import time
from utils import Utility
from ast import literal_eval
from argparse import ArgumentParser
from rclpy.logging import get_logger
from replay_buffer import ReplayBuffer
from recycRL import RecycRL, A2P, NRMDP


def main():
    # Define arguments
    description = ""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-path",
        dest="path",
        default="~/RecycRL",
        help="Path to save and load the RL model or actor",
    )
    parser.add_argument(
        "-objective",
        dest="objective",
        default="PAR",
        help="Objective used to optimize the policy",
    )
    parser.add_argument("-alpha", dest="alpha", default="1.0", help="Robustness coefficient")
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
        default="[0.03, 0.03, 0.03, 0.15, 0.15, 0.15]",
        help="Maximum noise value",
    )
    parser.add_argument("-seed", dest="seed", default="0", help="")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="")

    # Parse and assign arguments
    args = parser.parse_args()
    path = args.path
    alpha = float(args.alpha)
    objective = args.objective
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)
    expl_noise = literal_eval(args.expl_noise)
    noise_clip = literal_eval(args.noise_clip)
    seed = int(args.seed)
    deterministic = not bool(args.not_deterministic)

    # If seed is -1
    if seed == -1:
        # Get a random seed
        seed = int(time())

    # Define seeds for torch, numpy, and python randomness generators
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # If deterministic is set to True
    if deterministic:
        # Set GPU/CUDA to the corresponding seed for deterministic results
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Define paths and expand the user to handle "~"
    if objective == "PAR":
        rl_path = os.path.expanduser(path + f"/PAR; α={alpha}")
    elif objective != "PAR":
        rl_path = os.path.expanduser(path + f"/{objective}")
    online_buffer_path = os.path.expanduser(rl_path + "/Online_Buffer")

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

    # If the user wants to optimize with the PAR objective
    if objective == "PAR":
        # Initialize the RecycRL Class
        rl = RecycRL(state_dim, action_dim, min_action, max_action, expl_noise, noise_clip)

        # Define the search path
        search_path = rl_path + "/Policy"

    elif objective == "A2P":
        # Initialize the A2P Class
        rl = A2P(state_dim, action_dim, min_action, max_action, expl_noise, noise_clip)

        # Define the search path
        search_path = rl_path + "/Policy_A2P"

    # If the user wants to optimize with the NRMDP objective
    elif objective == "NRMDP":
        # Initialize the NRMDP Class
        rl = NRMDP(state_dim, action_dim, min_action, max_action, expl_noise, noise_clip)

        # Define the search path
        search_path = rl_path + "/Policy_NRMDP"

    # If the user provides and invalid objective, notify the user and return
    elif objective != "PAR" and objective != "A2P" and objective != "NRMDP":
        logger.error("Objective invalid")
        return

    # If the RL model has been saved previously, load the model
    if os.path.exists(search_path):
        rl.load(rl_path)
        logger.info(f"Policy from '{rl_path}' ready")

    # If the RL model does not exist, notify the user and return
    elif not os.path.exists(search_path):
        print(search_path)
        logger.error("Policy does not exist, provide correct path")
        return

    # Try the following
    try:
        # Initialize the Utility Node
        collect = Utility(active=True)

        # Move the robot to the bin position so that the workspace can be seen clearly
        collect.go_to(collect.bin)

        # Open the gripper to so that it is ready to grab an item
        collect.open_gripper()

        # Get the initial poses of the items in the workspace
        network_state, actual_state = collect.get_workspace_state(check=True)

        # Set 'run' to True initially to start the loop
        run = True

        # Start the loop
        while run:
            # If the workspace is empty
            if actual_state[-1] == 0:
                # Wait for the user to arrange items in the workspace
                collect.set_workspace_state()

                # Get the poses of the items after the user has rearranged the items
                network_state, actual_state = collect.get_workspace_state(check=True)

            # Move the robot to home
            collect.go_to(collect.home)

            # Pass the state through the actor network to get the action, add noise for exploration
            action = rl.select_action(network_state, add_noise=False)
            # action = collect.add_noise(action, expl_noise)

            # print("Action after noise:", action)

            # Go to the robot pose defined by the action
            executed, action = collect.execute_action(actual_state, action, penalize=False, train=True)

            # If the robot successfully moved to the desired pose
            if executed:
                # After the robot has moved output position, close the gripper, lift, and go to the bin
                collect.grab_and_go_to_bin()

                # Get the poses of the items after the action has been executed
                next_network_state, next_actual_state = collect.get_workspace_state(check=True)

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

            # Get the number of times the last state consecutively appeared in the buffer
            collect.get_last_state_amount(online_buffer)

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
