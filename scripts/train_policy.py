#!/usr/bin/env python3

"""
This script gives the training sequence for the policy after a reward model has been trained
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
from recycRL import RecycRL, A2P, NRMDP


def main():
    # Define arguments
    description = "Node that goes through the RL training process"
    parser = ArgumentParser(description=description)
    parser.add_argument("-training_steps", dest="training_steps", default="2000", help="Number of training steps")
    parser.add_argument("-alpha", dest="alpha", default="0.0", help="Robustness coefficient")
    parser.add_argument("-beta", dest="beta", default="2.0", help="Uncertainty coefficient")
    parser.add_argument(
        "-objective",
        dest="objective",
        default="PAR",
        help="Objective to optimize the policy ('PAR', 'A2P', 'NRMDP')",
    )
    parser.add_argument(
        "-path",
        dest="path",
        default="~/RecycRL",
        help="Path to save the RL model or actor",
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
    parser.add_argument("-explore_batch_size", dest="explore_batch_size", default="0", help="Batch size for explore buffer")
    parser.add_argument("-expert_batch_size", dest="expert_batch_size", default="200", help="Batch size for expert buffer")
    parser.add_argument("-online_batch_size", dest="online_batch_size", default="0", help="Batch size for online buffer")
    parser.add_argument("-seed", dest="seed", default="0", help="")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="")

    # Parse and assign arguments
    args = parser.parse_args()
    training_steps = int(args.training_steps)
    alpha = float(args.alpha)
    beta = float(args.beta)
    objective = args.objective
    path = args.path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)
    explore_batch_size = int(args.explore_batch_size)
    expert_batch_size = int(args.expert_batch_size)
    online_batch_size = int(args.online_batch_size)
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
    reward_model_path = os.path.expanduser(path)
    # reward_model_path = rl_path
    explore_buffer_path = os.path.expanduser(path + "/Explore_Buffer.npy")
    expert_buffer_path = os.path.expanduser(path + "/Expert_Buffer.npy")
    online_buffer_path = os.path.expanduser(rl_path + "/Online_Buffer.npy")

    # Define loggers
    logger = get_logger("train")

    # If the explore replay buffer does not exist, notify the user
    if not os.path.exists(explore_buffer_path):
        logger.error("Explore replay buffer does not exist, provide correct path")
        return

    # If the explore replay buffer exists, load the Class
    else:
        explore_buffer = np.load(explore_buffer_path, allow_pickle=True).item()

    # If the expert replay buffer does not exist, notify the user
    if not os.path.exists(expert_buffer_path):
        logger.error("Expert replay buffer does not exist, provide correct path")
        return

    # If the expert replay buffer exists, load the Class
    else:
        expert_buffer = np.load(expert_buffer_path, allow_pickle=True).item()

    # Define lists for buffers and batch sizes
    buffers = [explore_buffer, expert_buffer]
    batch_sizes = [explore_batch_size, expert_batch_size]

    # Logging for buffer sizes
    logger.info("Explore buffer ready with size of " + str(explore_buffer.size))
    logger.info("Expert buffer ready with size of " + str(expert_buffer.size))

    # If the online replay buffer does not exist, notify the user
    if not os.path.exists(online_buffer_path):
        logger.warn("Online replay buffer does not exist, will not be used")

    # If the online replay buffer exists, load the Class, add buffer and batch size to lists, and log
    else:
        online_buffer = np.load(online_buffer_path, allow_pickle=True).item()
        buffers.append(online_buffer)
        batch_sizes.append(online_batch_size)
        logger.info("Online buffer ready with size of " + str(online_buffer.size))

    # Initialize rclpy
    rclpy.init()

    # If the user wants to optimize with the PAR objective
    if objective == "PAR":
        # Initialize the RecycRL Class
        rl = RecycRL(state_dim, action_dim, min_action, max_action, model_path=reward_model_path)

        # Define the search path and training args
        search_path = rl_path + "/Policy"
        train_args = {"alpha": alpha, "beta": beta}

    # If the user wants to optimize with the A2P objective
    elif objective == "A2P":
        # Initialize the A2P Class
        rl = A2P(state_dim, action_dim, min_action, max_action, model_path=reward_model_path)

        # Define the search path and training args
        search_path = rl_path + "/Policy_A2P"
        train_args = {"alpha": alpha, "beta": beta, "gamma": 0.10}

    # If the user wants to optimize with the A2P objective
    elif objective == "NRMDP":
        # Initialize the A2P Class
        rl = NRMDP(state_dim, action_dim, min_action, max_action, model_path=reward_model_path)

        # Define the search path and training args
        search_path = rl_path + "/Policy_NRMDP"
        train_args = {"beta": beta}

    # If the user provides and invalid objective, notify the user and return
    elif objective != "PAR" and objective != "REINFORCE" and objective != "A2P":
        logger.error("Objective invalid")
        return

    # for p1 in rl.actor.parameters():
    #     print(p1.sum().item())

    # If the RL model has been saved previously, load it
    if os.path.exists(search_path):
        rl.load(rl_path)

    # If the RL model does not exist, create the save directory
    elif not os.path.exists(search_path):
        os.makedirs(os.path.dirname(search_path), exist_ok=True)

    # for p1 in rl.actor.parameters():
    #     print(p1.sum().item())

    # If the reward model in the policy was not loaded correctly, notify the user and return
    if rl.reward_model.total_it == 0:
        logger.error("Reward model does not exist, provide correct path")
        return

    # If the reward model in the policy was loaded correctly
    if rl.reward_model.total_it != 0:
        logger.info("Reward model ready")

    # Logging for policy
    logger.info("Policy ready with iteration step of " + str(rl.total_it))

    # Try the following
    try:
        # Initialize the Utility Node
        train = Utility(active=False)

        # Start the training loop for the specified number of training steps
        for _ in range(training_steps):
            # Sample a combined batch from the buffers
            batch = train.sample_from_buffers(buffers, batch_sizes)

            # Train the RL model
            rl.train(batch, **train_args)

            # If the training steps is divisible by 1000
            if rl.total_it % 1000 == 0:
                # Save the RL model incrementally
                rl.save(rl_path)
                print("Saved Policy\n")

        # Save the RL model after exiting the loop
        rl.save(rl_path)
        print("Saved Policy\n")

    # If there is an exception with the loop
    except Exception as e:
        print("\nTrain script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
