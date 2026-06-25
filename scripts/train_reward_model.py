#!/usr/bin/env python3

"""
This script gives the training sequence for the reward model after sufficient expert and online data has been collected
"""

import os
import torch
import rclpy
import random
import numpy as np
from time import time
from utils import Utility
from argparse import ArgumentParser
from rclpy.logging import get_logger
from reward_model import RewardModel


def main():
    # Define arguments
    description = "Training script for the reward model"
    parser = ArgumentParser(description=description)
    parser.add_argument("-training_steps", dest="training_steps", default="50000", help="Number of training steps")
    parser.add_argument("-network_amount", dest="network_amount", default="5", help="Number of networks for reward model")
    parser.add_argument(
        "-model_path",
        dest="model_path",
        default="~/RecycRL/PAR; α=0.5",
        help="Path to save the reward model",
    )
    parser.add_argument(
        "-explore_buffer_path",
        dest="explore_buffer_path",
        default="~/RecycRL/Explore_Buffer",
        help="Path to load and save the replay buffer with online policy demonstrations",
    )
    parser.add_argument(
        "-expert_buffer_path",
        dest="expert_buffer_path",
        default="~/RecycRL/Expert_Buffer",
        help="Path to load and save the replay buffer with expert demonstrations",
    )
    parser.add_argument(
        "-online_buffer_path",
        dest="online_buffer_path",
        default="~/RecycRL/PAR; α=0.5/Online_Buffer",
        help="Path to load and save the replay buffer with online policy demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="4", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument("-explore_batch_size", dest="explore_batch_size", default="136", help="Batch size for explore buffer")
    parser.add_argument("-expert_batch_size", dest="expert_batch_size", default="64", help="Batch size for expert buffer")
    parser.add_argument("-online_batch_size", dest="online_batch_size", default="32", help="Batch size for online buffer")
    parser.add_argument("-seed", dest="seed", default="0", help="")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="")

    # Parse and assign arguments
    args = parser.parse_args()
    training_steps = int(args.training_steps)
    network_amount = int(args.network_amount)
    model_path = args.model_path
    explore_buffer_path = args.explore_buffer_path
    expert_buffer_path = args.expert_buffer_path
    online_buffer_path = args.online_buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
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

    # Expand the user to handle "~"
    model_path = os.path.expanduser(model_path)
    explore_buffer_path = os.path.expanduser(explore_buffer_path)
    expert_buffer_path = os.path.expanduser(expert_buffer_path)
    online_buffer_path = os.path.expanduser(online_buffer_path)

    # Define loggers
    logger = get_logger("train")

    # If the explore replay buffer does not exist, notify the user and return
    if not os.path.exists(explore_buffer_path + ".npy"):
        logger.error("Explore replay buffer does not exist, provide correct path")
        return

    # If the explore replay buffer exists, load the Class
    else:
        explore_buffer = np.load(explore_buffer_path + ".npy", allow_pickle=True).item()

    # If the expert replay buffer does not exist, notify the user ad return
    if not os.path.exists(expert_buffer_path + ".npy"):
        logger.error("Expert replay buffer does not exist, provide correct path")
        return

    # If the expert replay buffer exists, load the Class
    else:
        expert_buffer = np.load(expert_buffer_path + ".npy", allow_pickle=True).item()

    # Define lists for buffers and batch sizes
    buffers = [explore_buffer, expert_buffer]
    batch_sizes = [explore_batch_size, expert_batch_size]

    # Logging for buffer sizes
    logger.info("Explore buffer ready with size of " + str(explore_buffer.size))
    logger.info("Expert buffer ready with size of " + str(expert_buffer.size))

    # If the online replay buffer does not exist, notify the user
    if not os.path.exists(online_buffer_path + ".npy"):
        logger.warn("Online replay buffer does not exist, will not be used")

    # If the online replay buffer exists, load the Class, add buffer and batch size to lists, and log
    else:
        online_buffer = np.load(online_buffer_path + ".npy", allow_pickle=True).item()
        buffers.append(online_buffer)
        batch_sizes.append(online_batch_size)
        logger.info("Online buffer ready with size of " + str(online_buffer.size))

    # Initialize rclpy
    rclpy.init()

    # Initialize the RewardModel Class
    reward_model = RewardModel(state_dim, action_dim, network_amount)

    # If the reward model has been saved previously
    if os.path.exists(f"{model_path}/Reward_Model"):
        # Load the reward model
        reward_model.load(model_path)

    # If the reward model does not exist, create the save directory
    elif not os.path.exists(f"{model_path}/Reward_Model"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Logging for reward model
    logger.info("Reward model ready with iteration step of " + str(reward_model.total_it))

    # Try the following
    try:
        # Initialize the RecycRL Node
        train = Utility(active=False)

        # Start the training loop for the specified number of training steps
        for _ in range(training_steps):
            # Sample a combined batch from the buffers
            batch = train.sample_from_buffers(buffers, batch_sizes)

            # Train the reward model with the sampled batch
            reward_model.train(batch)

            # If the training steps is divisible by 1000
            if reward_model.total_it % 1000 == 0:
                # Save the reward model incrementally
                reward_model.save(model_path)

        # Save the reward model after exiting the loop
        reward_model.save(model_path)

    # If there is an exception with the loop
    except Exception as e:
        print("\nTrain script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
