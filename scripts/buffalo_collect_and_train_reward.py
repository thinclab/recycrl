#!/usr/bin/env python3

"""
This script generates random samples from the environment and trains a reward model on those samples
"""

import os
import rclpy
import torch
import random
import numpy as np
from time import time
import gymnasium as gym
from utils import Utility
from buffalo_gym import buffalo_gym  # noqa: F401
from argparse import ArgumentParser
from rclpy.logging import get_logger
from reward_model import RewardModel
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_and_train(
    num_samples=200,
    training_steps=20000,
    explore_buffer_path="~/Buffalo/Explore_Buffer",
    model_path="~/Buffalo",
    network_amount=5,
    batch_size=100,
    seed=0,
    deterministic=True,
    degree=6,
    std_deviation=0.1,
    coef_range=2,
    max_val=10,
    shoulders=True,
    shoulder_leakage=0.1,
    predefined_polynomial=1,
    state_dim=1,
    action_dim=1,
):
    # Expand the user to handle "~"
    explore_buffer_path = os.path.expanduser(explore_buffer_path)
    model_path = os.path.expanduser(model_path)

    # Get the logger
    logger = get_logger("collect_and_train")

    # Make directories if necessary and initialize buffers and reward model
    os.makedirs(os.path.dirname(explore_buffer_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    explore_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
    reward_model = RewardModel(state_dim, action_dim, network_amount)

    # Logging for buffer and reward model
    logger.info("Explore buffer ready with size of " + str(explore_buffer.size))
    logger.info("Reward model ready with iteration step of " + str(reward_model.total_it))

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

    # Try the following
    try:
        # Initialize rclpy and Utility Node
        rclpy.init()
        cat = Utility(active=False)

        # Make the environment
        env = gym.make(
            "BoundlessBuffalo-v0",
            seed=seed,
            degree=degree,
            std_deviation=std_deviation,
            coef_range=coef_range,
            max_val=max_val,
            shoulders=shoulders,
            shoulder_leakage=shoulder_leakage,
            predefined_polynomial=predefined_polynomial,
            binary_reward=True,
        )

        # Reset the environment
        env.reset()

        # Define minimum and maximum action values based on shoulders
        min_action = [env.unwrapped.left_shoulder]
        max_action = [env.unwrapped.right_shoulder]

        # # Loop through number of random samples to generate
        for _ in range(num_samples):
            # Get a random action
            action = cat.select_random_action(min_action, max_action)

            # Add while loop to get an action that is not within given bounds
            while 0.3 < action < 0.4 or 0.8 < action < 0.875 or 0.225 < action < 0.275 or 0.675 < action < 0.75:
                action = cat.select_random_action(min_action, max_action)

            # Step the environment with the sampled action
            obs, reward, done, term, info = env.step(action)

            # Add the state, action, transition, and reward to the replay buffer
            cat.add_to_buffer(explore_buffer, obs, action, reward, check=False, log=False)

        # Save the explore buffer
        cat.save_buffer(explore_buffer, explore_buffer_path)

        # Start the training loop for the specified number of training steps
        for _ in range(training_steps):
            # Sample a batch from the buffer
            batch = explore_buffer.sample(batch_size)

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
        print("\nCollect and Train script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    # Define arguments and parse
    description = "This script generates random samples from the environment and trains a reward model on those samples"
    parser = ArgumentParser(description=description)
    parser.add_argument("-num_samples", dest="num_samples", default="200", help="Number of random samples to generate")
    parser.add_argument("-training_steps", dest="training_steps", default="20000", help="Number of training steps")
    parser.add_argument(
        "-explore_buffer_path",
        dest="explore_buffer_path",
        default="~/Buffalo/Explore_Buffer",
        help="Path to load and save the replay buffer with random samples",
    )
    parser.add_argument("-model_path", dest="model_path", default="~/Buffalo", help="Path to save trained reward model")
    parser.add_argument("-network_amount", dest="network_amount", default="5", help="Number of networks for reward model")
    parser.add_argument("-batch_size", dest="batch_size", default="100", help="Batch size for sampling from buffer")
    parser.add_argument("-seed", dest="seed", default="20", help="Seed for randomization and function generation")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="Turn deterministic optimization off")
    parser.add_argument("-degree", dest="degree", default="6", help="Power of generated function")
    parser.add_argument("-std_deviation", dest="std_deviation", default="0.1", help="Standard deviation for noise in function")
    parser.add_argument("-coef_range", dest="coef_range", default="2", help="Range for generating coefficients for function")
    parser.add_argument("-max_val", dest="max_val", default="10", help="Maximum value for function")
    parser.add_argument("-shoulders", dest="shoulders", default="True", help="Determines if there are bounds for function")
    parser.add_argument("-shoulder_leakage", dest="shoulder_leakage", default="0.1", help="Scaling for actions outside of bounds")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="Number of function to load")
    parser.add_argument("-state_dim", dest="state_dim", default="1", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="1", help="Dimension size of action")
    args = parser.parse_args()

    # Call training function
    collect_and_train(
        num_samples=int(args.num_samples),
        training_steps=int(args.training_steps),
        explore_buffer_path=args.explore_buffer_path,
        model_path=args.model_path,
        network_amount=int(args.network_amount),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        deterministic=not bool(args.not_deterministic),
        degree=int(args.degree),
        std_deviation=float(args.std_deviation),
        coef_range=float(args.coef_range),
        max_val=float(args.max_val),
        shoulders=bool(args.shoulders),
        shoulder_leakage=float(args.shoulder_leakage),
        predefined_polynomial=int(args.predefined_polynomial),
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
    )
