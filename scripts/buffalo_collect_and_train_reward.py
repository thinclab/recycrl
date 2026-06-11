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
from rclpy.logging import get_logger
from reward_model import RewardModel
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_and_train(
    num_samples=200,
    training_steps=20000,
    explore_buffer_path="~/Buffalo/Explore_Buffer",
    model_path="~/Buffalo",
    batch_size=100,
    holes=[[0.0, 0.05]],
    seed=0,
    deterministic=True,
    degree=6,
    coef_range=2,
    max_val=10,
    predefined_polynomial=1,
):
    # Expand the user to handle "~"
    explore_buffer_path = os.path.expanduser(explore_buffer_path)
    model_path = os.path.expanduser(model_path)

    # Get the logger
    logger = get_logger("collect_and_train")

    # Make directories if necessary and initialize buffers and reward model
    os.makedirs(os.path.dirname(explore_buffer_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    explore_buffer = ReplayBuffer(state_dim=1, action_dim=1, max_size=int(1e6))
    reward_model = RewardModel(state_dim=1, action_dim=1, network_amount=5)

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
            coef_range=coef_range,
            max_val=max_val,
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

            # Check that the action does not fall within any holes, and if it does, resample until it does not
            while any(low < action < high for low, high in holes):
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
    collect_and_train()
