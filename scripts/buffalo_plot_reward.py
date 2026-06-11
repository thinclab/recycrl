#!/usr/bin/env python3

"""
This script loads a trained reward function and plots it with varying values of beta
"""

import os
import torch
import random
import numpy as np
from time import time
import gymnasium as gym
import matplotlib.pyplot as plt
from buffalo_gym import buffalo_gym  # noqa: F401
from rclpy.logging import get_logger
from reward_model import RewardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot(
    model_path="~/Buffalo",
    betas=[0.0, 1.0, 2.0],
    seed=0,
    deterministic=True,
    degree=6,
    coef_range=2,
    max_val=10,
    predefined_polynomial=1,
):
    # Expand the user to handle "~"
    model_path = os.path.expanduser(model_path)

    # Get the logger
    logger = get_logger("plot")

    # Make directories if necessary and initialize buffers and reward model
    reward_model = RewardModel(state_dim=1, action_dim=1, network_amount=5)

    # Load the reward model
    reward_model.load(model_path)

    # Log that reward model is ready
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
        # Make the environment
        env = gym.make(
            "BoundlessBuffalo-v0",
            seed=seed,
            degree=degree,
            coef_range=coef_range,
            max_val=max_val,
            predefined_polynomial=predefined_polynomial,
        )

        # Get the reward model from the environment and define minimum and maximum action values based on shoulders
        true_reward_model = env.unwrapped.reward_model
        min_action = [env.unwrapped.left_shoulder]
        max_action = [env.unwrapped.right_shoulder]

        # Plot reward comparisons
        xs = np.linspace(min_action, max_action, 1000)
        ys = true_reward_model(xs)
        plt.plot(xs, ys, label="True")
        means, stds = reward_model.reward_model.predict(
            torch.zeros(1000, device=device).unsqueeze(1), torch.tensor(xs, dtype=torch.float32, device=device)
        )
        for beta in betas:
            rewards = (means - beta * stds).detach().cpu().numpy()
            plt.plot(xs, rewards, label=f"Approximate; β={beta}")
        plt.xlabel("Action")
        plt.ylabel("Expected Reward")
        plt.title("Expected Reward Comparisons")
        plt.legend()
        plt.grid()
        plt.show()

    # If there is an exception with the loop
    except Exception as e:
        print("\nPlot script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    plot()
