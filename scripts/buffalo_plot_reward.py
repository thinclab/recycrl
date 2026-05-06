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
from ast import literal_eval
import matplotlib.pyplot as plt
from buffalo_gym import buffalo_gym  # noqa: F401
from argparse import ArgumentParser
from rclpy.logging import get_logger
from reward_model import RewardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot(
    model_path="~/Buffalo",
    network_amount=5,
    betas=[0.0, 0.25, 0.50, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
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
    model_path = os.path.expanduser(model_path)

    # Get the logger
    logger = get_logger("plot")

    # Make directories if necessary and initialize buffers and reward model
    reward_model = RewardModel(state_dim, action_dim, network_amount)

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
            std_deviation=std_deviation,
            coef_range=coef_range,
            max_val=max_val,
            shoulders=shoulders,
            shoulder_leakage=shoulder_leakage,
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
    # Define arguments and parse
    description = "This script loads a trained reward function and plots it with varying values of beta"
    parser = ArgumentParser(description=description)
    parser.add_argument("-model_path", dest="model_path", default="~/Buffalo", help="Path to save trained reward model")
    parser.add_argument("-network_amount", dest="network_amount", default="5", help="Number of networks for reward model")
    parser.add_argument("-betas", dest="betas", default="[0.0, 0.50, 1.0, 1.5, 2.0, 2.5, 3.0]", help="")
    parser.add_argument("-seed", dest="seed", default="20", help="")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="")
    parser.add_argument("-degree", dest="degree", default="6", help="")
    parser.add_argument("-std_deviation", dest="std_deviation", default="0.1", help="")
    parser.add_argument("-coef_range", dest="coef_range", default="2", help="")
    parser.add_argument("-max_val", dest="max_val", default="10", help="")
    parser.add_argument("-shoulders", dest="shoulders", default="True", help="")
    parser.add_argument("-shoulder_leakage", dest="shoulder_leakage", default="0.1", help="")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="")
    parser.add_argument("-state_dim", dest="state_dim", default="1", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="1", help="Dimension size of action")
    args = parser.parse_args()

    # Call plot function
    plot(
        model_path=args.model_path,
        network_amount=int(args.network_amount),
        betas=literal_eval(args.betas),
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
