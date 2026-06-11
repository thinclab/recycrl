#!/usr/bin/env python3

"""
This script loads a number of policies and corresponding reward model and plots the actions on the reward function for comparison
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
from recycRL import RecycRL, A2P, NRMDP


def plot(
    objectives=["PAR", "PAR", "PAR"],
    rl_paths=["~/Buffalo/5/Alpha=0.0", "~/Buffalo/5/Alpha=0.5", "~/Buffalo/5/Alpha=1.0"],
    seed=0,
    deterministic=True,
    degree=6,
    coef_range=2,
    max_val=10,
    predefined_polynomial=1,
):
    # Expand the user to handle "~"
    rl_paths = [os.path.expanduser(rl_path) for rl_path in rl_paths]

    # Define logger
    logger = get_logger("plot")

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

        # DEfine the minimum and maximum action values based on shoulder values
        min_action = [env.unwrapped.left_shoulder]
        max_action = [env.unwrapped.right_shoulder]

        # Define list to hold actions of each loaded policy and labels
        actions = []
        labels = []

        # Loop through each rl path
        for i, rl_path in enumerate(rl_paths):
            # If the the current model is for the PAR objective
            if objectives[i] == "PAR":
                # Initialize the RecycRL Class
                rl = RecycRL(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path="")

                # Define the search path
                search_path = rl_path + "/Policy"

            # If the the current model is for the A2P objective
            elif objectives[i] == "A2P":
                # Initialize the A2P Class
                rl = A2P(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path="")

                # Define the search path
                search_path = rl_path + "/Policy_A2P"

            # If the the current model is for the NRMDP objective
            elif objectives[i] == "NRMDP":
                # Initialize the NRMDP Class
                rl = NRMDP(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path="")

                # Define the search path
                search_path = rl_path + "/Policy_NRMDP"

            # If the current model is for the invalid, notify the user and return
            elif objectives[i] != "PAR" and objectives[i] != "A2P" and objectives[i] != "NRMDP":
                logger.error("One or more objectives invalid")
                return

            # If the RL model has been saved previously, load it, get the action, and add it to the list
            if os.path.exists(search_path):
                rl.load(rl_path)
                action = rl.select_action([0])
                actions.append(action)
                label = rl_path.split("/")[-1]
                labels.append(label)

            # If one of the RL models does not exist, notify the user and return
            elif not os.path.exists(search_path):
                print(search_path)
                logger.error("One or more RL models does not exist")
                return

        # Plot the policies
        x = np.linspace(min_action, max_action, 1000)
        y = env.unwrapped.reward_model(x)
        plt.plot(x, y)
        for i, action in enumerate(actions):
            reward = env.unwrapped.reward_model(np.array([action]))
            plt.scatter(action, reward, label=labels[i])
        plt.xlabel("Action")
        plt.ylabel("Reward")
        plt.title("Reward Polynomial")
        plt.legend()
        plt.grid()
        plt.savefig(f"{'/'.join(rl_path.split('/')[:-1])}/plot.png")
        plt.show()
        plt.close()

    # If there is an exception with the loop
    except Exception as e:
        print("\nPlot script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    plot()
