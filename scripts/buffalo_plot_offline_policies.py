#!/usr/bin/env python3

"""
This script loads a number of policies and trained reward model and plots the actions on the reward function for comparison
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
from recycRL import RecycRL, A2P, NRMDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot(
    objectives=["PAR", "PAR", "PAR"],
    rl_paths=['~/Buffalo/Alpha=0.0', '~/Buffalo/Alpha=0.5', '~/Buffalo/Alpha=1.0'],
    reward_model_path="~/Buffalo",
    betas=[0.0, 0.50, 1.0],
    seed=0,
    deterministic=True,
    degree=6,
    coef_range=2,
    max_val=10,
    predefined_polynomial=1,
):
    # Expand the user to handle "~"
    reward_model_path = os.path.expanduser(reward_model_path)
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

        # Define the minimum and maximum action values based on shoulder values
        min_action = [env.unwrapped.left_shoulder]
        max_action = [env.unwrapped.right_shoulder]

        # Initialize the reward model Class
        reward_model = RewardModel(state_dim=1, action_dim=1)

        # If the reward model exists
        if os.path.exists(f"{reward_model_path}/Reward_Model"):
            # Load the reward model
            reward_model.load(reward_model_path)

        # If the reward model does not exist, notify the user and return
        elif not os.path.exists(f"{reward_model_path}/Reward_Model"):
            logger.error("Reward model does not exist, provide correct path")
            return

        # Define list to hold actions of each loaded policy and labels
        actions = []
        labels = []

        # Loop through each rl path
        for i, rl_path in enumerate(rl_paths):
            # If the the current model is for the PAR objective
            if objectives[i] == "PAR":
                # Initialize the RecycRL Class
                rl = RecycRL(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path=reward_model_path)

                # Define the search path
                search_path = rl_path + "/Policy"

            # If the the current model is for the A2P objective
            elif objectives[i] == "A2P":
                # Initialize the A2P Class
                rl = A2P(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path=reward_model_path)

                # Define the search path
                search_path = rl_path + "/Policy_A2P"

            # If the the current model is for the A2P objective
            elif objectives[i] == "NRMDP":
                # Initialize the A2P Class
                rl = NRMDP(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path=reward_model_path)

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

        # Get the reward model from the environment and define minimum and maximum action values based on shoulders
        true_reward_model = env.unwrapped.reward_model
        min_action = [env.unwrapped.left_shoulder]
        max_action = [env.unwrapped.right_shoulder]

        # Plot reward comparisons
        fig, ax = plt.subplots()
        xs = np.linspace(min_action, max_action, 1000)
        ys = true_reward_model(xs)
        plt.plot(xs, ys, label="True")
        means, stds = reward_model.reward_model.predict(
            torch.zeros(1000, device=device).unsqueeze(1), torch.tensor(xs, dtype=torch.float32, device=device)
        )
        for i, beta in enumerate(betas):
            rewards = (means - beta * stds).detach().cpu().numpy()
            plt.plot(xs, rewards, label=f"Approximate; β={beta}")
        with open(f"{'/'.join(rl_path.split('/')[:-1])}/log.txt", "w") as f:
            f.write("Experiments\n")
        for i, action in enumerate(actions):
            action = np.array([action])
            true_reward = env.unwrapped.reward_model(action)
            with open(f"{'/'.join(rl_path.split('/')[:-1])}/log.txt", "a") as f:
                f.write(f"{labels[i]}\n")
                f.write(f"True Reward: {true_reward.item()}\n")
            mean, std = reward_model.reward_model.predict(
                torch.tensor(np.array([[0]]), device=device), torch.tensor(action, device=device)
            )
            approx_reward = (mean - betas[i] * std).detach().cpu().numpy()
            plt.scatter(action, approx_reward, label=f"{labels[i]}")
            with open(f"{'/'.join(rl_path.split('/')[:-1])}/log.txt", "a") as f:
                f.write(f"Aproximate Reward: {approx_reward.item()}\n")
        plt.xlabel("Action")
        plt.ylabel("Expected Reward")
        plt.title("Expected Reward Comparisons")
        plt.legend()
        legend = ax.legend(loc="center", fontsize=12)
        legend.remove()
        plt.grid()
        plt.savefig(f"{'/'.join(rl_path.split('/')[:-1])}/plot.png")
        plt.show()
        fig_legend = plt.figure(figsize=(4, 2))
        fig_legend.legend(*ax.get_legend_handles_labels(), loc="center", fontsize=12)
        fig_legend.savefig(f"{'/'.join(rl_path.split('/')[:-1])}/legend.png", bbox_inches="tight")

    # If there is an exception with the loop
    except Exception as e:
        print("\nPlot script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    plot()
