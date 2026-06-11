#!/usr/bin/env python3

"""
This script loads a number of policies and logs the rewards for the base actions and noisy actions for comparison
"""

import os
import torch
import random
import numpy as np
from time import time
import gymnasium as gym
from buffalo_gym import buffalo_gym  # noqa: F401
from rclpy.logging import get_logger
from recycRL import RecycRL, A2P, NRMDP


def evaluate(
    num_samples=1000,
    max_noise=0.25,
    objectives=["PAR", "PAR", "PAR", "PAR", "PAR", "A2P", "NRMDP"],
    rl_paths=[
        "~/Buffalo/1/0.2/PAR; α=0.0",
        "~/Buffalo/1/0.2/PAR; α=0.35",
        "~/Buffalo/1/0.2/PAR; α=0.5",
        "~/Buffalo/1/0.2/PAR; α=0.65",
        "~/Buffalo/1/0.2/PAR; α=1.0",
        "~/Buffalo/1/0.2/A2P",
        "~/Buffalo/1/0.2/NRMDP",
    ],
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

        # Define the minimum and maximum action values based on shoulder values
        min_action = [env.unwrapped.left_shoulder]
        max_action = [env.unwrapped.right_shoulder]

        # Define lists
        actions = []
        rewards = []
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
                print(objectives[i])
                logger.error("One or more objectives invalid")
                return

            # If the RL model has been saved previously, load it, get the action, and add it to the list
            if os.path.exists(search_path):
                rl.load(rl_path)
                action = rl.select_action([0])
                actions.append(action)
                rewards.append(env.unwrapped.reward_model(np.array([action])).item())
                labels.append(rl_path.split("/")[-1])

            # If one of the RL models does not exist, notify the user and return
            elif not os.path.exists(search_path):
                print(search_path)
                logger.error("One or more RL models does not exist")
                return

        # Define empty lists for storage
        samples = []
        noisy_rewards_list = []

        # Generate noise samples for each action
        samples.append(np.random.normal(0, max_noise / 3, num_samples).clip(-max_noise, max_noise))
        samples.append(np.random.normal(0, max_noise / 2, num_samples).clip(-max_noise, max_noise))
        samples.append(np.random.uniform(-max_noise, max_noise, num_samples))

        # Add preliminary data to the log file
        with open(f"{'/'.join(rl_paths[0].split('/')[:-1])}/log.txt", "w") as f:
            f.write("Experiments\n")

        # Loop through each sample of noise
        for sample in samples:
            # Add the noise to the actions
            noisy_actions_list = [(action + sample).clip(min_action, max_action) for action in actions]

            # Get the average reward for the noisy rewards of the base action
            noisy_rewards = [
                env.unwrapped.reward_model(torch.tensor(noisy_actions).unsqueeze(1)).mean().item()
                for noisy_actions in noisy_actions_list
            ]

            # Append the noisy rewards to the list
            noisy_rewards_list.append(noisy_rewards)

        # Log the results
        with open(f"{'/'.join(rl_paths[0].split('/')[:-1])}/log.txt", "a") as f:
            for j, rl_path in enumerate(rl_paths):
                f.write(f"{rl_path.split('/')[-1]}\n")
                f.write(f"Reward: {rewards[j]}\n")
                for i in range(len(samples)):
                    f.write(f"Noise {i+1}: {noisy_rewards_list[i][j]}\n")

    # If there is an exception with the loop
    except Exception as e:
        print("\nEvaluate script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    evaluate()
