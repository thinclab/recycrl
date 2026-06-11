#!/usr/bin/env python3

"""
This script trains a policy on the approximate reward model
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


def train(
    training_steps=2000,
    objective="PAR",
    rl_path="~/Buffalo",
    reward_model_path="~/Buffalo",
    initial_action=None,
    seed=0,
    deterministic=True,
    degree=6,
    coef_range=2,
    max_val=10,
    predefined_polynomial=1,
    alpha=0.5,
    beta=2.0,
    step=[0.01],
    max_delta=[0.25],
):
    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)
    reward_model_path = os.path.expanduser(reward_model_path)

    # Define loggers
    logger = get_logger("train")

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

        # Define minimum and maximum action values based on shoulders
        min_action = [env.unwrapped.left_shoulder]
        max_action = [env.unwrapped.right_shoulder]

        # If the user wants to optimize with the PAR objective
        if objective == "PAR":
            # Initialize the RecycRL Class
            rl = RecycRL(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path=reward_model_path, initial_action=initial_action)

            # Define the search path
            search_path = rl_path + "/Policy"

        # If the user wants to optimize with the A2P objective
        elif objective == "A2P":
            # Initialize the A2P Class
            rl = A2P(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path=reward_model_path, initial_action=initial_action)

            # Define the search path
            search_path = rl_path + "/Policy_A2P"

        # If the user wants to optimize with the NRMDP objective
        elif objective == "NRMDP":
            # Initialize the NRMDP Class
            rl = NRMDP(state_dim=1, action_dim=1, min_action=min_action, max_action=max_action, model_path=reward_model_path, initial_action=initial_action)

            # Define the search path
            search_path = rl_path + "/Policy_NRMDP"

        # If the user provides and invalid objective, notify the user and return
        elif objective != "PAR" and objective != "A2P" and objective != "NRMDP":
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

        # Logging for policy
        logger.info("Policy ready with iteration step of " + str(rl.total_it))

        # Start the training loop for the specified number of training steps
        for _ in range(training_steps):
            # Train the RL model
            rl.train_for_buffalo(online=False, alpha=alpha, beta=beta, step=step, max_delta=max_delta)

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
    train()
