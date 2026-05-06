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
from ast import literal_eval
from buffalo_gym import buffalo_gym  # noqa: F401
from argparse import ArgumentParser
from rclpy.logging import get_logger
from recycRL import RecycRL, REINFORCE, A2P


def train(
    training_steps=2000,
    objective="PAR",
    rl_path="~/Buffalo",
    reward_model_path="",
    seed=0,
    deterministic=True,
    degree=6,
    std_deviation=0.1,
    coef_range=2,
    max_val=10,
    shoulders=True,
    shoulder_leakage=0.1,
    predefined_polynomial=1,
    alpha=0.75,
    beta=0.75,
    step=[0.01],
    max_delta=[0.25],
    state_dim=1,
    action_dim=1,
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
            std_deviation=std_deviation,
            coef_range=coef_range,
            max_val=max_val,
            shoulders=shoulders,
            shoulder_leakage=shoulder_leakage,
            predefined_polynomial=predefined_polynomial,
        )

        # Define minimum and maximum action values based on shoulders
        min_action = [env.unwrapped.left_shoulder]
        max_action = [env.unwrapped.right_shoulder]

        # If the user wants to optimize with the PAR objective
        if objective == "PAR":
            # Initialize the RecycRL Class
            rl = RecycRL(state_dim, action_dim, min_action, max_action, model_path=reward_model_path)

            # Define the search path
            search_path = rl_path + "/Policy"

        # If the user wants to optimize with the REINFORCE objective
        elif objective == "REINFORCE":
            # Initialize the REINFORCE Class
            rl = REINFORCE(action_dim, min_action, max_action)

            # Define the search path
            search_path = rl_path + "/Policy_REINFORCE"

        # If the user wants to optimize with the A2P objective
        elif objective == "A2P":
            # Initialize the A2P Class
            rl = A2P(action_dim, min_action, max_action)

            # Define the search path
            search_path = rl_path + "/Policy_A2P"

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
    # Define arguments and parse
    description = "This script trains a policy on the approximate reward model"
    parser = ArgumentParser(description=description)
    parser.add_argument("-training_steps", dest="training_steps", default="100000", help="Number of training steps")
    parser.add_argument(
        "-objective",
        dest="objective",
        default="PAR",
        help="Objective to optimize the policy ('PAR', 'REINFORCE', 'A2P')",
    )
    parser.add_argument(
        "-rl_path",
        dest="rl_path",
        default="~/Buffalo",
        help="Path to save the RL model or actor",
    )
    parser.add_argument("-reward_model_path", dest="reward_model_path", default="~/Buffalo", help="")
    parser.add_argument("-seed", dest="seed", default="20", help="")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="")
    parser.add_argument("-degree", dest="degree", default="6", help="")
    parser.add_argument("-std_deviation", dest="std_deviation", default="0.1", help="")
    parser.add_argument("-coef_range", dest="coef_range", default="2", help="")
    parser.add_argument("-max_val", dest="max_val", default="10", help="")
    parser.add_argument("-shoulders", dest="shoulders", default="True", help="")
    parser.add_argument("-shoulder_leakage", dest="shoulder_leakage", default="0.1", help="")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="")
    parser.add_argument("-alpha", dest="alpha", default="1.0", help="")
    parser.add_argument("-beta", dest="beta", default="1.0", help="")
    parser.add_argument("-step", dest="step", default="[0.01]", help="")
    parser.add_argument("-max_delta", dest="max_delta", default="[0.25]", help="")
    parser.add_argument("-state_dim", dest="state_dim", default="1", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="1", help="Dimension size of action")
    args = parser.parse_args()

    # Call training function
    train(
        training_steps=int(args.training_steps),
        objective=args.objective,
        rl_path=args.rl_path,
        reward_model_path=args.reward_model_path,
        seed=int(args.seed),
        deterministic=not bool(args.not_deterministic),
        degree=int(args.degree),
        std_deviation=float(args.std_deviation),
        coef_range=float(args.coef_range),
        max_val=float(args.max_val),
        shoulders=bool(args.shoulders),
        shoulder_leakage=float(args.shoulder_leakage),
        predefined_polynomial=int(args.predefined_polynomial),
        alpha=float(args.alpha),
        beta=float(args.beta),
        step=literal_eval(args.step),
        max_delta=literal_eval(args.max_delta),
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
    )
