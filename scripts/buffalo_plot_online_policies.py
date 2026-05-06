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
from ast import literal_eval
import matplotlib.pyplot as plt
from buffalo_gym import buffalo_gym  # noqa: F401
from argparse import ArgumentParser
from rclpy.logging import get_logger
from recycRL import RecycRL, REINFORCE, A2P


def plot(
    objectives=["PAR"],
    rl_paths=["~/Buffalo"],
    seed=0,
    deterministic=True,
    degree=-1,
    coef_range=1.0,
    max_val=10,
    shoulders=True,
    shoulder_leakage=0.1,
    predefined_polynomial=1,
    state_dim=1,
    action_dim=1,
    model_path="",
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
            shoulders=shoulders,
            shoulder_leakage=shoulder_leakage,
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
                rl = RecycRL(state_dim, action_dim, min_action, max_action, model_path=model_path)

                # Define the search path
                search_path = rl_path + "/Policy"

            # If the the current model is for the REINFORCE objective
            elif objectives[i] == "REINFORCE":
                # Initialize the REINFORCE Class
                rl = REINFORCE(action_dim, min_action, max_action)

                # Define the search path
                search_path = rl_path + "/Policy_REINFORCE"

            # If the the current model is for the A2P objective
            elif objectives[i] == "A2P":
                # Initialize the A2P Class
                rl = A2P(action_dim, min_action, max_action)

                # Define the search path
                search_path = rl_path + "/Policy_A2P"

            # If the current model is for the invalid, notify the user and return
            elif objectives[i] != "PAR" and objectives[i] != "REINFORCE" and objectives[i] != "A2P":
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
            plt.scatter(action, env.unwrapped.reward_model(np.array([action])), label=labels[i])
        plt.xlabel("Action")
        plt.ylabel("Reward")
        plt.title("Reward Polynomial")
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
    description = "This script loads a number of policies and plots the actions on the reward function for comparison"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-objectives",
        dest="objectives",
        default="['PAR', 'PAR', 'PAR']",
        help="List of objectives for policy ('PAR', 'REINFORCE', 'A2P')",
    )
    parser.add_argument(
        "-rl_paths",
        dest="rl_paths",
        default="['~/Buffalo/5/Alpha=0.0', '~/Buffalo/5/Alpha=0.5', '~/Buffalo/5/Alpha=1.0']",
        help="Paths to load the RL models or actors",
    )
    parser.add_argument("-seed", dest="seed", default="20", help="Seed for randomization and function generation")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="Turn deterministic optimization off")
    parser.add_argument("-degree", dest="degree", default="6", help="Power of generated function")
    parser.add_argument("-coef_range", dest="coef_range", default="2", help="Range for generating coefficients for function")
    parser.add_argument("-max_val", dest="max_val", default="10", help="Maximum value for function")
    parser.add_argument("-shoulders", dest="shoulders", default="True", help="Determines if there are bounds for function")
    parser.add_argument("-shoulder_leakage", dest="shoulder_leakage", default="0.1", help="Scaling for actions outside of bounds")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="Number of function to load")
    parser.add_argument("-alpha", dest="alpha", default="1.0", help="Coefficient for robustness factor in optimization")
    parser.add_argument("-step", dest="step", default="[0.01]", help="Step size for robustness calculation")
    parser.add_argument("-max_delta", dest="max_delta", default="[0.25]", help="Integral bounds for robustness calculation")
    parser.add_argument("-state_dim", dest="state_dim", default="1", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="1", help="Dimension size of action")
    parser.add_argument("-model_path", dest="model_path", default="", help="Path to reward model")
    args = parser.parse_args()

    # Call the main function
    plot(
        objectives=literal_eval(args.objectives),
        rl_paths=literal_eval(args.rl_paths),
        seed=int(args.seed),
        deterministic=not bool(args.not_deterministic),
        degree=int(args.degree),
        coef_range=float(args.coef_range),
        max_val=float(args.max_val),
        shoulders=bool(args.shoulders),
        shoulder_leakage=float(args.shoulder_leakage),
        predefined_polynomial=int(args.predefined_polynomial),
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
        model_path=args.model_path,
    )
