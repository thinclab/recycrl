#!/usr/bin/env python3

"""
This script automatically trains a number of policies and then plots the results for comparison
"""

import torch
import random
import numpy as np
from time import time
from ast import literal_eval
from argparse import ArgumentParser
from rclpy.logging import get_logger
from buffalo_train_online_policy import train
from buffalo_plot_online_policies import plot


def main(
    objectives=["PAR"],
    rl_paths=["~/Buffalo"],
    training_steps=2000,
    seed=0,
    deterministic=True,
    degree=6,
    std_deviation=0.1,
    coef_range=2,
    max_val=10,
    shoulders=True,
    shoulder_leakage=0.1,
    predefined_polynomial=1,
    alphas=[0.75],
    step=[0.01],
    max_delta=[0.25],
    state_dim=1,
    action_dim=1,
    model_path=""
):
    # Define logger
    logger = get_logger("test")

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
        # Set GPU/CUDA to the correponding seed for deterministic results
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Try the following
    try:
        # Loop through each policy
        for i, rl_path in enumerate(rl_paths):
            # Train each policy
            train(
                training_steps=training_steps,
                objective=objectives[i],
                rl_path=rl_path,
                seed=seed,
                deterministic=deterministic,
                degree=degree,
                std_deviation=std_deviation,
                coef_range=coef_range,
                max_val=max_val,
                shoulders=shoulders,
                shoulder_leakage=shoulder_leakage,
                predefined_polynomial=predefined_polynomial,
                binary_reward=False,
                alpha=alphas[i],
                step=step,
                max_delta=max_delta,
                state_dim=state_dim,
                action_dim=action_dim,
                model_path=model_path
            )
            logger.info(f"{rl_path} finished training\n")

        # Plot the trained policies on the reward model
        plot(
            objectives=objectives,
            rl_paths=rl_paths,
            seed=seed,
            deterministic=deterministic,
            degree=degree,
            coef_range=coef_range,
            max_val=max_val,
            shoulders=shoulders,
            shoulder_leakage=shoulder_leakage,
            predefined_polynomial=predefined_polynomial,
            state_dim=state_dim,
            action_dim=action_dim,
            model_path=model_path
        )

    # If there is an exception with the loop
    except Exception as e:
        print("\nTest script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    # Define arguments and parse
    description = "This script automatically trains a number of policies and then plots the results for comparison"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-objectives",
        dest="objectives",
        default="['PAR', 'PAR', 'PAR', 'PAR', 'PAR', 'PAR']",
        help="List of objectives for policy ('PAR', 'REINFORCE', 'A2P')",
    )
    parser.add_argument(
        "-rl_paths",
        dest="rl_paths",
        default="['~/Buffalo/5/Alpha=0.0', '~/Buffalo/5/Alpha=0.5', '~/Buffalo/5/Alpha=1.0', '~/Buffalo/5/Alpha=1.5', '~/Buffalo/5/Alpha=2.0', '~/Buffalo/5/Eta_Only']",
        help="Paths to load the RL models or actors",
    )
    parser.add_argument("-training_steps", dest="training_steps", default="2000", help="Number of training steps")
    parser.add_argument("-seed", dest="seed", default="20", help="")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="")
    parser.add_argument("-degree", dest="degree", default="6", help="")
    parser.add_argument("-std_deviation", dest="std_deviation", default="0.1", help="")
    parser.add_argument("-coef_range", dest="coef_range", default="10", help="")
    parser.add_argument("-max_val", dest="max_val", default="10", help="")
    parser.add_argument("-shoulders", dest="shoulders", default="True", help="")
    parser.add_argument("-shoulder_leakage", dest="shoulder_leakage", default="0.1", help="")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="")
    parser.add_argument("-alphas", dest="alphas", default="[0.0, 0.5, 1.0, 1.5, 2.0, 1000.0]", help="")
    parser.add_argument("-step", dest="step", default="[0.01]", help="")
    parser.add_argument("-max_delta", dest="max_delta", default="[0.25]", help="")
    parser.add_argument("-state_dim", dest="state_dim", default="1", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="1", help="Dimension size of action")
    parser.add_argument("-model_path", dest="model_path", default="", help="")
    args = parser.parse_args()

    # Call the main function
    main(
        objectives=literal_eval(args.objectives),
        rl_paths=literal_eval(args.rl_paths),
        training_steps=int(args.training_steps),
        seed=int(args.seed),
        deterministic=not bool(args.not_deterministic),
        degree=int(args.degree),
        std_deviation=float(args.std_deviation),
        coef_range=float(args.coef_range),
        max_val=float(args.max_val),
        shoulders=bool(args.shoulders),
        shoulder_leakage=float(args.shoulder_leakage),
        predefined_polynomial=literal_eval(args.predefined_polynomial),
        alphas=literal_eval(args.alphas),
        step=literal_eval(args.step),
        max_delta=literal_eval(args.max_delta),
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
        model_path=args.model_path
    )
