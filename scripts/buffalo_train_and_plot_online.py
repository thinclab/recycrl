#!/usr/bin/env python3

"""
This script automatically trains a number of policies and then plots the results for comparison
"""

import os
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
    path="~/Buffalo/Online",
    training_steps=400,
    initial_action=None,
    seed=0,
    deterministic=True,
    degree=6,
    coef_range=2,
    max_val=10,
    predefined_polynomial=1,
    alphas=[0.75],
    step=[0.01],
    max_delta=[0.25],
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
        # Set GPU/CUDA to the corresponding seed for deterministic results
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Expand user to handle "~"
    path = os.path.expanduser(path)

    # Determine function path
    if predefined_polynomial:
        function = predefined_polynomial
    else:
        function = f"Random_{int(time())}"

    # Determine path
    path = f"{path}/{function}/{initial_action}"

    # Define empty list for RL paths
    rl_paths = []

    # Generate rl_paths
    for i, objective in enumerate(objectives):
        if objective == "PAR":
            objective = f"PAR; α={alphas[i]}"
        elif objective == "A2P":
            objective = f"A2P"
        elif objective == "NRMDP":
            objective = f"NRMDP"
        rl_paths.append(f"{path}/{objective}")

    # Try the following
    try:
        # Loop through each policy
        for i, rl_path in enumerate(rl_paths):
            # Train each policy
            train(
                training_steps=training_steps,
                objective=objectives[i],
                rl_path=rl_path,
                initial_action=initial_action,
                seed=seed,
                deterministic=deterministic,
                degree=degree,
                coef_range=coef_range,
                max_val=max_val,
                predefined_polynomial=predefined_polynomial,
                alpha=alphas[i],
                step=step,
                max_delta=max_delta,
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
            predefined_polynomial=predefined_polynomial,
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
        default="['PAR', 'PAR', 'PAR', 'PAR', 'PAR', 'A2P', 'NRMDP']",
        help="List of objectives for policy ('PAR', 'REINFORCE', 'A2P', 'NRMDP')",
    )
    parser.add_argument(
        "-path",
        dest="path",
        default="~/Buffalo/Online",
        help="Path to load the RL models or actors",
    )
    parser.add_argument("-training_steps", dest="training_steps", default="400", help="Number of training steps")
    parser.add_argument("-initial_action", dest="initial_action", default=None, help="Sets bias in NN for first action")
    parser.add_argument("-seed", dest="seed", default="0", help="")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="")
    parser.add_argument("-degree", dest="degree", default="6", help="")
    parser.add_argument("-coef_range", dest="coef_range", default="10", help="")
    parser.add_argument("-max_val", dest="max_val", default="10", help="")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="")
    parser.add_argument("-alphas", dest="alphas", default="[0.0, 0.35, 0.5, 0.65, 1.0, 0.0, 0.0]", help="")
    parser.add_argument("-step", dest="step", default="[0.01]", help="")
    parser.add_argument("-max_delta", dest="max_delta", default="[0.25]", help="")
    args = parser.parse_args()

    # Call the main function
    main(
        objectives=literal_eval(args.objectives),
        path=args.path,
        training_steps=int(args.training_steps),
        initial_action=None if not args.initial_action else float(args.initial_action),
        seed=int(args.seed),
        deterministic=not bool(args.not_deterministic),
        degree=int(args.degree),
        coef_range=float(args.coef_range),
        max_val=float(args.max_val),
        predefined_polynomial=literal_eval(args.predefined_polynomial),
        alphas=literal_eval(args.alphas),
        step=literal_eval(args.step),
        max_delta=literal_eval(args.max_delta),
    )
