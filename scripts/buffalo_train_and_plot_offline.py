#!/usr/bin/env python3

"""
This scripts automatically trains a reward model on randomly generated samples, then
trains a number of policies on the reward model, and finally plots the results for comparison
"""

import os
import torch
import random
import numpy as np
from time import time
from ast import literal_eval
from argparse import ArgumentParser
from buffalo_train_offline_policy import train
from buffalo_plot_reward import plot as plot_reward
from buffalo_collect_and_train_reward import collect_and_train
from buffalo_plot_offline_policies import plot as plot_policies

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    num_samples=200,
    reward_training_steps=20000,
    policy_training_steps=400,
    objectives=["PAR"],
    path="~/Buffalo",
    batch_size=100,
    alpha=0.5,
    betas=[1.0],
    initial_action=None,
    seed=0,
    deterministic=True,
    degree=6,
    coef_range=2,
    max_val=10,
    predefined_polynomial=1,
    step=[0.01],
    max_delta=[0.25],
    holes=[[0.0, 0.05]],
):
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

    # Expand the user to handle "~"
    path = os.path.expanduser(path)

    # Generate path based on function
    if predefined_polynomial:
        function = predefined_polynomial
    else:
        function = f"Random_{int(time())}"
    path = f"{path}/{function}"

    # Define empty list for RL paths
    rl_paths = []

    # Generate rl_paths
    for i, objective in enumerate(objectives):
        if objective == "PAR":
            objective = f"PAR; α={alpha}; β={betas[i]}"
        elif objective == "A2P":
            objective = f"A2P; β={betas[i]}"
        elif objective == "NRMDP":
            objective = f"NRMDP; β={betas[i]}"
        rl_paths.append(f"{path}/{objective}")

    # Try the following
    try:
        # Collect random samples for environment and train reward function
        collect_and_train(
            num_samples=num_samples,
            training_steps=reward_training_steps,
            explore_buffer_path=f"{path}/Explore_Buffer",
            model_path=path,
            batch_size=batch_size,
            seed=seed,
            deterministic=deterministic,
            degree=degree,
            coef_range=coef_range,
            max_val=max_val,
            predefined_polynomial=predefined_polynomial,
            holes=holes,
        )

        # Plot the approximate reward function with varying beta values against the true reward function
        plot_reward(
            model_path=path,
            betas=betas,
            seed=seed,
            deterministic=deterministic,
            degree=degree,
            coef_range=coef_range,
            max_val=max_val,
            predefined_polynomial=predefined_polynomial,
        )

        # Loop through each policy
        for i, objective in enumerate(objectives):
            # Train each policy
            train(
                training_steps=policy_training_steps,
                objective=objective,
                rl_path=rl_paths[i],
                reward_model_path=path,
                seed=seed,
                deterministic=deterministic,
                degree=degree,
                coef_range=coef_range,
                max_val=max_val,
                predefined_polynomial=predefined_polynomial,
                alpha=alpha,
                beta=betas[i],
                initial_action=initial_action,
                step=step,
                max_delta=max_delta,
            )

        # PLot the converged policies from training with the true and approximate reward functions
        plot_policies(
            objectives=objectives,
            rl_paths=rl_paths,
            reward_model_path=path,
            betas=betas,
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
    description = "This scripts automatically trains a reward model, then trains a number of policies, and plots the results"
    parser = ArgumentParser(description=description)
    parser.add_argument("-num_samples", dest="num_samples", default="200", help="Number of random samples to generate")
    parser.add_argument(
        "-reward_training_steps", dest="reward_training_steps", default="20000", help="Number of training steps for reward model"
    )
    parser.add_argument(
        "-policy_training_steps", dest="policy_training_steps", default="400", help="Number of training steps for policy"
    )
    parser.add_argument(
        "-objectives",
        dest="objectives",
        default="['PAR', 'PAR', 'PAR', 'PAR', 'PAR', 'PAR']",
        help="List of objectives for policy ('PAR', 'A2P', 'NRMDP')",
    )
    parser.add_argument(
        "-path",
        dest="path",
        default="~/Buffalo/Offline",
        help="Paths to load the RL models or actors",
    )
    parser.add_argument("-batch_size", dest="batch_size", default="100", help="Batch size for sampling from buffer")
    parser.add_argument("-alpha", dest="alpha", default="0.5", help="Coefficient for robustness term")
    parser.add_argument("-betas", dest="betas", default="[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]", help="Coefficients for reward model uncertainty")
    parser.add_argument("-initial_action", dest="initial_action", default=None, help="Sets bias in NN for first action")
    parser.add_argument("-seed", dest="seed", default="0", help="Seed for randomization and function generation")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="Turn deterministic optimization off")
    parser.add_argument("-degree", dest="degree", default="6", help="Power of generated function")
    parser.add_argument("-coef_range", dest="coef_range", default="20", help="Range for generating coefficients for function")
    parser.add_argument("-max_val", dest="max_val", default="10", help="Maximum value for function")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="Number of function to load")
    parser.add_argument("-step", dest="step", default="[0.01]", help="")
    parser.add_argument("-max_delta", dest="max_delta", default="[0.25]", help="")
    parser.add_argument("-holes", dest="holes", default="[[0.0, 0.05]]", help="")
    args = parser.parse_args()

    # Call training function
    main(
        num_samples=int(args.num_samples),
        reward_training_steps=int(args.reward_training_steps),
        policy_training_steps=int(args.policy_training_steps),
        objectives=literal_eval(args.objectives),
        path=args.path,
        batch_size=int(args.batch_size),
        alpha=float(args.alpha),
        betas=literal_eval(args.betas),
        initial_action=None if not args.initial_action else float(args.initial_action),
        seed=int(args.seed),
        deterministic=not bool(args.not_deterministic),
        degree=int(args.degree),
        coef_range=float(args.coef_range),
        max_val=float(args.max_val),
        predefined_polynomial=int(args.predefined_polynomial),
        step=literal_eval(args.step),
        max_delta=literal_eval(args.max_delta),
        holes=literal_eval(args.holes),
    )
