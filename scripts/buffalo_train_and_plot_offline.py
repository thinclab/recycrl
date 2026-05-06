#!/usr/bin/env python3

"""
This scripts automatically trains a reward model on randomly generated samples, then
trains a number of policies on the reward model, and finally plots the results for comparison
"""

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
    policy_training_steps=2000,
    objectives=["PAR"],
    rl_paths=["~/Buffalo"],
    explore_buffer_path="~/Buffalo/Explore_Buffer",
    reward_model_path="~/Buffalo",
    network_amount=5,
    batch_size=100,
    betas=[0.0, 0.50, 1.0],
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
    step=[0.01],
    max_delta=[0.25],
    state_dim=1,
    action_dim=1,
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

    # Try the following
    try:
        # Collect random samples for environment and train reward function
        collect_and_train(
            num_samples=num_samples,
            training_steps=reward_training_steps,
            explore_buffer_path=explore_buffer_path,
            model_path=reward_model_path,
            network_amount=network_amount,
            batch_size=batch_size,
            seed=seed,
            deterministic=deterministic,
            degree=degree,
            std_deviation=std_deviation,
            coef_range=coef_range,
            max_val=max_val,
            shoulders=shoulders,
            shoulder_leakage=shoulder_leakage,
            predefined_polynomial=predefined_polynomial,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # Plot the approximate reward function with varying beta values against the true reward function
        plot_reward(
            model_path=reward_model_path,
            network_amount=network_amount,
            betas=betas,
            seed=seed,
            deterministic=deterministic,
            degree=degree,
            std_deviation=std_deviation,
            coef_range=coef_range,
            max_val=max_val,
            shoulders=shoulders,
            shoulder_leakage=shoulder_leakage,
            predefined_polynomial=predefined_polynomial,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # Loop through each policy
        for i, rl_path in enumerate(rl_paths):
            # Train each policy
            train(
                training_steps=policy_training_steps,
                objective=objectives[i],
                rl_path=rl_path,
                reward_model_path=reward_model_path,
                seed=seed,
                deterministic=deterministic,
                degree=degree,
                std_deviation=std_deviation,
                coef_range=coef_range,
                max_val=max_val,
                shoulders=shoulders,
                shoulder_leakage=shoulder_leakage,
                predefined_polynomial=predefined_polynomial,
                alpha=alpha,
                beta=betas[i],
                step=step,
                max_delta=max_delta,
                state_dim=state_dim,
                action_dim=action_dim,
            )

        # PLot the converged policies from training with the true and approximate reward functions
        plot_policies(
            objectives=objectives,
            rl_paths=rl_paths,
            reward_model_path=reward_model_path,
            betas=betas,
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
        "-policy_training_steps", dest="policy_training_steps", default="2000", help="Number of training steps for policy"
    )
    parser.add_argument(
        "-objectives",
        dest="objectives",
        default="['PAR', 'PAR', 'PAR']",
        help="List of objectives for policy ('PAR', 'REINFORCE', 'A2P')",
    )
    parser.add_argument(
        "-rl_paths",
        dest="rl_paths",
        default="['~/Buffalo/Alpha=1.0; Beta=0.0', '~/Buffalo/Alpha=1.0; Beta=1.0', '~/Buffalo/Alpha=1.0; Beta=2.0']",
        help="Paths to load the RL models or actors",
    )
    parser.add_argument(
        "-explore_buffer_path",
        dest="explore_buffer_path",
        default="~/Buffalo/Explore_Buffer",
        help="Path to load and save the replay buffer with random samples",
    )
    parser.add_argument(
        "-reward_model_path", dest="reward_model_path", default="~/Buffalo", help="Path to save trained reward model"
    )
    parser.add_argument("-network_amount", dest="network_amount", default="5", help="Number of networks for reward model")
    parser.add_argument("-batch_size", dest="batch_size", default="100", help="Batch size for sampling from buffer")
    parser.add_argument("-betas", dest="betas", default="[0.0, 1.0, 2.0]", help="Coefficients for reward model uncertainty")
    parser.add_argument("-seed", dest="seed", default="20", help="Seed for randomization and function generation")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="Turn deterministic optimization off")
    parser.add_argument("-degree", dest="degree", default="6", help="Power of generated function")
    parser.add_argument("-std_deviation", dest="std_deviation", default="0.1", help="Standard deviation for noise in function")
    parser.add_argument("-coef_range", dest="coef_range", default="2", help="Range for generating coefficients for function")
    parser.add_argument("-max_val", dest="max_val", default="10", help="Maximum value for function")
    parser.add_argument("-shoulders", dest="shoulders", default="True", help="Determines if there are bounds for function")
    parser.add_argument("-shoulder_leakage", dest="shoulder_leakage", default="0.1", help="Scaling for actions outside of bounds")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="Number of function to load")
    parser.add_argument("-alpha", dest="alpha", default="1.0", help="")
    parser.add_argument("-step", dest="step", default="[0.01]", help="")
    parser.add_argument("-max_delta", dest="max_delta", default="[0.25]", help="")
    parser.add_argument("-state_dim", dest="state_dim", default="1", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="1", help="Dimension size of action")
    args = parser.parse_args()

    # Call training function
    main(
        num_samples=int(args.num_samples),
        reward_training_steps=int(args.reward_training_steps),
        policy_training_steps=int(args.policy_training_steps),
        objectives=literal_eval(args.objectives),
        rl_paths=literal_eval(args.rl_paths),
        explore_buffer_path=args.explore_buffer_path,
        reward_model_path=args.reward_model_path,
        network_amount=int(args.network_amount),
        batch_size=int(args.batch_size),
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
        alpha=float(args.alpha),
        step=literal_eval(args.step),
        max_delta=literal_eval(args.max_delta),
        state_dim=int(args.state_dim),
        action_dim=int(args.action_dim),
    )
