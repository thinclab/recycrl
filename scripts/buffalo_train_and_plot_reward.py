#!/usr/bin/env python3

"""
This script automatically trains a reward model on randomly generated samples and plots the results
"""

import torch
import random
import numpy as np
from time import time
from ast import literal_eval
from argparse import ArgumentParser
from buffalo_plot_reward import plot
from buffalo_collect_and_train_reward import collect_and_train


def main(
    num_samples=200,
    training_steps=20000,
    explore_buffer_path="~/Buffalo/Explore_Buffer",
    model_path="~/Buffalo",
    batch_size=100,
    betas=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    seed=0,
    deterministic=True,
    degree=6,
    coef_range=2,
    max_val=10,
    predefined_polynomial=1,
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
            training_steps=training_steps,
            explore_buffer_path=explore_buffer_path,
            model_path=model_path,
            batch_size=batch_size,
            seed=seed,
            deterministic=deterministic,
            degree=degree,
            coef_range=coef_range,
            max_val=max_val,
            predefined_polynomial=predefined_polynomial,
        )

        # Plot the approximate reward function with varying beta values against the true reward function
        plot(
            model_path=model_path,
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
    description = "This script automatically trains a reward model on randomly generated samples and plots the results"
    parser = ArgumentParser(description=description)
    parser.add_argument("-num_samples", dest="num_samples", default="200", help="Number of random samples to generate")
    parser.add_argument("-training_steps", dest="training_steps", default="20000", help="Number of training steps")
    parser.add_argument(
        "-explore_buffer_path",
        dest="explore_buffer_path",
        default="~/Buffalo/Explore_Buffer",
        help="Path to load and save the replay buffer with random samples",
    )
    parser.add_argument("-model_path", dest="model_path", default="~/Buffalo", help="Path to save trained reward model")
    parser.add_argument("-batch_size", dest="batch_size", default="100", help="Batch size for buffer")
    parser.add_argument("-betas", dest="betas", default="[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]", help="")
    parser.add_argument("-seed", dest="seed", default="0", help="")
    parser.add_argument("--not_deterministic", action="store_true", default=False, help="")
    parser.add_argument("-degree", dest="degree", default="6", help="")
    parser.add_argument("-coef_range", dest="coef_range", default="2", help="")
    parser.add_argument("-max_val", dest="max_val", default="10", help="")
    parser.add_argument("-predefined_polynomial", dest="predefined_polynomial", default="1", help="")
    args = parser.parse_args()

    # Call the main function
    main(
        num_samples=int(args.num_samples),
        training_steps=int(args.training_steps),
        explore_buffer_path=args.explore_buffer_path,
        model_path=args.model_path,
        batch_size=int(args.batch_size),
        betas=literal_eval(args.betas),
        seed=int(args.seed),
        deterministic=not bool(args.not_deterministic),
        degree=int(args.degree),
        coef_range=float(args.coef_range),
        max_val=float(args.max_val),
        predefined_polynomial=int(args.predefined_polynomial),
    )
