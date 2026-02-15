#!/usr/bin/env python3

"""
This script simplifies the process of loading the tracking variables for the RL model
"""

import os
import numpy as np
from argparse import ArgumentParser


def main():
    # Define arguments
    description = "Script that loads iterations and average reward from RL model"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-rl_path",
        dest="rl_path",
        default="~/RecycRL",
        help="Path to load the RL model",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    rl_path = args.rl_path

    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)

    # Load the tracking variables
    total_it = int(np.load(rl_path + "/Policy_Iterations.npy"))
    prev_rewards = np.load(rl_path + "/Policy_Previous_Rewards.npy").tolist()

    # Print the tracking variables
    print(f"Total Timesteps: {total_it}")
    print(f"Previous Rewards: {prev_rewards}")
    print(f"Average Reward: {np.mean(prev_rewards)}")


if __name__ == "__main__":
    main()
