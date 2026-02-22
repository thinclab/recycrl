#!/usr/bin/env python3

"""
This script simplifies the process of loading the RL model and getting the actions for all of the states
"""

import os
from recycRL import RecycRL
from argparse import ArgumentParser


def main():
    # Define arguments
    description = "Script that loads the RL model and prints the actions for all states"
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

    # Initialize the RecycRL Class
    rl = RecycRL()

    # If the RL model has been saved previously, load the model
    if os.path.exists(f"{rl_path}/Policy"):
        rl.load(rl_path)
        print("Policy ready")

    # If the RL model does not exist, notify the user and return
    elif not os.path.exists(f"{rl_path}/Policy"):
        print("Policy does not exist, provide correct path")
        return

    # Loop through first dimension of state
    for i in range(1, 5):
        # Loop through third dimension of state
        for j in range(1, 5):
            # Loop through fourth dimension of state
            for k in range(0, 2):
                # Generate state from current loop variables
                state = [i, 1, j, k]

                # Pass state through network to get action
                action = rl.select_action(state, add_noise=False)

                # Print state actions pairs
                print("State:", state)
                print("Action", action, "\n")


if __name__ == "__main__":
    main()
