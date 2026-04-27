#!/usr/bin/env python3

"""
This script simplifies the process of loading the RL model and getting the actions for all of the states
"""

import os
from argparse import ArgumentParser
from recycRL import RecycRL, REINFORCE, A2P


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
    parser.add_argument(
        "-objective",
        dest="objective",
        default="Custom",
        help="Objective used to optimize the policy",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    rl_path = args.rl_path
    objective = args.objective

    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)

    # If the user wants to optimize with the PAR objective
    if objective == "PAR":
        # Initialize the RecycRL Class
        rl = RecycRL()

        # Define the search path
        search_path = rl_path + "/Policy"

    # If the user wants to optimize with the REINFORCE objective
    elif objective == "REINFORCE":
        # Initialize the REINFORCE Class
        rl = REINFORCE()

        # Define the search path
        search_path = rl_path + "/Policy_REINFORCE"

    elif objective == "A2P":
        # Initialize the A2P Class
        rl = A2P()

        # Define the search path
        search_path = rl_path + "/Policy_A2P"

    # If the user provides and invalid objective, notify the user and return
    elif objective != "PAR" and objective != "REINFORCE" and objective != "A2P":
        print("Objective invalid")
        return

    # If the RL model has been saved previously, load the model
    if os.path.exists(search_path):
        rl.load(rl_path)
        print("Policy ready")

    # If the RL model does not exist, notify the user and return
    elif not os.path.exists(search_path):
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
