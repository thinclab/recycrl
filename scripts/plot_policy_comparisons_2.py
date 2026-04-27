#!/usr/bin/env python3

"""
This script loads the trained reward model and PAR, REINFORCE, and A2P policies; then, it plots the reward of each
policies' actions as noise is added to a singular action dimension
"""

import os
import rclpy
import torch
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from rclpy.logging import get_logger
from reward_model import RewardModel
from recycRL import RecycRL, REINFORCE, A2P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Define arguments
    description = (
        "This script plots the reward model as a perturbation is added to a single dimension of the three policy actions"
    )
    parser = ArgumentParser(description=description)
    parser.add_argument("-dim_index", dest="dim_index", default="1", help="Index of action dimension to plot")
    parser.add_argument("-x_min", dest="x_min", default="-0.12", help="Minimum x value for plotting")
    parser.add_argument("-x_max", dest="x_max", default="0.12", help="Maximum x value for plotting")
    parser.add_argument("-action_num", dest="action_num", default="20", help="Number of actions to generate for given dimension")
    parser.add_argument("-state", dest="state", default="[1, 1, 1, 1]", help="State to pass through reward model")
    parser.add_argument("--add_noise", dest="add_noise", action="store_true", help="Add noise to the actions")
    parser.add_argument(
        "-noise_magnitude",
        dest="noise_magnitude",
        default="[0.02, 0.02, 0.02, 0.20, 0.20, 0.20]",
        help="Magnitude of noise to add",
    )
    parser.add_argument(
        "-beta", dest="beta", default="0.75", help="Coefficient to multiply standard deviation for penalized reward"
    )
    parser.add_argument(
        "-model_path",
        dest="model_path",
        default="~/RecycRL",
        help="Path to save the reward model",
    )
    parser.add_argument(
        "-rl_path",
        dest="rl_path",
        default="~/RecycRL",
        help="Path to load the RL model",
    )
    parser.add_argument("-network_amount", dest="network_amount", default="5", help="Number of networks for reward model")
    parser.add_argument("-state_dim", dest="state_dim", default="4", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")

    # Parse and assign arguments
    args = parser.parse_args()
    dim_index = int(args.dim_index)
    x_min = float(args.x_min)
    x_max = float(args.x_max)
    action_num = int(args.action_num)
    state = literal_eval(args.state)
    add_noise = args.add_noise
    noise_magnitude = literal_eval(args.noise_magnitude)
    beta = float(args.beta)
    network_amount = int(args.network_amount)
    model_path = args.model_path
    rl_path = args.rl_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)

    # Expand the user to handle "~"
    model_path = os.path.expanduser(model_path)
    rl_path = os.path.expanduser(rl_path)

    # Define loggers
    logger = get_logger("plot_comparisons")

    # Initialize the RecycRL(), REINFORCE(), and A2P() Classes
    par_rl = RecycRL()
    reinforce_rl = REINFORCE()
    a2p_rl = A2P()

    # If the RL models have been saved previously, load the models
    if (
        os.path.exists(f"{rl_path}/Policy")
        and os.path.exists(f"{rl_path}/Policy_REINFORCE")
        and os.path.exists(f"{rl_path}/Policy_A2P")
    ):
        par_rl.load(rl_path)
        reinforce_rl.load(rl_path)
        a2p_rl.load(rl_path)
        logger.info("Policies ready")

    # If one of the RL models does not exist, notify the user and return
    elif (
        not os.path.exists(f"{rl_path}/Policy")
        or not os.path.exists(f"{rl_path}/Policy_REINFORCE")
        or not os.path.exists(f"{rl_path}/Policy_A2P")
    ):
        logger.error("One or more policies do not exist, provide correct path")
        return

    # Initialize rclpy
    rclpy.init()

    # Initialize the RewardModel Class
    reward_model = RewardModel(state_dim, action_dim, network_amount)

    # If the reward model has been saved previously
    if os.path.exists(f"{model_path}/Reward_Model"):
        # Load the reward model
        reward_model.load(model_path)
        logger.info("Reward model ready")

    # If the reward model does not exist, create the save directory
    elif not os.path.exists(f"{model_path}/Reward_Model"):
        logger.error("Reward model does not exist, provide correct path")
        return

    # Try the following
    try:
        # Generate a matrix of size 'action_num' from 'x_min' to 'x_max' to generate values for the iterated action dimension
        iterated_dim = np.linspace(x_min, x_max, action_num)

        # Convert the state to a numpy array and tensor
        numpy_state = np.array(state)
        tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)

        # Get the actions from the par and standard policies for the current state
        par_action = par_rl.select_action(numpy_state, add_noise=False)
        reinforce_action = reinforce_rl.select_action(numpy_state, add_noise=False)
        a2p_action = a2p_rl.select_action(numpy_state, add_noise=False)

        # If the add noise flag is set
        if add_noise:
            # Define empty lists for determining direction to add noise
            par_noise_direction = []
            reinforce_noise_direction = []
            a2p_noise_direction = []

            # Loop through each dimension of the par action and append 1 for positive values and -1 for negative values
            for dim in par_action:
                if dim >= 0:
                    par_noise_direction.append(1)
                elif dim < 0:
                    par_noise_direction.append(-1)

            # Loop through each dimension of the reinforce action and append 1 for positive values and -1 for negative values
            for dim in reinforce_action:
                if dim >= 0:
                    reinforce_noise_direction.append(1)
                elif dim < 0:
                    reinforce_noise_direction.append(-1)

            # Loop through each dimension of the a2p action and append 1 for positive values and -1 for negative values
            for dim in a2p_action:
                if dim >= 0:
                    a2p_noise_direction.append(1)
                elif dim < 0:
                    a2p_noise_direction.append(-1)

            # Add noise to the actions by adding the product of the noise magnitude and noise direction to the original action
            par_action = par_action + np.array(par_noise_direction) * np.array(noise_magnitude)
            reinforce_action = reinforce_action + np.array(reinforce_noise_direction) * np.array(noise_magnitude)
            a2p_action = a2p_action + np.array(a2p_noise_direction) * np.array(noise_magnitude)

        # Get the value of the actions at the index value
        par_action_index_value = par_action[dim_index]
        reinforce_action_index_value = reinforce_action[dim_index]
        a2p_action_index_value = a2p_action[dim_index]

        # Get the mean and standard deviation of the policy actions
        par_mean, par_std = reward_model.reward_model.predict(tensor_state, torch.FloatTensor(par_action).unsqueeze(0).to(device))
        reinforce_mean, reinforce_std = reward_model.reward_model.predict(
            tensor_state, torch.FloatTensor(reinforce_action).unsqueeze(0).to(device)
        )
        a2p_mean, a2p_std = reward_model.reward_model.predict(tensor_state, torch.FloatTensor(a2p_action).unsqueeze(0).to(device))

        # Get the uncertainty penalized rewards for the par and standard policies
        par_reward = (par_mean - beta * par_std).item()
        reinforce_reward = (reinforce_mean - beta * reinforce_std).item()
        a2p_reward = (a2p_mean - beta * a2p_std).item()

        # Define empty lists to hold rewards
        par_rewards = []
        reinforce_rewards = []
        a2p_rewards = []

        # Loop through each generated value
        for value in iterated_dim:
            # Modify the action dimension with the iterated value and convert it to a tensor
            par_action[dim_index] = value
            iterated_action = torch.FloatTensor(par_action).unsqueeze(0).to(device)

            # With gradient disabled, pass the state and action through the model to get the reward
            with torch.no_grad():
                mean, std = reward_model.reward_model.predict(tensor_state, iterated_action)

                # Calculate the uncertainty penalized reward
                reward = mean - beta * std

            # Add the reward to the running list
            par_rewards.append(reward.item())

        # Loop through each generated value
        for value in iterated_dim:
            # Modify the action dimension with the iterated value and convert it to a tensor
            reinforce_action[dim_index] = value
            iterated_action = torch.FloatTensor(reinforce_action).unsqueeze(0).to(device)

            # With gradient disabled, pass the state and action through the model to get the reward
            with torch.no_grad():
                mean, std = reward_model.reward_model.predict(tensor_state, iterated_action)

                # Calculate the uncertainty penalized reward
                reward = mean - beta * std

            # Add the reward to the running list
            reinforce_rewards.append(reward.item())

        # Loop through each generated value
        for value in iterated_dim:
            # Modify the action dimension with the iterated value and convert it to a tensor
            a2p_action[dim_index] = value
            iterated_action = torch.FloatTensor(a2p_action).unsqueeze(0).to(device)

            # With gradient disabled, pass the state and action through the model to get the reward
            with torch.no_grad():
                mean, std = reward_model.reward_model.predict(tensor_state, iterated_action)

                # Calculate the uncertainty penalized reward
                reward = mean - beta * std

            # Add the reward to the running list
            a2p_rewards.append(reward.item())

        # Convert the rewards list to a numpy array
        par_rewards = np.array(par_rewards)
        reinforce_rewards = np.array(reinforce_rewards)
        a2p_rewards = np.array(a2p_rewards)

        # Plot the reward vs. action figure
        plt.figure()
        plt.plot(iterated_dim, par_rewards, label="PAR Objective Curve")
        plt.plot(iterated_dim, reinforce_rewards, label="Standard Objective Curve")
        plt.plot(iterated_dim, a2p_rewards, label="A2P Objective Curve")
        plt.scatter(par_action_index_value, par_reward, color="#1f4e79", label="PAR Objective Action")
        plt.scatter(reinforce_action_index_value, reinforce_reward, color="#cc6600", label="Standard Objective Action")
        plt.scatter(a2p_action_index_value, a2p_reward, color="#0066cc", label="A2P Objective Action")
        print_list = ["Δl", "Δw", "Δh", "Δφ", "Δω", "Δψ"]
        units = ["meters", "meters", "meters", "radians", "radians", "radians"]
        plt.xlabel(f"Value of '{print_list[dim_index]}' in Action ({units[dim_index]})")
        plt.ylabel("Expected Reward")
        if add_noise:
            plt.title("Expected Reward vs Noisy Action")
        else:
            plt.title("Expected Reward vs Action")
        plt.grid()
        # plt.legend(loc="lower left", fontsize=7)
        plt.show()

    # If there is an exception
    except Exception as e:
        print("\nPlot script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
