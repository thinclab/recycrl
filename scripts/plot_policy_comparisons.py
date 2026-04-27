#!/usr/bin/env python3

"""
This script loads the trained reward model and PAR, REINFORCE, and A2P policies; then, it plots the reward of each
policies' actions as noise is added to each action dimension
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
    description = "This script plots the reward model as a perturbations are added to the three policy actions"
    parser = ArgumentParser(description=description)
    parser.add_argument("-dim_index", dest="dim_index", default="1", help="Index of action dimension to plot")
    parser.add_argument("-x_min", dest="x_min", default="-0.12", help="Minimum x value for plotting")
    parser.add_argument("-x_max", dest="x_max", default="0.12", help="Maximum x value for plotting")
    parser.add_argument("-num_points", dest="num_points", default="1000", help="Number of points to plot")
    parser.add_argument("-state", dest="state", default="[1, 1, 1, 1]", help="State to pass through reward model")
    parser.add_argument("--add_noise", dest="add_noise", action="store_true", help="Add noise to the actions")
    parser.add_argument("-noise_magnitude", dest="max_noise", default="[0.05, 0.05, 0.05, 0.50, 0.50, 0.50]", help="Magnitude of noise to add")
    parser.add_argument("-beta", dest="beta", default="0.75", help="Coefficient to multiply standard deviation for penalized reward")
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
    num_points = int(args.num_points)
    state = literal_eval(args.state)
    add_noise = args.add_noise
    max_noise = literal_eval(args.max_noise)
    beta = float(args.beta)
    network_amount = int(args.network_amount)
    model_path = args.model_path
    rl_path = args.rl_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)

    # Expand the user to handle "~"
    model_path = os.path.expanduser(model_path)
    rl_path = os.path.expanduser(rl_path)

    # Define logger
    logger = get_logger("plot_comaprisons")

    # Initialize the RecycRL(), REINFORCE(), and A2P() Classes
    par_rl = RecycRL()
    reinforce_rl = REINFORCE()
    a2p_rl = A2P()

    # If the RL models have been saved previously, load the models
    if os.path.exists(f"{rl_path}/Policy") and os.path.exists(f"{rl_path}/Policy_REINFORCE") and os.path.exists(f"{rl_path}/Policy_A2P"):
        par_rl.load(rl_path)
        reinforce_rl.load(rl_path)
        a2p_rl.load(rl_path)
        logger.info("Policies ready")

    # If one of the RL models does not exist, notify the user and return
    elif not os.path.exists(f"{rl_path}/Policy") or not os.path.exists(f"{rl_path}/Policy_REINFORCE") or not os.path.exists(f"{rl_path}/Policy_A2P"):
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

    # Convert the state to a numpy array and tensor, then convert the max noise to a numpy array
    numpy_state = np.array(state)
    tensor_state = torch.FloatTensor(state).unsqueeze(0).to(device)
    numpy_max_noise = np.array(max_noise)

    # Generate a a range of noise levels from 0 to 1 based on 'num_points'
    alphas = np.linspace(0.0, 1.0, num_points)

    # Calculate the noise levels for each action dimension by multiplying the alphas with the max noise for each dimension
    noise_levels = alphas[:, None] * numpy_max_noise[None, :]

    # Get the base actions from each policy
    base_par_action = par_rl.select_action(numpy_state, add_noise=False)
    base_reinforce_action = reinforce_rl.select_action(numpy_state, add_noise=False)
    base_a2p_action = a2p_rl.select_action(numpy_state, add_noise=False)

    # Convert the actions to tensors
    base_par_tensor = torch.FloatTensor(base_par_action).unsqueeze(0).to(device)
    base_reinforce_tensor = torch.FloatTensor(base_reinforce_action).unsqueeze(0).to(device)
    base_a2p_tensor = torch.FloatTensor(base_a2p_action).unsqueeze(0).to(device)

    # Compute the mean and standard deviation of the base actions for each policy
    with torch.no_grad():
        par_mean, par_std = reward_model.reward_model.predict(tensor_state, base_par_tensor)
        reinforce_mean, reinforce_std = reward_model.reward_model.predict(tensor_state, base_reinforce_tensor)
        a2p_mean, a2p_std = reward_model.reward_model.predict(tensor_state, base_a2p_tensor)

    # Calculate the base reward for each policy with the uncertainty penalized reward
    base_par_reward = (par_mean - beta * par_std).item()
    base_reinforce_reward = (reinforce_mean - beta * reinforce_std).item()
    base_a2p_reward = (a2p_mean - beta * a2p_std).item()

    # Define empty lists for determining direction to add noise
    par_noise_direction = []
    reinforce_noise_direction = []
    a2p_noise_direction = []

    # Loop through each dimension of the PAR base action and append 1 for positive values and -1 for negative values
    for dim in base_par_action:
        par_noise_direction.append(1) if dim >= 0 else par_noise_direction.append(-1)

    # Loop through each dimension of the REINFORCE base action and append 1 for positive values and -1 for negative values
    for dim in base_reinforce_action:
        reinforce_noise_direction.append(1) if dim >= 0 else reinforce_noise_direction.append(-1)

    # Loop through each dimension of the A2P base action and append 1 for positive values and -1 for negative values
    for dim in base_a2p_action:
        a2p_noise_direction.append(1) if dim >= 0 else a2p_noise_direction.append(-1)

    # Define empty lists to store
    par_rewards = []
    reinforce_rewards = []
    a2p_rewards = []

    # Loop through each noise level
    for alpha in noise_levels:
        # Copy the base actions in order to modify them
        par_action = np.copy(base_par_action)
        reinforce_action = np.copy(base_reinforce_action)
        a2p_action = np.copy(base_a2p_action)

        # Apply the noise to the base action of each policy
        par_action += np.array(par_noise_direction) * alpha
        reinforce_action += np.array(reinforce_noise_direction) * alpha
        a2p_action += np.array(a2p_noise_direction) * alpha

        # Convert the actions to tensors
        par_tensor = torch.FloatTensor(par_action).unsqueeze(0).to(device)
        reinforce_tensor = torch.FloatTensor(reinforce_action).unsqueeze(0).to(device)
        a2p_tensor = torch.FloatTensor(a2p_action).unsqueeze(0).to(device)

        # With gradient disabled
        with torch.no_grad():
            # Get the mean and standard deviation of the reward for each action
            c_mean, c_std = reward_model.reward_model.predict(tensor_state, par_tensor)
            r_mean, r_std = reward_model.reward_model.predict(tensor_state, reinforce_tensor)
            a2p_mean, a2p_std = reward_model.reward_model.predict(tensor_state, a2p_tensor)

            # Calculate the uncertainty penalized reward for each policy
            c_reward = (c_mean - beta * c_std).item()
            r_reward = (r_mean - beta * r_std).item()
            a2p_reward = (a2p_mean - beta * a2p_std).item()

        # Clip the rewards to be between 0 and 1
        c_reward = np.clip(c_reward, 0, 1)
        r_reward = np.clip(r_reward, 0, 1)
        a2p_reward = np.clip(a2p_reward, 0, 1)

        # Add the reward to the corresponding list
        par_rewards.append(c_reward)
        reinforce_rewards.append(r_reward)
        a2p_rewards.append(a2p_reward)

    # Convert the list to numpy arrays for plotting
    par_rewards = np.array(par_rewards)
    reinforce_rewards = np.array(reinforce_rewards)
    a2p_rewards = np.array(a2p_rewards)

    # Plot the reward vs. action figure
    plt.figure()
    plt.plot(alphas, par_rewards, label="PAR Objective")
    plt.plot(alphas, reinforce_rewards, label="Standard Objective")
    plt.plot(alphas, a2p_rewards, label="A2P Objective")
    plt.xlabel(f"Magnitude of Noise Added to Action\nMax Noise: {max_noise}")
    plt.ylabel("Expected Reward")
    plt.title("Expected Reward vs Action Perturbation Magnitude")
    plt.grid()
    plt.legend(loc="lower left")
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

if __name__ == "__main__":
    main()