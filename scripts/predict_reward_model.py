#!/usr/bin/env python3

"""
This script loads the reward model and passes a sampled state and action through the model
"""

import os
import torch
import rclpy
import numpy as np
from argparse import ArgumentParser
from rclpy.logging import get_logger
from reward_model import RewardModel


def main():
    # Define arguments
    description = "Script that loads reward model and predicts on sampled state and action"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-model_path",
        dest="model_path",
        default="~/RecycRL",
        help="Path to save the reward model",
    )
    parser.add_argument(
        "-expert_buffer_path",
        dest="expert_buffer_path",
        default="~/RecycRL/Expert_Buffer",
        help="Path to load and save the replay buffer with expert demonstrations",
    )
    parser.add_argument(
        "-online_buffer_path",
        dest="online_buffer_path",
        default="~/RecycRL/Online_Buffer",
        help="Path to load and save the replay buffer with online policy demonstrations",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    model_path = args.model_path
    expert_buffer_path = args.expert_buffer_path
    online_buffer_path = args.online_buffer_path

    # Expand the user to handle "~"
    model_path = os.path.expanduser(model_path)
    expert_buffer_path = os.path.expanduser(expert_buffer_path)
    online_buffer_path = os.path.expanduser(online_buffer_path)

    # Define loggers
    logger = get_logger("t")

    # If the expert replay buffer does not exist, notify the user
    if not os.path.exists(expert_buffer_path + ".npy"):
        logger.error("Expert replay buffer does not exist, provide correct path")
        return

    # If the expert replay buffer exists, load the Class
    else:
        expert_buffer = np.load(expert_buffer_path + ".npy", allow_pickle=True).item()

    # If the online replay buffer does not exist, notify the user
    if not os.path.exists(online_buffer_path + ".npy"):
        logger.error("Online replay buffer does not exist, provide correct path")
        return

    # If the online replay buffer exists, load the Class
    else:
        online_buffer = np.load(online_buffer_path + ".npy", allow_pickle=True).item()

    # Logging for buffer sizes
    logger.info("Expert buffer ready with size of " + str(expert_buffer.size))
    logger.info("Online buffer ready with size of " + str(online_buffer.size))

    # Initialize rclpy
    rclpy.init()

    # Initialize the reward model Class
    reward_model = RewardModel(state_dim=4, action_dim=6, network_amount=5)

    # If the reward model has been saved previously
    if os.path.exists(f"{model_path}/Reward_Model"):
        # Load the reward model
        reward_model.load(model_path)

    # If the reward model does not exist, notify the user and return
    if not os.path.exists(f"{model_path}/Reward_Model"):
        logger.error("Reward model does not exist, provide correct path")
        return

    state, action, reward = expert_buffer.sample(1)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # state = torch.FloatTensor(np.array([2, 1, 2, 0]).reshape(1, -1)).to(device)
    # action = torch.FloatTensor(np.array([-0.028, -0.008, 0.02, 0.00396, -0.07946, 0.2]).reshape(1, -1)).to(device)

    with torch.no_grad():
        mean, std = reward_model.reward_model.predict(state, action)

    print("State:", state.cpu().detach().numpy().flatten())
    print("Action:", action.cpu().detach().numpy().flatten())
    print("Actual Reward:", reward.cpu().detach().numpy().flatten())
    print("Predicted Reward:", mean.cpu().detach().numpy().flatten())
    print("Predicted Reward Standard Deviation:", std.cpu().detach().numpy().flatten())
    print("Training Iterations:", reward_model.total_it)


if __name__ == "__main__":
    main()
