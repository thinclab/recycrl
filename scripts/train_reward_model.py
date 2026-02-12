#!/usr/bin/env python3

"""
This script gives the training sequence for the reward model after sufficient expert and online data has been collected
"""

import os
import rclpy
import numpy as np
from recycRL import RecycRL
from argparse import ArgumentParser
from rclpy.logging import get_logger
from reward_model import RewardModel
from replay_buffer import ReplayBuffer


def main():
    # Define arguments
    description = "Training script for the reward model"
    parser = ArgumentParser(description=description)
    parser.add_argument("-training_steps", dest="training_steps", default="100000", help="Number of training steps")
    parser.add_argument("-network_amount", dest="network_amount", default="5", help="Number of networks for reward model")
    parser.add_argument(
        "-model_path",
        dest="model_path",
        default="~/RecycRL/recycrl",
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
    parser.add_argument("-state_dim", dest="state_dim", default="4", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument("-expert_batch_size", dest="expert_batch_size", default="80", help="Batch size for expert buffer")
    parser.add_argument("-online_batch_size", dest="online_batch_size", default="80", help="Batch size for online buffer")

    # Parse and assign arguments
    args = parser.parse_args()
    training_steps = int(args.training_steps)
    network_amount = int(args.network_amount)
    model_path = args.model_path
    expert_buffer_path = args.expert_buffer_path
    online_buffer_path = args.online_buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    expert_batch_size = int(args.expert_batch_size)
    online_batch_size = int(args.online_batch_size)

    # Expand the user to handle "~"
    model_path = os.path.expanduser(model_path)
    expert_buffer_path = os.path.expanduser(expert_buffer_path)
    online_buffer_path = os.path.expanduser(online_buffer_path)

    # Define loggers
    logger = get_logger("train_reward")

    # If the expert replay buffer does not exist, notify the user
    if not os.path.exists(expert_buffer_path + ".npy"):
        logger.warn("Expert replay buffer does not exist, provide correct path")
        return

    # If the expert replay buffer exists, load the Class
    else:
        expert_buffer = np.load(expert_buffer_path + ".npy", allow_pickle=True).item()

    # If the online replay buffer does not exist, initialize a new ReplayBuffer() Class
    if not os.path.exists(online_buffer_path + ".npy"):
        os.makedirs(os.path.dirname(online_buffer_path), exist_ok=True)
        online_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    # If the online replay buffer exists, load the Class
    else:
        online_buffer = np.load(online_buffer_path + ".npy", allow_pickle=True).item()

    # Logging for buffer sizes
    logger.info("Expert buffer ready with size of " + str(expert_buffer.size))
    logger.info("Online buffer ready with size of " + str(online_buffer.size))

    # Initialize rclpy
    rclpy.init()

    # Initialize the RewardModel Class
    reward_model = RewardModel(state_dim, action_dim, network_amount)

    # If the reward model has been saved previously
    if os.path.exists(f"{model_path}_reward_model"):
        # Load the reward model
        reward_model.load(model_path)

    elif not os.path.exists(f"{model_path}_reward_model"):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Try the following
    try:
        # Initialize the RecycRL Node
        train = RecycRL(active=False)

        # Start the training loop for the specified number of training steps
        for _ in range(training_steps):
            # Sample a combined batch from the expert and online buffers
            batch = train.sample_from_buffers(expert_buffer, online_buffer, expert_batch_size, online_batch_size)

            # Train the reward model with the sampled batch
            reward_model.train(batch)

            # If the training steps is divisible by 1000
            if reward_model.total_it % 1000 == 0:
                # Save the reward model incrementally
                reward_model.save(model_path)

        # Save the reward model after exiting the loop
        reward_model.save(model_path)

    # If there is an exception with the loop
    except Exception as e:
        print("\nTrain script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
