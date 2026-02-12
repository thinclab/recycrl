#!/usr/bin/env python3

"""
This script gives the training sequence for the policy after a reward model has been trained
"""

import os
import rclpy
import numpy as np
from utils import Utility
from recycRL import RecycRL
from ast import literal_eval
from argparse import ArgumentParser
from rclpy.logging import get_logger
from replay_buffer import ReplayBuffer


def main():
    # Define arguments
    description = "Node that goes through the RL training process"
    parser = ArgumentParser(description=description)
    parser.add_argument("-training_steps", dest="training_steps", default="100000", help="Number of training steps")
    parser.add_argument(
        "-rl_path",
        dest="rl_path",
        default="~/RecycRL/recycrl",
        help="Path to save the RL model or actor",
    )
    parser.add_argument(
        "-reward_model_path",
        dest="reward_model_path",
        default="~/RecycRL/recycrl",
        help="Path to load the reward model",
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
    parser.add_argument(
        "-min_action",
        dest="min_action",
        default="[-0.1, -0.1, -0.01, -0.7853981634, -0.7853981634, -0.7853981634]",
        help="Minimum action values vector for x, y, and z position outputs",
    )
    parser.add_argument(
        "-max_action",
        dest="max_action",
        default="[0.1, 0.1, 0.1, 0.7853981634, 0.7853981634, 0.7853981634]",
        help="Maximum action values vector for x, y, and z position outputs",
    )
    parser.add_argument("-expert_batch_size", dest="expert_batch_size", default="80", help="Batch size for expert buffer")
    parser.add_argument("-online_batch_size", dest="online_batch_size", default="80", help="Batch size for online buffer")

    # Parse and assign arguments
    args = parser.parse_args()
    training_steps = int(args.training_steps)
    rl_path = args.rl_path
    reward_model_path = args.reward_model_path
    expert_buffer_path = args.expert_buffer_path
    online_buffer_path = args.online_buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)
    expl_noise = literal_eval(args.expl_noise)
    noise_clip = literal_eval(args.noise_clip)
    expert_batch_size = int(args.expert_batch_size)
    online_batch_size = int(args.online_batch_size)

    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)
    reward_model_path = os.path.expanduser(reward_model_path)
    expert_buffer_path = os.path.expanduser(expert_buffer_path)
    online_buffer_path = os.path.expanduser(online_buffer_path)

    # Define loggers
    logger = get_logger("train")

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

    # Initialize the RecycRL Class
    rl = RecycRL(reward_model_path, state_dim, action_dim, min_action, max_action, expl_noise, noise_clip)

    # If the RL model has been saved previously
    if os.path.exists(f"{rl_path}_actor"):
        # Load the RL model
        rl.load(rl_path)

    elif not os.path.exists(f"{rl_path}_actor"):
        os.makedirs(os.path.dirname(rl_path), exist_ok=True)

    # Try the following
    try:
        # Initialize the Utility Node
        train = Utility(active=False)

        # Start the training loop for the specified number of training steps
        for _ in range(training_steps):
            # Sample a combined batch from the expert and online buffers
            batch = train.sample_from_buffers(expert_buffer, online_buffer, expert_batch_size, online_batch_size)

            # Train the RL model
            rl.train(batch)

            # If the training steps is divisible by 1000
            if rl.total_it % 1000 == 0:
                # Save the RL model incrementally
                rl.save(rl_path)

        # Save the RL model after exiting the loop
        rl.save(rl_path)

    # If there is an exception with the loop
    except Exception as e:
        print("\nTrain script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
