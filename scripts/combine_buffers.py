#!/usr/bin/env python3

"""
This script combines two replay buffers into one
"""

import os
import numpy as np
from rclpy.logging import get_logger
from replay_buffer import ReplayBuffer  # noqa: F401


def main(
    buffer_path_1="~/RecycRL/NRMDP/Online_Buffer",
    buffer_path_2="~/RecycRL/Combined_Buffer",
    save_path="~/RecycRL/Combined_Buffer"
):

    # Expand the user to handle "~"
    buffer_path_1 = os.path.expanduser(buffer_path_1 + ".npy")
    buffer_path_2 = os.path.expanduser(buffer_path_2 + ".npy")
    save_path = os.path.expanduser(save_path + ".npy")

    # Define logger
    logger = get_logger("combine")

    # If the 1st replay buffer does not exist, notify the user
    if not os.path.exists(buffer_path_1):
        logger.error("1st replay buffer does not exist, provide correct path")
        return

    # If the 1st replay buffer exists, load the Class
    else:
        buffer_1 = np.load(buffer_path_1, allow_pickle=True).item()

    # If the 2nd replay buffer does not exist, notify the user
    if not os.path.exists(buffer_path_2):
        logger.error("2nd replay buffer does not exist, provide correct path")
        return

    # If the 2nd replay buffer exists, load the Class
    else:
        buffer_2 = np.load(buffer_path_2, allow_pickle=True).item()

    # Print the size of the buffers
    logger.info(f"1st Buffer Size: {buffer_1.size}")
    logger.info(f"2nd Buffer Size: {buffer_2.size}")

    # Initialize a new ReplayBuffer() Class
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined_buffer = ReplayBuffer(buffer_1.state.shape[1], buffer_1.action.shape[1], max_size=int(1e6))

    # Copy the 1st buffer
    combined_buffer.state[:buffer_1.size] = buffer_1.state[:buffer_1.size]
    combined_buffer.action[:buffer_1.size] = buffer_1.action[:buffer_1.size]
    combined_buffer.reward[:buffer_1.size] = buffer_1.reward[:buffer_1.size]

    # Copy the 2nd buffer
    start = buffer_1.size
    end = start + buffer_2.size
    combined_buffer.state[start:end] = buffer_2.state[:buffer_2.size]
    combined_buffer.action[start:end] = buffer_2.action[:buffer_2.size]
    combined_buffer.reward[start:end] = buffer_2.reward[:buffer_2.size]

    # Determine the pointer and size
    combined_buffer.ptr = buffer_1.ptr + buffer_2.ptr
    combined_buffer.size = buffer_1.size + buffer_2.size

    # Save the replay buffer and a copy of it
    np.save(save_path, combined_buffer)
    np.save(save_path + "_copy", combined_buffer)

    # Log the save
    logger.info(f"Saved the combined buffer of size {combined_buffer.size} successfully")


if __name__ == "__main__":
    main()
