#!/usr/bin/env python3

"""
This script simplifies the process of loading the Replay Buffer and sampling data from it
"""

import os
import numpy as np
from argparse import ArgumentParser


def main():
    # Define arguments
    description = "Script to simplify the data collection process for the replay buffer"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-buffer_path",
        dest="buffer_path",
        default="~/Replay_Buffer.npy",
        help="Path to save the replay buffer with demonstrations",
    )
    parser.add_argument("-sample_num", dest="sample_num", default="10", help="Number of samples to take")

    # Parse and assign arguments
    args = parser.parse_args()
    buffer_path = args.buffer_path
    sample_number = int(args.sample_num)

    # Expand the user to handle "~"
    buffer_path = os.path.expanduser(buffer_path)

    # Load the buffer
    buffer = np.load(buffer_path, allow_pickle=True).item()

    # Sample from the buffer
    tensors = buffer.sample(sample_number)

    # Print the samples
    for tensor in tensors:
        print("")
        samples = tensor.detach().cpu().numpy()
        for sample in samples:
            print(sample)


if __name__ == "__main__":
    main()
