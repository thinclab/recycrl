#!/usr/bin/env python3

"""
This script loads the replay buffer at the specified path and removes an entry
"""

import os
import numpy as np
from argparse import ArgumentParser


def main():
    # Define arguments
    description = "Script to remove entries from the replay buffer"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-buffer_path",
        dest="buffer_path",
        default="~/RD3/Expert_Buffer",
        help="Path to load and save the replay buffer with demonstrations",
    )
    parser.add_argument(
        "-entries",
        dest="entries",
        default="1",
        help="Number of entries to remove from the end of the buffer",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    buffer_path = args.buffer_path
    entries = int(args.entries)

    # Expand the user to handle "~"
    buffer_path = os.path.expanduser(buffer_path)

    # Load the buffer
    buffer = np.load(buffer_path + ".npy", allow_pickle=True).item()

    # Print the size of the buffer
    print(f"Buffer size before: {buffer.size}")

    # Loop for number of 'entries' specified
    for _ in range(0, entries):
        # Remove the last entry in the buffer
        buffer.remove_last()

    # Print that removal was successful
    print("Removed last entry")

    # Save the replay buffer and a copy of it
    np.save(buffer_path, buffer)
    np.save(buffer_path + "_copy", buffer)

    # Print the size of the buffer
    print(f"New buffer size: {buffer.size}")


if __name__ == "__main__":
    main()
