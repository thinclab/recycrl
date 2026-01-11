#!/usr/bin/env python3

"""
This script gives the RL training sequence for the local grasping policy
"""

import os
import numpy as np
from TD3.RD3 import RD3
from argparse import ArgumentParser
from recycrl.scripts.recycRL import RecycRL


def main():
    # Define arguments
    description = "Script to that goes through the training process"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-rl_path",
        dest="rl_path",
        default="~/RD3",
        help="Path to save the RL model",
    )
    parser.add_argument(
        "-buffer_path",
        dest="buffer_path",
        default="~/Replay_Buffer.npy",
        help="Path to save the replay buffer with demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="8", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument("-max_action", dest="max_action", default="1.0", help="Maximum action value")
    parser.add_argument("-batch_size", dest="batch_size", default="128", help="Batch size for training")

    # Parse and assign arguments
    args = parser.parse_args()
    rl_path = args.rl_path
    buffer_path = args.buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    max_action = float(args.max_action)
    batch_size = int(args.batch_size)

    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)
    buffer_path = os.path.expanduser(buffer_path)

    # Initialize the RD3 RL Class
    rl = RD3(state_dim, action_dim, max_action)

    # If the RL model has been saved previously
    if os.path.exists(rl_path):
        # Load the RL model
        rl.load(rl_path)

    # Try the following
    try:
        # Initialize the RecycRL Node
        train = RecycRL(buffer_path)

        # Move the robot to the hidden state so that the workspace can be seen clearly
        train.go_to(train.hidden)

        # Get the initial poses of the items in the workspace
        state = train.get_workspace_state()

        # Start the loop
        while True:
            # If the workspace is empty
            if state[-1] == 0:
                # Wait for the user to arrange items in the workspace
                train.set_workspace_state()

                # Get the poses of the items after the user has rearranged the items
                state = train.get_workspace_state()

            # Get the robot state to move to for item grabbing from the Neural Network
            action = RD3.select_noisy_action(state)

            # Go to the robot pose
            train.go_to(action)

            # After the robot has moved to neural net output position, grab and lift the gripper
            train.grab_and_lift()

            # Get the poses of the items after the action has been executed
            next_state = train.get_workspace_state()

            # Get the reward of the action
            reward, done = train.get_reward(state[-1], next_state[-1])

            # Save the state, action, transition, and reward to the replay buffer
            train.save_to_buffer(state, action, next_state, reward, done)

            # If the replay buffer has been populated enough
            if train.buffer.size >= batch_size:
                # Train the RL model with the replay buffer
                rl.train(train.buffer, batch_size)

            # Assign the next state to the current state for the next iteration, more efficient
            state = next_state

            # Every 25 timesteps
            if rl.total_it % 25 == 0:
                # Save the RL model
                rl.save(rl_path)

                # Save the replay buffer
                np.save(buffer_path, train.buffer)

    # If there is an exception with the loop
    except Exception as e:
        # Save the RL model
        rl.save(rl_path)

        # Save the replay buffer
        np.save(buffer_path, train.buffer)

        print("\nTrain script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        # Save the RL model
        rl.save(rl_path)

        # Save the replay buffer
        np.save(buffer_path, train.buffer)

        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
