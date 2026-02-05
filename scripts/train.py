#!/usr/bin/env python3

"""
This node gives the RL training sequence for the local grasping policy
"""

import os
import rclpy
import numpy as np
from TD3.RD3 import RD3
from recycRL import RecycRL
from ast import literal_eval
from TD3.utils import ReplayBuffer
from argparse import ArgumentParser
from rclpy.logging import get_logger


def main():
    # Define arguments
    description = "Node that goes through the RL training process"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-rl_path",
        dest="rl_path",
        default="~/RD3/RD3",
        help="Path to save the RL model",
    )
    parser.add_argument(
        "-expert_buffer_path",
        dest="expert_buffer_path",
        default="~/RD3/Expert_Buffer",
        help="Path to load and save the replay buffer with expert demonstrations",
    )
    parser.add_argument(
        "-online_buffer_path",
        dest="online_buffer_path",
        default="~/RD3/Online_Buffer",
        help="Path to load and save the replay buffer with online policy demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="9", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument(
        "-min_action",
        dest="min_action",
        default="[0.2, -0.35, 0.81, -0.7853981634, -0.7853981634, -0.7853981634]",
        help="Minimum action values vector for x, y, and z position outputs",
    )
    parser.add_argument(
        "-max_action",
        dest="max_action",
        default="[0.55, 0.35, 0.93, 0.7853981634, 0.7853981634, 0.7853981634]",
        help="Maximum action values vector for x, y, and z position outputs",
    )
    parser.add_argument("-discount", dest="discount", default="0.15", help="")
    parser.add_argument("-tau", dest="tau", default="0.05", help="")
    parser.add_argument(
        "-expl_noise",
        dest="expl_noise",
        # default="[0.04, 0.04, 0.03, 0.1, 0.1, 0.1]",
        default="[0.06, 0.06, 0.05, 0.15, 0.15, 0.15]",
        help="Exploration noise standard deviation",
    )
    parser.add_argument(
        "-policy_noise",
        dest="policy_noise",
        default="[0.02, 0.02, 0.01, 0.05, 0.05, 0.05]",
        help="Policy noise standard deviation",
    )
    parser.add_argument(
        "-noise_clip",
        dest="noise_clip",
        # default="[0.08, 0.08, 0.05, 0.2, 0.2, 0.2]",
        default="[0.1, 0.1, 0.1, 0.3, 0.3, 0.3]",
        help="Maximum noise value",
    )
    parser.add_argument("-expert_batch_size", dest="expert_batch_size", default="40", help="Batch size for expert buffer")
    parser.add_argument("-online_batch_size", dest="online_batch_size", default="40", help="Batch size for online buffer")

    # Parse and assign arguments
    args = parser.parse_args()
    rl_path = args.rl_path
    expert_buffer_path = args.expert_buffer_path
    online_buffer_path = args.online_buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)
    discount = float(args.discount)
    tau = float(args.tau)
    expl_noise = literal_eval(args.expl_noise)
    policy_noise = literal_eval(args.policy_noise)
    noise_clip = literal_eval(args.noise_clip)
    expert_batch_size = int(args.expert_batch_size)
    online_batch_size = int(args.online_batch_size)

    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)
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

    # Initialize the RD3 RL Class
    rl = RD3(state_dim, action_dim, min_action, max_action, discount, tau, expl_noise, policy_noise, noise_clip)

    # If the RL model has been saved previously
    if os.path.exists(f"{rl_path}_actor"):
        # Load the RL model
        rl.load(rl_path)

    elif not os.path.exists(f"{rl_path}_actor"):
        os.makedirs(os.path.dirname(rl_path), exist_ok=True)

    # Try the following
    try:
        # Initialize the RecycRL Node
        train = RecycRL()

        # Move the robot to the bin position so that the workspace can be seen clearly
        train.go_to(train.bin)

        # Open the gripper to so that it is ready to grab an item
        train.open_gripper()

        # Get the initial poses of the items in the workspace
        state = train.get_workspace_state()

        # Set 'run' to True initially to start the loop
        run = True

        # Start the loop
        while run:
            # If the workspace is empty
            if state[-1] == 0:
                # Wait for the user to arrange items in the workspace
                train.set_workspace_state()

                # Get the poses of the items after the user has rearranged the items
                state = train.get_workspace_state()

            # Move the robot to home
            train.go_to(train.home)

            # Pass the state through the actor network to get the action, add noise for exploration
            action = rl.select_action(state, add_noise=True)

            # Go to the robot pose defined by the action
            valid, executed = train.execute_action(state, action, True)

            # If the robot successfully moved to the desired pose
            if executed:
                # After the robot has moved output position, close the gripper, lift, and go to the bin
                train.grab_and_go_to_bin()

                # Get the poses of the items after the action has been executed
                next_state = train.get_workspace_state()

            # If the robot could not reach the desired pose
            elif not executed:
                # There is no change in state due to failed action, so next state is same as current state
                next_state = state

            # Get the reward of the action
            reward, done = train.get_reward(state, next_state, valid)

            # Open the gripper to so that it is ready to grab an item
            train.open_gripper()

            # Add the state, action, transition, and reward to the replay buffer
            train.add_to_buffer(online_buffer, state, action, next_state, reward, done, False)

            # If the replay buffer has been populated enough
            if online_buffer.size >= online_batch_size:
                # Sample a combined batch from the expert and online buffers
                batch = train.sample_from_buffers(expert_buffer, online_buffer, expert_batch_size, online_batch_size)

                # Train the RL model with the replay buffer
                rl.train(batch)

            # Assign the next state to the current state for the next iteration, more efficient
            state = next_state

            # Evaluate the policy
            rl.evaluate_policy(reward, rl_path)

            # If the training steps is divisible by 50
            if rl.total_it % 50 == 0:
                # Save the buffer and RL model incrementally
                train.save_buffer(online_buffer, online_buffer_path)
                rl.save(rl_path)

            # Check if the loop should continue
            run = train.loop_check()

        # Save the buffer and RL model after exiting the loop
        train.save_buffer(online_buffer, online_buffer_path)
        rl.save(rl_path)

    # If there is an exception with the loop
    except Exception as e:
        print("\nTrain script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
