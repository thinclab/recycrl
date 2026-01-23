#!/usr/bin/env python3

"""
This node gives the RL training sequence for the local grasping policy
"""

import os
import rclpy
from TD3.RD3 import RD3
from recycRL import RecycRL
from ast import literal_eval
from argparse import ArgumentParser


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
        "-buffer_path",
        dest="buffer_path",
        default="~/RD3/Replay_Buffer",
        help="Path to save the replay buffer with demonstrations",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="8", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument(
        "-min_action",
        dest="min_action",
        default="[0.2, -0.35, 0.81, -1.570796327, -1.570796327, -0.7853981634]",
        help="Minimum action values vector for x, y, and z position outputs",
    )
    parser.add_argument(
        "-max_action",
        dest="max_action",
        default="[0.55, 0.35, 0.93, 1.570796327, 1.570796327, 0.7853981634]",
        help="Maximum action values vector for x, y, and z position outputs",
    )
    parser.add_argument("-expl_noise", dest="expl_noise", default="0.012", help="Exploration noise standard deviation")
    parser.add_argument("-policy_noise", dest="policy_noise", default="0.008", help="Policy noise standard deviation")
    parser.add_argument("-noise_clip", dest="noise_clip", default="0.02", help="Maximum noise value")
    parser.add_argument("-batch_size", dest="batch_size", default="256", help="Batch size for training")

    # Parse and assign arguments
    args = parser.parse_args()
    rl_path = args.rl_path
    buffer_path = args.buffer_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)
    expl_noise = float(args.expl_noise)
    policy_noise = float(args.policy_noise)
    noise_clip = float(args.noise_clip)
    batch_size = int(args.batch_size)

    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)
    buffer_path = os.path.expanduser(buffer_path)

    # Initialize rclpy
    rclpy.init()

    # Initialize the RD3 RL Class
    rl = RD3(
        state_dim, action_dim, min_action, max_action, expl_noise=expl_noise, policy_noise=policy_noise, noise_clip=noise_clip
    )

    # If the RL model has been saved previously
    if os.path.exists(f"{rl_path}_actor"):
        # Load the RL model
        rl.load(rl_path)

    elif not os.path.exists(f"{rl_path}_actor"):
        os.makedirs(os.path.dirname(rl_path), exist_ok=True)

    # Try the following
    try:
        # Initialize the RecycRL Node
        train = RecycRL(buffer_path)

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
            executed = train.execute_action(state, action)

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
            reward, done = train.get_reward(state[-1], next_state[-1])

            # Add the state, action, transition, and reward to the replay buffer
            train.add_to_buffer(state, action, next_state, reward, done)

            # If the replay buffer has been populated enough
            if train.buffer.size >= batch_size:
                # Train the RL model with the replay buffer
                rl.train(train.buffer, batch_size)

            # Assign the next state to the current state for the next iteration, more efficient
            state = next_state

            # Evaluate the policy
            rl.evaluate_policy(reward, rl_path)

            # If the training steps is divisible by 50
            if rl.total_it % 50 == 0:
                # Save the buffer and RL model incrementally
                train.save_buffer()
                rl.save(rl_path)

            # Check if the loop should continue
            run = train.loop_check()

        # Save the buffer and RL model after exiting the loop
        train.save_buffer()
        rl.save(rl_path)

    # If there is an exception with the loop
    except Exception as e:
        print("\nTrain script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
