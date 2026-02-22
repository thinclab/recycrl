#!/usr/bin/env python3

"""
This node loads the trained policy and executes the policy continuously on the workspace
"""

import os
import rclpy
from utils import Utility
from recycRL import RecycRL
from ast import literal_eval
from argparse import ArgumentParser
from rclpy.logging import get_logger


def main():
    # Define arguments
    description = ""
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-rl_path",
        dest="rl_path",
        default="~/RecycRL",
        help="Path to save and load the RL model or actor",
    )
    parser.add_argument("-state_dim", dest="state_dim", default="4", help="Dimension size of state")
    parser.add_argument("-action_dim", dest="action_dim", default="6", help="Dimension size of action")
    parser.add_argument(
        "-min_action",
        dest="min_action",
        default="[-0.10, -0.10, -0.01, -0.7853981634, -0.7853981634, -0.7853981634]",
        help="Minimum action values vector for x, y, and z position outputs",
    )
    parser.add_argument(
        "-max_action",
        dest="max_action",
        default="[0.10, 0.10, 0.10, 0.7853981634, 0.7853981634, 0.7853981634]",
        help="Maximum action values vector for x, y, and z position outputs",
    )

    # Parse and assign arguments
    args = parser.parse_args()
    rl_path = args.rl_path
    state_dim = int(args.state_dim)
    action_dim = int(args.action_dim)
    min_action = literal_eval(args.min_action)
    max_action = literal_eval(args.max_action)

    # Expand the user to handle "~"
    rl_path = os.path.expanduser(rl_path)

    # Initialize rclpy
    rclpy.init()

    # Initialize the RecycRL Class
    rl = RecycRL(state_dim, action_dim, min_action, max_action)

    # Define loggers
    logger = get_logger("execute")

    # If the RL model has been saved previously, load the model
    if os.path.exists(f"{rl_path}/Policy"):
        rl.load(rl_path)
        logger.info("Policy ready")

    # If the RL model does not exist, notify the user and return
    elif not os.path.exists(f"{rl_path}/Policy"):
        logger.error("Policy does not exist, provide correct path")
        return

    # Try the following
    try:
        # Initialize the Utility Node
        execute = Utility(active=True)

        # Move the robot to the bin position so that the workspace can be seen clearly
        execute.go_to(execute.bin)

        # Open the gripper to so that it is ready to grab an item
        execute.open_gripper()

        # Get the initial poses of the items in the workspace
        network_state, actual_state = execute.get_workspace_state(check=False)

        # Set 'run' to True initially to start the loop
        run = True

        # Start the loop
        while run:
            # If the workspace is empty
            if actual_state[-1] == 0:
                # Wait for the user to arrange items in the workspace
                execute.set_workspace_state()

                # Get the poses of the items after the user has rearranged the items
                network_state, actual_state = execute.get_workspace_state(check=False)

            # Move the robot to home
            execute.go_to(execute.home)

            # Pass the state through the actor network to get the action
            action = rl.select_action(network_state, add_noise=False)

            # Go to the robot pose defined by the action
            execute.execute_action(actual_state, action, penalize=False, train=True)

            # After the robot has moved, close the gripper, lift, and go to the bin
            execute.grab_and_go_to_bin()

            # Get the poses of the items after the action has been executed
            network_state, actual_state = execute.get_workspace_state(check=False)

            # Open the gripper to so that it is ready to grab an item
            execute.open_gripper()

            # Check if the loop should continue
            run = execute.loop_check()

    # If there is an exception with the loop
    except Exception as e:
        print("\nExecute script failed: %r" % (e,))

    # If there is a Keyboard Interrupt
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


if __name__ == "__main__":
    main()
