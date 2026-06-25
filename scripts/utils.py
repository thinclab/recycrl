#!/usr/bin/env python3

"""
This script provides the main Class and all of the common functions for this package
"""

import rclpy
import torch
import numpy as np
from time import sleep
from rclpy.node import Node
from math import degrees, pi
from recycrl.srv import Recyclables
from lifecycle_msgs.msg import Transition
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, PoseStamped
from lifecycle_msgs.srv import GetState, ChangeState
from ament_index_python import get_package_share_directory
from rclpy.logging import set_logger_level, LoggingSeverity
from kuka_kontrol.grip_utils import gripper_to_pos, get_current_load
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_joint_constraint, construct_link_constraint


class Utility(Node):
    def __init__(self, active=True):
        # Register the ROS2 Node
        super().__init__("recycrl")

        # Declare parameter for 'run'
        self.declare_parameters(namespace="", parameters=[("run", True)])

        # Set all loggers to warning or above, but keep this Node at info level
        set_logger_level("", LoggingSeverity.ERROR)
        self.get_logger().set_level(LoggingSeverity.INFO)

        # Define global variables
        self.execution_status = None
        self.executed = False
        self.second_max_height = 0.0

        # Create a publisher to publish the Pose of the object in RViz
        self.publisher = self.create_publisher(PoseStamped, "/recycrl_pose", 10)

        # If we are in 'active' mode, initialize the services and MoveIt configuration
        if active:
            # Define requests and clients
            self.get_recyclables_request = Recyclables.Request()
            self.get_state_request = GetState.Request()
            self.change_state_request = ChangeState.Request()
            self.get_recyclables_client = self.create_client(Recyclables, "/get_recyclables")
            self.get_state_client = self.create_client(GetState, "/robot_manager/get_state")
            self.change_state_client = self.create_client(ChangeState, "/robot_manager/change_state")

            # Wait for the services to become available
            while not self.get_recyclables_client.wait_for_service(4.0):
                self.get_logger().info("Waiting for YOLO service ...")
            while not self.get_state_client.wait_for_service(4.0):
                self.get_logger().info("Waiting for robot get state service ...")
            while not self.change_state_client.wait_for_service(4.0):
                self.get_logger().info("Waiting for robot change state service ...")

            # Check if the robot is active, and activate it if it is inactive
            self.activate_external_control()

            # Define the MoveIt Configuration
            moveit_config = (
                MoveItConfigsBuilder("kuka_lbr_iisy")
                .robot_description(file_path="/urdf/lbr_iisy3_r760_combined.urdf")
                .robot_description_semantic(file_path=(get_package_share_directory("kuka_kontrol")) + "/urdf/lbr_iisy3_r760.srdf")
                .joint_limits(
                    file_path=get_package_share_directory("kuka_lbr_iisy_support") + "/config/lbr_iisy3_r760_joint_limits.yaml"
                )
                .moveit_cpp(file_path=get_package_share_directory("kuka_kontrol") + "/config/planning.yaml")
                .to_moveit_configs()
            )

            # Instantiate MoveItPy and the various components
            self.kuka_move = MoveItPy(node_name="moveit_py", config_dict=moveit_config.to_dict())
            self.kuka = self.kuka_move.get_planning_component("manipulator")
            self.robot_model = self.kuka_move.get_robot_model()
            self.robot_state = RobotState(self.robot_model)
            self.joint_group = self.robot_model.get_joint_model_group("manipulator")
            self.trajectory_manager = self.kuka_move.get_trajectory_execution_manager()
            self.plan_parameters = PlanRequestParameters(self.kuka_move, "pilz_solo")
            self.planning_scene = self.kuka_move.get_planning_scene_monitor()
            self.set_collision_scene()

            # Define pre-set positions for KUKA
            self.home = self.construct_joint_position([0.0, -1.74533, 1.5708, 0.0, 1.74533, 0.0])
            self.bin = self.construct_joint_position([-1.5708, -1.13446, 1.48353, 0.0, 1.22173, -1.5708])
            # self.bin = self.construct_joint_position([-1.65806, -1.13446, 1.48353, 0.0, 1.22173, -1.65806])
            self.hidden = self.construct_joint_position([-1.5708, -1.5708, 1.5708, 0.0, 1.5708, 0.0])

    def trajectory_callback(self, msg):
        # Assign the message to the global variable
        self.execution_status = msg.status

        # Set True to exit the while loop in the go_to() function
        self.executed = True

    def set_workspace_state(self):
        # Block the program until the user has notified that the workspace has been reset
        self.get_logger().info("Robot moved to hidden pose, reset workspace if sort has finished")
        self.get_logger().warn("Workspace ready? (Enter)")
        answer = input()

        # If they do not click "Enter", notify the user
        while answer != "":
            self.get_logger().warn("You typed '" + str(answer) + "', click 'Enter' to capture the current workspace and continue")
            answer = input()

    def get_workspace_state(self, check=True):
        # Call the get poses service, spin the Node until a response is received, and define the response
        get_recyclables_future = self.get_recyclables_client.call_async(self.get_recyclables_request)
        rclpy.spin_until_future_complete(self, get_recyclables_future)
        get_state_response = get_recyclables_future.result()

        # Extract the different parts of the response
        positions = get_state_response.positions
        pitches = get_state_response.pitches
        yaws = get_state_response.yaws
        types = get_state_response.types
        heights = get_state_response.heights
        items = len(positions)

        # Define variables to choose index or object for state
        index = 0
        tallest_index = 0

        # Check that the service returned at least one item
        if items > 0:
            # If there are more than two items
            if items > 1:
                # Get the index of the tallest object
                tallest_index = heights.index(max(heights))

                # Remove the tallest item from the list and get the new max for the "second max" value
                heights.pop(tallest_index)
                self.second_max_height = max(heights)

            # If there is one item
            elif items == 1:
                # Set the "second max" value to that of the only object (in case we miss)
                self.second_max_height = heights[0]

            # Iterate through each pitch angle
            for pitch in pitches:
                # If the pitch of the object is less than 0 (if the object is tilted and not lying flat)
                if pitch < 0.0:
                    # Set the index to that of the tallest object so that the tallest object is chosen
                    index = tallest_index
                    continue

            # Get the position, pitch, and yaw of the tallest object or the highest confidence
            position = positions[index]
            pitch = pitches[index]
            yaw = yaws[index]

            # Publish the pose for RViz visualization (we can do quaternion_from_euler here since roll is zero)
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "world"
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose.position = position
            qx, qy, qz, qw = quaternion_from_euler(0.0, pitch, yaw)
            pose_stamped.pose.orientation.x = qx
            pose_stamped.pose.orientation.y = qy
            pose_stamped.pose.orientation.z = qz
            pose_stamped.pose.orientation.w = qw
            self.publisher.publish(pose_stamped)

            # Determine which quadrant we are in based on coordinates of the centroid
            if position.x >= 0.375 and position.y >= 0:
                quadrant = 1
            elif position.x < 0.375 and position.y >= 0:
                quadrant = 2
            elif position.x < 0.375 and position.y < 0:
                quadrant = 3
            elif position.x >= 0.375 and position.y < 0:
                quadrant = 4

            # Determine which slice we are in based on pitch angle of the object
            if 0 >= pitch > -pi / 6:
                sector = 1
            elif -pi / 6 >= pitch > -pi / 3:
                sector = 2
            else:
                sector = 3

            # Determine which slice we are in based on yaw angle of the object
            if abs(yaw) <= pi / 4:
                slice = 1
            elif pi / 4 < yaw <= 3 * pi / 4:
                slice = 2
            elif abs(yaw) > 3 * pi / 4:
                slice = 3
            elif -pi / 4 > yaw >= -3 * pi / 4:
                slice = 4

            # Define the state for the neural network with quadrant, sector, slice, and object type
            network_state = [quadrant, sector, slice, types[index]]

            # Define the actual state with position, orientation, object type, and number of items
            actual_state = [
                round(position.x, 5),
                round(position.y, 5),
                round(position.z, 5),
                round(pitch, 5),
                round(yaw, 5),
                types[index],
                items,
            ]

            # Define a list for printing the state with the angles in degrees
            print_state = actual_state.copy()
            print_state[3] = round(degrees(print_state[3]), 5)
            print_state[4] = round(degrees(print_state[4]), 5)

        # Otherwise, return a list with all zeros
        else:
            network_state = [0, 0, 0, 0]
            actual_state = network_state
            print_state = network_state

        # Log the state
        self.get_logger().warn(f"Network State: {network_state}")
        self.get_logger().warn(f"Actual State: {print_state}; Index: {index + 1}")

        # If the 'check' argument is True
        if check:
            # Block the program until the user has notified that the  robot state has been set
            self.get_logger().warn("Is the state accurate? (y/n/f)")
            answer = input()

            # If they do not click "Enter", notify the user
            while answer != "y" and answer != "n" and answer != "f":
                self.get_logger().warn(
                    "You typed '" + str(answer) + "', type 'y' to use state, 'n' to retry, and 'f' to flip yaw"
                )
                answer = input()

            # If the user wants to recapture the state
            if answer == "n":
                network_state, actual_state = self.get_workspace_state()

            # If the user wants to flip the yaw angle
            if answer == "f":
                if network_state[2] == 1 or network_state[2] == 2:
                    network_state[2] += 2
                elif network_state[2] == 3 or network_state[2] == 4:
                    network_state[2] -= 2
                if actual_state[4] >= 0:
                    actual_state[4] -= pi
                    print_state[4] = round((print_state[4] - 180.000), 5)
                elif actual_state[4] < 0:
                    actual_state[4] += pi
                    print_state[4] = round((print_state[4] + 180.000), 5)
                self.get_logger().warn(f"Flipped Network State: {network_state}")
                self.get_logger().warn(f"Flipped Actual State: {print_state}; Index: {index + 1}")

        return network_state, actual_state

    def set_robot_state(self):
        # Block the program until the user has notified that the  robot state has been set
        self.get_logger().info("Robot moved to Home, provide pose for training")
        self.get_logger().warn("Finished hand-guiding robot? (Enter)")
        answer = input()

        # If they do not click "Enter", notify the user
        while answer != "":
            self.get_logger().warn("You typed '" + str(answer) + "', click 'Enter' if you have provided a hand-guided pose")
            answer = input()

    def get_robot_state(self, state):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("tcp")

        # Extract the roll, pitch, and yaw angles from the quaternion
        roll, pitch, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Flip the yaw angle if facing backwards (done in convert_action() so need to do here)
        if state[4] > pi / 2:
            state_yaw = state[4] - pi
        elif state[4] < -pi / 2:
            state_yaw = state[4] + pi
        else:
            state_yaw = state[4]

        # Convert the absolute pitch and yaw angles to relative angles based on the state (roll is already relative)
        pitch -= state[3]
        yaw -= state_yaw

        # Calculate the change in x, y, z and create an array from these values
        x_change = pose.position.x - state[0]
        y_change = pose.position.y - state[1]
        z_change = pose.position.z - state[2]
        xyz_change = np.array([x_change, y_change, z_change])

        # Get the rotation matrix from the quaternion
        rotation_matrix = Rotation.from_quat(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        ).as_matrix()

        # Take the dot product of each rotation matrix axis with the xyz change to get the change in the rotated frame
        length_change = np.dot(xyz_change, rotation_matrix[:, 0])
        width_change = np.dot(xyz_change, rotation_matrix[:, 1])
        height_change = np.dot(xyz_change, rotation_matrix[:, 2])

        # Define the list for the action
        pose_list = [length_change, width_change, height_change, roll, pitch, yaw]

        # Round all the values in the list to 5 decimals
        rounded_pose = [round(x, 5) for x in pose_list]

        # Log the action
        self.get_logger().warn(f"Action: {rounded_pose}")

        return rounded_pose

    def get_reward(self, state, next_state):
        # Get the current load (mA) of the gripper after lifting
        current_load = get_current_load()

        # Get the difference of items in the workspace after executing the action
        item_diff = state[-1] - next_state[-1]

        # If there's one less item and current load is >200 mA, we grabbed successfully, assign a reward of 1
        if item_diff == 1 and current_load > 160:
            reward = 1

        # If one of the two conditions for determining a successful grasp is False, query the user for reward
        elif (item_diff != 1 and current_load > 160) or (item_diff == 1 and current_load <= 160):
            # Block the program until the user has given the reward for the transition
            self.get_logger().warn(f"Unable to determine reward; Item Diff: {item_diff}; Current Load: {current_load} mA")
            self.get_logger().warn("Enter reward (0, 1, or other)")
            reward = input()

            # If the user does not enter a correct reward
            while reward != "0" and reward != "1":
                self.get_logger().warn(
                    "You typed '" + str(reward) + "', type '0' or '1' to record the reward or 'other' for further calculation"
                )
                reward = input()

        # In all other cases, assign a reward of 0
        else:
            reward = 0

        # Log the reward
        self.get_logger().warn(f"Reward: {float(reward)}")

        return float(reward)

    def add_to_buffer(self, buffer, state, action, reward, check=True, log=False):
        # If in 'check' mode
        if check:
            # Block the program until the user has given the reward for the transition
            self.get_logger().warn("Add previous state, action, transition, and reward to buffer? (Enter)")
            answer = input()

            # If they do not click "Enter", notify the user
            while answer != "":
                self.get_logger().warn("You typed '" + str(answer) + "', click 'Enter' to add to the buffer and continue")
                answer = input()

        # Add to the replay buffer
        buffer.add(state, action, reward)

        # If in 'log' mode
        if log:
            self.get_logger().info("State, action, and reward added to replay buffer")

    def save_buffer(self, buffer, buffer_path):
        # Save the replay buffer and a copy of it
        np.save(buffer_path, buffer)
        np.save(buffer_path + "_copy", buffer)

        self.get_logger().info("Saved the replay buffer successfully")

    def sample_from_buffers(self, buffers, batch_sizes):
        # Define empty lists
        states, actions, rewards = [], [], []

        # Loop through the buffers and batch sizes
        for buffer, batch_size in zip(buffers, batch_sizes):
            # Sample from the buffer
            s, a, r = buffer.sample(batch_size)

            # Append the values to lists
            states.append(s)
            actions.append(a)
            rewards.append(r)

        # Combine the samples
        state = torch.cat(states, dim=0)
        action = torch.cat(actions, dim=0)
        reward = torch.cat(rewards, dim=0)

        return (state, action, reward)

    def go_to(self, position):
        # Define variables
        self.executed = False

        # Set the start and goal states and plan the motion
        self.kuka.set_start_state_to_current_state()
        self.kuka.set_goal_state(motion_plan_constraints=[position])
        plan_result = self.kuka.plan(single_plan_parameters=self.plan_parameters)

        # Logic for successful planning
        if plan_result:
            # Convert the trajectory, add it to the list, and execute
            robot_trajectory = plan_result.trajectory
            robot_trajectory_msg = robot_trajectory.get_robot_trajectory_msg()
            self.trajectory_manager.push(robot_trajectory_msg)
            self.trajectory_manager.execute(self.trajectory_callback)

            # This block of code allows "Ctrl+C" to stop trajectory during execution
            try:
                while self.trajectory_manager.is_managing_controllers() and not self.executed:
                    sleep(0.05)
            except KeyboardInterrupt:
                self.get_logger().warn("Stopped execution")
                self.trajectory_manager.stop_execution()

        # If the most recent execution status is "SUCCEEDED", set executed to True; False otherwise
        executed = True if self.execution_status == "SUCCEEDED" else False

        # Reset the execution status variable
        self.execution_status = None

        return executed

    def select_random_action(self, min_action, max_action):
        # Convert the lists representing action bounds to arrays
        min_action = np.array(min_action)
        max_action = np.array(max_action)

        # Sample a random action
        action = np.random.uniform(min_action, max_action)

        # Round the action to 8 decimals
        action = np.round(action, decimals=5)

        return action

    def execute_action(
        self, state, action, penalize=False, train=True, initial_tolerance=0.01, tolerance_increase=0.01, max_tolerance=0.1
    ):
        # Set approached and executed to be False initially
        approached = False
        executed = False

        # Log the action to execute
        self.get_logger().warn("Action " + str(list(action)))

        # Convert the action from its 3-angle representation into a quaternion
        action, fixed_action = self.convert_action(state, action, penalize)

        # Log the actual action to execute
        self.get_logger().warn("Actual Action " + str(action))

        # First, go to the approach position
        approached = self.approach(action)

        # If we reached the approach position
        if approached:
            # Log successful approach
            self.get_logger().info("Successfully reached the approach position")

            # Encode the action position as a Pose() variable
            pose = Pose()
            pose.position.x = float(action[0])
            pose.position.y = float(action[1])
            pose.position.z = float(action[2])
            pose.orientation.x = action[3]
            pose.orientation.y = action[4]
            pose.orientation.z = action[5]
            pose.orientation.w = action[6]

            # Try to find an IK solution
            found_ik = self.robot_state.set_from_ik("manipulator", pose, "tcp", timeout=5.0)

            # If an IK solution is found
            if found_ik:
                # Extract the joint angles and define the joint goal
                joint_angles = self.robot_state.get_joint_group_positions("manipulator")
                action_position = self.construct_joint_position(joint_angles, initial_tolerance)

                # # If an IK solution is not found
                # if not found_ik:
                #     # Define the pose goal
                #     action_position = construct_link_constraint(
                #         "tcp",
                #         "world",
                #         [action[0], action[1], action[2]],
                #         initial_tolerance,
                #         [action[3], action[4], action[5], action[6]],
                #         initial_tolerance,
                #     )

                # Pass the goal position to the move function
                executed = self.go_to(action_position)

                # Define the tolerance
                tolerance = initial_tolerance + tolerance_increase

                # Loop the following until the action succeeds or we reach the max retry attempts
                while (not executed) and (tolerance <= max_tolerance):
                    # If an IK solution is found
                    if found_ik:
                        # Reconstruct the joint goal with incrementally increasing tolerance
                        action_position = self.construct_joint_position(joint_angles, tolerance)

                    # # If an IK solution is not found
                    # if not found_ik:
                    #     # Redefine the lift position with an increased tolerance
                    #     action_position = construct_link_constraint(
                    #         "tcp",
                    #         "world",
                    #         [action[0], action[1], action[2]],
                    #         tolerance,
                    #         [action[3], action[4], action[5], action[6]],
                    #         tolerance,
                    #     )

                    # Pass the action position to the move function
                    executed = self.go_to(action_position)

                    # Increase the tolerance and try the approach again
                    tolerance += tolerance_increase

            elif not found_ik:
                self.get_logger().error("Could not find an IK solution for action")
                executed = False

        # If the approach position could not be reached
        elif not approached:
            # If we are not in 'train' mode
            if not train:
                # Query the user if they would still like to execute the action
                self.get_logger().warn("Could not reach approach, execute action anyway (y/n)")
                answer = input()

                # If they do not type "y" or "n", notify the user
                while answer != "y" and answer != "n":
                    self.get_logger().warn("You typed '" + str(answer) + "', type 'y' to execute the action or 'n' to cancel")
                    answer = input()

            # If we are in 'train' mode or if the user wants to execute the action despite not being in 'train' mode
            if train or (not train and answer == "y"):
                # Encode the action position as a Pose() variable
                pose = Pose()
                pose.position.x = float(action[0])
                pose.position.y = float(action[1])
                pose.position.z = float(action[2])
                pose.orientation.x = action[3]
                pose.orientation.y = action[4]
                pose.orientation.z = action[5]
                pose.orientation.w = action[6]

                # Try to find an IK solution
                found_ik = self.robot_state.set_from_ik("manipulator", pose, "tcp", timeout=5.0)

                # If an IK solution is found
                if found_ik:
                    # Extract the joint angles and define the joint goal
                    joint_angles = self.robot_state.get_joint_group_positions("manipulator")
                    action_position = self.construct_joint_position(joint_angles, initial_tolerance)

                    # # If an IK solution is not found
                    # if not found_ik:
                    #     # Define the pose goal
                    #     action_position = construct_link_constraint(
                    #         "tcp",
                    #         "world",
                    #         [action[0], action[1], action[2]],
                    #         initial_tolerance,
                    #         [action[3], action[4], action[5], action[6]],
                    #         initial_tolerance,
                    #     )

                    # Pass the goal position to the move function
                    executed = self.go_to(action_position)

                    # Define the tolerance
                    tolerance = initial_tolerance + tolerance_increase

                    # Loop the following until the action succeeds or we reach the max retry attempts
                    while (not executed) and (tolerance <= max_tolerance):
                        # If an IK solution is found
                        if found_ik:
                            # Reconstruct the joint goal with incrementally increasing tolerance
                            action_position = self.construct_joint_position(joint_angles, tolerance)

                        # If an IK solution is not found
                        if not found_ik:
                            # Redefine the lift position with an increased tolerance
                            action_position = construct_link_constraint(
                                "tcp",
                                "world",
                                [action[0], action[1], action[2]],
                                tolerance,
                                [action[3], action[4], action[5], action[6]],
                                tolerance,
                            )

                        # Pass the action position to the move function
                        executed = self.go_to(action_position)

                        # Increase the tolerance and try the approach again
                        tolerance += tolerance_increase

                elif not found_ik:
                    self.get_logger().error("Could not find an IK solution for action")
                    executed = False

        return executed, fixed_action

    def approach(self, action, approach_dist=0.12, initial_tolerance=0.1, tolerance_increase=0.01, max_tolerance=0.3):
        # Position vector
        pos = np.array(action[0:3])

        # Get the rotation matrix from the quaternion
        rotation_matrix = Rotation.from_quat([action[3], action[4], action[5], action[6]]).as_matrix()

        # Get the TCP x-axis in world frame
        tcp_x = rotation_matrix @ np.array([0.0, 0.0, -1.0])

        # Calculate the approach position by multiplying the TCP x-axis components by the approach distance
        approach_pos = pos - approach_dist * tcp_x
        x, y, z = approach_pos.tolist()

        # Encode the approach position as a Pose() variable
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = action[3]
        pose.orientation.y = action[4]
        pose.orientation.z = action[5]
        pose.orientation.w = action[6]

        # Try to find an IK solution
        found_ik = self.robot_state.set_from_ik("manipulator", pose, "tcp", timeout=5.0)

        # If an IK solution is found
        if found_ik:
            # Extract the joint angles and define the joint goal
            joint_angles = self.robot_state.get_joint_group_positions("manipulator")
            approach_position = self.construct_joint_position(joint_angles, initial_tolerance)

            # Pass the goal position to the move function
            approached = self.go_to(approach_position)

            # Define the tolerance
            tolerance = initial_tolerance + tolerance_increase

            # Loop the following until the approach action succeeds or we reach the max retry attempts
            while (not approached) and (tolerance <= max_tolerance):
                # If an IK solution is found
                if found_ik:
                    # Reconstruct the joint goal with incrementally increasing tolerance
                    approach_position = self.construct_joint_position(joint_angles, tolerance)

                # Pass the approach position to the move function
                approached = self.go_to(approach_position)

                # Increase the tolerance and try the approach again
                tolerance += tolerance_increase

        elif not found_ik:
            self.get_logger().error("Could not find an IK solution for approach position")
            approached = False

        return approached

    def lift(self, z_extra=0.02, initial_tolerance=0.01, tolerance_increase=0.01, max_tolerance=0.3):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("tcp")

        # Calculate the desired height to lift the gripper
        pose.position.z = self.second_max_height + (pose.position.z - 0.75) + z_extra

        # Try to find an IK solution
        found_ik = self.robot_state.set_from_ik("manipulator", pose, "tcp", timeout=5.0)

        # If an IK solution is found
        if found_ik:
            # Extract the joint angles and define the joint goal
            joint_angles = self.robot_state.get_joint_group_positions("manipulator")
            lift_position = self.construct_joint_position(joint_angles, initial_tolerance)

            # # If an IK solution is not found
            # if not found_ik:
            #     # Define the pose goal
            #     lift_position = construct_link_constraint(
            #         "tcp",
            #         "world",
            #         [pose.position.x, pose.position.y, pose.position.z],
            #         initial_tolerance,
            #         [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            #         0.01,
            #     )

            # Pass the lift position to the move function
            lifted = self.go_to(lift_position)

            # Define the initial tolerance for the position constraint
            tolerance = initial_tolerance + tolerance_increase

            # Loop the following until the lift action succeeds or we reach the max retry attempts
            while (not lifted) and (tolerance <= max_tolerance):
                # If an IK solution is found
                if found_ik:
                    # Reconstruct the joint goal with incrementally increasing tolerance
                    lift_position = self.construct_joint_position(joint_angles, tolerance)

                # If an IK solution is not found
                if not found_ik:
                    # Redefine the lift position with an increased tolerance
                    lift_position = construct_link_constraint(
                        "tcp",
                        "world",
                        [pose.position.x, pose.position.y, pose.position.z],
                        tolerance,
                        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                        0.01,
                    )

                # Pass the lift position to the move function
                lifted = self.go_to(lift_position)

                # Increase the tolerance and try the lift again
                tolerance += tolerance_increase

        elif not found_ik:
            self.get_logger().error("Could not find an IK solution for lift position")
            lifted = False

        return lifted

    def grab_and_go_to_bin(self, pos=55, sleep_time=0.2):
        # Close the gripper
        closed = gripper_to_pos(pos, sleep_time=sleep_time)

        # If the gripper closed
        if closed:
            # Log successful grab
            self.get_logger().info("Successfully closed gripper")

            # Lift the object slightly
            lifted = self.lift()

            # Check that the lift action succeeded
            if lifted:
                # Log successful lift
                self.get_logger().info("Successfully reached the lift position")

                # Go to the bin
                self.go_to(self.bin)

            elif not lifted:
                # Query the user if they would still like to execute the action
                self.get_logger().warn("Could not reach lift, go to bin anyway (y/n)")
                answer = input()

                # If they do not type "y" or "n", notify the user
                while answer != "y" and answer != "n":
                    self.get_logger().warn("You typed '" + str(answer) + "', type 'y' to go to the bin or 'n' to cancel")
                    answer = input()

                # If the user still wants to go to the bin
                if answer == "y":
                    # Go to the bin
                    self.go_to(self.bin)

        # If the gripper did not close, notify the user
        elif not closed:
            self.get_logger().warn("Failed to close gripper, check that gripper is powered\nGoing back home")

    def distance_check(self, state, action, height=0.07, radius=0.07):
        # Define the gripper position and object position
        gripper_position = np.array([action[0], action[1], action[2]])
        object_position = np.array([state[0], state[1], state[2]])

        # Get the gripper rotation matrix from the gripper's quaternion
        gripper_rotation = Rotation.from_quat([action[3], action[4], action[5], action[6]]).as_matrix()

        # Get the rotation matrix from the object's quaternion
        object_rotation = Rotation.from_quat([state[3], state[4], state[5], state[6]]).as_matrix()

        # Get the slope of the 3D line for the cylinder by getting the third column vector (negative z-axis)
        gripper_slope = -gripper_rotation[:, 2]

        # Get the slope of the 3D line for the object by getting the first column vector (negative x-axis)
        object_slope = -object_rotation[:, 0]

        # Get the vector difference between the object and gripper position
        difference = object_position - gripper_position

        # Get the correlation of the different vectors
        slope_correlation = np.dot(object_slope, gripper_slope)
        grip_difference_correlation = np.dot(difference, gripper_slope)
        object_difference_correlation = np.dot(difference, object_slope)

        # Calculate the denominator
        denom = 1.0 - (slope_correlation**2)

        # If the denominator is less than 0.000001, there is a large correlation so lines are nearly parallel
        if abs(denom) < 1e-6:
            # Set _ to 0
            t_star = 0.0

        # In all other cases calculate 't_star' for closest point calculation
        else:
            t_star = -(object_difference_correlation - grip_difference_correlation * slope_correlation) / denom

        # If the object is a bottle, set the half length to the following (cm)
        if state[7] == 0:
            half_length = 0.12

        # If the object is a can, set the half length to the following (cm)
        elif state[7] == 1:
            half_length = 0.07

        # Clamp to the segment
        t_star = np.clip(t_star, -half_length, half_length)

        # Get the closest point of the object to the gripper
        closest_point = object_position + t_star * object_slope

        # Calculate the projection of the closest point onto the gripper slope
        closest_projection = np.dot(gripper_slope, closest_point - gripper_position)

        # Convert the closest point projection back into 3D space
        closest_projection_3d = gripper_position + closest_projection * gripper_slope

        # Calculate the distance between the closest and its projection
        closest_projection_distance = np.linalg.norm(closest_point - closest_projection_3d)

        # If the projection distance is less than the radius and the projection is within the height, return True
        if (closest_projection_distance <= radius) and (-0.05 <= closest_projection <= height):
            result = True
            self.get_logger().info(
                f"Distance Check Passed; Radial Distance: {closest_projection_distance}; Length distance: {closest_projection}"
            )

        # If the check is not passed, return False and notify the user
        else:
            result = False
            self.get_logger().info(
                f"Distance Check Failed; Radial Distance: {closest_projection_distance}; Length distance: {closest_projection}"
            )

        return result

    def open_gripper(self, pos=25, sleep_time=0.2):
        # Open the gripper
        gripper_to_pos(pos, sleep_time=sleep_time)

    def deactivate_external_control(self):
        # Call the Get State Service, spin the Node until a response is received, and define the response
        get_state_future = self.get_state_client.call_async(self.get_state_request)
        rclpy.spin_until_future_complete(self, get_state_future)
        get_state_response = get_state_future.result()

        # Check that the Get State Service request completed
        if get_state_response:
            #  Define the current state
            current_state = get_state_response.current_state.label

            # If the robot is already inactive, no need to deactivate
            if current_state != "active":
                self.get_logger().info("Robot is already inactive, proceed to hand-guiding")

            # If the robot is active, proceed
            elif current_state == "active":
                self.get_logger().info("Robot is active, proceeding to deactivation")

                # Define the Change State request
                self.change_state_request.transition.id = Transition.TRANSITION_DEACTIVATE

                # Call the Change State Service, spin the Node until a response is received, and define the response
                change_state_future = self.change_state_client.call_async(self.change_state_request)
                rclpy.spin_until_future_complete(self, change_state_future)
                change_state_response = change_state_future.result()

                # Check that the Change State Service request completed
                if change_state_response:
                    # If the Change State Service request completed, notify the user
                    if change_state_response.success:
                        self.get_logger().info("Deactivation successful")

                    # If the Change State Service request completed, but did not deactivate the robot, notify the user and retry
                    else:
                        self.get_logger().info("Transition request sent successfully, but transition failed. Retrying now")
                        self.deactivate_external_control()

                # If the Change State Service request failed, notify the user
                else:
                    self.get_logger().error("Transition request failed")

        # If the Get State Service request failed, notify the user
        else:
            self.get_logger().error("Request for get state service failed")

    def activate_external_control(self):
        # Call the Get State Service, spin the Node until a response is received, and define the response
        get_state_future = self.get_state_client.call_async(self.get_state_request)
        rclpy.spin_until_future_complete(self, get_state_future)
        get_state_response = get_state_future.result()

        # Check that the Get State Service request completed
        if get_state_response:
            #  Define the current state
            current_state = get_state_response.current_state.label

            # If the robot is already active, no need to activate
            if current_state == "active":
                self.get_logger().info("Robot is already active")

            # If the robot is inactive, proceed
            elif current_state != "active":
                self.get_logger().info("Robot is inactive, proceeding to activation")

                # Define the Change State request
                self.change_state_request.transition.id = Transition.TRANSITION_ACTIVATE

                # Call the Change State Service, spin the Node until a response is received, and define the response
                change_state_future = self.change_state_client.call_async(self.change_state_request)
                rclpy.spin_until_future_complete(self, change_state_future)
                change_state_response = change_state_future.result()

                # Check that the Change State Service request completed
                if change_state_response:
                    # If the Change State Service request completed, notify the user
                    if change_state_response.success:
                        self.get_logger().info("Activation was successful")

                    # If the Change State Service request completed, but did not activate the robot, notify the user and retry
                    else:
                        self.get_logger().info("Transition request sent successfully, but transition failed. Retrying now")
                        self.activate_external_control()

                # If the Change State Service request failed, notify the user
                else:
                    self.get_logger().error("Transition request failed")

        # If the Get State Service request failed, notify the user
        else:
            self.get_logger().error("Request for get state service failed")

    def construct_joint_position(self, angles, tolerance=0.01):
        # Take the passed angles and assign them as a dictionary
        joint_angles = {
            "joint_1": angles[0],
            "joint_2": angles[1],
            "joint_3": angles[2],
            "joint_4": angles[3],
            "joint_5": angles[4],
            "joint_6": angles[5],
        }

        # Assign the dictionary to the Robot State joint positions
        self.robot_state.joint_positions = joint_angles

        # Build the joint goal position as a joint constraint
        joint_position = construct_joint_constraint(self.robot_state, self.joint_group, tolerance)

        return joint_position

    def convert_action(
        self,
        state,
        action,
        penalize=False,
        min_bound=[0.2, -0.35, 0.805, -0.7854, -0.7854, -1.9635],
        max_bound=[0.55, 0.35, 0.93, 0.7854, 0.7854, 1.9635],
    ):
        # Define modified variable to be False initially
        modified = False

        # Define fixed action to be the same as the passed action initially
        fixed_action = action.copy()

        # Flip the yaw angle if facing backwards to avoid unnecessary large rotations
        if state[4] > pi / 2:
            flipped_yaw = state[4] - pi
        elif state[4] < -pi / 2:
            flipped_yaw = state[4] + pi
        else:
            flipped_yaw = state[4]

        # Convert the relative angles to absolute angles, roll is already relative
        unclamped_roll = action[3]
        unclamped_pitch = action[4] + state[3]
        unclamped_yaw = action[5] + flipped_yaw

        # If we are penalizing out-of-bounds actions,
        if penalize:
            pass

        # If we are not penalizing out-of-bounds actions
        elif not penalize:
            # Clamp the absolute angles to be within the bounds
            roll = max(min_bound[3], min(max_bound[3], unclamped_roll))
            pitch = max(min_bound[4], min(max_bound[4], unclamped_pitch))
            yaw = max(min_bound[5], min(max_bound[5], unclamped_yaw))

            # If the angular part of the action is clamped
            if roll != unclamped_roll or pitch != unclamped_pitch or yaw != unclamped_yaw:
                # Redefine the network action that was executed to have the clamped angles
                fixed_pitch = pitch - state[3]
                fixed_yaw = yaw - flipped_yaw
                fixed_action = [fixed_action[0], fixed_action[1], fixed_action[2], roll, fixed_pitch, fixed_yaw]

                # Set modified to True
                modified = True

        # Convert the Euler angles to a quaternion
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw, "sxyz")

        # Get the rotation matrix from the quaternion
        rotation_matrix = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()

        # Calculate the change in position along each axis
        length_change = action[0] * rotation_matrix[:, 0]
        width_change = action[1] * rotation_matrix[:, 1]
        height_change = action[2] * rotation_matrix[:, 2]

        # Calculate the change in x, y, and z by summing the components
        x_change = length_change[0] + width_change[0] + height_change[0]
        y_change = length_change[1] + width_change[1] + height_change[1]
        z_change = length_change[2] + width_change[2] + height_change[2]

        # Convert the relative action to absolute coordinates
        unclamped_x = x_change + state[0]
        unclamped_y = y_change + state[1]
        unclamped_z = z_change + state[2]

        # If we are penalizing out-of-bounds actions,
        if penalize:
            pass

        # If we are not penalizing out-of-bounds actions
        elif not penalize:
            # Clamp the absolute position to be within the bounds
            x = max(min_bound[0], min(max_bound[0], unclamped_x))
            y = max(min_bound[1], min(max_bound[1], unclamped_y))
            z = max(min_bound[2], min(max_bound[2], unclamped_z))

            # If the positional part of the action is clamped
            if x != unclamped_x or y != unclamped_y or z != unclamped_z:
                # Calculate the change in x, y, z and create an array from these values
                x_change = x - state[0]
                y_change = y - state[1]
                z_change = z - state[2]
                xyz_change = np.array([x_change, y_change, z_change])

                # Take the dot product of each rotation matrix axis with the xyz change to get the change in the rotated frame
                length_change = np.dot(xyz_change, rotation_matrix[:, 0])
                width_change = np.dot(xyz_change, rotation_matrix[:, 1])
                height_change = np.dot(xyz_change, rotation_matrix[:, 2])

                # Redefine the action that was executed to have the clamped positions
                fixed_action = [length_change, width_change, height_change, fixed_action[3], fixed_action[4], fixed_action[5]]

                # Set modified to True
                modified = True

        # If the action was modified and we are not penalizing actions, log the modified state
        if modified and not penalize:
            rounded_fixed_action = [round(x, 5) for x in fixed_action]
            self.get_logger().warn(f"Action is OoB; modified to {rounded_fixed_action}")

        # Define the action to execute
        action = [x, y, z, qx, qy, qz, qw]

        # Round all the values in the list to 5 decimals
        rounded_action = [round(x, 5) for x in action]

        return rounded_action, fixed_action

    def normalize(self, matrix):
        # Define a variable to be assigned locally in upcoming loop
        sum = 0

        # Loop through element in the matrix
        for element in matrix:
            # Square the element and add to the running sum
            sum += element**2

        # Get the square root of the sum to acquire the magnitude of the vector
        magnitude = np.sqrt(sum)

        # Divide the matrix by the magnitude to normalize it
        normalized = matrix / magnitude

        return normalized

    def set_collision_scene(self):
        # Define the Collision objects initially
        conveyor_collision = CollisionObject()
        conveyor_collision.id = "conveyor_collision"
        conveyor_collision.header.frame_id = "world"
        conveyor_collision.operation = CollisionObject.ADD

        bar_collision = CollisionObject()
        bar_collision.id = "bar_collision"
        bar_collision.header.frame_id = "world"
        bar_collision.operation = CollisionObject.ADD

        cam_collision = CollisionObject()
        cam_collision.id = "cam_collision"
        cam_collision.header.frame_id = "world"
        cam_collision.operation = CollisionObject.ADD

        pillar_collision = CollisionObject()
        pillar_collision.id = "pillar_collision"
        pillar_collision.header.frame_id = "world"
        pillar_collision.operation = CollisionObject.ADD

        computer_collision = CollisionObject()
        computer_collision.id = "computer_collision"
        computer_collision.header.frame_id = "world"
        computer_collision.operation = CollisionObject.ADD

        ur_collision = CollisionObject()
        ur_collision.id = "ur_collision"
        ur_collision.header.frame_id = "world"
        ur_collision.operation = CollisionObject.ADD

        # Define the collision bounds for each box
        conveyor_box = SolidPrimitive()
        conveyor_box.type = SolidPrimitive.BOX
        conveyor_box.dimensions = [0.8, 1.0, 0.4]
        conveyor_collision.primitives.append(conveyor_box)

        bar_box = SolidPrimitive()
        bar_box.type = SolidPrimitive.BOX
        bar_box.dimensions = [1.45, 0.09, 0.06]
        bar_collision.primitives.append(bar_box)

        cam_box = SolidPrimitive()
        cam_box.type = SolidPrimitive.BOX
        cam_box.dimensions = [0.25, 0.2, 0.04]
        cam_collision.primitives.append(cam_box)

        pillar_box = SolidPrimitive()
        pillar_box.type = SolidPrimitive.BOX
        pillar_box.dimensions = [0.11, 0.09, 1.55]
        pillar_collision.primitives.append(pillar_box)

        computer_box = SolidPrimitive()
        computer_box.type = SolidPrimitive.BOX
        computer_box.dimensions = [0.5, 0.05, 0.5]
        computer_collision.primitives.append(computer_box)

        ur_box = SolidPrimitive()
        ur_box.type = SolidPrimitive.BOX
        ur_box.dimensions = [0.2, 0.3, 0.5]
        ur_collision.primitives.append(ur_box)

        # Define the transforms for each collision box
        conveyor_pose = Pose()
        conveyor_pose.position.x = 0.545
        conveyor_pose.position.y = 0.0
        conveyor_pose.position.z = 0.52
        conveyor_pose.orientation.w = 1.0
        conveyor_collision.primitive_poses.append(conveyor_pose)

        bar_pose = Pose()
        bar_pose.position.x = 0.4
        bar_pose.position.y = 0.0
        bar_pose.position.z = 1.81
        bar_pose.orientation.w = 1.0
        bar_collision.primitive_poses.append(bar_pose)

        cam_pose = Pose()
        cam_pose.position.x = 0.32
        cam_pose.position.y = 0.0
        cam_pose.position.z = 1.76
        cam_pose.orientation.w = 1.0
        cam_collision.primitive_poses.append(cam_pose)

        pillar_pose = Pose()
        pillar_pose.position.x = -0.36
        pillar_pose.position.y = 0.0
        pillar_pose.position.z = 1.2
        pillar_pose.orientation.w = 1.0
        pillar_collision.primitive_poses.append(pillar_pose)

        computer_pose = Pose()
        computer_pose.position.x = 0.0
        computer_pose.position.y = -0.865
        computer_pose.position.z = 1.02
        computer_pose.orientation.w = 1.0
        computer_collision.primitive_poses.append(computer_pose)

        ur_pose = Pose()
        ur_pose.position.x = 0.91
        ur_pose.position.y = 0.0
        ur_pose.position.z = 1.0
        ur_pose.orientation.w = 1.0
        ur_collision.primitive_poses.append(ur_pose)

        # Add the objects to the scene
        self.planning_scene.process_collision_object(conveyor_collision)
        self.planning_scene.process_collision_object(bar_collision)
        self.planning_scene.process_collision_object(cam_collision)
        self.planning_scene.process_collision_object(pillar_collision)
        self.planning_scene.process_collision_object(computer_collision)
        self.planning_scene.process_collision_object(ur_collision)

    def add_noise(self, action, magnitude):
        # Determine the noise direction by the sign of the action
        noise_direction = np.sign(action)

        noise_direction[2] = 1.0

        # Multiply the magnitude by the noise direction to get adversarial noise
        noise = np.array(magnitude) * noise_direction

        # Add the noise to the action
        action += noise

        return action

    def loop_check(self):
        # Print statement for looks
        print("")

        # Spin the node to process parameter changes
        rclpy.spin_once(self, timeout_sec=0.01)

        # Get the current parameter value
        run = self.get_parameter("run").value

        return run
