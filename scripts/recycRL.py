#!/usr/bin/env python3

"""
This script provides the main Class and all of the functions for capturing state transitions for
the replay buffer and training the robot with RD3 RL to sort recyclables (train.py, imitate.py, and generate.py)
"""

import rclpy
import torch
import numpy as np
from time import sleep
from math import degrees
from rclpy.node import Node
from geometry_msgs.msg import Pose
from recycrl.srv import Recyclables
from lifecycle_msgs.msg import Transition
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from scipy.spatial.transform import Rotation
from lifecycle_msgs.srv import GetState, ChangeState
from tf_transformations import euler_from_quaternion
from ament_index_python import get_package_share_directory
from rclpy.logging import set_logger_level, LoggingSeverity
from kuka_kontrol.grip_utils import gripper_to_pos, get_current_load
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_joint_constraint, construct_link_constraint


class RecycRL(Node):
    def __init__(self, state_dim=9, action_dim=6):
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

    def get_workspace_state(self):
        # Call the get poses service, spin the Node until a response is received, and define the response
        get_recyclables_future = self.get_recyclables_client.call_async(self.get_recyclables_request)
        rclpy.spin_until_future_complete(self, get_recyclables_future)
        get_state_response = get_recyclables_future.result()
        
        # Extract the different parts of the response
        poses = get_state_response.poses
        types = get_state_response.types
        heights = get_state_response.heights

        # Define variables to choose index or object for state
        pose_index = 0
        tallest_index = 0

        # Check that the service returned at least one pose
        if poses:
            # If there are more than two objects
            if len(heights) > 1:
                # Get the index of the tallest object
                tallest_index = heights.index(max(heights))

                # Remove the tallest item from the list and get the new max for the "second max" value
                heights.pop(tallest_index)
                self.second_max_height = max(heights)

            # If there is one item
            elif len(heights) == 1:
                # Set the "second max" value to 0.75 (conveyor height)
                self.second_max_height = 0.75

            # Iterate through each pose
            for i, pose in enumerate(poses):
                # Extract the pitch angle from the quaternion
                _, pitch, _ = euler_from_quaternion(
                    [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                )

                # If the pitch of the object is less than 0 (if the object is tilted and not lying flat)
                if pitch < 0.0:
                    # Set the pose index to that of the tallest object so that this object is chosen
                    pose_index = tallest_index
                    continue

            # Get the pose of the tallest object or the highest confidence
            pose = poses[pose_index]

            # Convert the pose into a list and add the object type and number of items in the workspace
            state = [
                round(pose.position.x, 4),
                round(pose.position.y, 4),
                round(pose.position.z, 4),
                round(pose.orientation.x, 4),
                round(pose.orientation.y, 4),
                round(pose.orientation.z, 4),
                round(pose.orientation.w, 4),
                types[pose_index],
                len(poses),
            ]

            # Convert the quaternion to pitch and yaw angles and define a list for printing the state
            _, pitch, yaw = euler_from_quaternion(
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            )
            print_state = [state[0], state[1], state[2], round(degrees(pitch), 3), round(degrees(yaw), 3), state[7], state[8]]

        # Otherwise, return a list with all zeros (invalid position and quaternion, bottle type, and empty workspace)
        else:
            state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0]
            print_state = state

        # Log the state
        self.get_logger().warn(f"State: {print_state}; Index: {pose_index + 1}")

        # Block the program until the user has notified that the  robot state has been set
        self.get_logger().warn("Is the state accurate? (y/n)")
        answer = input()

        # If they do not click "Enter", notify the user
        while answer != "y" and answer != "n":
            self.get_logger().warn("You typed '" + str(answer) + "', type 'y' to use the current state and 'n' to retry")
            answer = input()

        # If the user wants to recapture the state
        if answer == "n":
            state = self.get_workspace_state()

        return state

    def set_robot_state(self):
        # Block the program until the user has notified that the  robot state has been set
        self.get_logger().info("Robot moved to Home, provide pose for training")
        self.get_logger().warn("Finished hand-guiding robot? (Enter)")
        answer = input()

        # If they do not click "Enter", notify the user
        while answer != "":
            self.get_logger().warn("You typed '" + str(answer) + "', click 'Enter' if you have provided a hand-guided pose")
            answer = input()

    def get_robot_state(self):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("tcp")

        # Define vectors to represent x and z coordinate axes
        x_axis = np.array([1.0, 0.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        # Convert the quaternion to a transformation matrix
        rotation_matrix = Rotation.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Extract the rotation vector for the z-axis
        z_vector = rotation_matrix.apply(z_axis)

        # Extract the rotation vector for the x-axis
        x_vector = rotation_matrix.apply(x_axis)

        # Normalize the x vector
        x_vector /= np.linalg.norm(x_vector)

        # Get the projection of the x-axis onto the z vector
        x_axis_proj = x_axis - np.dot(x_axis, z_vector) * z_vector

        # Normalize the projected x axis
        x_axis_proj /= np.linalg.norm(x_axis_proj)

        # Calculate a plane perpendicular to the z vector
        plane_normal = np.cross(x_axis_proj, x_vector)

        # Calculate the yaw angle
        yaw = np.arctan2(np.dot(plane_normal, z_vector), np.dot(x_axis_proj, x_vector))

        # Multiply the yaw by the third column vector to the get the transformation matrix
        # In essence: Rotate about the z axis in the local gripper frame
        yaw_matrix = Rotation.from_rotvec(yaw * z_vector)

        # Multiply the inverse of the yaw matrix with the original transformation matrix to remove the yaw angle
        tilt_matrix = (yaw_matrix.inv() * rotation_matrix).as_matrix()

        # Calculate the roll by taking the inverse tangent of the z-component and y-component of the y-axis
        roll = np.arctan2(tilt_matrix[2, 1], tilt_matrix[1, 1])

        # Calculate the pitch by taking the inverse tangent of the z-axis and the x-axis (z-components)
        pitch = np.arctan2(tilt_matrix[0, 2], tilt_matrix[0, 0])

        # Convert the position with the 3-angle orientation representation to a list
        pose_list = [pose.position.x, pose.position.y, pose.position.z, roll, pitch, yaw]

        # Round all the values in the list to 4 decimals
        rounded_pose = [round(x, 4) for x in pose_list]

        self.get_logger().warn(f"Action: {rounded_pose}")

        return rounded_pose

    def get_reward(self, state, action, next_state, valid=True, executed=True):
        # If the action returned was valid, so the action was attempted, but it could not be executed,
        # penalize the 'impossible' action with a reward of 0 and return
        if valid and not executed:
            return 0, False

        # Define "done" variable to be False initially, tracks whether episode has ended
        done = False

        # Get the current load (mA) of the gripper after lifting
        current_load = get_current_load()

        # Get the difference of items in the workspace after executing the action
        item_diff = state[-1] - next_state[-1]

        # If there's one less item and current load is >200 mA, we grabbed successfully, assign a reward of 1
        if item_diff == 1 and current_load > 160:
            reward = 1

            # If there are no more items, we have finished sorting, so set "done" to True
            if next_state[-1] == 0:
                done = True

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

            # If there are no more items, we have finished sorting, so set "done" to True
            if next_state[-1] == 0:
                done = True

        # In all other cases assign a reward of 0
        else:
            # Calculate the distance between the gripper and object centroid
            distance = np.linalg.norm(np.array(action[:3]) - np.array(state[:3]))

            # Subtract the distance from 0.4, the target reach
            target = 0.4 - distance

            # If the target reach is positive, calculate the distance-based reward
            if target > 0:
                reward = 0.5 * (target / 0.4)

            # If the target reach is negative, assign a reward of 0
            else:
                reward = 0

        self.get_logger().warn(f"Reward: {float(reward)}")

        return float(reward), done

    # def get_reward(self, state, action, next_state, valid=True, approached=True, executed=True, radius=0.0, length=0.0):
    #     # If the action returned was valid, so the action was attempted, it reached the approach, 
    #     # but it could not be executed, penalize the 'impossible' action with a reward of 0 and return
    #     # if valid and not executed:
    #     #     self.get_logger().warn(f"Reward: 0")
    #     #     return 0, False

    #     # Define "done" variable to be False initially, tracks whether episode has ended
    #     done = False

    #     # Get the current load (mA) of the gripper after lifting
    #     current_load = get_current_load()

    #     # Get the difference of items in the workspace after executing the action
    #     item_diff = state[-1] - next_state[-1]

    #     # If there's one less item and current load is >200 mA, we grabbed successfully, assign a reward of 1
    #     if item_diff == 1 and current_load > 160:
    #         reward = 1

    #         # If there are no more items, we have finished sorting, so set "done" to True
    #         if next_state[-1] == 0:
    #             done = True

    #     # If one of the two conditions for determining a successful grasp is False, query the user for reward
    #     elif (item_diff != 1 and current_load > 160) or (item_diff == 1 and current_load <= 160):
    #         # Block the program until the user has given the reward for the transition
    #         self.get_logger().warn(f"Unable to determine reward; Item Diff: {item_diff}; Current Load: {current_load} mA")
    #         self.get_logger().warn("Enter reward (0, 1, or other)")
    #         reward = input()

    #         # If the user does not enter a correct reward
    #         while reward != "0" and reward != "1":
    #             self.get_logger().warn(
    #                 "You typed '" + str(reward) + "', type '0' or '1' to record the reward or 'other' for further calculation"
    #             )
    #             reward = input()

    #         # If there are no more items, we have finished sorting, so set "done" to True
    #         if next_state[-1] == 0:
    #             done = True

    #     # In all other cases
    #     else:
    #         distance_reward = 0
    #         distance = np.linalg.norm(np.array(action[:3]) - np.array(state[:3]))
    #         distance_target = 0.3 - distance
    #         if distance_target >= 0:
    #             distance_reward = distance_target / 0.3
    #         # radius_target = 0.1 - radius
    #         # length_target = 0.08 - abs(length)
    #         # if radius_target > 0 and length_target > 0:
    #         #     distance_reward = 0.5 * ((radius_target / 0.1) + (length_target / 0.08))
            
    #         orientation_reward = 0
    #         action = self.convert_action(action)
    #         object_rotation = Rotation.from_quat([state[3], state[4], state[5], state[6]]).as_matrix()
    #         gripper_rotation = Rotation.from_quat([action[3], action[4], action[5], action[6]]).as_matrix()
    #         gripper_slope = gripper_rotation[:, 2]
    #         object_slope = object_rotation[:, 2]
    #         orientation_target = np.dot(gripper_slope, object_slope) - 0.75
    #         if orientation_target >= 0:
    #             orientation_reward = orientation_target / 0.75

    #         reward = 0.25 * (distance_reward + orientation_reward)
    #         # # If the action returned was valid, so the action was attempted, but it could not reach the approach,
    #         # # but it could be executed, penalize actions with bad grab orientations, only for unsuccessful grab
    #         # if valid and not approached and executed:
    #         #     reward *= 0.2

    #     self.get_logger().warn(f"Reward: {float(reward)}")

    #     return float(reward), done

    def add_to_buffer(self, buffer, state, action, next_state, reward, done, imitate=True):
        # If in 'imitate' mode
        if imitate:
            # Block the program until the user has given the reward for the transition
            self.get_logger().warn("Add previous state, action, transition, and reward to buffer? (Enter)")
            answer = input()

            # If they do not click "Enter", notify the user
            while answer != "":
                self.get_logger().warn("You typed '" + str(answer) + "', click 'Enter' to add to the buffer and continue")
                answer = input()

        # Add to the replay buffer
        buffer.add(state, action, next_state, reward, done)

        self.get_logger().info("State, action, transition, and reward added to replay buffer")

    def save_buffer(self, buffer, buffer_path):
        # Save the replay buffer and a copy of it
        np.save(buffer_path, buffer)
        np.save(buffer_path + "_copy", buffer)

        self.get_logger().info("Saved the replay buffer successfully")

    def sample_from_buffers(self, expert_buffer, online_buffer, expert_batch_size=100, online_batch_size=100):
        # Sample from each buffer the corresponding amount of samples
        e_state, e_action, e_next_state, e_reward, e_not_done = expert_buffer.sample(expert_batch_size)
        o_state, o_action, o_next_state, o_reward, o_not_done = online_buffer.sample(online_batch_size)

        # Combine the samples
        state = torch.cat([e_state, o_state], dim=0)
        action = torch.cat([e_action, o_action], dim=0)
        next_state = torch.cat([e_next_state, o_next_state], dim=0)
        reward = torch.cat([e_reward, o_reward], dim=0)
        not_done = torch.cat([e_not_done, o_not_done], dim=0)
        
        return (state, action, next_state, reward, not_done)

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

        executed = True if self.execution_status == "SUCCEEDED" else False
        self.execution_status = None

        return executed

    def select_random_action(self, min_action, max_action):
        # Convert the lists representing action bounds to arrays
        min_action = np.array(min_action)
        max_action = np.array(max_action)
        
        # Sample a random action
        action = np.random.uniform(min_action, max_action)
        
        # Round the action to 8 decimals
        action = np.round(action, decimals=8)
        
        return action

    def execute_action(self, state, action, initial_tolerance=0.01, tolerance_increase=0.01, max_tolerance=0.1):
        # Set approached and executed to be False initially
        approached = False
        executed = False

        # Log the action to execute
        self.get_logger().warn("Action " + str(list(action)))

        # Convert the action from its 3-angle representation into a quaternion
        action = self.convert_action(action)

        # First check that the action is valid with a distance check
        valid, radius, length = self.distance_check(state, action)

        # If the action is valid
        if valid:
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

                # If an IK solution is not found
                if not found_ik:
                    # Define the pose goal
                    action_position = construct_link_constraint(
                        "tcp",
                        "world",
                        [action[0], action[1], action[2]],
                        initial_tolerance,
                        [action[3], action[4], action[5], action[6]],
                        initial_tolerance,
                    )

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

            # If the approach position could not be reached
            elif not approached:
                # Query the user if they would still like to execute the action
                self.get_logger().warn("Could not reach approach, execute action anyway (y/n)")
                answer = input()

                # If they do not type "y" or "n", notify the user
                while answer != "y" and answer != "n":
                    self.get_logger().warn("You typed '" + str(answer) + "', type 'y' to execute the action or 'n' to cancel")
                    answer = input()

                # If the user still wants to execute the action
                if answer == "y":
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

                    # If an IK solution is not found
                    if not found_ik:
                        # Define the pose goal
                        action_position = construct_link_constraint(
                            "tcp",
                            "world",
                            [action[0], action[1], action[2]],
                            initial_tolerance,
                            [action[3], action[4], action[5], action[6]],
                            initial_tolerance,
                        )

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

        return valid, approached, executed, radius, length

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

        # If an IK solution is not found
        if not found_ik:
            # Define the pose goal
            approach_position = construct_link_constraint(
                "tcp",
                "world",
                [x, y, z],
                initial_tolerance,
                [action[3], action[4], action[5], action[6]],
                initial_tolerance,
            )

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

            # If an IK solution is not found
            if not found_ik:
                # Redefine the lift position with an increased tolerance
                approach_position = construct_link_constraint(
                    "tcp",
                    "world",
                    [x, y, z],
                    tolerance,
                    [action[3], action[4], action[5], action[6]],
                    tolerance,
                )

            # Pass the approach position to the move function
            approached = self.go_to(approach_position)

            # Increase the tolerance and try the approach again
            tolerance += tolerance_increase

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

        # If an IK solution is not found
        if not found_ik:
            # Define the pose goal
            lift_position = construct_link_constraint(
                "tcp",
                "world",
                [pose.position.x, pose.position.y, pose.position.z],
                initial_tolerance,
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                0.01,
            )

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

    # def distance_check(self, state, action, height=0.08, radius=0.06):
    #     # Define the gripper position and object position
    #     gripper_position = np.array([action[0], action[1], action[2]])
    #     object_position = np.array([state[0], state[1], state[2]])

    #     # Get the rotation matrix from the quaternion
    #     rotation_matrix = Rotation.from_quat([action[3], action[4], action[5], action[6]]).as_matrix()

    #     # Get the slope of the 3D line for the cylinder by getting the third column vector (negative z-axis)
    #     slope = -rotation_matrix[:, 2]

    #     # Calculate the projection of the object position onto the slope line
    #     projection = np.dot(slope, object_position - gripper_position)

    #     # Convert projection back into 3D space
    #     projection_3d = gripper_position + projection * slope

    #     # Calculate the distance between the object position and its projection
    #     proj_distance = np.linalg.norm(object_position - projection_3d)

    #     # If the projection distance is less than the radius and the projection is within the height, return True
    #     if (proj_distance <= radius) and (0.0 <= projection <= height):
    #         result = True
    #         self.get_logger().info(f"Distance check passed: radial distance: {proj_distance}; length distance: {projection}")

    #     # If the check is not passed, return False and notify the user
    #     else:
    #         result = False
    #         self.get_logger().info(f"Distance check failed: radial distance: {proj_distance}; length distance: {projection}")

    #     return result

    def distance_check(self, state, action, height=0.08, radius=0.10):
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

        # In all other cases calculate
        else:
            #
            t_star = -(object_difference_correlation - grip_difference_correlation * slope_correlation) / denom

        # If the object is a bottle, set the half length to the following (cm)
        if state[7] == 0:
            half_length = 0.12

        # If the object is a can, set the half length to the following (cm)
        elif state[7] == 1:
            half_length = 0.07

        # Clamp to segment
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

    def convert_action(self, action):
        # Define vectors to represent the coordinate axes
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, 1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])

        # Multiply the roll by the x-axis vector and convert to a transformation matrix
        # In essence: Rotate about the x-axis equivalent to roll, convert roll to a transformation matrix
        roll_matrix = Rotation.from_rotvec(action[3] * x_axis)

        # Multiply the pitch by the y-axis vector and convert to a transformation matrix
        pitch_matrix = Rotation.from_rotvec(action[4] * y_axis)

        # Multiply the pitch and roll transformation matrices to get the tilt orientation of the gripper
        tilt_matrix = roll_matrix * pitch_matrix

        # Get the third column vector from the tilt matrix
        # In essence: Extract the rotation vector for the z axis, apply the rotations from vector to z axis
        approach = tilt_matrix.apply(z_axis)

        # Multiply the yaw by the third column vector to the get the transformation matrix
        # In essence: Rotate about the z axis in the local gripper frame
        yaw_matrix = Rotation.from_rotvec(action[5] * approach)

        # Multiply the yaw rotation matrix by the tilt rotation matrix to get the final transformation matrix
        rotation_matrix = yaw_matrix * tilt_matrix

        # Convert the transformation matrix to a quaternion
        quaternion = rotation_matrix.as_quat()

        # Define the action to execute
        action = [action[0], action[1], action[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]

        return action

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
        conveyor_pose.position.z = 0.535
        conveyor_pose.orientation.w = 1.0
        conveyor_collision.primitive_poses.append(conveyor_pose)

        bar_pose = Pose()
        bar_pose.position.x = 0.4
        bar_pose.position.y = 0.0
        bar_pose.position.z = 1.91
        bar_pose.orientation.w = 1.0
        bar_collision.primitive_poses.append(bar_pose)

        cam_pose = Pose()
        cam_pose.position.x = 0.32
        cam_pose.position.y = 0.0
        cam_pose.position.z = 1.86
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

    def loop_check(self):
        # Print statement for looks
        print("")

        # Spin the node to process parameter changes
        rclpy.spin_once(self, timeout_sec=0.01)

        # Get the current parameter value
        run = self.get_parameter("run").value

        return run
