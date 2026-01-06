#!/usr/bin/env python3

"""
This script provides the main Class and all of the functions for capturing state transitions for
the replay buffer and training the robot with RD3 RL to sort recyclables (train.py and imitate.py)
"""

import os
import torch
import rclpy
import numpy as np
from time import sleep
from rclpy.node import Node
from recycrl.srv import Poses
from TD3.utils import ReplayBuffer
from lifecycle_msgs.msg import Transition
from scipy.spatial.transform import Rotation
from lifecycle_msgs.srv import GetState, ChangeState
from ament_index_python import get_package_share_directory
from rclpy.logging import set_logger_level, LoggingSeverity
from kuka_kontrol.grip_utils import gripper_to_pos, get_current_load
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_joint_constraint, construct_link_constraint


class RecycRL(Node):
    def __init__(self, buffer_path, state_dim=8, action_dim=6):
        # Register the ROS2 Node
        super().__init__("recycrl")

        # Set all loggers to warning or above, but keep this Node at info level
        set_logger_level("", LoggingSeverity.ERROR)
        self.get_logger().set_level(LoggingSeverity.INFO)

        # Define global variables
        self.buffer_path = buffer_path
        self.execution_status = None
        self.executed = False
        self.timed_out = False

        # If the replay buffer does not exist, initialize a new ReplayBuffer() Class
        if not os.path.exists(buffer_path):
            self.buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
        # If the replay buffer exists, load the Class
        else:
            self.buffer = np.load(buffer_path, allow_pickle=True).item()

        """
        (JK) See get_robot_state()
        # Get the GPU if it is available
        self.processor = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Define the model and put it on the GPU
        self.network = NeuralNetwork().to(self.processor)
        """

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

        # Define pre-set positions for KUKA
        self.home = self.construct_joint_position([0.0, -1.74533, 1.5708, 0.0, 1.74533, 0.0])
        self.bin = self.construct_joint_position([-1.5708, -1.13446, 1.48353, 0.0, 1.22173, -1.5708])
        self.hidden = self.construct_joint_position([-1.5708, -1.5708, 1.5708, 0.0, 1.5708, 0.0])

        # Define requests and clients
        self.get_poses_request = Poses.Request()
        self.get_state_request = GetState.Request()
        self.change_state_request = ChangeState.Request()
        self.get_poses_client = self.create_client(Poses, "/get_poses")
        self.get_state_client = self.create_client(GetState, "/robot_manager/get_state")
        self.change_state_client = self.create_client(ChangeState, "/robot_manager/change_state")

        # Wait for the services to become available
        while not self.get_poses_client.wait_for_service(4.0):
            self.get_logger().info("Waiting for YOLO service ...")
        while not self.get_state_client.wait_for_service(4.0):
            self.get_logger().info("Waiting for robot get state service ...")
        while not self.change_state_client.wait_for_service(4.0):
            self.get_logger().info("Waiting for robot change state service ...")

    def trajectory_callback(self, msg):
        # Assign the message to the global variable
        self.execution_status = msg.status

        # Set True to exit the while loop in the execute_action() function
        self.executed = True

    def set_workspace_state(self):
        # Block the program until the user has notified that the workspace has been reset
        self.get_logger().info("Robot moved to hidden pose, reset workspace if sort has finished")
        self.get_logger().warn("Workspace ready? (yes)")
        answer = input()

        # If they do not say "yes", make sure the user is sure
        while answer != "yes":
            self.get_logger().warn("You typed '" + str(answer) + "', type 'yes' to capture the current workspace and continue")
            answer = input()

    # (JK) Need to pass one pose and remaining number of items
    def get_workspace_state(self):
        # Call the get poses service, spin the Node until a response is received, and define the response
        get_poses_future = self.get_poses_client.call_async(self.get_poses_request)
        rclpy.spin_until_future_complete(self, get_poses_future)
        get_state_response = get_poses_future.result()

        # Get the first pose from the response, convert the pose into a list, and round all the values
        pose = get_state_response.poses[0]
        pose_list = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rounded_pose = [round(x, 4) for x in pose_list]

        return rounded_pose

    def set_robot_state(self):
        # Block the program until the user has notified that the  robot state has been set
        self.get_logger().info("Robot moved to Home, provide pose for training")
        self.get_logger().warn("Finished hand-guiding robot? (yes)")
        answer = input()

        # If they do not say "yes", notify the user
        while answer != "yes":
            self.get_logger().warn("You typed '" + str(answer) + "', type 'yes' if you have provided a hand-guided pose")
            answer = input()

    # (JK) Transformation matrix for rotation?
    def get_robot_state(self):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("gripper_base_link")

        # Convert the Pose message into a list and round all the values in the list to 4 decimals
        pose_list = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rounded_pose = [round(x, 4) for x in pose_list]

        return rounded_pose

    # (JK) Is this necessary? Transformation matrix?
    def get_action(self, pose):
        # Convert the workspace state into a list and round all the values in the list to 4 decimals
        pose_list = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]
        rounded_pose = [round(x, 4) for x in pose_list]

        # Convert the pose representation to a tensor
        pose_tensor = torch.tensor(rounded_pose).to(self.processor)

        # Pass the pose through the neural network and get the output from the GPU
        output = self.network(pose_tensor).detach().cpu().numpy()

        # Extract and normalize the first column vector of the 6D output orientation
        b1 = self.normalize(np.array([output[3], output[4], output[5]]))

        # Define the second column vector of the 6D output orientation
        a2 = np.array([output[6], output[7], output[8]])

        # Calculate an orthogonal vector of a2 to b1 and normalize it
        b2 = self.normalize(a2 - np.dot(b1, a2) * b1)

        # Recover the third column vector by computing the cross product of b1 and b2
        b3 = np.cross(b1, b2)

        # Define the rotation matrix from the column vectors
        rotation_matrix = [[b1[0], b2[0], b3[0]], [b1[1], b2[1], b3[1]], [b1[2], b2[2], b3[2]]]

        # Transform the rotation matrix into a quaternion
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

        # Define the action to execute
        action = [output[0], output[1], output[2], quaternion[0], quaternion[1], quaternion[2], quaternion[3]]

        return action

    # (JK) Debug
    def get_reward(self, prev_items, items):
        # Define "done" variable to be False initially, tracks whether episode has ended
        done = False

        # Get the current load (mA) of the gripper after lifting
        current_load = get_current_load()

        # Get the difference of items in the workspace after executing the action
        item_diff = prev_items - items

        # If there is one less item and the current load is greater than 200 mA, we have grasped something
        if item_diff == 1 and current_load > 200 and items == 0:
            # If we there are no more items, we have finished sorting, so assign a large reward and set "done" to True
            if items == 0:
                reward = 10
                done = True

            # If there are items remaining, give a smaller reward
            else:
                reward = 1

        # If one of the two conditions for determining a successful grasp is False, query the user for reward
        if (item_diff != 1 and current_load > 200) or (item_diff == 1 and current_load <= 200 and items == 0):
            # Block the program until the user has given the reward for the transition
            self.get_logger().warn("Unable to determine reward for action")
            self.get_logger().warn("Enter reward (0, 1, or 10)")
            reward = int(input())

            # If the user does not enter a correct reward
            while reward != 0 or reward != 1 or reward != 10:
                self.get_logger().warn(
                    "You typed '" + str(reward) + "', type '0', '1', or '10' to record the reward and continue"
                )
                reward = input()

        # In all other cases assign a reward of 0
        else:
            reward = 0

        return reward, done

    def save_to_buffer(self, state, action, next_state, reward, done):
        # Add to the replay buffer
        self.buffer.add(state, action, next_state, reward, done)

        self.get_logger().info("State, action, transition, and reward saved to replay buffer\n")

    # (JK) Flush out
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

        # Print error if planning fails
        else:
            self.get_logger().error("Planning failed")

    # Add distance check (JK)
    def execute_action(self, action):
        # Convert the action and pass it to the go_to() function
        position = construct_link_constraint(
            "gripper_base_link",
            "world",
            [action[0], action[1], action[2]],
            0.0,
            [action[3], action[4], action[5], action[6]],
            0.01,
        )
        self.go_to(position)

    # (JK) Need to add invalid position handling
    def lift(self, height=0.12):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("gripper_base_link")

        # Calculate the desired height to lift the gripper
        z_after_lift = pose.position.z + height

        # Build the pose goal position as a pose constraint
        lift_position = construct_link_constraint(
            "gripper_base_link",
            "world",
            [pose.position.x, pose.position.y, z_after_lift],
            0.0,
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            0.01,
        )

        # Pass the lift position to the move function
        self.go_to(lift_position)

    def grab_and_lift(self):
        # Close the gripper
        closed = gripper_to_pos(55)

        # If the gripper closed
        if closed:
            # Lift the object slightly
            self.lift()

            # Check that the lift action succeeded
            if self.execution_status == "SUCCEEDED":
                # Go to the bin
                self.go_to(self.bin)

                # Open the gripper to drop the item
                release_result = gripper_to_pos(25)

            elif self.execution_status != "SUCCEEDED":
                self.get_logger().error("Unable to lift gripper, trying again")

        # If the gripper did not close, notify the user
        elif not closed:
            self.get_logger().warn("Failed to close gripper, check that gripper is powered\nGoing back home")

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

    def construct_joint_position(self, angles):
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
        joint_position = construct_joint_constraint(self.robot_state, self.joint_group, 0.01)

        return joint_position

    def normalize(self, matrix):
        # Define a variable to be assigned locally in upcoming loop
        sum = 0

        # Loop through element in the matrix
        for element in matrix:
            # Square the element and add t the running sum
            sum += element**2

        # Get the square root of the sum to acquire the magnitude of the vector
        magnitude = np.sqrt(sum)

        # Divide the matrix by the magnitude to normalize it
        normalized = matrix / magnitude

        return normalized
