#!/usr/bin/env python3

"""
This script provides the ability to save Images and matching robot poses for Imitation Learning Data
"""

import os
import cv2
import rclpy
from csv import writer
from time import sleep
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import GetState, ChangeState
from ament_index_python import get_package_share_directory
from rclpy.logging import set_logger_level, LoggingSeverity
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_joint_constraint


def main():
    # Define variables
    data_dir = os.path.expanduser("~/Recycrl_Data/Offline")

    # Initialize rclpy and the Imitate Node
    rclpy.init()
    imitate = Imitate(data_dir)

    # imitate.go_to(imitate.home)
    # Start the loop, ends after time out reached
    while not imitate.timed_out:
        # Move the robot to the hidden state so that an image can be taken clearly
        imitate.go_to(imitate.hidden)

        # Blocking call that asks user to rearrange state for new image
        imitate.set_workspace_state()

        # Take a picture of the state after the user has rearranged the items, and save it locally
        img_name = imitate.capture_workspace_state()

        # Move the robot to home
        imitate.go_to(imitate.home)

        # Deactivate the KUKA robot's external control to allow operator to hand guide robot
        imitate.deactivate_external_control()

        # Wait for the operator to move the robot to the desired pose
        imitate.set_robot_state()

        # Once the operator has provided a pose, reactivate the robot's external control
        imitate.activate_external_control()

        # Get the current state of the robot now that external control is active
        robot_state = imitate.get_robot_state()

        # Match the image with the provided pose and save to provide a data pair for IL
        imitate.save_data(img_name, robot_state)


class Imitate(Node):
    def __init__(self, data_dir):
        # Register the ROS2 Node
        super().__init__("recycle")

        # Set all loggers to warning or above, but keep this Node at info level
        set_logger_level("", LoggingSeverity.ERROR)
        self.get_logger().set_level(LoggingSeverity.INFO)

        # Define the Cv Bridge and empty image variable
        self.bridge = CvBridge()

        # Define global variables
        self.data_dir = data_dir
        self.rgb = None
        self.image_name = None
        self.execution_status = None
        self.executed = False
        self.timed_out = False
        self.image_received = False

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

        # Create a subscriber for RGB Images
        self.sub = self.create_subscription(Image, "/oak/rgb/image_raw", self.sub_callback, 1)

        # Define pre-set positions for KUKA
        self.home = self.construct_joint_position([0.0, -1.74533, 1.5708, 0.0, 1.74533, 0.0])
        self.hidden = self.construct_joint_position([-1.5708, -1.5708, 1.5708, 0.0, 1.5708, 0.0])

        # Define a pair of requests and clients to get and change the current state of the KUKA robot
        self.get_state_request = GetState.Request()
        self.change_state_request = ChangeState.Request()
        self.get_state_client = self.create_client(GetState, "/robot_manager/get_state")
        self.change_state_client = self.create_client(ChangeState, "/robot_manager/change_state")

        # Wait for the robot state services to become available
        while not self.get_state_client.wait_for_service(4.0):
            self.get_logger().info("Make sure Robot is Active ...")
        while not self.change_state_client.wait_for_service(4.0):
            self.get_logger().info("Make sure Robot is Active ...")

    def sub_callback(self, msg):
        # Assign the message to the global variable
        self.rgb = msg

    def trajectory_callback(self, msg):
        # Assign the message to the global variable
        self.execution_status = msg.status

        # Set True to exit the while loop in the execute_action() function
        self.executed = True

    def capture_workspace_state(self):
        # Check that the camera is publishing, and notify the user if it is not
        while not self.get_publishers_info_by_topic("/oak/rgb/image_raw"):
            self.get_logger().error("Camera is inactive")
            sleep(5.0)

        # Reset the RGB image so that spin gets the most recent
        self.rgb = None

        # Spin until the messages are received
        while not self.rgb:
            rclpy.spin_once(self)

        # Convert the message to a cv image
        cv_image = self.bridge.imgmsg_to_cv2(self.rgb, desired_encoding="bgr8")

        # Get the list of previously saved images
        images = os.listdir(self.data_dir)

        # If there is an annotations file in the folder, get rid of it
        if "annotations.csv" in images:
            images.pop(images.index("annotations.csv"))

        # If images exist
        if images:
            # Define variable to be assigned locally
            largest_image_number = 0

            # Go through each image
            for image in images:
                # Extract the number from the image name
                image_number = int(image[len("image_") : -len(".png")])

                # If the current image number is greater than the largest, assign this to be the new largest
                if image_number > largest_image_number:
                    largest_image_number = image_number

            # Define the image name
            self.image_name = f"image_{largest_image_number + 1}"

        # If there are no images, create the first one
        else:
            self.image_name = "image_0"

        # Save the image
        cv2.imwrite(f"{self.data_dir}/{self.image_name}.png", cv_image)

        return self.image_name

    # Flush out (JK)
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
                self.get_logger().warn("Stopped Execution")
                self.trajectory_manager.stop_execution()

        # Print error if planning fails
        else:
            self.get_logger().error("Planning failed")

    # GUI? (JK)
    def set_workspace_state(self):
        # Block the program until the user has to notified that the workspace has been reset
        self.get_logger().info("Robot moved to hidden pose, reset workspace")
        self.get_logger().warn("Finished resetting workspace? (yes)")
        answer = input()

        # If they do not say "yes", make sure the user is sure
        while answer != "yes":
            self.get_logger().warn("You typed '" + str(answer) + "', type 'yes' to capture the current workspace and continue")
            answer = input()

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
            self.get_logger().error("Request for Get State Service failed")

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
            self.get_logger().error("Request for Get State Service failed")

    # GUI?(JK)
    def set_robot_state(self):
        # Block the program until the user has notified that the  robot state has been set
        self.get_logger().info("Robot moved to Home, provide pose for training")
        self.get_logger().warn("Finished hand-guiding robot? (yes)")
        answer = input()

        # If they do not say "yes", make sure the user is sure
        while answer != "yes":
            self.get_logger().warn("You typed '" + str(answer) + "', type 'yes' to capture the current workspace and continue")
            answer = input()

    # Add orientation filtering? (JK)
    def get_robot_state(self):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("gripper_base_link")

        # Transform the quaternion to euler angles
        roll, pitch, yaw = euler_from_quaternion([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Convert euler angles back to quaternion with roll and pitch set to zero to preserve only yaw angle
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        # Convert the Pose message into a list
        pose_list = [pose.position.x, pose.position.y, pose.position.z, qx, qy, qz, qw]

        # # Convert the Pose message into a list
        # pose_list = [
        #     pose.position.x, pose.position.y, pose.position.z,
        #     pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        # ]

        # Round all the values in the list to 4 decimals
        rounded_pose = [round(x, 4) for x in pose_list]

        return rounded_pose

    def save_data(self, img_name, robot_state):
        # Outputs the data to a .csv file
        file_location = f"{self.data_dir}/annotations.csv"
        with open(file_location, mode="a", newline="") as file:
            appendor = writer(file)
            appendor.writerow([f"{self.data_dir}/{img_name}.png,{robot_state}"])

        self.get_logger().info("Image and robot state recorded\n")

        # Reset the image name
        self.image_name = None

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


if __name__ == "__main__":
    main()
