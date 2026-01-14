#!/usr/bin/env python3

"""
This node takes detected objects from the detection service and disposes them
"""

import rclpy
from time import sleep
from math import radians
from rclpy.node import Node
from recycrl.srv import Poses
from sensor_msgs.msg import Image
from tf_transformations import quaternion_from_euler
from ament_index_python import get_package_share_directory
from rclpy.logging import set_logger_level, LoggingSeverity
from kuka_kontrol.grip_utils import gripper_to_pos, get_finger_pos
from message_filters import Subscriber, ApproximateTimeSynchronizer

from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_link_constraint, construct_joint_constraint


def main():
    # Define variables, gripper length and approach height in meters
    gripper_length = 0.16
    approach_height = 0.06

    # Initialize rclpy and the PickandPlace Node
    rclpy.init()
    pnp = PickandPlace(gripper_length, approach_height)

    # Start the loop
    while True:
        # Move the robot to the home
        pnp.go_home()

        # Get the location of the object to grab
        object_location = pnp.get_object_location()

        # If there were any detected objects
        if object_location:
            # Query the operator for the desired yaw angle of pickup
            object_yaw = pnp.query_operator_for_angle()

            # Go to the approach point for the object location and specified yaw
            pnp.approach(object_location, object_yaw)

            # Drop, grab, lift, take to bin, and drop the item
            pnp.grab_and_dispose()

        # If no objects detected
        else:
            break


class PickandPlace(Node):
    def __init__(self, gripper_length, approach_height):
        # Register the ROS2 Node
        super().__init__("pick_and_place")

        # Set all loggers to warning or above, but keep this Node at info level
        set_logger_level("", LoggingSeverity.ERROR)
        self.get_logger().set_level(LoggingSeverity.INFO)

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

        # Create a subscriber for RGB and Depth images and synchronize them
        self.rgb_sub = Subscriber(self, Image, "/oak/rgb/image_raw")
        self.depth_sub = Subscriber(self, Image, "/oak/stereo/image_raw")
        self.sync_sub = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 1, 0.1)
        self.sync_sub.registerCallback(self.sub_callback)

        # Define global variables
        self.gripper_length = gripper_length
        self.approach_height = approach_height

        self.rgb = None
        self.depth = None
        self.execution_status = None

        self.timed_out = False

        self.request = Poses.Request()

        # Define pre-set positions for KUKA
        self.home = self.construct_joint_position([0.0, -1.74533, 1.5708, 0.0, 1.74533, 0.0])
        self.bin = self.construct_joint_position([-1.5708, -1.13446, 1.48353, 0.0, 1.22173, -1.5708])

        # Create a client to the detection service
        self.client = self.create_client(Poses, "/get_poses")

        # Wait until the service is running
        while not self.client.wait_for_service(4.0):
            self.get_logger().info("Waiting for detection service ...")

    def sub_callback(self, rgb_msg, depth_msg):
        # Assign the most recent Images to the global variables
        self.rgb = rgb_msg
        self.depth = depth_msg

    def trajectory_callback(self, msg):
        # Assign the message to the global variable
        self.execution_status = msg.status

    def go_home(self):
        # Go to the home position
        self.go_to(self.home)

        # Check that the homing action succeeded
        if self.execution_status == "SUCCEEDED":
            sleep(0.5)

        # If the homing action failed
        elif self.execution_status != "SUCCEEDED":
            self.get_logger().error("Unable to go to the home position")

    # Flush out, add operator query on failure for constraint or different spot (JK)
    def go_to(self, position):
        # Set for blocking call
        self.execution_status = None

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
                while self.trajectory_manager.is_managing_controllers() and not self.execution_status:
                    sleep(0.05)
            except KeyboardInterrupt:
                self.get_logger().warn("Stopped Execution")
                self.trajectory_manager.stop_execution()

        # Print error if planning fails
        else:
            self.get_logger().error("Planning failed")

    def query_operator_for_angle(self):
        self.get_logger().warn("Enter yaw angle [-180, 180]")
        answer = input()  # Takes key input, but need to press "Enter"

        # If they do not provide a valid response, try again
        if 180 < float(answer) < -180:
            self.get_logger().warn("Enter a value between -180 and 180 degrees")
            answer = input()  # Takes key input, but need to press "Enter"

        return float(answer)

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

    def construct_pose_position(self, pose):
        # Take the passed list and build a pose position
        pose_position = construct_link_constraint(
            "gripper_base_link",
            "world",
            [pose[0], pose[1], pose[2]],
            0.0,
            [pose[3], pose[4], pose[5], pose[6]],
            0.01,
        )

        return pose_position

    # Error on failed planning, hardcoded values (JK)
    def lift(self):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("gripper_base_link")

        # # Calculate the desired height to lift the gripper
        # z_after_lift = pose.position.z + self.approach_height

        # For now lets just hardcode the z
        z_after_lift = 0.99

        # Define the lift pose by replacing the current position with the desired lift amount
        lift_pose = [
            pose.position.x,
            pose.position.y,
            z_after_lift,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]

        # Pass the list above to the pose position constructor to get the lift position
        lift_position = self.construct_pose_position(lift_pose)

        # Pass the lift position to the move function
        self.go_to(lift_position)

    # Error on failed planning, hardcoded values (JK)
    def dip(self):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("gripper_base_link")

        # # Calculate the desired height to lift the gripper
        # z_after_dip = pose.position.z - self.approach_height

        # For now lets hardcode this
        z_after_dip = 0.94

        # Define the lift pose by replacing the current position with the desired lift amount
        dip_pose = [
            pose.position.x,
            pose.position.y,
            z_after_dip,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]

        # Pass the list above to the pose position constructor to get the lift position
        dip_position = self.construct_pose_position(dip_pose)

        # Pass the lift position to the move function
        self.go_to(dip_position)

    def grab_and_dispose(self):
        # Dip the gripper
        self.dip()

        # If the dip action was successful
        if self.execution_status == "SUCCEEDED":
            # Close the gripper
            closed = gripper_to_pos(65)

            # Get the gripper position and block until it is closed
            gripper_position = get_finger_pos("a")
            while gripper_position == 22:
                gripper_position = get_finger_pos("a", sleep_time=0.1)

            # If the gripper closed
            if closed:
                # Lift the object slightly
                self.lift()

                # Check that the lift action succeeded
                if self.execution_status == "SUCCEEDED":
                    # Go to the bin
                    self.go_to(self.bin)

                    # Open the gripper to drop the item
                    gripper_to_pos(22, sleep_time=0.5)

                    # Get the current gripper position and block until it is open
                    open_pos = get_finger_pos("a")
                    while open_pos != 22:
                        open_pos = get_finger_pos("a", sleep_time=0.1)

                elif self.execution_status != "SUCCEEDED":
                    self.get_logger().error("Unable to lift gripper, trying again")

            # If the gripper did not close, notify the user
            elif not closed:
                self.get_logger().warn("Failed to close gripper, check that gripper is powered\nGoing back home")

        # If the dip action was unsuccessful
        elif self.execution_status != "SUCCEEDED":
            self.get_logger().error("Unable to dip gripper, trying again")

    def get_object_location(self):
        # Call the detection service
        future = self.client.call_async(self.request)

        # Spin the node until the request has been complete
        rclpy.spin_until_future_complete(self, future)

        # Get the list of poses from the response
        poses = future.result().poses

        # If there are any detected objects
        if len(poses) > 0:
            # Define the pose from the passed Pose
            pose = [poses[0].position.x, poses[0].position.y, poses[0].position.z]

            # Print the object location
            self.get_logger().info("Going to X: " + str(pose[0]) + "; Y: " + str(pose[1]) + "; Z: " + str(pose[2]))

            return pose

        # If there aren't any objects detected
        else:
            self.get_logger().info("No bottles detected in the workspace")

            return None

    # Error on failed planning, hardcoded values (JK)
    def approach(self, position, yaw):
        # Convert the yaw angle from degrees to radians
        yaw = radians(yaw)

        # Build a quaternion from the yaw angle
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw)

        # # Add the approach height and gripper length to get the desired z value
        # z = position[2] + self.gripper_length + self.approach_height

        # For now lets hard code this
        z = 0.99

        # Pass the position and quaternion to the construct pose function
        approach = self.construct_pose_position([position[0], position[1], z, qx, qy, qz, qw])

        # Go the the approach point
        self.go_to(approach)


if __name__ == "__main__":
    main()
