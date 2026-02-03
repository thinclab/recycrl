#!/usr/bin/env python3

"""
This Node resets the robot by reactivating it, opening the gripper, and sending it home
"""

import rclpy
from time import sleep
from rclpy.node import Node
from geometry_msgs.msg import Pose
from lifecycle_msgs.msg import Transition
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject
from kuka_kontrol.grip_utils import gripper_to_pos
from lifecycle_msgs.srv import GetState, ChangeState
from ament_index_python import get_package_share_directory
from rclpy.logging import set_logger_level, LoggingSeverity
from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.kinematic_constraints import construct_joint_constraint, construct_link_constraint


def main():
    # Initialize rclpy and the Reset() Node, then call the reset() function
    rclpy.init()
    reset = Reset()
    reset.reset()


class Reset(Node):
    def __init__(self):
        # Register the ROS2 Node
        super().__init__("reset")

        # Set all loggers to warning or above, but keep this Node at info level
        set_logger_level("", LoggingSeverity.ERROR)
        self.get_logger().set_level(LoggingSeverity.INFO)

        # Define global variables
        self.execution_status = None
        self.executed = False

        # Define requests and clients
        self.get_state_request = GetState.Request()
        self.change_state_request = ChangeState.Request()
        self.get_state_client = self.create_client(GetState, "/robot_manager/get_state")
        self.change_state_client = self.create_client(ChangeState, "/robot_manager/change_state")

        # Wait for the services to become available
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

    def trajectory_callback(self, msg):
        # Assign the message to the global variable
        self.execution_status = msg.status

        # Set True to exit the while loop in the go_to() function
        self.executed = True

    def reset(self):
        # Open the gripper
        opened = self.open_gripper()

        # If the gripper opened successfully
        if opened:
            self.get_logger().info("Gripper opened successfully")

            # Get the current pose of the robot
            with self.planning_scene.read_only() as scene:
                pose = scene.current_state.get_pose("tcp")

            # If the current height of the gripper is less than 92.5 cm
            if pose.position.z < 0.925:
                # Lift the gripper
                lifted = self.lift()

            # If the current height of the gripper is greater than or equal to 92.5 cm
            elif pose.position.z >= 0.925:
                # Do not lift the gripper and set the logic variable to True
                lifted = True

            # If the gripper lifted successfully
            if lifted:
                self.get_logger().info("Gripper lifted successfully")

                # Go to the home position
                self.go_to(self.home)

            # If the gripper was not lifted
            elif not lifted:
                self.get_logger().warn("Gripper could not reach lift position")

        # If the gripper did not open notify the user
        elif not opened:
            self.get_logger().warn("Gripper could not be opened")

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

            return True

        # Otherwise, return False
        else:
            return False

    def lift(self, initial_tolerance=0.05, tolerance_increase=0.05, max_tolerance=0.4):
        # Get the current pose of the robot
        with self.planning_scene.read_only() as scene:
            pose = scene.current_state.get_pose("tcp")

        # Calculate the desired height to lift the gripper
        pose.position.z = 0.925

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

    def open_gripper(self, pos=25, sleep_time=0.2):
        # Open the gripper
        opened = gripper_to_pos(pos, sleep_time=sleep_time)

        return opened

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
        conveyor_pose.position.x = 0.43
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


if __name__ == "__main__":
    main()
