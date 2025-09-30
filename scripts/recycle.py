#!/usr/bin/env python3

"""
This script provides the Global Policy for Recycling with the KUKA
"""

import rclpy
from time import sleep
from rclpy.node import Node
from sensor_msgs.msg import Image
from lifecycle_msgs.msg import Transition
from kuka_kontrol.grip_utils import gripper_to_pos, get_current_load
from lifecycle_msgs.srv import GetState, ChangeState
from ament_index_python import get_package_share_directory
from rclpy.logging import set_logger_level, LoggingSeverity
from message_filters import Subscriber, ApproximateTimeSynchronizer

from moveit.core.robot_state import RobotState
from moveit_configs_utils import MoveItConfigsBuilder
from moveit.planning import MoveItPy, PlanRequestParameters, MultiPipelinePlanRequestParameters
from moveit.core.kinematic_constraints import construct_link_constraint, construct_joint_constraint


def main():
    # Initialize rclpy and the Recycle Node
    rclpy.init()
    recycle = Recycle()

    # Start the loop, ends after time out reached
    while not recycle.timed_out:
        # Define variables that are assigned in local blocks
        action = None
        teach_response = None
        pose_response = None

        # Go to home if not already
        recycle.go_to(recycle.home)

        # If in training, need to allow for optional hand-guiding for teaching
        if recycle.train:
            # Ask the operator if they would like to provide hand-guiding
            teach_response = recycle.query_operator()

            # If the operator would like to provide hand-guiding teaching
            if teach_response == "yes":
                # Deactivate the KUKA robot's external control to allow operator to hand guide robot
                recycle.deactivate_external_control()

                # Wait for the operator to move the robot to the desired pose
                pose_response = recycle.wait_for_operator()

                # Once the operator has provided a pose, reactivate the robot's external control
                recycle.activate_external_control()

                # Get the current pose of the robot
                with recycle.planning_scene.read_only() as scene:
                    pose = scene.current_state.get_pose("gripper_base_link")
                recycle.get_logger().info(str(pose))

            # If the operator would like the Neural Network to provide its own exploratory actions
            elif teach_response == "no":
                # Process the current Image to get the current state of the workspace
                recycle.process_image()
                # rclpy.spin_once(recycle)

                # Get the action to execute from the Neural Network based on the current state
                action = recycle.get_action()

                # Once the action has been received, pass it to the execute() function
                action_result = recycle.execute_action(action)

        # If not in training, all actions come from Neural Network
        elif not recycle.train:
            # Process the current Image to get the current state of the workspace
            recycle.process_image()

            # Get the action to execute from the Neural Network based on the current state
            action = recycle.get_action()

            # Once the action has been received, pass it to the execute() function
            recycle.execute_action(action)

        # If the action was successful or the action was provided by a user
        if recycle.execution_status == "SUCCEEDED" or pose_response == "finished":
            # Grab the item with the gripper, move to the bin, and drop the item
            recycle.grab_and_dispose()

            # Return to home
            recycle.go_to(recycle.home)


class Recycle(Node):
    def __init__(self):
        # Register the ROS2 Node
        super().__init__("recycle")

        # Set all loggers to warning or above, but keep this Node at info level
        # set_logger_level('', LoggingSeverity.WARN)
        # self.get_logger().set_level(LoggingSeverity.INFO)

        # Declare and get the parameter for the Recycle Node
        self.declare_parameters("", [("train", True)])
        self.train = self.get_parameter("train").get_parameter_value().bool_value

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
        """
        # Create a subscriber to the PointCloud data
        self.pc_sub = self.create_subscription(PointCloud2, "/add_topc", self.pc_callback, 1)
        """

        # Define global variables
        self.rgb = None
        self.depth = None
        self.execution_status = None

        self.executed = False
        self.timed_out = False

        """
        # Define a variable to hold the most recent PointCloud
        self.pointcloud
        """

        # Define pre-set positions for KUKA
        self.home = self.construct_joint_position([0.0, -1.74533, 1.5708, 0.0, 1.74533, 0.0])
        self.bin = self.construct_joint_position([-1.5708, -1.13446, 1.48353, 0.0, 1.22173, -1.5708])

        # If in training mode, need to allow for activation and deactivation of the KUKA robot
        if self.train:
            # Define a pair of requests and clients to get and change the current state of the KUKA robot
            self.get_state_request = GetState.Request()
            self.change_state_request = ChangeState.Request()
            self.get_state_client = self.create_client(GetState, "/robot_manager/get_state")
            self.change_state_client = self.create_client(ChangeState, "/robot_manager/change_state")

            # Wait for the robot state services to become available
            while not self.get_state_client.wait_for_service(4.0):
                self.get_logger().info("Waiting for Get State Service ...")
            while not self.change_state_client.wait_for_service(4.0):
                self.get_logger().info("Waiting for Change State Service ...")

    # Decide on RGB/Depth and PointCloud (JK)
    def sub_callback(self, rgb_msg, depth_msg):
        # Assign the most recent Images to the global variables
        self.rgb = rgb_msg
        self.depth = depth_msg

    """
    def pc_callback(self, msg):
        # Assign the most recent PointCloud to the global variable
        self.pointcloud = msg
    """

    def trajectory_callback(self, msg):
        # Assign the message to the global variable
        self.execution_status = msg.status

        # Set True to exit the while loop in the execute_action() function
        self.executed = True

    def process_image(self):
        # Check that the camera is publishing, and notify the user if it is not
        while not self.get_publishers_info_by_topic("/oak/rgb/image_raw"):
            self.get_logger().error("Camera is inactive")
            sleep(5.0)

        # Spin until the messages are received
        while not self.rgb and not self.depth:
            rclpy.spin_once(self)

    # Add API call to Neural Net and revise variable assignment (if necessary now) (JK)
    def get_action(self):
        # Define variable to be assigned in loops
        output = None

        # If rgb and depth variable are empty, notify the user that the camera is inactive
        if not self.rgb or not self.depth:
            self.get_logger().warn("Camera is inactive, providing No-op action")
            return "No-op"

        # if self.train:
        #     NeuralNet(image_input, goal_pose, reward)
        # elif not self.train:
        #     neural_net_output = NeuralNet(image_input)

        # # After passing the image to the neural net, reset the rgb and depth variables
        # self.rgb = None
        # self.depth = None

        # # Define the action variable as Revision message
        # action = Revision()
        # action.no_op = True
        # action.pose.position.x = neural_net_output
        # action.pose.position.y = neural_net_output
        # action.pose.position.z = neural_net_output
        # action.pose.orientation.w = neural_net_output
        # action.pose.orientation.x = neural_net_output
        # action.pose.orientation.y = neural_net_output
        # action.pose.orientation.z = neural_net_output

        # action.pose.position.x = neural_net_output
        # action.pose.position.y = neural_net_output
        # action.pose.position.z = neural_net_output
        # action.pose.orientation.w = neural_net_output
        # action.pose.orientation.x = neural_net_output
        # action.pose.orientation.y = neural_net_output
        # action.pose.orientation.z = neural_net_output

        neural_net_output = [
            0.3029718845865429,
            -0.0293896569710944,
            0.9429706111948394,
            -0.03278797900140336,
            0.016672964307594622,
            -0.1233876877720587,
            0.9916765799394811,
        ]

        return neural_net_output

    def execute_action(self, action):
        # If the action is not a motion
        if action == "No-op":
            # Pause momentarily, and then return
            sleep(1.0)
            self.get_logger().info("No-op executed")
            return

        # If the action is a motion
        elif action != "No-op":
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
            self.get_logger().warn("Plan Result")
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

    # Decide on readkey() vs. input(); include return?; GUI? (JK)
    def query_operator(self):
        # Ask the operator if they would like to provide hand guiding teachings
        # self.get_logger().warn("Provide hand-guided pose?")
        # key = readkey()  # Takes first key input no matter what
        # self.get_logger().warn("You pressed: " + str(key))
        self.get_logger().info("IT WORKS")
        self.get_logger().warn("Provide hand-guided pose? (yes/no)")
        answer = input()  # Takes key input, but need to press "Enter"
        self.get_logger().warn("You entered: " + str(answer))

        # If they do not provide a valid response, try again
        if answer != "yes" and answer != "no":
            answer = self.query_operator()

        return answer

    def deactivate_external_control(self):
        # Call the Get State Service, spin the Node until a response is received, and define the response
        get_state_future = self.get_state_client.call_async(self.get_state_request)
        rclpy.spin_until_future_complete(self, get_state_future)
        get_state_response = get_state_future.result()

        # Check that the Get State Service request completed
        if get_state_response:
            #  Define the current state
            current_state = get_state_response.current_state.label

            # Fix the following logic (JK)
            # If the robot is already inactive, no need to deactivate
            if current_state != "active":
                self.get_logger().info("Robot is already inactive, proceed to hand-guiding")

            # If the robot is active, proceed
            elif current_state == "active":
                self.get_logger().info("Robot is active, proceeding to deactivation")

                # Fix the transition (JK)
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
                        self.get_logger().info("Deactivation was successful, proceed to hand-guiding")

                    # If the Change State Service request completed, but did not deactivate the robot, notify the user and retry
                    else:
                        self.get_logger().info("Transition request sent successfully, but transition failed\nRetrying now")
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
                        self.get_logger().info("Transition request sent successfully, but transition failed\nRetrying now")
                        self.activate_external_control()

                # If the Change State Service request failed, notify the user
                else:
                    self.get_logger().error("Transition request failed")

        # If the Get State Service request failed, notify the user
        else:
            self.get_logger().error("Request for Get State Service failed")

    #  GUI?(JK)
    def wait_for_operator(self):
        # Prompt the user to move the robot, and ask them to notify the machine when complete
        self.get_logger().warn("Provide pose for training, type 'finished' when done")
        answer = input()

        # If the user indicates a hand-guided pose has been provided
        if answer == "finished":
            # Pass the pose and corresponding state as training data to the Neural Network
            # NeuralNet(self.rgb, self.depth, robot_state)
            self.get_logger().info("Hand-guided pose received and passed to Neural Net")

        # If the user does not provide confirmation of hand-guided pose
        elif answer != "finished":
            self.wait_for_operator()

        return answer

    def grab_and_dispose(self):
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

    # Need to add invalid position
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

    #
    def get_reward(self):
        # Define reward variable to be assigned in local loops
        reward = None

        # Get the current load (mA) of the gripper after lifting
        current_load = get_current_load()

        # If the current load is greater than 200 mA, the gripper has something in its grasp
        if current_load > 200:
            reward = 1

        # If the current load is less than 200 mA, the gripper has nothing in its grasp
        elif current_load < 200:
            reward = 0

        return reward


if __name__ == "__main__":
    main()
