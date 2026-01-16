#!/usr/bin/env python3

"""
This script provides a ROS2 service to get the position and orientation of recyclables. It first utilizes a YOLO instance
segmentation model to identify the masks of the detected bottles and cans. Then, it computes the yaw angle of the objects
using Principal Component Analysis (PCA). Finally, it calculates the pitch angle of the bottle by getting the 3D position
of two endpoints on the bottle.
"""

import rclpy
import matplotlib
import numpy as np
from time import sleep
from pathlib import Path
import supervision as sv
from rclpy.node import Node
from rclpy.time import Time
from ultralytics import YOLO
from math import degrees, pi
from threading import Thread
from recycrl.srv import Poses
from cv_bridge import CvBridge
from cv2 import imwrite, imread
from argparse import ArgumentParser
from rclpy.duration import Duration
from sklearn.decomposition import PCA
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from tf2_geometry_msgs import do_transform_pose
from rclpy.wait_for_message import wait_for_message
from tf_transformations import quaternion_from_euler
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # Get the /recycrl directory
    directory = Path(__file__).parents[4] / "src" / "recycrl"

    # Define arguments
    description = "Service Node that calculates the 3D position and orientation of detected objects"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-weights",
        dest="weights",
        default=directory / "yolo" / "weights" / "all_finetuned.pt",
        help="Location of weights for the instance segmentation model",
    )
    parser.add_argument(
        "-output", dest="output", default=directory / "output", help="Location to save the most recent masks and PCA analysis"
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Print the yaw and pitch angles for debugging")
    parser.add_argument(
        "--save_test_data",
        dest="save_test_data",
        action="store_true",
        help="Save the most recent RGB and Depth images as test data",
    )
    parser.add_argument(
        "--use_test_data", dest="use_test_data", action="store_true", help="Use the test data rather than live images"
    )

    # Parse and assign arguments
    args = parser.parse_args()
    weights = args.weights
    output = args.output
    debug = args.debug
    save_test_data = args.save_test_data
    use_test_data = args.use_test_data

    # Initialize rclpy and the YOLOService Node
    rclpy.init()
    yolo_srv = YOLOService(weights, output, debug, save_test_data, use_test_data)

    # Try to spin the node
    try:
        rclpy.spin(yolo_srv)

    # If there is an exception with spinning the Node, notify the user
    except Exception as e:
        print("\nYolo service failed: %r" % (e,))

    # If there is a Keyboard Interrupt, gracefully shut down the Node
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt, shutting down")


class YOLOService(Node):
    def __init__(self, weights, output, debug, save_test_data, use_test_data):
        # Register the ROS2 Node
        super().__init__("yolo_service")

        # Define global variables from passed arguments
        self.debug = debug
        self.model = YOLO(weights)
        self.output_path = output
        self.save_test_data = save_test_data
        self.use_test_data = use_test_data

        # Define the Cv Bridge, image annotators, and camera model
        self.bridge = CvBridge()
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_position=sv.Position.CENTER)
        self.mask_annotator = sv.MaskAnnotator()
        self.camera_model = PinholeCameraModel()

        # Create a subscriber for RGB and Depth images and the camera info and synchronize all of them
        self.rgb_sub = Subscriber(self, Image, "/oak/rgb/image_raw")
        self.depth_sub = Subscriber(self, Image, "/oak/stereo/image_raw")
        self.sync_sub = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.2)
        self.sync_sub.registerCallback(self.sub_callback)

        # Create a service for getting the 3D object locations
        self.service = self.create_service(Poses, "/get_poses", self.get_poses)

        # Define global variable for storing the most recent images and camera data
        self.rgb = None
        self.depth = None
        self.depth_height = None
        self.depth_width = None

        # Define empty variable for camera transform, create a Transform Buffer, clear it, then create a Transform Listener
        self.cam_tf = None
        self.tf_buffer = Buffer()
        self.tf_buffer.clear()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Check that the camera is publishing, and notify the user if it is not
        while not self.get_publishers_info_by_topic("/oak/rgb/image_raw"):
            self.get_logger().info("Camera topics currently unavailable")
            sleep(5.0)

        # Wait for the RGB and depth camera info messages
        rgb_info = wait_for_message(CameraInfo, self, "/oak/rgb/camera_info")[1]
        depth_info = wait_for_message(CameraInfo, self, "/oak/stereo/camera_info")[1]

        # Define the camera model, get its frame, and define the depth image height and width
        self.camera_model.from_camera_info(rgb_info)
        self.tf = self.camera_model.get_tf_frame()
        self.depth_height = depth_info.height
        self.depth_width = depth_info.width

    def sub_callback(self, rgb_msg, depth_msg):
        # Assign the most recent RGB and depth images to the global variables
        self.rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        self.depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        # Get the transform of the camera frame to the root/world frame now that we are spinning
        if not self.cam_tf:
            # Try statement to handle exceptions
            try:
                # Get the camera transform
                self.cam_tf = self.tf_buffer.lookup_transform("root", self.tf, Time(seconds=0.0), Duration(seconds=2.0))

                # If the transform has bad values for the translation or rotation (Camera node needs time to process and publish)
                if self.cam_tf.transform.translation.z == 0.0 or self.cam_tf.transform.rotation.x == 0.5:
                    # Reset the variable that holds the transform to retry, notify user, and pause slightly
                    self.cam_tf = None
                    self.get_logger().info("Received incorrect Transform, trying again")
                    sleep(2.0)

                # If the transform is correct, notify the user that the service is ready
                else:
                    self.get_logger().info("YOLO Service Ready")

            # If we have a Lookup Exception (due to timeout generally), notify the user
            except LookupException:
                self.get_logger().info("Transform Lookup Exception, trying again")

            # If we have a Connectivity Exception (not sure why this occurs), notify the user
            except ConnectivityException:
                self.get_logger().info("Transform Connectivity Exception, trying again")

    def get_poses(self, msg, response):
        # If debug mode was specified
        if self.debug:
            self.get_logger().info("Request received")

        # If no new image has been received or if the camera transform has not been defined, return an empty response
        if self.rgb is None or not self.cam_tf:
            return response

        # Make a copy of the most recent RGB image
        rgb_image = self.rgb.copy()
        depth_image = self.depth.copy()

        # If the user sets "save_test_data" to True, the script will save the most recent images to the output directory
        if self.save_test_data:
            imwrite(f"{self.output_path}/rgb_test.png", self.rgb)
            imwrite(f"{self.output_path}/depth_test.png", self.depth)

        # If the user sets "use_test_data" to True, the script will use the test data from the output directory
        if self.use_test_data:
            rgb_image = imread(f"{self.output_path}/rgb_test.png")
            depth_image = imread(f"{self.output_path}/depth_test.png")

        # Reset the global image so that it represents the most recent image
        self.rgb = None
        self.depth = None

        # Pass the copied image to the model to detect objects (This takes 1 extra second to load on first request)
        prediction = self.model.predict(rgb_image, conf=0.5, verbose=False)[0]

        # Get the labels and detections from the model output
        detections = sv.Detections.from_ultralytics(prediction)

        # Annotate the image with the masks and labels
        image = self.mask_annotator.annotate(rgb_image, detections=detections)
        image = self.label_annotator.annotate(rgb_image, detections=detections)

        # Save the annotated image
        imwrite(f"{self.output_path}/masks.png", image)

        # If no objects are detected, return an empty response
        if not prediction.masks:
            return response

        # Get the masks from the predictions and move them from GPU -> CPU as numpy arrays
        masks = prediction.masks.data.detach().cpu().numpy()

        # If debug mode was specified
        if self.debug:
            self.get_logger().info(f"Confidence Scores:\n{prediction.boxes.conf.detach().cpu().tolist()}")

        # Define lists for upcoming "for" loop
        centroid_points = []
        plot_centroids = []
        yaw_angles = []
        plot_axes = []
        endpoint_pairs = []
        plot_endpoints = []
        variance_ratios = []

        """Initial Mask Processing Loop"""
        # Iterate through each detected mask
        for i, mask in enumerate(masks):
            # Take the boolean array, remove all values that are zero, and transform it into a stack that represents the points
            mask_points = np.column_stack(np.nonzero(mask))

            # Define the PCA model to be in 2 Dimensions, fit the model to the mask, and append the current variance ratio
            pca = PCA(n_components=2)
            pca.fit(mask_points)
            variance_ratios.append(pca.explained_variance_ratio_[0])

            # Get computed centroid, compute point in image (due to transforms, masks have different shape; also centroid: [y,x])
            centroid = pca.mean_
            plot_centroids.append(centroid)
            centroid_point = [centroid[1] * (self.depth_width / mask.shape[1]), centroid[0] * (self.depth_height / mask.shape[0])]
            centroid_points.append(centroid_point)

            # Get the vector components of the principal axis and calculate the yaw angle of the mask. We do arctan(-x, -y)
            # because origin of image is in top left and world axis is rotated +90 degrees about Z axis from image
            principal_axis = pca.components_[0]
            plot_axes.append(principal_axis)
            yaw_angle = np.arctan2(-principal_axis[1], -principal_axis[0])

            # Constrict the yaw angle to be between -90 and 90 degrees
            if yaw_angle > pi / 2:
                yaw_angle -= pi
            elif yaw_angle < -pi / 2:
                yaw_angle += pi

            # Append the yaw angle to the list
            yaw_angles.append(yaw_angle)

            # Find the projection of the points onto the principal axis, transform from mask -> principal axis coordinates
            # We subtract the mean from every point so that the points are centered at the mean
            projection_points = np.dot(mask_points - pca.mean_, principal_axis)

            # Find the min and max values of the projection to find end points of the bottle, use these to get the length
            min_projection_point = projection_points.min()
            max_projection_point = projection_points.max()
            length = max_projection_point - min_projection_point

            # Move the endpoints inward by 20% of the length to avoid noisy endpoints
            min_projection_point = min_projection_point + length * 0.2
            max_projection_point = max_projection_point - length * 0.2

            # Convert from principal axis -> mask shape coordinates -> image coordinates, then append the endpoint pair
            min_end = pca.mean_ + min_projection_point * principal_axis
            max_end = pca.mean_ + max_projection_point * principal_axis
            plot_endpoints.append([min_end, max_end])
            min_endpoint = [min_end[1] * (self.depth_width / mask.shape[1]), min_end[0] * (self.depth_height / mask.shape[0])]
            max_endpoint = [max_end[1] * (self.depth_width / mask.shape[1]), max_end[0] * (self.depth_height / mask.shape[0])]
            endpoint_pairs.append([min_endpoint, max_endpoint])

        # Create a thread for plotting and start it in the background
        Thread(target=self.plot, args=(masks, plot_centroids, plot_axes, plot_endpoints)).start()

        # Define an array to hold the 3D positions of the centroids with-respect-to the camera
        centroid_positions = []

        """Centroid Depth Calculation Loop"""
        # Loop through each masks centroid
        for i, centroid_point in enumerate(centroid_points):
            # Get the depth of the point in the image
            depth = self.get_depth(depth_image, centroid_point, 5, 5, "mean")

            # If a valid depth is extracted from the image
            if depth:
                # Get the position of the point with-respect-to the camera
                position = self.get_position_wrt_cam(centroid_point, depth)

                # Append the current position to the running list
                centroid_positions.append(position)

            # If a valid depth is not extracted
            elif not depth:
                # Delete the corresponding yaw angle and endpoints from the list and notify the user
                del yaw_angles[i]
                del endpoint_pairs[i]
                self.get_logger().error("Unable to extract depth info for centroid")

        # Define an array to hold the calculated pitch angles
        pitch_angles = []

        # If debug mode was specified
        if self.debug:
            self.get_logger().info(f"Variance Ratios:\n{variance_ratios}")

        """Pitch Calculation Loop"""
        # Loop through each pair of endpoints
        for i, endpoint_pair in enumerate(endpoint_pairs):
            # Define a list to hold the positions of the current pair of endpoints
            endpoint_poses = []

            # If the object is a can and its variance ratio is less than 0.65
            if prediction.boxes.cls[i] == 1 and variance_ratios[i] < 0.64:
                # Append a pitch angle of -90 degrees (standing upright) and skip to the next endpoints
                pitch_angles.append((-pi / 2))
                continue

            # Loop through each endpoint in the pair
            for endpoint in endpoint_pair:
                # Get the depth of the current endpoint in the image
                depth = self.get_depth(depth_image, endpoint, 1, 2, "mean")

                # If a valid depth is extracted
                if depth:
                    # Get the position of the endpoint with-respect-to the camera
                    position = self.get_position_wrt_cam(endpoint, depth)

                    # Transform the 3D pose of the object in the camera frame to the world frame
                    pose = self.transform_to_world(position, 0.0, 0.0)

                    # Append the pose to the list of endpoint poses
                    endpoint_poses.append(pose)

                # If no valid depth is extracted
                elif not depth:
                    self.get_logger().error("Unable to extract depth info for endpoint")

            # If two endpoint poses were computed
            if len(endpoint_poses) == 2:
                # Calculate the pitch angle between the two endpoints
                pitch_angle = self.calculate_pitch(endpoint_poses)

                # If the pitch angle is less than 20 degrees, lets say it is flat
                if -degrees(pitch_angle) < 20.0:
                    pitch_angle = 0.0

                # If the pitch angle is greater than 70 degrees, lets say it is upright
                elif -degrees(pitch_angle) > 70:
                    pitch_angle = -pi / 2

                # Append the calculated pitch for the pair of endpoints to the running list
                pitch_angles.append(pitch_angle)

        # If the user specifies debug mode
        if self.debug:
            # Print statements for the extracted yaw and pitch angles
            for i in range(len(centroid_positions)):
                self.get_logger().info(f"Yaw: {round(degrees(yaw_angles[i]), 4)}, Pitch: {round(degrees(pitch_angles[i]), 4)}")

        """Final Pose Calculation Loop"""
        # Loop through each centroid position with-respect-to the camera
        for i, centroid_position in enumerate(centroid_positions):
            # Transform the 3D position of the centroid in camera frame to world frame, pass corresponding pitch and yaw
            pose = self.transform_to_world(centroid_position, pitch_angles[i], yaw_angles[i])

            # If the pitch is -90 degrees (standing upright) or if the z position is too high
            if pitch_angles[i] == -pi / 2 or pose.position.z > 0.9:
                # Get the maximum height of the mask
                max_height, _ = self.get_mask_max(depth_image, masks[i])

                # The centroid is top of object, so set z position to be halfway between conveyor and top of object
                pose.position.z = 0.75 + ((max_height - 0.75) / 2)

            # Add the poses to the list
            response.poses.append(pose)

        return response

    def get_mask_max(self, depth_image, mask):
        # Take the boolean array, remove all values that are zero, and transform it into a stack that represents the points
        points = np.column_stack(np.nonzero(mask))

        # Convert the points from mask -> depth image coordinates
        points[:, 1] = points[:, 1] * (self.depth_width / mask.shape[1])
        points[:, 0] = points[:, 0] * (self.depth_height / mask.shape[0])

        # Apply the mask to the depth image
        masked_depth = depth_image[points[:, 0], points[:, 1]].copy()

        # Remove values that are 0 in the mask
        masked_depth = np.ma.masked_equal(masked_depth, 0)

        # Find the minimum value location in the masked depth (Because of overhead camera, the closer an object, the taller it is)
        min_index = np.argmin(masked_depth)

        # Use the index to find the minimum depth point which corresponds to the maximum height
        min_point = points[min_index]

        # Get the height of the minimum point
        depth = self.get_depth(depth_image, [min_point[1], min_point[0]], 1, 1, "mean")
        position = self.get_position_wrt_cam(min_point, depth)
        pose = self.transform_to_world(position, 0.0, 0.0)

        # Flip the coordinates and convert from depth image -> mask coordinates
        min_point[1] = min_point[1] * (mask.shape[1] / self.depth_width)
        min_point[0] = min_point[0] * (mask.shape[0] / self.depth_height)

        return pose.position.z, min_point

    def get_depth(self, depth_img, img_point, initial_box_width, box_increment, mode):
        # Define a depth variable in case no valid depths are extracted
        depth = None

        # Start an iterative loop at the specified width with specified increments to define ROI, stops at 1/3 image height
        for box_width in range(initial_box_width, int(self.depth_height / 3), box_increment):
            # Calculate the minimum and maximum x and y values for the bounding box
            min_x = img_point[0] - (box_width / 2)
            min_y = img_point[1] - (box_width / 2)
            max_x = img_point[0] + (box_width / 2)
            max_y = img_point[1] + (box_width / 2)

            # Clamp the minimum and maximum x and y values to make sure they stay within the image
            min_x = int(min(self.depth_width, max(0, min_x)))
            min_y = int(min(self.depth_height, max(0, min_y)))
            max_x = int(max(0, min(self.depth_width, max_x)))
            max_y = int(max(0, min(self.depth_height, max_y)))

            # Define the region of interest within the depth image using the clamped values
            roi = depth_img[min_y:max_y, min_x:max_x].copy()

            # Create a masked array to exclude depth values that are 0
            roi = np.ma.masked_equal(roi, 0)

            # If there are any valid depth values in the region of interest
            if np.any(roi):
                # If the mode is "mean"
                if mode == "mean":
                    # Average the depth values in the region of interest and exit the loop
                    depth = np.mean(roi)
                    break
                elif mode == "median":
                    # Find the median of the depth values in the region of interest and exit the loop
                    depth = np.median(roi)
                    break

        return depth

    def get_position_wrt_cam(self, img_point, depth):
        # Get the ray of the point
        ray = self.camera_model.project_pixel_to_3d_ray(self.camera_model.rectify_point((img_point[0], img_point[1])))

        # Use the ray and depth to calculate the pose, convert to array for division, divide by 1000 to convert mm -> m
        position = np.array([(i * depth) / ray[2] for i in ray]) / 1000

        return position

    def transform_to_world(self, position, pitch, yaw):
        # Define a Pose Stamped message and fill the header and position values
        pose = PoseStamped()
        pose.header.frame_id = self.tf
        pose.header.stamp = Time(seconds=0)
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]

        # Transform the pose from the camera frame to the root frame
        tf_pose = do_transform_pose(pose.pose, self.cam_tf)

        # Transform the passed pitch and yaw to a quaternion
        qx, qy, qz, qw = quaternion_from_euler(0.0, pitch, yaw)

        # Fill the orientation values (it is easier to fill them here because we know them in the world frame)
        tf_pose.orientation.x = qx
        tf_pose.orientation.y = qy
        tf_pose.orientation.z = qz
        tf_pose.orientation.w = qw

        return tf_pose

    def calculate_pitch(self, endpoint_poses):
        # Extract the poses from the passed list
        pose_1 = endpoint_poses[0]
        pose_2 = endpoint_poses[1]

        # Calculate the difference in depth between the endpoints
        depth_diff = abs(pose_1.position.z - pose_2.position.z)

        # Calculate the distance between the points in the XY plane
        xy_dist = np.sqrt((pose_1.position.x - pose_2.position.x) ** 2 + (pose_1.position.y - pose_2.position.y) ** 2)

        # Use the inverse tangent function to calculate the pitch angle
        pitch_angle = -np.arctan2(depth_diff, xy_dist)

        return pitch_angle

    def plot(self, masks, centroids, axes, endpoints):
        # Initialize subplots
        fig, ax = plt.subplots()

        # This is only necessary for scaling
        ax.imshow(masks[0], cmap="gray")

        # Loop through each of the masks
        for i, mask in enumerate(masks):
            # Take boolean array, remove all zero values, and transform it into stack that represents the points
            mask_points = np.column_stack(np.nonzero(mask))

            # Plot the current mask, centroid, endpoints, and principal axis
            ax.scatter(mask_points[:, 1], mask_points[:, 0], s=1)
            ax.scatter(centroids[i][1], centroids[i][0], color="red", label="Center")
            ax.text(centroids[i][1] + 5, centroids[i][0] - 5, f"{i + 1}", color="white")
            ax.plot(
                [centroids[i][1], centroids[i][1] + 30 * axes[i][1]],
                [centroids[i][0], centroids[i][0] + 30 * axes[i][0]],
                color="blue",
                linewidth=2,
                label="Principal Axis",
            )
            ax.scatter(endpoints[i][0][1], endpoints[i][0][0], color="yellow", label="End 1")
            ax.scatter(endpoints[i][1][1], endpoints[i][1][0], color="orange", label="End 2")

        # After everything has been added, save the figure
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(f"{self.output_path}/pca.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
