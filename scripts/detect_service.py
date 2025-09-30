#!/usr/bin/env python3

"""
This script utilizes a pre-trained Roboflow vision backbone to identify bottles
"""

import os
import cv2
import rclpy
import numpy as np
from time import sleep
import supervision as sv
from rclpy.node import Node
from roboflow import Roboflow
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from image_geometry import PinholeCameraModel
from tf2_geometry_msgs.tf2_geometry_msgs import PoseStamped
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from tf2_geometry_msgs import do_transform_pose
from recycrl.srv import Poses

def main():
    # Define the RoboFlow model and version
    model = "bottles-8bf70"
    version = 8
    # model = "bottles-and-cans-finetunning"
    # version = 3
    # model = "grocery-rfn8l"
    # version = 3
    # model = "beverage-containers-3atxb"
    # version = 3
    # model = "plastic-recyclable-detection"
    # version = 2
    # model = "plastic-bottles-5zduz"
    # version = 2
    
    # Initialize rclpy and the Detect Node
    rclpy.init()
    detect_srv = DetectService(model, version)
    rclpy.spin(detect_srv)


class DetectService(Node):
    def __init__(self, model, version):
        # Register the ROS2 Node
        super().__init__("detector_service")
        
        # Define the Cv Bridge and empty image variable
        self.bridge = CvBridge()
        self.label_annotator = sv.LabelAnnotator()
        self.box_annotator = sv.BoxAnnotator()
        self.camera_model = PinholeCameraModel()
        
        # Create a subscriber for RGB and Depth images and synchronize them
        self.rgb_sub = Subscriber(self, Image, "/oak/rgb/image_raw")
        self.depth_sub = Subscriber(self, Image, "/oak/stereo/image_raw")
        self.info_sub = Subscriber(self, CameraInfo, "/oak/rgb/camera_info")
        self.sync_sub = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub], 1, 0.2)
        self.sync_sub.registerCallback(self.sub_callback)
        
        # Create a service for getting the 3D object locations
        self.service = self.create_service(Poses, "/get_object_locations", self.get_object_locations)
        
        # # Define Roboflow, the image dataset, and the model
        rf = Roboflow(api_key="97C4ApY6BQXzTP1VZZjK")
        project = rf.workspace().project(model)
        self.model = project.version(version).model
        
        # Define global variable for storing the most recent images and camera data
        self.rgb = None
        self.depth = None
        self.depth_height = None
        self.depth_width = None
        
        # Create a Transform Buffer, clear it, then create a Transform Broadcaster and Listener
        self.tf = None
        self.tf_buffer = Buffer()
        self.tf_buffer.clear()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_listener = TransformListener(self.tf_buffer, self)
    
    def sub_callback(self, rgb_msg, depth_msg, info_msg):
        # Assign the most recent images to the global variables
        self.rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        self.depth = self.bridge.imgmsg_to_cv2(depth_msg, '32FC1')
        
        # If unassigned, or unequal to the previous depth image dimensions, define the depth image height and width
        if not self.depth_height or self.depth_height != self.depth.shape[0] or self.depth_width != self.depth.shape[1]:
            self.depth_height = self.depth.shape[0]
            self.depth_width = self.depth.shape[1]
        
        # If unassigned, define the camera info and Transform
        if not self.tf:
            self.camera_model.from_camera_info(info_msg)
            self.tf = self.camera_model.get_tf_frame()
            print("Ready")
    
    def get_object_locations(self, msg, response):
        # Get the desired directory for saving
        directory = os.path.expanduser("~/images")
        
        # Get the list of previously saved images
        images = os.listdir(f"{directory}/raw")
        
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
            image_name = f"image_{largest_image_number + 1}"

        # If there are no images, create the first one
        else:
            image_name = "image_0"

        # Save the image
        cv2.imwrite(f"{directory}/raw/{image_name}.png", self.rgb)
        
        # Pass the image to the model to detect objects
        print("\nGetting Predictions")
        prediction = self.model.predict(f"{directory}/raw/{image_name}.png").json()
        print("Predictions Received")
        
        # Get the labels and detections from the model output
        labels = [item["class"] for item in prediction["predictions"]]
        detections = sv.Detections.from_inference(prediction)
        
        # Define the image and annotate it
        image = cv2.imread(f"{directory}/raw/{image_name}.png")
        annotated_image = self.box_annotator.annotate(image, detections)
        annotated_image = self.label_annotator.annotate(annotated_image, detections, labels)
        
        # Save the annotated image
        cv2.imwrite(f"{directory}/annotated/{image_name}.png", annotated_image)
        
        # Extract the bounding boxes from the detections
        bounding_boxes = detections.xyxy.tolist()
        # bounding_boxes = [[738. , 265.5, 856. , 596.5], [408. , 456. , 548. , 600. ], [548. , 127. , 708. , 317. ]]
        # bounding_boxes = [[546. , 130. , 712. , 322. ],[734. , 264.5, 838. , 609.5],[398. , 453. , 556. , 595. ]]
        
        # Define an array to hold the 3D poses
        object_poses = []
        
        # Loop through each detection    H: 720 W:1280
        for box in bounding_boxes:
            # Define a list to hold the region of interest
            roi = []
            
            # Calculate the central x and y values from the bounding box
            x = (box[2] + box[0]) / 2
            y = (box[3] + box[1]) / 2
            
            # Start an iterative loop at 5 with increments of 5 pixels until you get to a third of the image height to define region of interest
            for box_width in range(5, int(self.depth_height/3), 5):
                # Calculate the minimum and maximum x and y values for the bounding box
                min_x = x - (box_width / 2)
                min_y = y - (box_width / 2)
                max_x = x + (box_width / 2)
                max_y = y + (box_width / 2)
                
                # Clamp the minimum and maximum x and y values to make sure they stay within the image 
                min_x = int(min(self.depth_width, max(0, min_x)))
                min_y = int(min(self.depth_height, max(0, min_y)))
                max_x = int(max(0, min(self.depth_width, max_x)))
                max_y = int(max(0, min(self.depth_height, max_y)))
                
                # Define the region of interest within the depth image using the clamped values
                roi = self.depth[min_y:max_y, min_x:max_x].copy()

                # Create a masked array to exclude depth values that are 0
                roi = np.ma.masked_equal(roi, 0)
                
                # Define a depth variable in case no valid depths are extracted
                depth = None
                
                # If there are any valid depth values in the region of interest
                if np.any(roi):
                    # Average the depth values in the region of interest and exit the loop
                    depth = np.mean(roi)
                    break
            
            # If a valid depth is extracted
            if depth:
                # Get the ray of the point
                ray = self.camera_model.project_pixel_to_3d_ray(self.camera_model.rectify_point((x, y)))
                # Use the ray and depth to calculate the pose
                pose = [(i * depth) / ray[2] for i in ray]
                # Once the pose has been calculated, append it to the list, divide by 1000 to convert to meters
                object_poses.append(np.array(pose)/1000)
            
            # If no valid depth is extracted, notify the user
            elif not depth:
                self.get_logger().error("Unable to extract depth info for object")

        # # Define a pose list to hold transformed poses
        # poses = []

        # Loop through each extracted 3D pose
        for object_pose in object_poses:
            # Define a Pose Stamped message
            pose = PoseStamped()
            pose.header.frame_id = self.tf
            pose.header.stamp = Time(seconds=0)
            pose.pose.position.x = object_pose[0]
            pose.pose.position.y = object_pose[1]
            pose.pose.position.z = object_pose[2]
            
            # Get the transform between the camera and the root frames
            cam_tf = self.tf_buffer.lookup_transform("root", self.tf, Time(seconds=0), Duration(seconds=1.0))
            
            # Transform the pose from the camera frame to the root frame
            tf_pose = do_transform_pose(pose.pose, cam_tf)

            # Add the poses to the list
            response.poses.append(tf_pose)
        
        return response


if __name__ == "__main__":
    main()