#!/usr/bin/env python3

"""
Uses a trained YOLO model to detect objects in an image
"""

from cv2 import imread
import supervision as sv
from random import randint
from ultralytics import YOLO
from os import getcwd, listdir
from argparse import ArgumentParser


# Get the current directory
dir = getcwd()

# Define the image directory
image_dir = f"{dir}/dataset/train/images/"

# Get the images in the training set
images = listdir(image_dir)

# Generate a random index to get a random image name from the training set for the default value
image_name = images[randint(0, (len(images) - 1))]

# Define arguments
description = "Begins training on a specified dataset and YOLO model"
parser = ArgumentParser(description=description)
parser.add_argument(
    "-weights", dest="weights", default=f"{dir}/weights/best.pt", help="Location of weights to resume training on"
)
parser.add_argument(
    "-image", dest="image", default=f"{image_dir}/{image_name}", help="Location of the image to pass through model"
)

# Parse and assign arguments
args = parser.parse_args()
weights = args.weights
image = args.image

# Define the model with the weights
model = YOLO(weights)

# Use the model to predict the instances in the image
result = model.predict(image, conf=0.25)[0]

# result.boxes.xyxy, result.boxes.conf, result.boxes.cls, result.masks.data

# Get the raw image
raw_image = imread(image)

# Transform the ultralytics result to a supervision compatible result
detections = sv.Detections.from_ultralytics(result)

# Define annotators
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_position=sv.Position.CENTER)

# Create a copy of the image and annotate it
annotated_image = raw_image.copy()
annotated_image = mask_annotator.annotate(annotated_image, detections=detections)
annotated_image = label_annotator.annotate(annotated_image, detections=detections)

# Show the annotated image
sv.plot_image(annotated_image, size=(10, 10))
