#!/usr/bin/env python3

"""
Script that resumes the training on a specified YOLO model
"""

from os import getcwd
from ultralytics import YOLO
from argparse import ArgumentParser


# Get the current directory
dir = getcwd()

# Define arguments
description = "Continues training on a specified dataset and YOLO model"
parser = ArgumentParser(description=description)
parser.add_argument(
    "-dataset", dest="dataset", default=f"{dir}/dataset/data.yaml", help="Location of the dataset's data.yaml file"
)
parser.add_argument(
    "-weights", dest="weights", default=f"{dir}/weights/best.pt", help="Location of weights to resume training on"
)
parser.add_argument("--validate", dest="validate", action="store_true", help="Whether to validate on separate data")

# Parse and assign arguments
args = parser.parse_args()
dataset = args.dataset
weights = args.weights
validate = args.validate

# Load the model with the partially trained weights
model = YOLO(weights)

# Train the model with the resume argument set to True
results = model.train(data=dataset, epochs=100, imgsz=640, device="0", batch=8, resume=True)

# If the validate flag is set
if validate:
    # Validate on separate data
    metrics = model.val(data=dataset)

# Get the path of export
path = model.export(format="onnx")
print(f"Export Path: {path}")
