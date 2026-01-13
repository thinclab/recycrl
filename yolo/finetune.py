#!/usr/bin/env python3

"""
Script that takes a previously trained YOLO model and trains on a new data
"""

from os import getcwd
from ultralytics import YOLO
from argparse import ArgumentParser


# Get the current directory
dir = getcwd()

# Define arguments
description = "Finetunes a preexisting YOLO model with new data"
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

# Load the model
model = YOLO(weights)

# Train the model
results = model.train(data=dataset, epochs=100, imgsz=640, device="0", batch=4, plots=True)

# If the validate flag is set
if validate:
    # Validate on separate data
    metrics = model.val(data=dataset)

# Get the path of export
path = model.export(format="onnx")
print(f"Export Path: {path}")
