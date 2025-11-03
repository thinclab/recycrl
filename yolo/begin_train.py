#!/usr/bin/env python3

"""
Script that begins the training on a specified dataset and YOLO model
"""

from os import getcwd
from ultralytics import YOLO
from argparse import ArgumentParser


# Get the current directory
dir = getcwd()

# Define arguments
description = "Trains a YOLO model from scratch"
parser = ArgumentParser(description=description)
parser.add_argument(
    "-dataset", dest="dataset", default=f"{dir}/dataset/data.yaml", help="Location of the dataset's data.yaml file"
)
parser.add_argument("-model", dest="model", default="yolo11m-seg.yaml", help="Define which YOLO model type to train")

# Parse and assign arguments
args = parser.parse_args()
dataset = args.dataset
model_type = args.model

# Load the model
model = YOLO(model_type)

# Train the model
results = model.train(data=dataset, epochs=100, imgsz=640, device="0", batch=4, plots=True)

# Validate the model
metrics = model.val(data=dataset)

# Get the path of export
path = model.export(format="onnx")
print(f"Export Path: {path}")
