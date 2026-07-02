"""
train.py

Train a YOLOv8 model for Tomato Leaf Disease Detection.
"""

from ultralytics import YOLO
import os


# ==========================================================
# Project Configuration
# ==========================================================

DATASET_CONFIG = "../dataset/data.yaml"

MODEL_NAME = "yolov8n.pt"

EPOCHS = 50

IMAGE_SIZE = 640

BATCH_SIZE = 16

PROJECT_NAME = "../results"

RUN_NAME = "plant_disease_training"


# ==========================================================
# Create Results Directory
# ==========================================================

os.makedirs(PROJECT_NAME, exist_ok=True)


# ==========================================================
# Load YOLOv8 Model
# ==========================================================

print("=" * 60)
print("Loading YOLOv8 Model...")
print("=" * 60)

model = YOLO(MODEL_NAME)


# ==========================================================
# Start Training
# ==========================================================

print("\nStarting Training...\n")

model.train(
    data=DATASET_CONFIG,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    project=PROJECT_NAME,
    name=RUN_NAME,
    pretrained=True,
    save=True,
    device="cpu",      # Change to "0" if using NVIDIA GPU
    workers=4
)


# ==========================================================
# Training Completed
# ==========================================================

print("\nTraining Completed Successfully!")

print(f"\nResults saved in: {PROJECT_NAME}/{RUN_NAME}")

print("\nBest Model Location:")
print(f"{PROJECT_NAME}/{RUN_NAME}/weights/best.pt")
