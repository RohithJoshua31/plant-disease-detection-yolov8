"""
evaluate.py

Evaluate the trained YOLOv8 model.
"""

from ultralytics import YOLO

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "../results/plant_disease_training/weights/best.pt"

DATASET_CONFIG = "../dataset/data.yaml"

# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading Model...")
print("=" * 60)

model = YOLO(MODEL_PATH)

# ==========================================================
# Validate Model
# ==========================================================

print("\nEvaluating Model...\n")

metrics = model.val(
    data=DATASET_CONFIG
)

print("=" * 60)
print("Evaluation Completed")
print("=" * 60)

print(f"mAP50      : {metrics.box.map50:.4f}")
print(f"mAP50-95   : {metrics.box.map:.4f}")
print(f"Precision  : {metrics.box.mp:.4f}")
print(f"Recall     : {metrics.box.mr:.4f}")
