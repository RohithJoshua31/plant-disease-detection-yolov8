"""
predict.py

Run inference using the trained YOLOv8 model.
"""

from ultralytics import YOLO

# ==========================================================
# Configuration
# ==========================================================

MODEL_PATH = "../results/plant_disease_training/weights/best.pt"

IMAGE_PATH = "../assets/sample.jpg"

OUTPUT_DIR = "../results/predictions"

# ==========================================================
# Load Model
# ==========================================================

print("=" * 60)
print("Loading Trained Model...")
print("=" * 60)

model = YOLO(MODEL_PATH)

# ==========================================================
# Predict
# ==========================================================

print("\nRunning Prediction...\n")

results = model.predict(
    source=IMAGE_PATH,
    save=True,
    project=OUTPUT_DIR,
    name="prediction",
    conf=0.25
)

print("\nPrediction Completed Successfully!")
print(f"\nResults saved in: {OUTPUT_DIR}/prediction")
