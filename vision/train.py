from ultralytics import YOLO

# 1. Load a pretrained Nano model (good for small datasets)
model = YOLO("yolov8n.pt")

# 2. Train the model
results = model.train(
    data="dataset.yaml",
    epochs=100,  # Start with 100, it will stop early if it plateaus
    imgsz=640,  # Standard resolution
    batch=8,  # Small batch size for a small dataset
    name="tube_detector",
    # --- Augmentations for small data ---
    degrees=15.0,  # Rotate tubes slightly
    flipud=0.5,  # Flip upside down (if applicable to your angles)
    hsv_h=0.015,  # Randomly shift colors slightly to make it robust
    mosaic=1.0,  # Combines 4 images into 1 (excellent for detection)
)
