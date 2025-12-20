from ultralytics import YOLO
import cv2
import numpy as np
# load your test tube model
model = YOLO("test-tubes.pt")  # Need to change this
COLOR_RANGES = {
    "red":    [(np.array([0, 120, 70]),  np.array([10, 255, 255])),
               (np.array([170, 120, 70]), np.array([180, 255, 255]))],
    "green":  [(np.array([35,  80, 80]), np.array([85, 255, 255]))],
    "blue":   [(np.array([90,  80, 80]), np.array([130, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
}


def get_test_tube_rois(frame_bgr):
    results = model(frame_bgr, imgsz=640, conf=0.4)[0]
    rois = []
    h, w = frame_bgr.shape[:2]

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        roi = frame_bgr[y1:y2, x1:x2].copy()
        rois.append({
            "bbox": (x1, y1, x2, y2),
            "class_id": cls_id,
            "conf": conf,
            "roi": roi
        })

    return rois

def filter_color_bgr_image(img_bgr, color_name):
    if color_name not in COLOR_RANGES:
        raise ValueError(f"Unknown color '{color_name}'")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)  # BGR -> HSV [web:50]

    ranges = COLOR_RANGES[color_name]
    full_mask = None
    for lower, upper in ranges:
        mask = cv2.inRange(hsv, lower, upper)  # binary mask [web:43]
        full_mask = mask if full_mask is None else cv2.bitwise_or(full_mask, mask)

    result = cv2.bitwise_and(img_bgr, img_bgr, mask=full_mask)  # keep only selected color [web:38]
    return full_mask, result

def image_to_coords():
    raise NotImplementedError("Yet to be implemented")
