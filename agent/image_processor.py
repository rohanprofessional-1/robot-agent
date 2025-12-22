import cv2
import numpy as np
import roboflow
from datetime import datetime
import os
# load your test tube model

COLOR_RANGES = {
    "red":    [(np.array([0, 120, 70]),  np.array([10, 255, 255])),
               (np.array([170, 120, 70]), np.array([180, 255, 255]))],
    "green":  [(np.array([35,  80, 80]), np.array([85, 255, 255]))],
    "blue":   [(np.array([90,  80, 80]), np.array([130, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
}

class ImageProcessor:
    def __init__(self, model_api):
        rf = roboflow.Roboflow(api_key=model_api)
        project = rf.workspace("researchworker").project("tube-detection-x52mi-9pgxe")
        self.model = project.version(1).model  # Loads YOLOv8

    
    def get_test_tube_rois(self, frame_bgr):
        # Use Roboflow model inference (same YOLO format)
        results = self.model.predict(frame_bgr, confidence=0.4, image_size=640)[0]
        
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

    def filter_color_bgr_image(self, img_bgr, color_name):
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


    def capture_image(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        cur_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = "logs"
        os.makedirs(dir_path, exist_ok=True)
        filename = os.path.join(dir_path, f"{cur_datetime}.jpg")
        if not cap.isOpened():
            return f"Error: Could not open video device {camera_index}. Check the camera index."

        print(f"Accessing camera at index {camera_index}...")
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        
        if ret:
            cv2.imwrite(filename, frame)
            print(f"Successfully captured image and saved as {filename}")
            return filename
        else:
            return "Error: Could not read a frame from the camera."
    



    def image_to_coords(self, x, y):
        
        raise NotImplementedError("Yet to be implemented")

   
    
    
    
_image_processor = None


def get_image_processor(model_api):
    """Get or create the global robot controller instance."""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor(model_api)
    return _image_processor
