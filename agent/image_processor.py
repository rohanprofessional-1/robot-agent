import cv2
import numpy as np
import roboflow
from datetime import datetime
import os
from dataclasses import dataclass
# load your test tube model
COLOR_RANGES = {
    "red":    [(np.array([0, 120, 70]),  np.array([10, 255, 255])),
               (np.array([170, 120, 70]), np.array([180, 255, 255]))],
    "green":  [(np.array([35,  80, 80]),  np.array([85, 255, 255]))],
    "blue":   [(np.array([90,  80, 80]),  np.array([130, 255, 255]))],
    "yellow": [(np.array([20, 100, 100]), np.array([30, 255, 255]))],

    # add these:
    "white":  [(np.array([0,   0, 200]),  np.array([180, 40, 255]))],
    "purple": [(np.array([125, 50, 50]),  np.array([150, 255, 255]))],
}


@dataclass
class TestTube:
    tube_id: int
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    class_id: int
    conf: float
    color: str | None = None
    image_path: str | None = None  # original image path
class ImageProcessor:
    def __init__(self, model_api):
        rf = roboflow.Roboflow(api_key=model_api)
        project = rf.workspace("researchworker").project("tube-detection-x52mi-9pgxe")
        self.model = project.version(1).model  # Loads YOLOv8

    def detect_test_tubes(self, image_path: str) -> list[TestTube]:
        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Roboflow expects path or RGB; easiest: pass path
        prediction = self.model.predict(image_path, confidence=40).json()  # confidence in %, e.g. 40 = 0.4

        tubes = []
        for i, det in enumerate(prediction["predictions"]):
            x = det["x"]
            y = det["y"]
            w = det["width"]
            h = det["height"]
            conf = det["confidence"]
            class_name = det["class"]

            # Convert center x,y,w,h to x1,y1,x2,y2
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            roi = frame_bgr[y1:y2, x1:x2].copy()

            tubes.append({
                "tube_id": i,
                "bbox": (x1, y1, x2, y2),
                "class_name": class_name,
                "conf": conf,
                "roi": roi,
            })

        return tubes

    # 3) Classify tube color from ROI ------------------------------------
    def classify_tube_color(self, frame_bgr, tube: TestTube, color_name_list: list[str]) -> str | None:
 
        x1, y1, x2, y2 = tube["bbox"]
        roi = frame_bgr[y1:y2, x1:x2].copy()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        best_color = None
        best_count = 0
        for color_name in color_name_list:
            ranges = COLOR_RANGES[color_name]
            full_mask = None
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                full_mask = mask if full_mask is None else cv2.bitwise_or(full_mask, mask)
            count = int(full_mask.sum())
            if count > best_count:
                best_count = count
                best_color = color_name

        return best_color
    
    def get_colored_tubes(self, image_path: str, color_name: str) -> list[TestTube]:
        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise ValueError(f"Could not read image at {image_path}")

        tubes = self.detect_test_tubes(image_path)
        colored = []
        for tube in tubes:
            color = self.classify_tube_color(frame_bgr, tube, [color_name])
            if color == color_name:
                tube["color"] = color
                colored.append(tube)
        return colored



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
