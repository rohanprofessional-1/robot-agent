from image_processor import get_image_processor
import os

image_processor = get_image_processor(model_api=os.getenv("ROBOFLOW_API_KEY"))

detect = image_processor.detect_test_tubes("/Users/rohannair/Desktop/Research/RobotAgent/agent/logs/IMG20230408125614_BURST000_COVER.jpg")
print(len(detect))