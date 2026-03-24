"""
Tile Detection Module
---------------------
Uses YOLOv8 to detect tile regions in uploaded images.
Crops detected tiles and saves them for embedding/recommendation.
"""

from ultralytics import YOLO
import cv2
import os
import uuid
import glob

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "tile_yolov8_best.pt")
model = YOLO(MODEL_PATH)

# class name → id mapping from model
CLASS_NAMES = model.names  # {0:'tiles', 1:'other'}


def detect_tiles(image_path: str, crop_dir: str) -> list:
    """
    Detect tiles in the given image using YOLOv8.

    Args:
        image_path: Path to the input image.
        crop_dir: Directory to save cropped tile images.

    Returns:
        List of cropped tile filenames.
    """
    # clear old crops to avoid stale results
    for old in glob.glob(os.path.join(crop_dir, "*.jpg")):
        os.remove(old)

    results = model(image_path)[0]
    img = cv2.imread(image_path)

    crop_files = []

    for i, box in enumerate(results.boxes.xyxy):
        class_id = int(results.boxes.cls[i])

        # skip non-tile detections
        if CLASS_NAMES[class_id] != "tiles":
            continue

        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]
        crop_name = f"{uuid.uuid4().hex[:8]}_tile_{i}.jpg"
        crop_path = os.path.join(crop_dir, crop_name)

        cv2.imwrite(crop_path, crop)
        crop_files.append(crop_name)

    return crop_files
