from ultralytics import YOLO
import cv2
import os
import uuid
import glob

MODEL_PATH = "models/tile_yolov8_best.pt"
model = YOLO(MODEL_PATH)

# class name → id mapping from model
CLASS_NAMES = model.names   # {0:'tiles', 1:'other'}

def detect_tiles(image_path, crop_dir):
    # clear old crops to avoid stale results
    for old in glob.glob(os.path.join(crop_dir, "*.jpg")):
        os.remove(old)

    results = model(image_path)[0]
    img = cv2.imread(image_path)

    crop_files = []

    for i, box in enumerate(results.boxes.xyxy):

        class_id = int(results.boxes.cls[i])

        # ❌ skip "other"
        if CLASS_NAMES[class_id] != "tiles":
            continue

        x1, y1, x2, y2 = map(int, box)

        crop = img[y1:y2, x1:x2]
        crop_name = f"{uuid.uuid4().hex[:8]}_tile_{i}.jpg"
        crop_path = os.path.join(crop_dir, crop_name)

        cv2.imwrite(crop_path, crop)
        crop_files.append(crop_name)

    return crop_files

