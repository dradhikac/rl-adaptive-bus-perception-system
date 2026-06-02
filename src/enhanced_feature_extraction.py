import cv2
import numpy as np
import pandas as pd
import json
import os
from ultralytics import YOLO

# =========================================================
# PATHS
# =========================================================

IMAGE_FOLDER = "datasets/bdd100k/images"

LABEL_FILE = (
    "datasets/bdd100k/labels/"
    "bdd100k_labels_images_train.json"
)

OUTPUT_FILE = (
    "outputs/enhanced_bdd100k_features.csv"
)

# =========================================================
# LOAD YOLO
# =========================================================

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO loaded successfully.")

# =========================================================
# LOAD LABELS
# =========================================================

with open(LABEL_FILE, "r") as f:
    label_data = json.load(f)

label_lookup = {}

for item in label_data:
    label_lookup[item["name"]] = item

# =========================================================
# IMAGE LIST
# =========================================================

images = os.listdir(IMAGE_FOLDER)

print(f"\nFound {len(images)} images")

results = []

# =========================================================
# PROCESS IMAGES
# =========================================================

for idx, image_name in enumerate(images):

    image_path = os.path.join(
        IMAGE_FOLDER,
        image_name
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    # =====================================================
    # RESIZE
    # =====================================================

    image = cv2.resize(
        image,
        (640, 480)
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # =====================================================
    # BRIGHTNESS
    # =====================================================

    brightness = np.mean(gray)

    # =====================================================
    # CONTRAST
    # =====================================================

    contrast = np.std(gray)

    # =====================================================
    # BLUR SCORE
    # =====================================================

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # =====================================================
    # EDGE DETECTION
    # =====================================================

    edges = cv2.Canny(
        gray,
        100,
        200
    )

    edge_count = np.sum(edges > 0)

    edge_density = (
        edge_count /
        (640 * 480)
    )

    # =====================================================
    # ENTROPY
    # =====================================================

    hist = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist = hist.ravel()

    hist = hist / hist.sum()

    entropy = -np.sum(
        hist * np.log2(hist + 1e-10)
    )

    # =====================================================
    # YOLO OBJECT DETECTION
    # =====================================================

    detections = model(
        image,
        verbose=False
    )

    object_count = 0
    person_count = 0
    vehicle_count = 0

    vehicle_classes = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle"
    }

    for box in detections[0].boxes:

        object_count += 1

        class_id = int(box.cls[0])

        class_name = (
            model.names[class_id]
        )

        if class_name == "person":
            person_count += 1

        if class_name in vehicle_classes:
            vehicle_count += 1

    # =====================================================
    # LABEL INFORMATION
    # =====================================================

    weather = "unknown"
    timeofday = "unknown"

    label = label_lookup.get(
        image_name,
        None
    )

    if label:

        attributes = label.get(
            "attributes",
            {}
        )

        weather = attributes.get(
            "weather",
            "unknown"
        )

        timeofday = attributes.get(
            "timeofday",
            "unknown"
        )

    # =====================================================
    # STORE RESULTS
    # =====================================================

    results.append({
        "image_name": image_name,
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
        "edge_count": edge_count,
        "edge_density": edge_density,
        "entropy": entropy,
        "object_count": object_count,
        "person_count": person_count,
        "vehicle_count": vehicle_count,
        "weather": weather,
        "timeofday": timeofday
    })

    if idx % 10 == 0:

        print(
            f"Processed "
            f"{idx}/{len(images)}"
        )

# =========================================================
# SAVE DATASET
# =========================================================

df = pd.DataFrame(results)

os.makedirs(
    "outputs",
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n================================")
print("FEATURE EXTRACTION COMPLETE")
print("================================")

print(f"\nSaved to:\n{OUTPUT_FILE}")

print("\nDataset Shape:")
print(df.shape)

print("\nPreview:")
print(df.head())