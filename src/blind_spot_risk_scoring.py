from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO
import cv2
import os

# ====================================
# LOAD DATASET
# ====================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot="datasets/nuscenes",
    verbose=False
)

model = YOLO("yolov8n.pt")

print("Dataset loaded.")

# ====================================
# LOAD SAMPLE
# ====================================

scene = nusc.scene[0]

sample = nusc.get(
    "sample",
    scene["first_sample_token"]
)

# ====================================
# SIDE CAMERAS
# ====================================

left_cameras = [
    "CAM_FRONT_LEFT",
    "CAM_BACK_LEFT"
]

right_cameras = [
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT"
]

# ====================================
# COUNTERS
# ====================================

left_vehicles = 0
left_people = 0

right_vehicles = 0
right_people = 0

vehicle_classes = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle"
]

# ====================================
# LEFT SIDE ANALYSIS
# ====================================

for cam in left_cameras:

    token = sample["data"][cam]

    sd = nusc.get(
        "sample_data",
        token
    )

    path = os.path.join(
        "datasets/nuscenes",
        sd["filename"]
    )

    image = cv2.imread(path)

    results = model(
        image,
        verbose=False
    )

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        name = model.names[cls_id]

        if name in vehicle_classes:
            left_vehicles += 1

        if name == "person":
            left_people += 1

# ====================================
# RIGHT SIDE ANALYSIS
# ====================================

for cam in right_cameras:

    token = sample["data"][cam]

    sd = nusc.get(
        "sample_data",
        token
    )

    path = os.path.join(
        "datasets/nuscenes",
        sd["filename"]
    )

    image = cv2.imread(path)

    results = model(
        image,
        verbose=False
    )

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        name = model.names[cls_id]

        if name in vehicle_classes:
            right_vehicles += 1

        if name == "person":
            right_people += 1

# ====================================
# RISK LOGIC
# ====================================

def evaluate_side(
    vehicles,
    people
):

    score = 0

    score += vehicles * 30

    score += people * 50

    if score >= 100:
        level = "HIGH"

    elif score >= 50:
        level = "MEDIUM"

    else:
        level = "LOW"

    return score, level

left_score, left_level = evaluate_side(
    left_vehicles,
    left_people
)

right_score, right_level = evaluate_side(
    right_vehicles,
    right_people
)

# ====================================
# LANE CHANGE DECISION
# ====================================

left_safe = left_level == "LOW"

right_safe = right_level == "LOW"

# ====================================
# REPORT
# ====================================

print("\n==============================")
print(" BLIND SPOT RISK REPORT ")
print("==============================")

print("\nLEFT SIDE")

print("Vehicles:", left_vehicles)
print("People:", left_people)
print("Risk Score:", left_score)
print("Risk Level:", left_level)

print(
    "Lane Change Safe:",
    left_safe
)

print("\nRIGHT SIDE")

print("Vehicles:", right_vehicles)
print("People:", right_people)
print("Risk Score:", right_score)
print("Risk Level:", right_level)

print(
    "Lane Change Safe:",
    right_safe
)

print("\n==============================")