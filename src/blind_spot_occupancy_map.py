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
# CAMERA GROUPS
# ====================================

camera_groups = {

    "FRONT_LEFT":
        "CAM_FRONT_LEFT",

    "REAR_LEFT":
        "CAM_BACK_LEFT",

    "FRONT_RIGHT":
        "CAM_FRONT_RIGHT",

    "REAR_RIGHT":
        "CAM_BACK_RIGHT"
}

vehicle_classes = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle"
]

# ====================================
# OCCUPANCY RESULTS
# ====================================

occupancy = {}

# ====================================
# ANALYZE EACH CAMERA
# ====================================

for zone, camera in camera_groups.items():

    token = sample["data"][camera]

    sd = nusc.get(
        "sample_data",
        token
    )

    image_path = os.path.join(
        "datasets/nuscenes",
        sd["filename"]
    )

    image = cv2.imread(
        image_path
    )

    results = model(
        image,
        verbose=False
    )

    vehicles = 0
    people = 0

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        name = model.names[cls_id]

        if name in vehicle_classes:
            vehicles += 1

        if name == "person":
            people += 1

    occupancy[zone] = {

        "vehicles": vehicles,
        "people": people,
        "total": vehicles + people
    }

# ====================================
# DISPLAY MAP
# ====================================

print("\n====================================")
print(" BLIND SPOT OCCUPANCY MAP ")
print("====================================")

print("\n           FRONT")

print(
    f"      FL:{occupancy['FRONT_LEFT']['total']}"
)

print(
    f"      FR:{occupancy['FRONT_RIGHT']['total']}"
)

print("\n            BUS")

print(
    f"      RL:{occupancy['REAR_LEFT']['total']}"
)

print(
    f"      RR:{occupancy['REAR_RIGHT']['total']}"
)

print("\n            REAR")

print("\n====================================")

# ====================================
# RISK ESTIMATION
# ====================================

for zone in occupancy:

    total = occupancy[zone]["total"]

    if total >= 3:
        risk = "HIGH"

    elif total >= 1:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    print(
        zone,
        "| Objects:",
        total,
        "| Risk:",
        risk
    )

print("====================================")