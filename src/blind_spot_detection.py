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

print("Dataset loaded.")

# ====================================
# LOAD YOLO
# ====================================

model = YOLO("yolov8n.pt")

# ====================================
# LOAD FIRST SAMPLE
# ====================================

scene = nusc.scene[0]

sample_token = scene["first_sample_token"]

sample = nusc.get(
    "sample",
    sample_token
)

# ====================================
# BLIND SPOT CAMERAS
# ====================================

cameras = {

    "LEFT_FRONT": "CAM_FRONT_LEFT",
    "LEFT_REAR": "CAM_BACK_LEFT",

    "RIGHT_FRONT": "CAM_FRONT_RIGHT",
    "RIGHT_REAR": "CAM_BACK_RIGHT"
}

# ====================================
# DETECTION REPORT
# ====================================

report = []

print("\n==============================")
print(" BLIND SPOT OBJECT DETECTION ")
print("==============================")

for position, camera_name in cameras.items():

    token = sample["data"][camera_name]

    sample_data = nusc.get(
        "sample_data",
        token
    )

    image_path = os.path.join(
        "datasets/nuscenes",
        sample_data["filename"]
    )

    image = cv2.imread(image_path)

    if image is None:
        continue

    results = model(
        image,
        verbose=False
    )

    object_count = 0

    vehicles = 0

    pedestrians = 0

    print(f"\n{position}")

    print("-" * 30)

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        object_count += 1

        if class_name in [
            "car",
            "truck",
            "bus",
            "motorcycle"
        ]:
            vehicles += 1

        if class_name == "person":
            pedestrians += 1

        print(class_name)

    print(
        f"Objects: {object_count}"
    )

    print(
        f"Vehicles: {vehicles}"
    )

    print(
        f"Pedestrians: {pedestrians}"
    )

    report.append({

        "camera": position,

        "objects": object_count,

        "vehicles": vehicles,

        "pedestrians": pedestrians
    })

print("\n==============================")
print(" DETECTION COMPLETE ")
print("==============================")