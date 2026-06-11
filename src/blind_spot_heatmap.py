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
# CAMERA ZONES
# ====================================

zones = {

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

risk_map = {}

# ====================================
# PROCESS CAMERAS
# ====================================

for zone, camera in zones.items():

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

    risk_score = (

        vehicles * 30 +

        people * 50

    )

    risk_map[zone] = risk_score

# ====================================
# FIND DANGEROUS ZONE
# ====================================

danger_zone = max(
    risk_map,
    key=risk_map.get
)

# ====================================
# HEATMAP DISPLAY
# ====================================

def level(score):

    if score >= 100:
        return "HIGH"

    elif score >= 50:
        return "MEDIUM"

    return "LOW"

print("\n====================================")
print(" BLIND SPOT DANGER HEATMAP ")
print("====================================")

for zone in risk_map:

    print(

        zone,

        "| Score:",

        risk_map[zone],

        "|",

        level(risk_map[zone])

    )

print("\n====================================")

print(
    "MOST DANGEROUS ZONE:",
    danger_zone
)

print(
    "RISK SCORE:",
    risk_map[danger_zone]
)

print("====================================")

# ====================================
# SAVE REPORT
# ====================================

os.makedirs(
    "outputs",
    exist_ok=True
)

with open(
    "outputs/blind_spot_heatmap.txt",
    "w"
) as f:

    f.write(
        "BLIND SPOT DANGER HEATMAP\n\n"
    )

    for zone in risk_map:

        f.write(

            f"{zone} | "

            f"{risk_map[zone]}\n"
        )

    f.write(
        f"\nMost Dangerous Zone: {danger_zone}"
    )

print(
    "\nSaved:"
)

print(
    "outputs/blind_spot_heatmap.txt"
)