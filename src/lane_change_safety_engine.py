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

left_cameras = [
    "CAM_FRONT_LEFT",
    "CAM_BACK_LEFT"
]

right_cameras = [
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT"
]

vehicle_classes = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle"
]

# ====================================
# SIDE ANALYSIS
# ====================================

def analyze_side(camera_list):

    vehicles = 0
    people = 0

    for cam in camera_list:

        token = sample["data"][cam]

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

    return {

        "vehicles": vehicles,

        "people": people,

        "risk_score": risk_score
    }

# ====================================
# ANALYZE BOTH SIDES
# ====================================

left = analyze_side(
    left_cameras
)

right = analyze_side(
    right_cameras
)

# ====================================
# SAFETY LOGIC
# ====================================

def lane_status(score):

    if score >= 100:

        return (
            "UNSAFE",
            "HIGH"
        )

    elif score >= 50:

        return (
            "CAUTION",
            "MEDIUM"
        )

    else:

        return (
            "SAFE",
            "LOW"
        )

left_status, left_risk = lane_status(
    left["risk_score"]
)

right_status, right_risk = lane_status(
    right["risk_score"]
)

# ====================================
# RECOMMENDED ACTION
# ====================================

if left_status == "SAFE":

    recommendation = (
        "CHANGE LANE LEFT"
    )

elif right_status == "SAFE":

    recommendation = (
        "CHANGE LANE RIGHT"
    )

else:

    recommendation = (
        "KEEP CURRENT LANE"
    )

# ====================================
# COLLISION RISK
# ====================================

collision_risk = max(

    left["risk_score"],

    right["risk_score"]

)

# ====================================
# REPORT
# ====================================

print("\n====================================")
print(" LANE CHANGE SAFETY REPORT ")
print("====================================")

print("\nLEFT SIDE")

print(
    "Vehicles:",
    left["vehicles"]
)

print(
    "People:",
    left["people"]
)

print(
    "Risk Score:",
    left["risk_score"]
)

print(
    "Lane Status:",
    left_status
)

print(
    "Risk Level:",
    left_risk
)

print("\nRIGHT SIDE")

print(
    "Vehicles:",
    right["vehicles"]
)

print(
    "People:",
    right["people"]
)

print(
    "Risk Score:",
    right["risk_score"]
)

print(
    "Lane Status:",
    right_status
)

print(
    "Risk Level:",
    right_risk
)

print("\n====================================")

print(
    "Collision Risk:",
    collision_risk
)

print(
    "Recommended Action:",
    recommendation
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
    "outputs/lane_change_report.txt",
    "w"
) as f:

    f.write(
        "LANE CHANGE SAFETY REPORT\n\n"
    )

    f.write(
        f"Left Status: {left_status}\n"
    )

    f.write(
        f"Right Status: {right_status}\n"
    )

    f.write(
        f"Recommendation: {recommendation}\n"
    )

print(
    "\nSaved:"
)

print(
    "outputs/lane_change_report.txt"
)