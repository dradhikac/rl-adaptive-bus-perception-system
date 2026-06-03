from ultralytics import YOLO
import cv2
import numpy as np

# ====================================
# LOAD MODEL
# ====================================

model = YOLO("yolov8n.pt")

# Change this to your image
IMAGE_PATH = "test_image.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

# ====================================
# OBJECT DETECTION
# ====================================

results = model(
    image,
    conf=0.4
)

object_count = 0
vehicle_count = 0
pedestrian_count = 0

vehicle_classes = {
    "car",
    "truck",
    "bus",
    "motorcycle"
}

for box in results[0].boxes:

    cls_id = int(box.cls[0])

    cls_name = model.names[cls_id]

    object_count += 1

    if cls_name == "person":
        pedestrian_count += 1

    if cls_name in vehicle_classes:
        vehicle_count += 1

# ====================================
# CAMERA RELIABILITY
# ====================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

brightness = np.mean(gray)

contrast = gray.std()

blur_score = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

camera_reliability = 1.0

if brightness < 60:
    camera_reliability -= 0.2

if contrast < 40:
    camera_reliability -= 0.2

if blur_score < 100:
    camera_reliability -= 0.2

camera_reliability = max(
    0.0,
    min(camera_reliability, 1.0)
)

# ====================================
# FUSION CONFIDENCE
# ====================================

lidar_reliability = 1.0

fusion_confidence = (
    camera_reliability +
    lidar_reliability
) / 2

# ====================================
# TRAFFIC LIGHT STATE
# ====================================

# Temporary placeholder
# Change to RED/GREEN manually

traffic_state = "GREEN"

# ====================================
# RISK SCORING
# ====================================

risk_score = 0

# Pedestrians are high risk
risk_score += pedestrian_count * 5

if object_count > 15:
    risk_score += 4

elif object_count > 10:
    risk_score += 3

elif object_count > 5:
    risk_score += 2

# Low confidence penalty

if fusion_confidence < 0.7:
    risk_score += 2

# ====================================
# DECISION ENGINE
# ====================================

decision = "MOVE"

if traffic_state == "RED":

    decision = "STOP"

elif pedestrian_count > 0:

    decision = "STOP"

elif risk_score >= 5:

    decision = "STOP"

elif risk_score >= 2:

    decision = "SLOW DOWN"

else:

    decision = "MOVE"

# ====================================
# SPEED CONTROL
# ====================================

speed = 40

if decision == "STOP":

    speed = 0

elif decision == "SLOW DOWN":

    speed = 15

# ====================================
# REPORT
# ====================================

print("\n====================================")
print(" REAL-TIME DECISION ENGINE ")
print("====================================")

print(f"\nObjects Detected     : {object_count}")
print(f"Vehicles Detected    : {vehicle_count}")
print(f"Pedestrians Detected : {pedestrian_count}")

print()

print(f"Brightness           : {brightness:.2f}")
print(f"Contrast             : {contrast:.2f}")
print(f"Blur Score           : {blur_score:.2f}")

print()

print(f"Camera Reliability   : {camera_reliability:.3f}")
print(f"LiDAR Reliability    : {lidar_reliability:.3f}")
print(f"Fusion Confidence    : {fusion_confidence:.3f}")

print()

print(f"Traffic Signal       : {traffic_state}")
print(f"Risk Score           : {risk_score}")

print()

print(f"Decision             : {decision}")
print(f"Recommended Speed    : {speed} km/h")

print("\n====================================")