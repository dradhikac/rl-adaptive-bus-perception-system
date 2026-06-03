from ultralytics import YOLO
import cv2
import numpy as np

# ====================================
# LOAD MODEL
# ====================================

model = YOLO("yolov8n.pt")

IMAGE_PATH = "test_image.jpg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Failed to load image.")
    exit()

# ====================================
# YOLO DETECTION
# ====================================

results = model(image)

object_count = 0
vehicle_count = 0
pedestrian_count = 0

vehicle_classes = [
    "car",
    "truck",
    "bus",
    "motorcycle"
]

# ====================================
# TRAFFIC LIGHT ANALYSIS
# ====================================

traffic_state = "UNKNOWN"

for box in results[0].boxes:

    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]

    if class_name == "traffic light":

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        traffic_crop = image[y1:y2, x1:x2]

        if traffic_crop.size == 0:
            continue

        hsv = cv2.cvtColor(
            traffic_crop,
            cv2.COLOR_BGR2HSV
        )

        # RED
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        red_mask = (
            cv2.inRange(
                hsv,
                lower_red1,
                upper_red1
            )
            +
            cv2.inRange(
                hsv,
                lower_red2,
                upper_red2
            )
        )

        # YELLOW
        yellow_mask = cv2.inRange(
            hsv,
            np.array([20, 100, 100]),
            np.array([35, 255, 255])
        )

        # GREEN
        green_mask = cv2.inRange(
            hsv,
            np.array([40, 50, 50]),
            np.array([90, 255, 255])
        )

        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)

        if (
            red_pixels > yellow_pixels
            and red_pixels > green_pixels
        ):
            traffic_state = "RED"

        elif (
            yellow_pixels > red_pixels
            and yellow_pixels > green_pixels
        ):
            traffic_state = "YELLOW"

        elif (
            green_pixels > red_pixels
            and green_pixels > yellow_pixels
        ):
            traffic_state = "GREEN"

# ====================================
# OBJECT COUNTING
# ====================================

for box in results[0].boxes:

    cls_id = int(box.cls[0])

    class_name = model.names[cls_id]

    object_count += 1

    if class_name == "person":
        pedestrian_count += 1

    if class_name in vehicle_classes:
        vehicle_count += 1

# ====================================
# VEHICLE DISTANCE ESTIMATION
# ====================================

closest_vehicle = "NONE"

largest_area = 0

for box in results[0].boxes:

    cls_id = int(box.cls[0])

    class_name = model.names[cls_id]

    if class_name not in vehicle_classes:
        continue

    x1, y1, x2, y2 = box.xyxy[0]

    width = x2 - x1
    height = y2 - y1

    area = width * height

    if area > largest_area:

        largest_area = area

        closest_vehicle = class_name

if largest_area > 150000:

    distance_state = "VERY CLOSE"

elif largest_area > 50000:

    distance_state = "MEDIUM"

else:

    distance_state = "FAR"

# ====================================
# RELIABILITY ANALYSIS
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
    min(
        camera_reliability,
        1.0
    )
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
# DECISION ENGINE V2
# ====================================

decision = "MOVE"

speed = 40

reason = []

# Highest priority

if traffic_state == "RED":

    decision = "STOP"

    speed = 0

    reason.append(
        "Red traffic signal detected"
    )

elif pedestrian_count > 0:

    decision = "STOP"

    speed = 0

    reason.append(
        "Pedestrian detected"
    )

elif distance_state == "VERY CLOSE":

    decision = "STOP"

    speed = 0

    reason.append(
        "Vehicle too close"
    )

elif traffic_state == "YELLOW":

    decision = "SLOW DOWN"

    speed = 10

    reason.append(
        "Yellow signal detected"
    )

elif distance_state == "MEDIUM":

    decision = "SLOW DOWN"

    speed = 15

    reason.append(
        "Vehicle ahead at medium distance"
    )

else:

    decision = "MOVE"

    speed = 40

    reason.append(
        "Road appears clear"
    )

# Additional context

if fusion_confidence < 0.7:

    reason.append(
        "Low fusion confidence"
    )

# ====================================
# REPORT
# ====================================

print("\n")
print("=" * 50)
print(" AUTONOMOUS DECISION REPORT ")
print("=" * 50)

print(
    f"Traffic Signal      : {traffic_state}"
)

print(
    f"Lead Vehicle        : {closest_vehicle}"
)

print(
    f"Distance            : {distance_state}"
)

print(
    f"Objects Detected    : {object_count}"
)

print(
    f"Vehicles Detected   : {vehicle_count}"
)

print(
    f"Pedestrians         : {pedestrian_count}"
)

print(
    f"Camera Reliability  : "
    f"{camera_reliability:.3f}"
)

print(
    f"Fusion Confidence   : "
    f"{fusion_confidence:.3f}"
)

print()

print(
    f"Decision            : {decision}"
)

print(
    f"Recommended Speed   : {speed} km/h"
)

print("\nReason:")

for r in reason:
    print(f"- {r}")

print("=" * 50)

# ====================================
# SAVE REPORT
# ====================================

with open(
    "outputs/decision_report_v2.txt",
    "w"
) as f:

    f.write(
        f"Traffic Signal: {traffic_state}\n"
    )

    f.write(
        f"Lead Vehicle: {closest_vehicle}\n"
    )

    f.write(
        f"Distance: {distance_state}\n"
    )

    f.write(
        f"Decision: {decision}\n"
    )

    f.write(
        f"Speed: {speed}\n"
    )

    f.write(
        f"Fusion Confidence: "
        f"{fusion_confidence:.3f}\n"
    )

print(
    "\nSaved: outputs/decision_report_v2.txt"
)