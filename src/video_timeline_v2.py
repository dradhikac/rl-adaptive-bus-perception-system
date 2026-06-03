from ultralytics import YOLO
from risk_based_decision_engine import calculate_risk

import cv2
import pandas as pd
import numpy as np

# ====================================
# CONFIGURATION
# ====================================

VIDEO_PATH = "videos/sample_drive.mp4"

FRAME_SKIP = 30

model = YOLO("yolov8n.pt")

# ====================================
# OPEN VIDEO
# ====================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("Failed to open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)

timeline = []

frame_number = 0

# ====================================
# PROCESS VIDEO
# ====================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    if frame_number % FRAME_SKIP != 0:
        continue

    timestamp = round(
        frame_number / fps,
        2
    )

    # ====================================
    # YOLO DETECTION
    # ====================================

    results = model(
        frame,
        verbose=False
    )

    object_count = 0
    vehicle_count = 0
    pedestrian_count = 0

    vehicle_classes = [
        "car",
        "truck",
        "bus",
        "motorcycle"
    ]

    traffic_state = "UNKNOWN"

    largest_area = 0

    distance_state = "FAR"

    # ====================================
    # PROCESS DETECTIONS
    # ====================================

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        object_count += 1

        if class_name == "person":

            pedestrian_count += 1

        if class_name in vehicle_classes:

            vehicle_count += 1

            x1, y1, x2, y2 = box.xyxy[0]

            width = x2 - x1

            height = y2 - y1

            area = width * height

            if area > largest_area:

                largest_area = area

        # ====================================
        # TRAFFIC LIGHT ANALYSIS
        # ====================================

        if class_name == "traffic light":

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            hsv = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2HSV
            )

            # RED

            red_mask1 = cv2.inRange(
                hsv,
                np.array([0,70,50]),
                np.array([10,255,255])
            )

            red_mask2 = cv2.inRange(
                hsv,
                np.array([170,70,50]),
                np.array([180,255,255])
            )

            red_pixels = cv2.countNonZero(
                red_mask1 + red_mask2
            )

            # YELLOW

            yellow_pixels = cv2.countNonZero(

                cv2.inRange(
                    hsv,
                    np.array([20,100,100]),
                    np.array([35,255,255])
                )
            )

            # GREEN

            green_pixels = cv2.countNonZero(

                cv2.inRange(
                    hsv,
                    np.array([40,50,50]),
                    np.array([90,255,255])
                )
            )

            if (

                red_pixels > yellow_pixels
                and
                red_pixels > green_pixels

            ):

                traffic_state = "RED"

            elif (

                yellow_pixels > red_pixels
                and
                yellow_pixels > green_pixels

            ):

                traffic_state = "YELLOW"

            elif (

                green_pixels > red_pixels
                and
                green_pixels > yellow_pixels

            ):

                traffic_state = "GREEN"

    # ====================================
    # DISTANCE ESTIMATION
    # ====================================

    if largest_area > 150000:

        distance_state = "VERY CLOSE"

    elif largest_area > 50000:

        distance_state = "MEDIUM"

    else:

        distance_state = "FAR"

    # ====================================
    # RELIABILITY
    # ====================================

    gray = cv2.cvtColor(
        frame,
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
    # RISK ENGINE
    # ====================================

    (
        risk_score,
        risk_level,
        decision,
        speed,
        reasons

    ) = calculate_risk(

        traffic_state,

        pedestrian_count,

        distance_state,

        fusion_confidence,

        camera_reliability
    )

    # ====================================
    # SAVE RECORD
    # ====================================

    timeline.append({

        "time_sec":
            timestamp,

        "traffic_signal":
            traffic_state,

        "vehicle_distance":
            distance_state,

        "objects":
            object_count,

        "vehicles":
            vehicle_count,

        "pedestrians":
            pedestrian_count,

        "camera_reliability":
            round(
                camera_reliability,
                3
            ),

        "fusion_confidence":
            round(
                fusion_confidence,
                3
            ),

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "decision":
            decision,

        "recommended_speed":
            speed,

        "reason":
            " | ".join(reasons)
    })

    print(

        f"{timestamp}s | "

        f"{traffic_state} | "

        f"{risk_score} | "

        f"{risk_level} | "

        f"{decision} | "

        f"{speed} km/h"
    )

# ====================================
# SAVE CSV
# ====================================

cap.release()

df = pd.DataFrame(timeline)

df.to_csv(

    "outputs/video_risk_timeline.csv",

    index=False
)

print("\n================================")
print("VIDEO ANALYSIS COMPLETE")
print("================================")

print(
    "\nSaved:"
)

print(
    "outputs/video_risk_timeline.csv"
)

print(
    "\nFrames Analyzed:",
    len(df)
)