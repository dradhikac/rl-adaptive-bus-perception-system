from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np

# =====================================
# CONFIGURATION
# =====================================

VIDEO_PATH = "videos/sample_drive.mp4"

FRAME_SKIP = 30

model = YOLO("yolov8n.pt")

# =====================================
# OPEN VIDEO
# =====================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("Failed to open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

print("FPS:", fps)

frame_number = 0

timeline = []

# =====================================
# PROCESS VIDEO
# =====================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # Process every Nth frame
    if frame_number % FRAME_SKIP != 0:
        continue

    timestamp = round(
        frame_number / fps,
        2
    )

    # =================================
    # YOLO DETECTION
    # =================================

    results = model(frame, verbose=False)

    object_count = 0
    vehicle_count = 0
    pedestrian_count = 0

    traffic_state = "UNKNOWN"

    vehicle_classes = [
        "car",
        "truck",
        "bus",
        "motorcycle"
    ]

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[cls_id]

        object_count += 1

        if class_name == "person":
            pedestrian_count += 1

        if class_name in vehicle_classes:
            vehicle_count += 1

        # =============================
        # TRAFFIC LIGHT ANALYSIS
        # =============================

        if class_name == "traffic light":

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = frame[
                y1:y2,
                x1:x2
            ]

            if crop.size > 0:

                hsv = cv2.cvtColor(
                    crop,
                    cv2.COLOR_BGR2HSV
                )

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

                green_pixels = cv2.countNonZero(
                    cv2.inRange(
                        hsv,
                        np.array([40,50,50]),
                        np.array([90,255,255])
                    )
                )

                yellow_pixels = cv2.countNonZero(
                    cv2.inRange(
                        hsv,
                        np.array([20,100,100]),
                        np.array([35,255,255])
                    )
                )

                if (
                    red_pixels > green_pixels
                    and
                    red_pixels > yellow_pixels
                ):
                    traffic_state = "RED"

                elif (
                    green_pixels > red_pixels
                    and
                    green_pixels > yellow_pixels
                ):
                    traffic_state = "GREEN"

                elif (
                    yellow_pixels > red_pixels
                    and
                    yellow_pixels > green_pixels
                ):
                    traffic_state = "YELLOW"

    # =================================
    # DECISION LOGIC
    # =================================

    decision = "MOVE"

    speed = 40

    if traffic_state == "RED":

        decision = "STOP"

        speed = 0

    elif pedestrian_count > 0:

        decision = "STOP"

        speed = 0

    elif vehicle_count > 5:

        decision = "SLOW DOWN"

        speed = 15

    # =================================
    # STORE RESULT
    # =================================

    timeline.append({

        "time_sec":
            timestamp,

        "traffic_signal":
            traffic_state,

        "objects":
            object_count,

        "vehicles":
            vehicle_count,

        "pedestrians":
            pedestrian_count,

        "decision":
            decision,

        "recommended_speed":
            speed
    })

    print(
        f"{timestamp}s | "
        f"{traffic_state} | "
        f"{decision} | "
        f"{speed} km/h"
    )

# =====================================
# SAVE CSV
# =====================================

cap.release()

df = pd.DataFrame(timeline)

df.to_csv(

    "outputs/video_decision_report.csv",

    index=False
)

print("\nSaved:")
print(
    "outputs/video_decision_report.csv"
)

print(
    "\nFrames Analyzed:",
    len(df)
)