from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os

# =========================================================
# Blind Spot Risk Prediction Foundation
# =========================================================

# Create outputs folder if not exists
os.makedirs("outputs", exist_ok=True)

# =========================================================
# LOAD YOLO MODEL
# =========================================================

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("YOLO model loaded successfully.")

# =========================================================
# OPEN VIDEO
# =========================================================

video_path = "videos/sample_drive.mp4"

cap = cv2.VideoCapture(video_path)

# Check video
if not cap.isOpened():
    print("Error opening video.")
    exit()

print("Blind spot analysis started.")

# =========================================================
# FRAME SETTINGS
# =========================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# =========================================================
# BLIND SPOT ZONES
# =========================================================

LEFT_BLIND = (0, 0, 150, FRAME_HEIGHT)

RIGHT_BLIND = (490, 0, 640, FRAME_HEIGHT)

# =========================================================
# DATA STORAGE
# =========================================================

blind_results = []

critical_events = 0

frame_count = 0

# =========================================================
# VIDEO LOOP
# =========================================================

while True:

    # Read frame
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        break

    frame_count += 1

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # =====================================================
    # DRAW BLIND SPOT REGIONS
    # =====================================================

    # Left blind zone
    cv2.rectangle(
        frame,
        (LEFT_BLIND[0], LEFT_BLIND[1]),
        (LEFT_BLIND[2], LEFT_BLIND[3]),
        (0, 0, 255),
        2
    )

    # Right blind zone
    cv2.rectangle(
        frame,
        (RIGHT_BLIND[0], RIGHT_BLIND[1]),
        (RIGHT_BLIND[2], RIGHT_BLIND[3]),
        (0, 0, 255),
        2
    )

    # =====================================================
    # YOLO OBJECT DETECTION
    # =====================================================

    results = model(frame)

    # =====================================================
    # PROCESS DETECTED OBJECTS
    # =====================================================

    for box in results[0].boxes:

        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # =================================================
        # OBJECT CENTER
        # =================================================

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # =================================================
        # CHECK BLIND SPOT ENTRY
        # =================================================

        in_left_blind = (
            LEFT_BLIND[0] <= cx <= LEFT_BLIND[2]
        )

        in_right_blind = (
            RIGHT_BLIND[0] <= cx <= RIGHT_BLIND[2]
        )

        # =================================================
        # OBJECT SIZE / PROXIMITY ESTIMATION
        # =================================================

        width = x2 - x1
        height = y2 - y1

        area = width * height

        print("Object Area:", area)

        # =================================================
        # RISK ANALYSIS
        # =================================================

        risk_level = "LOW"

        if in_left_blind or in_right_blind:
            risk_level = "HIGH"

        # Critical danger
        if (
            in_left_blind or
            in_right_blind
        ) and area > 15000:

            risk_level = "CRITICAL"

        # =================================================
        # DANGER EVENT COUNT
        # =================================================

        if risk_level == "CRITICAL":
            critical_events += 1

        # =================================================
        # BOX COLOR
        # =================================================

        color = (0, 255, 0)

        if risk_level == "HIGH":
            color = (0, 165, 255)

        if risk_level == "CRITICAL":
            color = (0, 0, 255)

        # =================================================
        # DRAW BOUNDING BOX
        # =================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # =================================================
        # DRAW CENTER POINT
        # =================================================

        cv2.circle(
            frame,
            (cx, cy),
            5,
            color,
            -1
        )

        # =================================================
        # DISPLAY RISK LABEL
        # =================================================

        cv2.putText(
            frame,
            f"Risk: {risk_level}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # =================================================
        # SAVE ANALYSIS DATA
        # =================================================

        blind_results.append({
            "frame": frame_count,
            "center_x": cx,
            "center_y": cy,
            "risk_level": risk_level,
            "area": area
        })

    # =====================================================
    # DISPLAY CRITICAL EVENT COUNT
    # =====================================================

    cv2.putText(
        frame,
        f"Critical Events: {critical_events}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # =====================================================
    # DISPLAY FRAME
    # =====================================================

    cv2.imshow(
        "Blind Spot Analysis",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# RELEASE RESOURCES
# =========================================================

cap.release()

cv2.destroyAllWindows()

# =========================================================
# SAVE CSV RESULTS
# =========================================================

df = pd.DataFrame(blind_results)

df.to_csv(
    "outputs/blind_spot_results.csv",
    index=False
)

print("\nBlind spot CSV saved.")

print("Critical Blind Spot Events:", critical_events)

print("\n===================================================")
print("Blind Spot Risk Prediction System Completed")
print("===================================================")