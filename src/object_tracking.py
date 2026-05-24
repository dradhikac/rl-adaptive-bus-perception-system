from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os

# =========================================================
# Dynamic Object Tracking & Trajectory Intelligence System
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
# VIDEO PATH
# =========================================================

video_path = "videos/sample_drive.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

# Check if video opened correctly
if not cap.isOpened():
    print("Error opening video.")
    exit()

print("Tracking system started.")

# =========================================================
# TRACKING VARIABLES
# =========================================================

# Object ID counter
object_id = 0

# Store trajectories
trajectories = {}

# Tracking results storage
tracking_results = []

# =========================================================
# VIDEO LOOP
# =========================================================

while True:

    # Read frame
    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        break

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

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
        # CENTROID CALCULATION
        # =================================================

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # =================================================
        # OBJECT ID ASSIGNMENT
        # =================================================

        current_id = object_id

        object_id += 1

        # =================================================
        # STORE TRAJECTORY
        # =================================================

        if current_id not in trajectories:
            trajectories[current_id] = []

        trajectories[current_id].append((cx, cy))

        # =================================================
        # DRAW BOUNDING BOX
        # =================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # =================================================
        # DRAW CENTROID
        # =================================================

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 0, 255),
            -1
        )

        # =================================================
        # DISPLAY OBJECT ID
        # =================================================

        cv2.putText(
            frame,
            f"ID: {current_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # =================================================
        # DRAW TRAJECTORY PATH
        # =================================================

        for point in trajectories[current_id]:

            cv2.circle(
                frame,
                point,
                2,
                (255, 255, 0),
                -1
            )

        # =================================================
        # MOTION SPEED ESTIMATION
        # =================================================

        if len(trajectories[current_id]) >= 2:

            prev_x, prev_y = trajectories[current_id][-2]

            speed = np.sqrt(
                (cx - prev_x) ** 2 +
                (cy - prev_y) ** 2
            )

            print(f"Object {current_id} Speed:", speed)

        else:
            speed = 0

        # =================================================
        # SAVE TRACKING DATA
        # =================================================

        tracking_results.append({
            "object_id": current_id,
            "center_x": cx,
            "center_y": cy,
            "speed": speed
        })

    # =====================================================
    # DISPLAY TRACKING FRAME
    # =====================================================

    cv2.imshow("Object Tracking", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# RELEASE RESOURCES
# =========================================================

cap.release()

cv2.destroyAllWindows()

# =========================================================
# SAVE TRACKING CSV
# =========================================================

df = pd.DataFrame(tracking_results)

df.to_csv(
    "outputs/object_tracking.csv",
    index=False
)

print("\nTracking CSV saved.")

print("\n===================================================")
print("Dynamic Object Tracking System Completed")
print("===================================================")