from ultralytics import YOLO
import cv2
import time
import pandas as pd
import os

# =========================================================
# Real-Time Video Perception Engine
# =========================================================

# Create output folder if not exists
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

# Check video opened
if not cap.isOpened():
    print("Error opening video.")
    exit()

print("Video loaded successfully.")

# =========================================================
# VIDEO WRITER SETUP
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    "outputs/annotated_video.mp4",
    fourcc,
    20.0,
    (640, 480)
)

# =========================================================
# FPS VARIABLES
# =========================================================

prev_time = 0

# =========================================================
# VIDEO ANALYSIS STORAGE
# =========================================================

video_results = []

frame_count = 0

# =========================================================
# FRAME PROCESSING LOOP
# =========================================================

while True:

    # Read frame
    ret, frame = cap.read()

    # Stop when video ends
    if not ret:
        break

    frame_count += 1

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # =====================================================
    # YOLO OBJECT DETECTION
    # =====================================================

    results = model(frame)

    # Get annotated frame
    annotated_frame = results[0].plot()

    # =====================================================
    # FPS CALCULATION
    # =====================================================

    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # Draw FPS text
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # =====================================================
    # OBJECT COUNTING
    # =====================================================

    object_count = {}

    for box in results[0].boxes:

        class_id = int(box.cls[0])

        class_name = model.names[class_id]

        if class_name not in object_count:
            object_count[class_name] = 0

        object_count[class_name] += 1

    # Print detected objects
    print(f"\nFrame {frame_count}")
    print("Detected Objects:", object_count)

    # =====================================================
    # SAVE VIDEO ANALYTICS
    # =====================================================

    video_results.append({
        "frame": frame_count,
        "fps": fps,
        "objects_detected": len(object_count)
    })

    # =====================================================
    # SAVE ANNOTATED FRAME
    # =====================================================

    out.write(annotated_frame)

    # =====================================================
    # DISPLAY VIDEO
    # =====================================================

    cv2.imshow(
        "YOLO Video Detection",
        annotated_frame
    )

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# =========================================================
# RELEASE RESOURCES
# =========================================================

cap.release()

out.release()

cv2.destroyAllWindows()

# =========================================================
# SAVE CSV ANALYSIS
# =========================================================

df = pd.DataFrame(video_results)

df.to_csv(
    "outputs/video_analysis.csv",
    index=False
)

print("\nVideo analysis CSV saved.")

print("\nAnnotated video saved to:")
print("outputs/annotated_video.mp4")

print("\n===================================================")
print("Real-Time Video Perception Engine Completed")
print("===================================================")