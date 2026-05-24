from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import os

# =========================================================
# Adaptive Environmental Risk Intelligence Engine
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

print("Adaptive Risk Intelligence Started.")

# =========================================================
# FRAME SETTINGS
# =========================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Blind spot regions
LEFT_BLIND = (0, 0, 150, 480)

RIGHT_BLIND = (490, 0, 640, 480)

# =========================================================
# DATA STORAGE
# =========================================================

adaptive_results = []

frame_count = 0

# =========================================================
# VIDEO LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Resize frame
    frame = cv2.resize(frame, (640, 480))

    # =====================================================
    # ENVIRONMENTAL RELIABILITY ANALYSIS
    # =====================================================

    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Brightness estimation
    brightness = np.mean(gray)

    # Contrast estimation
    contrast = gray.std()

    # =====================================================
    # RELIABILITY ESTIMATION
    # =====================================================

    reliability = 1.0

    # Dark environment penalty
    if brightness < 60:
        reliability -= 0.3

    # Low contrast penalty
    if contrast < 40:
        reliability -= 0.3

    # Clamp reliability
    reliability = max(
        0.0,
        min(reliability, 1.0)
    )

    # =====================================================
    # DRAW BLIND SPOT ZONES
    # =====================================================

    cv2.rectangle(
        frame,
        (LEFT_BLIND[0], LEFT_BLIND[1]),
        (LEFT_BLIND[2], LEFT_BLIND[3]),
        (0, 0, 255),
        2
    )

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
        # DETECTION CONFIDENCE
        # =================================================

        confidence = float(box.conf[0])

        # =================================================
        # BLIND SPOT CHECK
        # =================================================

        in_left_blind = (
            LEFT_BLIND[0] <= cx <= LEFT_BLIND[2]
        )

        in_right_blind = (
            RIGHT_BLIND[0] <= cx <= RIGHT_BLIND[2]
        )

        # =================================================
        # OBJECT SIZE / PROXIMITY
        # =================================================

        width = x2 - x1
        height = y2 - y1

        area = width * height

        # =================================================
        # ADAPTIVE RISK SCORING
        # =================================================

        adaptive_risk = 0

        # Blind spot danger
        if in_left_blind or in_right_blind:
            adaptive_risk += 3

        # Close object danger
        if area > 15000:
            adaptive_risk += 2

        # Environmental unreliability
        if reliability < 0.5:
            adaptive_risk += 3

        # Weak detection confidence
        if confidence < 0.5:
            adaptive_risk += 2

        # =================================================
        # RISK LEVEL CLASSIFICATION
        # =================================================

        risk_level = "LOW"

        if adaptive_risk >= 4:
            risk_level = "HIGH"

        if adaptive_risk >= 7:
            risk_level = "CRITICAL"

        # =================================================
        # COLOR SELECTION
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
        # DRAW OBJECT CENTER
        # =================================================

        cv2.circle(
            frame,
            (cx, cy),
            5,
            color,
            -1
        )

        # =================================================
        # DISPLAY RISK INFORMATION
        # =================================================

        cv2.putText(
            frame,
            f"{risk_level} | Rel:{reliability:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        # =================================================
        # SAVE ANALYSIS DATA
        # =================================================

        adaptive_results.append({
            "frame": frame_count,
            "confidence": confidence,
            "reliability": reliability,
            "adaptive_risk": adaptive_risk,
            "risk_level": risk_level,
            "area": area
        })

    # =====================================================
    # DISPLAY ENVIRONMENT RELIABILITY
    # =====================================================

    cv2.putText(
        frame,
        f"Env Reliability: {reliability:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # =====================================================
    # DISPLAY VIDEO
    # =====================================================

    cv2.imshow(
        "Adaptive Risk Intelligence",
        frame
    )

    # Quit with Q
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

df = pd.DataFrame(adaptive_results)

df.to_csv(
    "outputs/adaptive_risk_results.csv",
    index=False
)

print("\nAdaptive risk CSV saved.")

print("\n===================================================")
print("Adaptive Environmental Risk Intelligence Completed")
print("===================================================")