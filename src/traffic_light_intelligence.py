from ultralytics import YOLO
import cv2
import numpy as np

# ====================================
# LOAD MODEL
# ====================================

model = YOLO("yolov8n.pt")

IMAGE_PATH = "test_image.jpg"

image = cv2.imread(IMAGE_PATH)

results = model(image)

traffic_states = []

# ====================================
# FIND TRAFFIC LIGHT
# ====================================

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

        # ===============================
        # COLOR ANALYSIS
        # ===============================

        hsv = cv2.cvtColor(
            traffic_crop,
            cv2.COLOR_BGR2HSV
        )

        # RED
        lower_red1 = np.array([0,70,50])
        upper_red1 = np.array([10,255,255])

        lower_red2 = np.array([170,70,50])
        upper_red2 = np.array([180,255,255])

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
            np.array([20,100,100]),
            np.array([35,255,255])
        )

        # GREEN
        green_mask = cv2.inRange(
            hsv,
            np.array([40,50,50]),
            np.array([90,255,255])
        )

        red_pixels = cv2.countNonZero(
            red_mask
        )

        yellow_pixels = cv2.countNonZero(
            yellow_mask
        )

        green_pixels = cv2.countNonZero(
            green_mask
        )

        print("\nTraffic Light Analysis")

        print(
            "Red Pixels:",
            red_pixels
        )

        print(
            "Yellow Pixels:",
            yellow_pixels
        )

        print(
            "Green Pixels:",
            green_pixels
        )

        if red_pixels > yellow_pixels and red_pixels > green_pixels:
            traffic_states.append("RED")
            traffic_state = "RED"
        elif yellow_pixels > red_pixels and yellow_pixels > green_pixels:
            traffic_states.append("YELLOW")
            traffic_state = "YELLOW"
        elif green_pixels > red_pixels and green_pixels > yellow_pixels:
            traffic_states.append("GREEN")
            traffic_state = "GREEN"
        else:
            traffic_state = "UNKNOWN"

        break

if "RED" in traffic_states:
    traffic_state = "RED"
elif "YELLOW" in traffic_states:
    traffic_state = "YELLOW"
elif "GREEN" in traffic_states:
    traffic_state = "GREEN"
else:
    traffic_state = "UNKNOWN"

print("\n========== TRAFFIC DEBUG ==========")
print("Traffic States Found:", traffic_states)
print("Final Traffic State :", traffic_state)
print("===================================\n")

print("\n======================")
print("TRAFFIC LIGHT STATUS")
print("======================")
print("State:", traffic_state)
print("======================")