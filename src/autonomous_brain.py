import cv2
import numpy as np

from ultralytics import YOLO
from src.risk_based_decision_engine import calculate_risk
from src.traffic_light_intelligence import traffic_states

# ====================================
# LOAD MODEL ONCE
# ====================================

model = YOLO("yolov8n.pt")

# ====================================
# MAIN FUNCTION
# ====================================

def analyze_scene(image):

    results = model(
        image,
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

    # ====================================
    # LEAD VEHICLE INTELLIGENCE
    # ====================================

    best_vehicle = None

    best_score = -999999

    lead_area = 0

    largest_area = 0

    distance_state = "FAR"

    image_height, image_width = image.shape[:2]

    image_center = image_width / 2

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

            center_x = (x1 + x2) / 2

            center_distance = abs(
                center_x - image_center
            )

            score = (
                area / 1000
            ) - (
                center_distance
            )

            if score > best_score:
                best_score = score
                best_vehicle = class_name
                lead_area = area

            if area > largest_area:
                largest_area = area

        # ====================================
        # TRAFFIC LIGHT INTELLIGENCE
        # ====================================

        if class_name == "traffic light":
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

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

            yellow_pixels = cv2.countNonZero(
                cv2.inRange(
                    hsv,
                    np.array([20,100,100]),
                    np.array([35,255,255])
                )
            )

            green_pixels = cv2.countNonZero(
                cv2.inRange(
                    hsv,
                    np.array([40,50,50]),
                    np.array([90,255,255])
                )
            )

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

            print("\nTraffic Light Found")

            print(
                "Box:",
                x1, y1, x2, y2
            )

            print(
                "Red:",
                red_pixels
            )

            print(
                "Yellow:",
                yellow_pixels
            )

            print(
                "Green:",
                green_pixels
            )

            print(
                "State:",
                traffic_state
            )

    # ====================================
    # LEAD VEHICLE DISTANCE
    # ====================================

    if lead_area > 120000:

        distance_state = "VERY_CLOSE"

    elif lead_area > 60000:

        distance_state = "MEDIUM"

    else:

        distance_state = "FAR"

    print("\n===== LEAD VEHICLE =====")

    print(
        "Vehicle:",
        best_vehicle
    )

    print(
        "Area:",
        int(lead_area)
    )

    print(
        "Distance:",
        distance_state
    )

    print("========================\n")

    # ====================================
    # RELIABILITY
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
    # ANNOTATED IMAGE
    # ====================================

    annotated_image = results[0].plot()
    
    

    # ====================================
    # RETURN EVERYTHING
    # ====================================

    print("\n========== TRAFFIC DEBUG ==========")
    print("Traffic States Found:", traffic_states)
    print("Final Traffic State :", traffic_state)
    print("===================================\n")

    return {

        "annotated_image":
            annotated_image,

        "lead_vehicle":
            best_vehicle,

        "lead_vehicle_area":
            int(lead_area),

        "object_count":
            object_count,

        "vehicle_count":
            vehicle_count,

        "pedestrian_count":
            pedestrian_count,

        "traffic_signal":
            traffic_state,

        "vehicle_distance":
            distance_state,

        "brightness":
            round(brightness, 2),

        "contrast":
            round(contrast, 2),

        "blur_score":
            round(blur_score, 2),

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

        "reasons":
            reasons
    }