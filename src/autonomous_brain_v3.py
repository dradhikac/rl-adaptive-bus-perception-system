from unittest import result

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO

from src.risk_based_decision_engine import calculate_risk
from src.blind_spot_intelligence import analyze_blind_spots
from src.decision_fusion_engine import ( fuse_decisions)
from src.pedestrian_intelligence import (analyze_pedestrians)
# ====================================
# LOAD MODELS
# ====================================

model = YOLO("yolov8n.pt")

vehicle_classes = [

    "car",
    "truck",
    "bus",
    "motorcycle"
]

# ====================================
# RL POLICY LOOKUP
# ====================================

def get_rl_action(

    traffic_signal,
    vehicle_distance,
    risk_level,
    left_status,
    right_status

):

    try:

        policy = pd.read_csv(
            "outputs/scenario_rl_policy.csv"
        )

        match = policy[

            (policy["traffic_signal"] == traffic_signal)

            &

            (policy["vehicle_distance"] == vehicle_distance)

            &

            (policy["risk_level"] == risk_level)

            &

            (policy["blind_left"] == left_status)

            &

            (policy["blind_right"] == right_status)

        ]

        if len(match) > 0:

            return match.iloc[0][
                "best_action"
            ]

    except:
        pass

    # ====================================
    # FALLBACK POLICY
    # ====================================

    if traffic_signal == "RED":

        return "STOP"

    if right_status == "UNSAFE" and left_status == "SAFE":

        return "CHANGE_LEFT"

    if left_status == "UNSAFE" and right_status == "SAFE":

        return "CHANGE_RIGHT"

    if risk_level == "HIGH":

        return "STOP"

    if vehicle_distance == "MEDIUM":

        return "SLOW_DOWN"

    return "MOVE"

# ====================================
# MAIN FUNCTION
# ====================================

def analyze_scene(

    front_image,

    front_left,

    front_right,

    rear_left,

    rear_right

):

    # ====================================
    # FRONT CAMERA ANALYSIS
    # ====================================

    results = model(

        front_image,

        verbose=False
    )
    
    front_left_result = model(
    front_left,
    verbose=False
    )

    front_right_result = model(
    front_right,
    verbose=False
)

    rear_left_result = model(
    rear_left,
    verbose=False
)

    rear_right_result = model(
    rear_right,
    verbose=False
)
    
    front_left_objects = len(
        front_left_result[0].boxes
)

    front_right_objects = len(
        front_right_result[0].boxes
)

    rear_left_objects = len(
        rear_left_result[0].boxes
)

    rear_right_objects = len(
        rear_right_result[0].boxes
)


    object_count = 0

    vehicle_count = 0

    pedestrian_count = 0

    traffic_signal = "UNKNOWN"

    lead_vehicle = None

    lead_area = 0

    best_score = -999999

    image_height, image_width = (
        front_image.shape[:2]
    )

    image_center = image_width / 2
    detection_list = []
    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        class_name = model.names[
            cls_id
        ]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detection_list.append({

        "class": class_name,

        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
        })

        object_count += 1

        if class_name == "person":

            pedestrian_count += 1

        # ==========================
        # VEHICLES
        # ==========================

        if class_name in vehicle_classes:

            vehicle_count += 1

            x1, y1, x2, y2 = box.xyxy[0]

            width = x2 - x1

            height = y2 - y1

            area = width * height

            center_x = (
                x1 + x2
            ) / 2

            center_distance = abs(

                center_x -
                image_center

            )

            score = (

                area / 1000

                -

                center_distance

            )

            if score > best_score:

                best_score = score

                lead_vehicle = (
                    class_name
                )

                lead_area = area

        # ==========================
        # TRAFFIC SIGNAL
        # ==========================

        if class_name == "traffic light":

            x1, y1, x2, y2 = map(

                int,

                box.xyxy[0]
            )

            crop = front_image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            hsv = cv2.cvtColor(

                crop,

                cv2.COLOR_BGR2HSV
            )

            red1 = cv2.inRange(

                hsv,

                np.array([0,70,50]),

                np.array([10,255,255])
            )

            red2 = cv2.inRange(

                hsv,

                np.array([170,70,50]),

                np.array([180,255,255])
            )

            red_pixels = cv2.countNonZero(
                red1 + red2
            )

            yellow_pixels = cv2.countNonZero(

                cv2.inRange(

                    hsv,

                    np.array(
                        [20,100,100]
                    ),

                    np.array(
                        [35,255,255]
                    )
                )
            )

            green_pixels = cv2.countNonZero(

                cv2.inRange(

                    hsv,

                    np.array(
                        [40,50,50]
                    ),

                    np.array(
                        [90,255,255]
                    )
                )
            )

            if (

                red_pixels >
                yellow_pixels

                and

                red_pixels >
                green_pixels

            ):

                traffic_signal = "RED"

            elif (

                yellow_pixels >
                red_pixels

                and

                yellow_pixels >
                green_pixels

            ):

                traffic_signal = "YELLOW"

            elif (

                green_pixels >
                red_pixels

                and

                green_pixels >
                yellow_pixels

            ):

                traffic_signal = "GREEN"

    
    # ====================================
    # PEDESTRIAN INTELLIGENCE
    # ====================================

    pedestrian_info = analyze_pedestrians(

        detection_list,

        image_width,

        image_height
    )

    print("\n===== PEDESTRIAN DEBUG =====")
    print(pedestrian_info)
    print("===========================\n")
    
    # ====================================
    # DISTANCE
    # ====================================

    if lead_area > 120000:

        vehicle_distance = (
            "VERY_CLOSE"
        )

    elif lead_area > 60000:

        vehicle_distance = (
            "MEDIUM"
        )

    else:

        vehicle_distance = (
            "FAR"
        )

    # ====================================
    # RELIABILITY
    # ====================================

    gray = cv2.cvtColor(

        front_image,

        cv2.COLOR_BGR2GRAY
    )

    brightness = np.mean(
        gray
    )

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

        0,

        min(
            camera_reliability,
            1
        )
    )

    # ====================================
    # FUSION
    # ====================================

    fusion_confidence = (

        camera_reliability +
        1.0

    ) / 2

    # ====================================
    # BLIND SPOTS
    # ====================================

    blind_spot = analyze_blind_spots(

        front_left,

        rear_left,

        front_right,

        rear_right
    )

    # ====================================
    # RISK
    # ====================================

    (
    risk_score,
    risk_level,
    risk_decision,
    risk_speed,
    reasons

) = calculate_risk(

    traffic_signal,

    pedestrian_info[
        "pedestrian_count"
    ],

    pedestrian_info[
        "pedestrian_status"
    ],

    vehicle_distance,

    fusion_confidence,

    camera_reliability,

    blind_spot["collision_risk"],

    blind_spot["left_status"],

    blind_spot["right_status"]
)
    print("\n===== AUTONOMOUS BRAIN DEBUG =====")
    print("Pedestrians:", pedestrian_count)
    print("Risk Score:", risk_score)
    print("Risk Level:", risk_level)
    print("Risk Decision:", risk_decision)
    print("=================================\n")

    # ====================================
    # RL POLICY
    # ====================================

    rl_action = get_rl_action(

        traffic_signal,

        vehicle_distance,

        risk_level,

        blind_spot[
            "left_status"
        ],

        blind_spot[
            "right_status"
        ]
    )
        # ====================================
    # DECISION FUSION ENGINE
    # ====================================

    fusion_result = fuse_decisions(

        traffic_signal=
            traffic_signal,

        risk_decision=
            risk_decision,

        risk_speed=
            risk_speed,

        rl_action=
            rl_action,

        left_lane_status=
            blind_spot["left_status"],

        right_lane_status=
            blind_spot["right_status"],

        lane_recommendation=
            blind_spot["recommendation"],

        collision_risk=
            blind_spot["collision_risk"]
    )
    # ====================================
    # DECISION FUSION
    # ====================================

       

        # ====================================
    # OUTPUT
    # ====================================

    return {

        "annotated_image":
            results[0].plot(),
            
        "front_annotated":
            results[0].plot(),

        "front_left_annotated":
            front_left_result[0].plot(),

        "front_right_annotated":
            front_right_result[0].plot(),

        "rear_left_annotated":
            rear_left_result[0].plot(),

        "rear_right_annotated":
            rear_right_result[0].plot(),
            
        "front_left_objects":
            front_left_objects,

        "front_right_objects":
            front_right_objects,

        "rear_left_objects":
            rear_left_objects,

        "rear_right_objects":
            rear_right_objects,

        "object_count":
            object_count,

        "vehicle_count":
            vehicle_count,

        "pedestrian_count":
            pedestrian_count,
            
        "pedestrian_status":
            pedestrian_info[
                "pedestrian_status"
            ],

        "in_lane_count":
            pedestrian_info[
                "in_lane_count"
            ],

        "near_lane_count":
            pedestrian_info[
                "near_lane_count"
            ],

        "sidewalk_count":
            pedestrian_info[
                "sidewalk_count"
            ],

        "traffic_signal":
            traffic_signal,

        "lead_vehicle":
            lead_vehicle,
            
        "lead_vehicle_area":
            int(lead_area),

        "vehicle_distance":
            vehicle_distance,

        "brightness":
            round(brightness,2),

        "contrast":
            round(contrast,2),

        "blur_score":
            round(blur_score,2),

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

        "left_lane_status":
            blind_spot[
                "left_status"
            ],

        "right_lane_status":
            blind_spot[
                "right_status"
            ],

        "collision_risk":
            blind_spot[
                "collision_risk"
            ],

        "lane_recommendation":
            blind_spot[
                "recommendation"
            ],

                "rl_action":
            rl_action,

        "decision":
            fusion_result[
                "final_decision"
            ],

        "recommended_speed":
            fusion_result[
                "final_speed"
            ],

        "lane_action":
            fusion_result[
                "lane_action"
            ],

        "fusion_reason":
            fusion_result[
                "reason"
            ],

        "reasons":
            reasons
    }