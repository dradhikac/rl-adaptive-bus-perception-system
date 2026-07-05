import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

vehicle_classes = [

    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle"
]

# ====================================
# SINGLE CAMERA ANALYSIS
# ====================================

def analyze_camera(image):

    results = model(
        image,
        verbose=False
    )

    vehicles = 0
    people = 0

    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        name = model.names[cls_id]

        if name in vehicle_classes:
            vehicles += 1

        if name == "person":
            people += 1

    return vehicles, people

# ====================================
# SIDE ANALYSIS
# ====================================

def analyze_side(

    front_image,

    rear_image

):

    v1, p1 = analyze_camera(
        front_image
    )

    v2, p2 = analyze_camera(
        rear_image
    )

    vehicles = v1 + v2

    people = p1 + p2

    risk_score = (

    vehicles * 40 +

    people * 10

)

    return {

        "vehicles": vehicles,

        "people": people,

        "risk_score": risk_score
    }

# ====================================
# STATUS
# ====================================

def lane_status(score):

    if score >= 120:

        return "UNSAFE"

    elif score >= 40:

        return "CAUTION"

    else:

        return "SAFE"

# ====================================
# MAIN API
# ====================================

def analyze_blind_spots(

    front_left,

    rear_left,

    front_right,

    rear_right

):

    left = analyze_side(

        front_left,

        rear_left
    )

    right = analyze_side(

        front_right,

        rear_right
    )

    left_status = lane_status(left["risk_score"])
    right_status = lane_status(right["risk_score"])

    if left_status == "SAFE" and right_status == "SAFE":
        recommendation = "KEEP CURRENT LANE"
    elif left_status == "SAFE":
        recommendation = "CHANGE LEFT"
    elif right_status == "SAFE":
        recommendation = "CHANGE RIGHT"
    else:
        recommendation = "KEEP CURRENT LANE"

    collision_risk = left["risk_score"] + right["risk_score"]

    if left_status != "SAFE" or right_status != "SAFE":
        print("\n===== BLIND SPOT DEBUG =====")
        print("Left Status:", left_status)
        print("Right Status:", right_status)
        print("Left Risk:", left["risk_score"])
        print("Right Risk:", right["risk_score"])
        print("Collision Risk:", collision_risk)
        print("Recommendation:", recommendation)
        print("============================\n")

    return {
        "left_status": left_status,
        "right_status": right_status,
        "left_risk": left["risk_score"],
        "right_risk": right["risk_score"],
        "collision_risk": collision_risk,
        "recommendation": recommendation
    }
    
    # ====================================
# TEST
# ====================================

if __name__ == "__main__":

    import cv2

    front_left = cv2.imread(
        "datasets/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_LEFT__1532402927604844.jpg"
    )

    rear_left = cv2.imread(
        "datasets/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402927647423.jpg"
    )

    front_right = cv2.imread(
        "datasets/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_FRONT_RIGHT__1532402927620339.jpg"
    )

    rear_right = cv2.imread(
        "datasets/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-24-11-22-45+0800__CAM_BACK_RIGHT__1532402927627893.jpg"
    )

    result = analyze_blind_spots(

        front_left,
        rear_left,

        front_right,
        rear_right
    )

    print("\n====================")
    print("BLIND SPOT TEST")
    print("====================")

    for key, value in result.items():

        print(
            key,
            ":",
            value
        )