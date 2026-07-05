# ====================================
# RISK BASED DECISION ENGINE
# ====================================


def calculate_risk(
    traffic_state,
    pedestrian_count,
    pedestrian_status,
    distance_state,
    fusion_confidence,
    camera_reliability,
    collision_risk,
    left_lane_status,
    right_lane_status
):

    risk_score = 0

    reasons = []

    # ==========================
    # TRAFFIC SIGNAL
    # ==========================

    if traffic_state == "RED":

        risk_score += 100

        reasons.append(
            "Red signal"
        )

    elif traffic_state == "YELLOW":

        risk_score += 40

        reasons.append(
            "Yellow signal"
        )

    # ==========================
    # PEDESTRIAN
    # ==========================

    if pedestrian_status == "IN_LANE":

        risk_score += 100

        reasons.append(
            "Pedestrian in driving lane"
        )

    elif pedestrian_status == "NEAR_LANE":

        risk_score += 30

        reasons.append(
            "Pedestrian near lane"
        )

    elif pedestrian_status == "SIDEWALK":

        risk_score += 10

        reasons.append(
            "Pedestrian on sidewalk"
        )
    # ==========================
    # DISTANCE
    # ==========================

    # ==========================
# DISTANCE
# ==========================

    if distance_state == "VERY_CLOSE":

        risk_score += 80

        reasons.append(
            "Very close vehicle"
        )

    elif distance_state == "MEDIUM":

        risk_score += 30

        reasons.append(
            "Vehicle ahead"
        )

    elif distance_state == "FAR":

        risk_score += 5

        reasons.append(
            "Vehicle far ahead"
        )   
    # ==========================
    # BLIND SPOT INTELLIGENCE
    # ==========================

    if collision_risk >= 100:

        risk_score += 40

        reasons.append(
            "High blind spot collision risk"
        )

    if left_lane_status == "UNSAFE":

        risk_score += 20

        reasons.append(
            "Left lane unsafe"
    )

    if right_lane_status == "UNSAFE":

        risk_score += 20

        reasons.append(
            "Right lane unsafe"
    )
        
    # ==========================
    # RELIABILITY
    # ==========================

    if camera_reliability < 0.7:

        risk_score += 20

        reasons.append(
            "Low camera reliability"
        )

    # ==========================
    # FUSION
    # ==========================

    if fusion_confidence < 0.7:

        risk_score += 20

        reasons.append(
            "Low fusion confidence"
        )
# ==========================
# BLIND SPOT INTELLIGENCE
# ==========================
#
# V4 ARCHITECTURE:
# Blind Spot Intelligence is now
# handled by the Lane Change Engine.
#
# We keep the inputs for future
# Decision Fusion Engine V2.
#
# DO NOT add blind spot penalties
# to forward-driving risk.
# ==========================

    blind_spot_info = {

    "collision_risk":
        collision_risk,

    "left_lane_status":
        left_lane_status,

    "right_lane_status":
        right_lane_status

}

# Reserved for:
#
# decision_fusion_engine.py
#
# Future lane-change decisions:
#
# CHANGE_LEFT
# CHANGE_RIGHT
# KEEP_LANE
    # ==========================
    # DECISION
    # ==========================

    if risk_score >= 80:

        decision = "STOP"
        speed = 0
        level = "HIGH"

    elif risk_score >= 40:

        decision = "SLOW DOWN"
        speed = 15
        level = "MEDIUM"

    else:

        decision = "MOVE"
        speed = 40
        level = "LOW"
        
    lane_data = {

    "collision_risk":
        collision_risk,

    "left_lane_status":
        left_lane_status,

    "right_lane_status":
        right_lane_status

}
    return (
        risk_score,
        level,
        decision,
        speed,
        reasons
    )


# ====================================
# TEST CASE
# ====================================

if __name__ == "__main__":

    risk_score, level, decision, speed, reasons = (

        calculate_risk(
            traffic_state="RED",
            pedestrian_count=0,
            pedestrian_status="IN_LANE",
            distance_state="MEDIUM",
            fusion_confidence=1.0,
            camera_reliability=1.0,
            collision_risk=0,
            left_lane_status="SAFE",
            right_lane_status="SAFE"
        )
    )

    print("\n")
    print("=" * 50)
    print(" RISK-BASED DECISION REPORT ")
    print("=" * 50)

    print(f"Risk Score : {risk_score}")
    print(f"Risk Level : {level}")
    print(f"Decision   : {decision}")
    print(f"Speed      : {speed} km/h")

    print("\nReasons:")

    for r in reasons:
        print(f"- {r}")

    print("=" * 50)