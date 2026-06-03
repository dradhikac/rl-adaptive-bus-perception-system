# ====================================
# RISK BASED DECISION ENGINE
# ====================================

def calculate_risk(
    traffic_state,
    pedestrian_count,
    distance_state,
    fusion_confidence,
    camera_reliability
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

    if pedestrian_count > 0:

        risk_score += 25

        reasons.append(
            f"{pedestrian_count} pedestrian(s)"
        )

    # ==========================
    # DISTANCE
    # ==========================

    if distance_state == "VERY CLOSE":

        risk_score += 50

        reasons.append(
            "Very close vehicle"
        )

    elif distance_state == "MEDIUM":

        risk_score += 15

        reasons.append(
            "Vehicle ahead"
        )

    elif distance_state == "FAR":

        risk_score += 5

        reasons.append(
            "Vehicle far ahead"
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
            distance_state="MEDIUM",
            fusion_confidence=1.0,
            camera_reliability=1.0
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