def explain_decision(
    traffic_state,
    pedestrian_count,
    distance_state,
    fusion_confidence
):

    reasons = []

    risk_score = 0

    # ==========================
    # TRAFFIC LIGHT
    # ==========================

    if traffic_state == "RED":

        risk_score += 100

        reasons.append(
            "Red traffic signal detected"
        )

    elif traffic_state == "YELLOW":

        risk_score += 40

        reasons.append(
            "Yellow signal detected"
        )

    # ==========================
    # PEDESTRIANS
    # ==========================

    if pedestrian_count > 0:

        risk_score += 80

        reasons.append(
            f"{pedestrian_count} pedestrian(s) detected"
        )

    # ==========================
    # VEHICLE DISTANCE
    # ==========================

    if distance_state == "VERY CLOSE":

        risk_score += 60

        reasons.append(
            "Lead vehicle very close"
        )

    elif distance_state == "MEDIUM":

        risk_score += 30

        reasons.append(
            "Lead vehicle ahead"
        )

    # ==========================
    # SENSOR CONFIDENCE
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

    elif risk_score >= 40:

        decision = "SLOW DOWN"

        speed = 15

    else:

        decision = "MOVE"

        speed = 40

    return (
        risk_score,
        decision,
        speed,
        reasons
    )


# ====================================
# TEST
# ====================================

risk_score, decision, speed, reasons = (

    explain_decision(

        traffic_state="RED",

        pedestrian_count=0,

        distance_state="MEDIUM",

        fusion_confidence=1.0
    )
)

print("\n")
print("=" * 50)
print(" DECISION EXPLANATION REPORT ")
print("=" * 50)

print(
    f"Risk Score : {risk_score}"
)

print(
    f"Decision   : {decision}"
)

print(
    f"Speed      : {speed} km/h"
)

print("\nReasons:")

for reason in reasons:

    print(
        f"- {reason}"
    )

print("=" * 50)