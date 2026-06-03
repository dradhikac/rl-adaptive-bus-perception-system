import pandas as pd

# ==========================================
# LOAD EXISTING RESULTS
# ==========================================

camera_reliability = 0.8
lidar_reliability = 1.0

fusion_confidence = 0.911

object_count = 8
vehicle_count = 6
pedestrian_count = 0

# ==========================================
# RISK SCORING
# ==========================================

risk_score = 0

if object_count > 10:
    risk_score += 3

elif object_count > 5:
    risk_score += 2

if pedestrian_count > 0:
    risk_score += 5

if fusion_confidence < 0.5:
    risk_score += 3

# ==========================================
# DECISION ENGINE
# ==========================================

decision = "MOVE"

if pedestrian_count > 0:

    decision = "STOP"

elif risk_score >= 5:

    decision = "STOP"

elif risk_score >= 2:

    decision = "SLOW DOWN"

elif fusion_confidence < 0.5:

    decision = "CAUTION"

# ==========================================
# SPEED RECOMMENDATION
# ==========================================

speed = 40

if decision == "STOP":
    speed = 0

elif decision == "SLOW DOWN":
    speed = 15

elif decision == "CAUTION":
    speed = 10

else:
    speed = 40

# ==========================================
# OUTPUT
# ==========================================

print("\n")
print("=" * 50)
print(" AUTONOMOUS DECISION ENGINE ")
print("=" * 50)

print(
    f"Objects Detected     : {object_count}"
)

print(
    f"Vehicles Detected    : {vehicle_count}"
)

print(
    f"Pedestrians Detected : {pedestrian_count}"
)

print(
    f"Fusion Confidence    : {fusion_confidence:.3f}"
)

print(
    f"Risk Score           : {risk_score}"
)

print()

print(
    f"Vehicle Action       : {decision}"
)

print(
    f"Recommended Speed    : {speed} km/h"
)

print("=" * 50)

# ==========================================
# SAVE REPORT
# ==========================================

with open(
    "outputs/decision_report.txt",
    "w"
) as f:

    f.write(
        f"Decision: {decision}\n"
    )

    f.write(
        f"Speed: {speed}\n"
    )

    f.write(
        f"Risk Score: {risk_score}\n"
    )

    f.write(
        f"Fusion Confidence: "
        f"{fusion_confidence:.3f}\n"
    )

print(
    "\nSaved:"
)

print(
    "outputs/decision_report.txt"
)