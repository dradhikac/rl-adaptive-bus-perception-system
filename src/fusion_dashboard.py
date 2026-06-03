import pandas as pd
import os

# =====================================
# LOAD CAMERA REPORT
# =====================================

camera_df = pd.read_csv(
    "outputs/camera_reliability_report.csv"
)

# =====================================
# CAMERA RELIABILITY
# =====================================

camera_reliability = (
    camera_df["reliability"]
    .mean()
)

# =====================================
# LIDAR RELIABILITY
# =====================================

lidar_reliability = 1.0

# =====================================
# ADAPTIVE WEIGHTS
# =====================================

total = (
    camera_reliability +
    lidar_reliability
)

camera_weight = (
    camera_reliability / total
)

lidar_weight = (
    lidar_reliability / total
)

# =====================================
# GLOBAL CONFIDENCE
# =====================================

global_confidence = (
    (
        camera_reliability *
        camera_weight
    )
    +
    (
        lidar_reliability *
        lidar_weight
    )
)

# =====================================
# ENVIRONMENT DIFFICULTY
# =====================================

environment_difficulty = (
    1.0 - global_confidence
)

# =====================================
# SENSOR STATUS
# =====================================

if global_confidence > 0.90:
    status = "EXCELLENT"

elif global_confidence > 0.75:
    status = "GOOD"

elif global_confidence > 0.50:
    status = "MODERATE"

else:
    status = "POOR"

# =====================================
# DASHBOARD
# =====================================

print("\n")
print("=" * 50)
print(" AUTONOMOUS BUS PERCEPTION DASHBOARD ")
print("=" * 50)

print(
    f"\nAverage Camera Reliability : "
    f"{camera_reliability:.3f}"
)

print(
    f"LiDAR Reliability          : "
    f"{lidar_reliability:.3f}"
)

print()

print(
    f"Camera Weight              : "
    f"{camera_weight:.3f}"
)

print(
    f"LiDAR Weight               : "
    f"{lidar_weight:.3f}"
)

print()

print(
    f"Global Confidence          : "
    f"{global_confidence:.3f}"
)

print(
    f"Environment Difficulty     : "
    f"{environment_difficulty:.3f}"
)

print()

print(
    f"System Status              : "
    f"{status}"
)

print("=" * 50)

# =====================================
# SAVE REPORT
# =====================================

os.makedirs(
    "outputs",
    exist_ok=True
)

with open(
    "outputs/fusion_dashboard.txt",
    "w"
) as file:

    file.write(
        f"Camera Reliability: "
        f"{camera_reliability:.3f}\n"
    )

    file.write(
        f"LiDAR Reliability: "
        f"{lidar_reliability:.3f}\n"
    )

    file.write(
        f"Camera Weight: "
        f"{camera_weight:.3f}\n"
    )

    file.write(
        f"LiDAR Weight: "
        f"{lidar_weight:.3f}\n"
    )

    file.write(
        f"Global Confidence: "
        f"{global_confidence:.3f}\n"
    )

    file.write(
        f"Environment Difficulty: "
        f"{environment_difficulty:.3f}\n"
    )

    file.write(
        f"Status: "
        f"{status}\n"
    )

print("\nSaved:")
print("outputs/fusion_dashboard.txt")
