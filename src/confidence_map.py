import pandas as pd

# ============================================
# LOAD CAMERA RELIABILITY REPORT
# ============================================

df = pd.read_csv(
    "outputs/camera_reliability_report.csv"
)

print("Camera Reliability Report Loaded.\n")

print(df)

# ============================================
# CREATE CAMERA SCORE DICTIONARY
# ============================================

camera_scores = {}

for _, row in df.iterrows():

    camera_scores[
        row["camera"]
    ] = row["reliability"]

# ============================================
# ADD LIDAR RELIABILITY
# ============================================

lidar_reliability = 1.0

camera_scores["LIDAR_TOP"] = (
    lidar_reliability
)

# ============================================
# PRINT 360° CONFIDENCE MAP
# ============================================

print("\n")
print("=" * 40)
print(" 360 SENSOR CONFIDENCE MAP ")
print("=" * 40)

print(
    f"\n          FRONT"
)

print(
    f"          {camera_scores['CAM_FRONT']:.2f}"
)

print(
    f"\n{camera_scores['CAM_FRONT_LEFT']:.2f}"
    f"                "
    f"{camera_scores['CAM_FRONT_RIGHT']:.2f}"
)

print(
    "LEFT               RIGHT"
)

print(
    f"\n          LiDAR"
)

print(
    f"          {camera_scores['LIDAR_TOP']:.2f}"
)

print(
    f"\n{camera_scores['CAM_BACK_LEFT']:.2f}"
    f"                "
    f"{camera_scores['CAM_BACK_RIGHT']:.2f}"
)

print(
    "REAR LEFT      REAR RIGHT"
)

print(
    f"\n          BACK"
)

print(
    f"          {camera_scores['CAM_BACK']:.2f}"
)

# ============================================
# GLOBAL CONFIDENCE SCORE
# ============================================

global_confidence = sum(
    camera_scores.values()
) / len(camera_scores)

print("\n")
print("=" * 40)

print(
    f"Global Confidence : "
    f"{global_confidence:.3f}"
)

print("=" * 40)

# ============================================
# BEST SENSOR
# ============================================

best_sensor = max(
    camera_scores,
    key=camera_scores.get
)

print(
    f"\nBest Sensor : "
    f"{best_sensor}"
)

print(
    f"Score       : "
    f"{camera_scores[best_sensor]:.3f}"
)

# ============================================
# SAVE SUMMARY REPORT
# ============================================

with open(
    "outputs/confidence_map_report.txt",
    "w"
) as f:

    f.write("360 SENSOR CONFIDENCE MAP\n")
    f.write("=" * 40 + "\n")

    for sensor, score in camera_scores.items():

        f.write(
            f"{sensor}: {score:.3f}\n"
        )

    f.write(
        f"\nGlobal Confidence: "
        f"{global_confidence:.3f}\n"
    )

    f.write(
        f"Best Sensor: "
        f"{best_sensor}\n"
    )

print(
    "\nSaved:"
)

print(
    "outputs/confidence_map_report.txt"
)