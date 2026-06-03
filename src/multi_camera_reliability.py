from nuscenes.nuscenes import NuScenes
import cv2
import numpy as np
import pandas as pd
import os

# ============================================
# LOAD DATASET
# ============================================

DATASET_PATH = "datasets/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATASET_PATH,
    verbose=False
)

scene = nusc.scene[0]

sample = nusc.get(
    "sample",
    scene["first_sample_token"]
)

print("Dataset loaded.")

# ============================================
# CAMERA LIST
# ============================================

camera_names = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT"
]

# ============================================
# RELIABILITY FUNCTION
# ============================================

def calculate_reliability(image):

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

    edge_count = np.sum(
        cv2.Canny(
            gray,
            100,
            200
        ) > 0
    )

    reliability = 1.0

    if brightness < 60:
        reliability -= 0.20

    if contrast < 40:
        reliability -= 0.20

    if blur_score < 100:
        reliability -= 0.20

    if edge_count < 5000:
        reliability -= 0.20

    reliability = max(
        0.0,
        min(reliability, 1.0)
    )

    return (
        reliability,
        brightness,
        contrast,
        blur_score,
        edge_count
    )

# ============================================
# ANALYZE ALL CAMERAS
# ============================================

results = []

for camera_name in camera_names:

    camera_token = sample["data"][camera_name]

    camera_data = nusc.get(
        "sample_data",
        camera_token
    )

    image_path = os.path.join(
        DATASET_PATH,
        camera_data["filename"]
    )

    image = cv2.imread(
        image_path
    )

    (
        reliability,
        brightness,
        contrast,
        blur_score,
        edge_count
    ) = calculate_reliability(image)

    results.append({

        "camera": camera_name,

        "reliability": round(reliability, 3),

        "brightness": round(brightness, 2),

        "contrast": round(contrast, 2),

        "blur_score": round(blur_score, 2),

        "edge_count": int(edge_count)
    })

# ============================================
# CREATE DATAFRAME
# ============================================

df = pd.DataFrame(results)

df = df.sort_values(
    by="reliability",
    ascending=False
)

print("\n==============================")
print(" CAMERA RELIABILITY TABLE ")
print("==============================")

print(df.to_string(index=False))

# ============================================
# BEST & WORST CAMERA
# ============================================

best_camera = df.iloc[0]

worst_camera = df.iloc[-1]

print("\n==============================")

print(
    f"Best Camera : "
    f"{best_camera['camera']}"
)

print(
    f"Reliability : "
    f"{best_camera['reliability']:.3f}"
)

print()

print(
    f"Worst Camera : "
    f"{worst_camera['camera']}"
)

print(
    f"Reliability  : "
    f"{worst_camera['reliability']:.3f}"
)

print("==============================")

# ============================================
# SAVE REPORT
# ============================================

os.makedirs(
    "outputs",
    exist_ok=True
)

df.to_csv(
    "outputs/camera_reliability_report.csv",
    index=False
)

print("\nSaved:")
print("outputs/camera_reliability_report.csv")