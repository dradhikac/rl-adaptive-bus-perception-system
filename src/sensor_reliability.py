from nuscenes.nuscenes import NuScenes
import cv2
import numpy as np

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

print("Sample loaded.")

# ============================================
# LOAD FRONT CAMERA
# ============================================

camera_token = sample["data"]["CAM_FRONT"]

camera_data = nusc.get(
    "sample_data",
    camera_token
)

image_path = (
    DATASET_PATH +
    "/" +
    camera_data["filename"]
)

image = cv2.imread(image_path)

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

print(
    "\nImage Shape:",
    image.shape
)

# ============================================
# CAMERA FEATURES
# ============================================

brightness = np.mean(gray)

contrast = gray.std()

blur_score = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

edges = cv2.Canny(
    gray,
    100,
    200
)

edge_count = np.sum(
    edges > 0
)

print("\n===== CAMERA FEATURES =====")

print(
    "Brightness:",
    round(brightness, 2)
)

print(
    "Contrast:",
    round(contrast, 2)
)

print(
    "Blur Score:",
    round(blur_score, 2)
)

print(
    "Edge Count:",
    edge_count
)

# ============================================
# CAMERA RELIABILITY
# ============================================

camera_reliability = 1.0

if brightness < 60:
    camera_reliability -= 0.20

if contrast < 40:
    camera_reliability -= 0.20

if blur_score < 100:
    camera_reliability -= 0.20

if edge_count < 5000:
    camera_reliability -= 0.20

camera_reliability = max(
    0.0,
    min(camera_reliability, 1.0)
)

print(
    "\nCamera Reliability:",
    round(camera_reliability, 3)
)

# ============================================
# LOAD LIDAR
# ============================================

lidar_token = sample["data"]["LIDAR_TOP"]

lidar_data = nusc.get(
    "sample_data",
    lidar_token
)

lidar_path = (
    DATASET_PATH +
    "/" +
    lidar_data["filename"]
)

points = np.fromfile(
    lidar_path,
    dtype=np.float32
).reshape(-1, 5)

print(
    "\nLiDAR Shape:",
    points.shape
)

# ============================================
# LIDAR FEATURES
# ============================================

point_count = len(points)

avg_intensity = np.mean(
    points[:, 3]
)

print(
    "Point Count:",
    point_count
)

print(
    "Average Intensity:",
    round(avg_intensity, 2)
)

# ============================================
# LIDAR RELIABILITY
# ============================================

lidar_reliability = 1.0

if point_count < 20000:
    lidar_reliability -= 0.20

if avg_intensity < 5:
    lidar_reliability -= 0.10

lidar_reliability = max(
    0.0,
    min(lidar_reliability, 1.0)
)

print(
    "\nLiDAR Reliability:",
    round(lidar_reliability, 3)
)

# ============================================
# ENVIRONMENT DIFFICULTY
# ============================================

environment_difficulty = 1.0 - (
    (
        camera_reliability +
        lidar_reliability
    ) / 2
)

print(
    "\nEnvironment Difficulty:",
    round(environment_difficulty, 3)
)

# ============================================
# FUSION CONFIDENCE
# ============================================

fusion_confidence = (
    camera_reliability * 0.5 +
    lidar_reliability * 0.5
)

print(
    "\nFusion Confidence:",
    round(fusion_confidence, 3)
)

# ============================================
# FINAL REPORT
# ============================================

print("\n==============================")
print(" SENSOR RELIABILITY REPORT ")
print("==============================")

print(
    f"Camera Reliability      : {camera_reliability:.3f}"
)

print(
    f"LiDAR Reliability       : {lidar_reliability:.3f}"
)

print(
    f"Environment Difficulty  : {environment_difficulty:.3f}"
)

print(
    f"Fusion Confidence       : {fusion_confidence:.3f}"
)

print("==============================")