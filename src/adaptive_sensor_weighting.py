from nuscenes.nuscenes import NuScenes
import cv2
import numpy as np

DATASET_PATH = "datasets/nuscenes"

# =====================================
# LOAD DATASET
# =====================================

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

# =====================================
# CAMERA ANALYSIS
# =====================================

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

brightness = np.mean(gray)
contrast = gray.std()

blur_score = cv2.Laplacian(
    gray,
    cv2.CV_64F
).var()

edge_count = np.sum(
    cv2.Canny(gray,100,200) > 0
)

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
    min(
        camera_reliability,
        1.0
    )
)

# =====================================
# LiDAR ANALYSIS
# =====================================

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
).reshape(-1,5)

point_count = len(points)

avg_intensity = np.mean(
    points[:,3]
)

lidar_reliability = 1.0

if point_count < 20000:
    lidar_reliability -= 0.20

if avg_intensity < 5:
    lidar_reliability -= 0.10

lidar_reliability = max(
    0.0,
    min(
        lidar_reliability,
        1.0
    )
)

# =====================================
# ADAPTIVE WEIGHTING
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

print("\n==============================")
print(" ADAPTIVE SENSOR WEIGHTS ")
print("==============================")

print(
    f"Camera Reliability : "
    f"{camera_reliability:.3f}"
)

print(
    f"LiDAR Reliability  : "
    f"{lidar_reliability:.3f}"
)

print()

print(
    f"Camera Weight      : "
    f"{camera_weight:.3f}"
)

print(
    f"LiDAR Weight       : "
    f"{lidar_weight:.3f}"
)

print("==============================")