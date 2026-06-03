from nuscenes.nuscenes import NuScenes
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------------
# Load Dataset
# -----------------------------------

DATASET_PATH = "datasets/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATASET_PATH,
    verbose=False
)

print("Dataset loaded.")

# -----------------------------------
# First Scene
# -----------------------------------

scene = nusc.scene[0]

sample = nusc.get(
    "sample",
    scene["first_sample_token"]
)

# -----------------------------------
# Camera Image
# -----------------------------------

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

print("Image Shape:", image.shape)

# -----------------------------------
# LiDAR Point Cloud
# -----------------------------------

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

print("Point Cloud Shape:", points.shape)

# -----------------------------------
# XYZ Coordinates
# -----------------------------------

xyz = points[:, :3]

# -----------------------------------
# Front Facing Points
# -----------------------------------

front_points = xyz[
    xyz[:, 0] > 0
]

print("Front Points:", len(front_points))

# -----------------------------------
# Overlay
# -----------------------------------

height, width = image.shape[:2]

for point in front_points[:5000]:

    x = point[0]
    y = point[1]

    px = int(
        width / 2 +
        y * 25
    )

    py = int(
        height -
        x * 8
    )

    if (
        0 <= px < width and
        0 <= py < height
    ):
        cv2.circle(
            image,
            (px, py),
            1,
            (0, 255, 0),
            -1
        )

# -----------------------------------
# Display Using Matplotlib
# -----------------------------------

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(14, 8))
plt.imshow(image_rgb)
plt.title("LiDAR + Camera Overlay")
plt.axis("off")
plt.show()