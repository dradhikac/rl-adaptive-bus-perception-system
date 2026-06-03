from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

import numpy as np
import cv2

DATASET_PATH = "datasets/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATASET_PATH,
    verbose=False
)

print("Dataset loaded.")

scene = nusc.scene[0]

sample = nusc.get(
    "sample",
    scene["first_sample_token"]
)

camera_token = sample["data"]["CAM_FRONT"]

camera_data = nusc.get(
    "sample_data",
    camera_token
)

camera_path = (
    DATASET_PATH +
    "/" +
    camera_data["filename"]
)

image = cv2.imread(camera_path)

print(
    "Camera Loaded:",
    image.shape
)

lidar_token = sample["data"]["LIDAR_TOP"]

lidar_data = nusc.get(
    "sample_data",
    lidar_token
)

point_cloud = LidarPointCloud.from_file(
    DATASET_PATH +
    "/" +
    lidar_data["filename"]
)

print(
    "LiDAR Points:",
    point_cloud.points.shape
)

camera_calib = nusc.get(
    "calibrated_sensor",
    camera_data["calibrated_sensor_token"]
)

lidar_calib = nusc.get(
    "calibrated_sensor",
    lidar_data["calibrated_sensor_token"]
)

point_cloud.rotate(
    Quaternion(
        lidar_calib["rotation"]
    ).rotation_matrix
)

point_cloud.translate(
    np.array(
        lidar_calib["translation"]
    )
)

point_cloud.translate(
    -np.array(
        camera_calib["translation"]
    )
)

point_cloud.rotate(
    Quaternion(
        camera_calib["rotation"]
    ).rotation_matrix.T
)

points = point_cloud.points[:3, :]

mask = points[2, :] > 1

points = points[:, mask]

intrinsic = np.array(
    camera_calib["camera_intrinsic"]
)

projected = intrinsic @ points

projected[:2] /= projected[2]

for i in range(
    projected.shape[1]
):

    x = int(
        projected[0, i]
    )

    y = int(
        projected[1, i]
    )

    if (
        0 <= x < image.shape[1]
        and
        0 <= y < image.shape[0]
    ):

        cv2.circle(
            image,
            (x, y),
            1,
            (0, 255, 0),
            -1
        )
        
cv2.imshow(
    "Real Camera-LiDAR Fusion",
    image
)

cv2.waitKey(0)

cv2.destroyAllWindows()

