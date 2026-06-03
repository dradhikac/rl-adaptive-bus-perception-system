from nuscenes.nuscenes import NuScenes
import numpy as np

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
camera_calib = nusc.get(
    "calibrated_sensor",
    camera_data["calibrated_sensor_token"]
)
camera_intrinsic = np.array(
    camera_calib["camera_intrinsic"]
)

print("\nCamera Intrinsic Matrix:\n")

print(camera_intrinsic)
print("\nImage Width:")

print(camera_data["width"])

print("\nImage Height:")

print(camera_data["height"])

point_3d = np.array([
    [10],
    [2],
    [1]
])

projected = camera_intrinsic @ point_3d

print("\nProjected Coordinates:")

print(projected)

