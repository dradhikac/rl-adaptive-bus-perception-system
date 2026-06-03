from nuscenes.nuscenes import NuScenes

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
lidar_token = sample["data"]["LIDAR_TOP"]

lidar_data = nusc.get(
    "sample_data",
    lidar_token
)
camera_calib = nusc.get(
    "calibrated_sensor",
    camera_data["calibrated_sensor_token"]
)

lidar_calib = nusc.get(
    "calibrated_sensor",
    lidar_data["calibrated_sensor_token"]
)
print("\nCAMERA CALIBRATION")

print(
    "\nTranslation:"
)

print(
    camera_calib["translation"]
)

print(
    "\nRotation:"
)

print(
    camera_calib["rotation"]
)
print("\nLIDAR CALIBRATION")

print(
    "\nTranslation:"
)

print(
    lidar_calib["translation"]
)

print(
    "\nRotation:"
)

print(
    lidar_calib["rotation"]
)
print("\n===================")
print("POSITION COMPARISON")
print("===================")

print(
    "\nCamera Position:"
)

print(
    camera_calib["translation"]
)

print(
    "\nLiDAR Position:"
)

print(
    lidar_calib["translation"]
)
