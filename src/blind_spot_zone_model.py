from nuscenes.nuscenes import NuScenes

# ====================================
# LOAD DATASET
# ====================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot="datasets/nuscenes",
    verbose=False
)

print("Dataset loaded.")

# ====================================
# FIRST SCENE
# ====================================

scene = nusc.scene[0]

sample_token = scene["first_sample_token"]

sample = nusc.get(
    "sample",
    sample_token
)

# ====================================
# BLIND SPOT CAMERAS
# ====================================

blind_spot_cameras = [

    "CAM_FRONT_LEFT",
    "CAM_BACK_LEFT",

    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT"
]

print("\n========================")
print(" BLIND SPOT CAMERAS ")
print("========================")

for camera in blind_spot_cameras:

    token = sample["data"][camera]

    data = nusc.get(
        "sample_data",
        token
    )

    print(camera)

    print(
        "File:",
        data["filename"]
    )

    print()