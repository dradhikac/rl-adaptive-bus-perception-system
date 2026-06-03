from nuscenes.nuscenes import NuScenes

# Dataset path
DATASET_PATH = "datasets/nuscenes"

print("Loading nuScenes Mini...")

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATASET_PATH,
    verbose=True
)

print("\nDataset loaded successfully.")

print("\nAvailable Scenes:\n")

for scene in nusc.scene:

    print(
        scene["name"],
        "|",
        scene["description"]
    )