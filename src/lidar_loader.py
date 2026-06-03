from nuscenes.nuscenes import NuScenes
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "datasets/nuscenes"

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=DATASET_PATH,
    verbose=True
)

print("nuScenes loaded.")

scene = nusc.scene[0]

sample = nusc.get(
    "sample",
    scene["first_sample_token"]
)

lidar_token = sample["data"]["LIDAR_TOP"]

print("\nLiDAR Token:")
print(lidar_token)

lidar_data = nusc.get(
    "sample_data",
    lidar_token
)

lidar_path = (
    DATASET_PATH +
    "/" +
    lidar_data["filename"]
)

print("\nLiDAR File:")
print(lidar_path)

points = np.fromfile(
    lidar_path,
    dtype=np.float32
)

print(
    "\nRaw Values:",
    len(points)
)

points = points.reshape(
    -1,
    5
)

print(
    "\nPoint Cloud Shape:"
)

print(points.shape)

print("\nFirst 10 Points:\n")

print(
    points[:10]
)

print(
    "\nTotal Points:"
)

print(
    len(points)
)

plt.figure(
    figsize=(8,8)
)

plt.scatter(
    points[:,0],
    points[:,1],
    s=0.5
)

plt.title(
    "LiDAR Top View"
)

plt.xlabel("X")

plt.ylabel("Y")

plt.axis("equal")

plt.show()


