from nuscenes.nuscenes import NuScenes
import numpy as np
import open3d as o3d

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
    "Point Cloud Loaded:",
    points.shape
)
xyz = points[:, :3]

print(
    "XYZ Shape:",
    xyz.shape
)
pcd = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(
    xyz
)
o3d.visualization.draw_geometries(
    [pcd],
    window_name="nuScenes LiDAR Point Cloud"
)