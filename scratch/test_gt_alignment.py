import os
import json
import numpy as np

def quat_to_rot_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

dataset_root = r"datasets\nuscenes"
meta_dir = os.path.join(dataset_root, "v1.0-mini")

with open(os.path.join(meta_dir, "sample.json"), "r") as f:
    samples = json.load(f)
with open(os.path.join(meta_dir, "sample_data.json"), "r") as f:
    sample_data = json.load(f)
with open(os.path.join(meta_dir, "ego_pose.json"), "r") as f:
    ego_poses = {ep["token"]: ep for ep in json.load(f)}
with open(os.path.join(meta_dir, "calibrated_sensor.json"), "r") as f:
    calibrated_sensors = {cs["token"]: cs for cs in json.load(f)}
with open(os.path.join(meta_dir, "sample_annotation.json"), "r") as f:
    annotations = json.load(f)
with open(os.path.join(meta_dir, "instance.json"), "r") as f:
    instances = {inst["token"]: inst["category_token"] for inst in json.load(f)}
with open(os.path.join(meta_dir, "category.json"), "r") as f:
    categories = {cat["token"]: cat["name"] for cat in json.load(f)}

# Take sample 0
sample_0 = samples[0]
sd_0 = next(sd for sd in sample_data if sd["sample_token"] == sample_0["token"] and "LIDAR_TOP" in sd["filename"] and sd["is_key_frame"])

pts_path = os.path.join(dataset_root, sd_0["filename"].replace("/", os.sep))
pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)

ego_pose = ego_poses[sd_0["ego_pose_token"]]
cal_sensor = calibrated_sensors[sd_0["calibrated_sensor_token"]]

r_sensor = quat_to_rot_mat(cal_sensor["rotation"])
t_sensor = np.array(cal_sensor["translation"])
r_ego = quat_to_rot_mat(ego_pose["rotation"])
t_ego = np.array(ego_pose["translation"])

# Filter points with r < 1m
r_pts = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
pts_env = pts[r_pts >= 1.0]

sample_anns = [a for a in annotations if a["sample_token"] == sample_0["token"]]
print(f"Total annotations for sample 0: {len(sample_anns)}")

# For boxes with > 10 points, check point containment
verified_boxes = 0
for a in sample_anns:
    if a["num_lidar_pts"] < 10:
        continue
    
    p_global = np.array(a["translation"])
    p_ego = r_ego.T @ (p_global - t_ego)
    p_sensor = r_sensor.T @ (p_ego - t_sensor)
    
    r_box_global = quat_to_rot_mat(a["rotation"])
    r_box_sensor = r_sensor.T @ r_ego.T @ r_box_global
    
    # Check dimensions: in nuScenes size is [w, l, h]
    w, l, h = a["size"]
    
    # Transform all LiDAR points into box's local frame
    # p_local = R_box_sensor.T @ (pts_sensor - p_sensor)
    pts_sensor = pts_env[:, :3]
    diff = pts_sensor - p_sensor
    pts_local = diff @ r_box_sensor # (N, 3) since (R.T @ diff.T).T = diff @ R
    
    # Check containment:
    # x in [-w/2, w/2], y in [-l/2, l/2], z in [-h/2, h/2]
    in_x = np.abs(pts_local[:, 0]) <= w / 2.0
    in_y = np.abs(pts_local[:, 1]) <= l / 2.0
    in_z = np.abs(pts_local[:, 2]) <= h / 2.0
    
    in_box = in_x & in_y & in_z
    measured_pts = in_box.sum()
    gt_pts = a["num_lidar_pts"]
    
    cat = categories.get(instances.get(a["instance_token"], ""), "unknown")
    print(f"Box: {cat:<25} | Size [w,l,h]=[{w:.2f}, {l:.2f}, {h:.2f}] | Center Sensor=[{p_sensor[0]:+6.2f}, {p_sensor[1]:+6.2f}, {p_sensor[2]:+6.2f}] | GT Pts: {gt_pts:>3} | Measured Inside: {measured_pts:>3}")
    verified_boxes += 1
    if verified_boxes >= 10:
        break
