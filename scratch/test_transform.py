import os
import json
import numpy as np
# Simple quaternion to rotation matrix helper
# Check if pyquaternion is installed or implement simple quaternion math
def quat_to_rot_mat(q):
    # q is [w, x, y, z]
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

meta_dir = r"datasets\nuscenes\v1.0-mini"
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

# Find LIDAR_TOP for first sample
sample_0 = samples[0]
sd_0 = next(sd for sd in sample_data if sd["sample_token"] == sample_0["token"] and "LIDAR_TOP" in sd["filename"] and sd["is_key_frame"])

ego_pose = ego_poses[sd_0["ego_pose_token"]]
cal_sensor = calibrated_sensors[sd_0["calibrated_sensor_token"]]

# Sensor extrinsics
r_sensor = quat_to_rot_mat(cal_sensor["rotation"])
t_sensor = np.array(cal_sensor["translation"])

# Ego pose
r_ego = quat_to_rot_mat(ego_pose["rotation"])
t_ego = np.array(ego_pose["translation"])

print("Ego translation:", t_ego)
print("Sensor translation:", t_sensor)

# Find annotations for sample 0
sample_anns = [a for a in annotations if a["sample_token"] == sample_0["token"]]
print(f"Annotations for sample 0: {len(sample_anns)}")

# Transform bounding box centers to sensor frame
# Global -> Ego: r_ego.T @ (p_global - t_ego)
# Ego -> Sensor: r_sensor.T @ (p_ego - t_sensor)
boxes_in_sensor = []
for a in sample_anns:
    p_global = np.array(a["translation"])
    p_ego = r_ego.T @ (p_global - t_ego)
    p_sensor = r_sensor.T @ (p_ego - t_sensor)
    
    cat_name = categories.get(instances.get(a["instance_token"], ""), "unknown")
    boxes_in_sensor.append({
        "category": cat_name,
        "center_sensor": p_sensor.tolist(),
        "size_wlh": a["size"],
        "num_pts": a["num_lidar_pts"]
    })

print(f"\nTransformed {len(boxes_in_sensor)} boxes to LIDAR sensor frame.")
print("Sample nearby boxes in front of the vehicle (Sensor frame):")
for b in sorted(boxes_in_sensor, key=lambda x: np.linalg.norm(x["center_sensor"][:2]))[:8]:
    c = b["center_sensor"]
    dist = np.hypot(c[0], c[1])
    print(f"  Class: {b['category']:<26} | X={c[0]:+6.2f}m, Y={c[1]:+6.2f}m, Z={c[2]:+6.2f}m | Dist={dist:5.1f}m | Pts={b['num_pts']}")
