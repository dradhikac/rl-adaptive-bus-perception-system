import os
import json
import numpy as np
from collections import defaultdict

dataset_root = r"datasets\nuscenes"
meta_dir = os.path.join(dataset_root, "v1.0-mini")

def load_json(name):
    with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as f:
        return json.load(f)

scenes = load_json("scene.json")
samples = load_json("sample.json")
sample_data = load_json("sample_data.json")
calibrated_sensors = {cs["token"]: cs for cs in load_json("calibrated_sensor.json")}
ego_poses = {ep["token"]: ep for ep in load_json("ego_pose.json")}
annotations = load_json("sample_annotation.json")
instances = {inst["token"]: inst["category_token"] for inst in load_json("instance.json")}
categories = {cat["token"]: cat["name"] for cat in load_json("category.json")}

def quat_to_rot_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

# Identify Train vs Val scenes
val_scene_names = {"scene-0553", "scene-0921"}
scene_token_to_name = {sc["token"]: sc["name"] for sc in scenes}
sample_to_scene_name = {sm["token"]: scene_token_to_name[sm["scene_token"]] for sm in samples}

print("============================================================")
print("STAGE 4: CLASS DISTRIBUTION & DETECTION TASK FEASIBILITY")
print("============================================================")
print(f"Total keyframes: {len(samples)} (Train: {len([s for s in samples if sample_to_scene_name[s['token']] not in val_scene_names])}, Val: {len([s for s in samples if sample_to_scene_name[s['token']] in val_scene_names])})")
print(f"Total annotations: {len(annotations)}")

# Group annotations by category
cat_stats = defaultdict(lambda: {
    "total": 0,
    "train_total": 0,
    "val_total": 0,
    "pts_gt_0": 0,
    "pts_gte_3": 0,
    "pts_gte_5": 0,
    "in_roi_45m": 0,
    "in_roi_and_pts_gte_3": 0,
    "frame_tokens": set(),
    "sizes": []
})

# Precompute sensor coordinates for all annotations to check distance
sample_data_lidar = {
    sd["sample_token"]: sd for sd in sample_data
    if "LIDAR_TOP" in sd.get("filename", "") and sd.get("is_key_frame")
}

for a in annotations:
    s_tok = a["sample_token"]
    sc_name = sample_to_scene_name[s_tok]
    is_val = sc_name in val_scene_names
    cat_name = categories.get(instances.get(a["instance_token"], ""), "unknown")
    
    st = cat_stats[cat_name]
    st["total"] += 1
    if is_val:
        st["val_total"] += 1
    else:
        st["train_total"] += 1
        
    num_pts = a.get("num_lidar_pts", 0)
    if num_pts > 0:
        st["pts_gt_0"] += 1
    if num_pts >= 3:
        st["pts_gte_3"] += 1
    if num_pts >= 5:
        st["pts_gte_5"] += 1
        
    st["frame_tokens"].add(s_tok)
    st["sizes"].append(a["size"])
    
    # Calculate distance to sensor
    sd = sample_data_lidar.get(s_tok)
    if sd:
        ego_pose = ego_poses[sd["ego_pose_token"]]
        cal_sensor = calibrated_sensors[sd["calibrated_sensor_token"]]
        r_sensor = quat_to_rot_mat(cal_sensor["rotation"])
        t_sensor = np.array(cal_sensor["translation"])
        r_ego = quat_to_rot_mat(ego_pose["rotation"])
        t_ego = np.array(ego_pose["translation"])
        
        p_global = np.array(a["translation"])
        p_ego = r_ego.T @ (p_global - t_ego)
        p_sensor = r_sensor.T @ (p_ego - t_sensor)
        
        dist = np.hypot(p_sensor[0], p_sensor[1])
        if dist <= 45.0:
            st["in_roi_45m"] += 1
            if num_pts >= 3:
                st["in_roi_and_pts_gte_3"] += 1

print("\n--- 1. RAW CATEGORY DISTRIBUTION (ALL 23 CLASSES) ---")
print(f"{'Category Name':<36} | {'Total':>5} | {'Train':>5} | {'Val':>4} | {'Frame %':>7} | {'In 45m ROI':>10} | {'Pts>=3':>7} | {'Trainable':>9}")
print("-" * 98)

sorted_cats = sorted(cat_stats.items(), key=lambda x: x[1]["total"], reverse=True)
for cat, st in sorted_cats:
    frame_pct = len(st["frame_tokens"]) / len(samples) * 100
    trainable = st["in_roi_and_pts_gte_3"]
    print(f"{cat:<36} | {st['total']:>5} | {st['train_total']:>5} | {st['val_total']:>4} | {frame_pct:>6.1f}% | {st['in_roi_45m']:>10} | {st['pts_gte_3']:>7} | {trainable:>9}")

# Now let's analyze CANONICAL GROUPINGS
# In autonomous vehicle perception (including nuScenes 10-class benchmark and bus perception):
# We can evaluate candidate groupings:
group_map = {
    # 1. Car
    "vehicle.car": "CAR",
    
    # 2. Pedestrian
    "human.pedestrian.adult": "PEDESTRIAN",
    "human.pedestrian.child": "PEDESTRIAN",
    "human.pedestrian.construction_worker": "PEDESTRIAN",
    "human.pedestrian.police_officer": "PEDESTRIAN",
    
    # 3. Large Vehicle (Truck / Bus / Construction)
    "vehicle.truck": "TRUCK",
    "vehicle.bus.rigid": "BUS",
    "vehicle.bus.bendy": "BUS",
    "vehicle.construction": "CONSTRUCTION_VEHICLE",
    "vehicle.trailer": "TRAILER",
    
    # 4. Cyclist / Two-Wheeler
    "vehicle.motorcycle": "MOTORCYCLE",
    "vehicle.bicycle": "BICYCLE",
    
    # 5. Static / Road Infrastructure
    "movable_object.barrier": "BARRIER",
    "movable_object.trafficcone": "TRAFFIC_CONE",
    "movable_object.pushable_pullable": "PUSHABLE_PULLABLE",
    "movable_object.debris": "DEBRIS",
    
    # Rare / Ignored
    "static_object.bicycle_rack": "BICYCLE_RACK",
    "vehicle.emergency.ambulance": "EMERGENCY_VEHICLE",
    "vehicle.emergency.police": "EMERGENCY_VEHICLE",
    "human.pedestrian.wheelchair": "WHEELCHAIR",
    "human.pedestrian.personal_mobility": "PERSONAL_MOBILITY",
    "human.pedestrian.stroller": "STROLLER"
}

grouped_stats = defaultdict(lambda: {
    "total": 0,
    "train_total": 0,
    "val_total": 0,
    "pts_gte_3": 0,
    "in_roi_45m": 0,
    "trainable": 0,
    "frame_tokens": set(),
    "sizes": []
})

for cat, st in cat_stats.items():
    grp = group_map.get(cat, "OTHER")
    gst = grouped_stats[grp]
    gst["total"] += st["total"]
    gst["train_total"] += st["train_total"]
    gst["val_total"] += st["val_total"]
    gst["pts_gte_3"] += st["pts_gte_3"]
    gst["in_roi_45m"] += st["in_roi_45m"]
    gst["trainable"] += st["in_roi_and_pts_gte_3"]
    gst["frame_tokens"].update(st["frame_tokens"])
    gst["sizes"].extend(st["sizes"])

print("\n--- 2. CANONICAL GROUPED DISTRIBUTION ---")
print(f"{'Grouped Class':<22} | {'Total':>5} | {'Train':>5} | {'Val':>4} | {'Frame %':>7} | {'Trainable (ROI & Pts>=3)':>24} | {'Mean Size [W x L x H] (m)'}")
print("-" * 98)

for grp, gst in sorted(grouped_stats.items(), key=lambda x: x[1]["total"], reverse=True):
    frame_pct = len(gst["frame_tokens"]) / len(samples) * 100
    sizes = np.array(gst["sizes"])
    mean_w = sizes[:, 0].mean() if len(sizes) > 0 else 0
    mean_l = sizes[:, 1].mean() if len(sizes) > 0 else 0
    mean_h = sizes[:, 2].mean() if len(sizes) > 0 else 0
    size_str = f"[{mean_w:.2f} x {mean_l:.2f} x {mean_h:.2f}]"
    print(f"{grp:<22} | {gst['total']:>5} | {gst['train_total']:>5} | {gst['val_total']:>4} | {frame_pct:>6.1f}% | {gst['trainable']:>24} | {size_str}")

# Save detailed summary json
output_summary = {
    "total_samples": len(samples),
    "train_samples": len([s for s in samples if sample_to_scene_name[s['token']] not in val_scene_names]),
    "val_samples": len([s for s in samples if sample_to_scene_name[s['token']] in val_scene_names]),
    "raw_categories": {
        cat: {
            "total": st["total"],
            "train": st["train_total"],
            "val": st["val_total"],
            "frame_presence_pct": len(st["frame_tokens"]) / len(samples) * 100,
            "trainable_roi_pts3": st["in_roi_and_pts_gte_3"]
        } for cat, st in sorted_cats
    },
    "grouped_classes": {
        grp: {
            "total": gst["total"],
            "train": gst["train_total"],
            "val": gst["val_total"],
            "frame_presence_pct": len(gst["frame_tokens"]) / len(samples) * 100,
            "trainable_roi_pts3": gst["trainable"]
        } for grp, gst in sorted(grouped_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    }
}

os.makedirs(r"visualizations\stage4_task", exist_ok=True)
with open(r"visualizations\stage4_task\class_distribution_summary.json", "w") as f:
    json.dump(output_summary, f, indent=2)

print("\nSTAGE 4 CLASS DISTRIBUTION ANALYSIS COMPLETE.")
