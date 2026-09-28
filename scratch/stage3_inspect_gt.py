import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Setup paths
dataset_root = r"datasets\nuscenes"
meta_dir = os.path.join(dataset_root, "v1.0-mini")
output_dir = r"visualizations\stage3_labels"
artifact_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\1c0087c6-dac4-4253-a522-0bdbc5b00eb8"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

print("============================================================")
print("STAGE 3: GROUND-TRUTH 3D BOUNDING BOX INSPECTION & ALIGNMENT")
print("============================================================")

def quat_to_rot_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_box_8_corners(center, size, rot_mat):
    # center: [x,y,z], size: [w, l, h], rot_mat: 3x3
    w, l, h = size
    # 8 corners in box local coordinates:
    # 0: -w/2, -l/2, -h/2 (bottom rear left)
    # 1:  w/2, -l/2, -h/2 (bottom rear right)
    # 2:  w/2,  l/2, -h/2 (bottom front right)
    # 3: -w/2,  l/2, -h/2 (bottom front left)
    # 4: -w/2, -l/2,  h/2 (top rear left)
    # 5:  w/2, -l/2,  h/2 (top rear right)
    # 6:  w/2,  l/2,  h/2 (top front right)
    # 7: -w/2,  l/2,  h/2 (top front left)
    x_corners = [-w/2,  w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2]
    y_corners = [-l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2,  l/2]
    z_corners = [-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2]
    corners_local = np.vstack([x_corners, y_corners, z_corners]) # (3, 8)
    corners_sensor = rot_mat @ corners_local + np.array(center)[:, None]
    return corners_sensor # (3, 8)

# Load metadata
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
scene_map = {sc["token"]: sc["name"] for sc in scenes}

# 1. Coordinate System Verification & Documentation
print("\n--- 1. COORDINATE SYSTEM VERIFICATION ---")
print("Raw annotation coordinate system: GLOBAL WORLD FRAME (UTM coordinates in meters)")
print("Sensor coordinate system: LIDAR_TOP SENSOR FRAME (Right-handed, origin at LiDAR optical center)")
print("  +X = Lateral Right")
print("  +Y = Longitudinal Forward (heading of ego bus)")
print("  +Z = Vertical Upward")

sample_to_anns = {}
for a in annotations:
    s_tok = a["sample_token"]
    if s_tok not in sample_to_anns:
        sample_to_anns[s_tok] = []
    sample_to_anns[s_tok].append(a)

# Select 4 representative samples across different scenes
target_scenes = ["scene-0061", "scene-0103", "scene-0553", "scene-0655"]
selected_samples = []

for sc_name in target_scenes:
    sc = next((s for s in scenes if s["name"] == sc_name), None)
    if not sc:
        continue
    # Pick a sample around middle of scene
    curr = sc["first_sample_token"]
    scene_samples = []
    while curr:
        sm = next(s for s in samples if s["token"] == curr)
        scene_samples.append(sm)
        curr = sm["next"]
    chosen_sample = scene_samples[len(scene_samples) // 2]
    selected_samples.append((sc_name, chosen_sample))

print(f"\nEvaluating Ground-Truth Annotations on {len(selected_samples)} test scenes:")

# Color mapping for classes
def get_class_color(cat_name):
    if "vehicle.car" in cat_name:
        return "#f85149" # Bright Red
    elif "vehicle.truck" in cat_name or "vehicle.bus" in cat_name:
        return "#ff7b72" # Coral Red / Orange
    elif "human" in cat_name:
        return "#3fb950" # Green
    elif "barrier" in cat_name:
        return "#d29922" # Yellow
    elif "trafficcone" in cat_name:
        return "#f0883e" # Amber
    elif "motorcycle" in cat_name or "bicycle" in cat_name:
        return "#a371f7" # Purple
    return "#8b949e" # Grey

# Detailed evaluation loop
sample_inspection_results = []

for idx, (sc_name, sample_obj) in enumerate(selected_samples):
    s_token = sample_obj["token"]
    sd = next(d for d in sample_data if d["sample_token"] == s_token and "LIDAR_TOP" in d["filename"] and d["is_key_frame"])
    
    # Load LiDAR points
    pts_path = os.path.join(dataset_root, sd["filename"].replace("/", os.sep))
    pts_raw = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)
    r_pts = np.sqrt(pts_raw[:, 0]**2 + pts_raw[:, 1]**2)
    pts_env = pts_raw[r_pts >= 1.0] # remove ego self-reflections
    
    # Extrinsics & Pose
    ego_pose = ego_poses[sd["ego_pose_token"]]
    cal_sensor = calibrated_sensors[sd["calibrated_sensor_token"]]
    
    r_sensor = quat_to_rot_mat(cal_sensor["rotation"])
    t_sensor = np.array(cal_sensor["translation"])
    r_ego = quat_to_rot_mat(ego_pose["rotation"])
    t_ego = np.array(ego_pose["translation"])
    
    # Transform all annotations
    raw_anns = sample_to_anns.get(s_token, [])
    boxes = []
    
    for a in raw_anns:
        cat_name = categories.get(instances.get(a["instance_token"], ""), "unknown")
        p_global = np.array(a["translation"])
        # Global to sensor transform:
        p_ego = r_ego.T @ (p_global - t_ego)
        p_sensor = r_sensor.T @ (p_ego - t_sensor)
        
        # Orientation transform:
        r_box_global = quat_to_rot_mat(a["rotation"])
        r_box_sensor = r_sensor.T @ r_ego.T @ r_box_global
        
        # Heading vector in sensor XY plane
        v_forward = r_box_sensor @ np.array([0, 1, 0])
        yaw_rad = np.arctan2(v_forward[1], v_forward[0]) # yaw angle in radians
        yaw_deg = np.degrees(yaw_rad)
        
        # Dimensions: [width, length, height]
        w, l, h = a["size"]
        
        # Calculate points strictly inside 3D bounding box
        # Transform points to local box frame
        diff = pts_env[:, :3] - p_sensor
        pts_local = diff @ r_box_sensor
        in_box = (np.abs(pts_local[:, 0]) <= w / 2.0) & \
                 (np.abs(pts_local[:, 1]) <= l / 2.0) & \
                 (np.abs(pts_local[:, 2]) <= h / 2.0)
        num_inside = int(in_box.sum())
        
        corners_3d = get_box_8_corners(p_sensor, [w, l, h], r_box_sensor)
        
        boxes.append({
            "token": a["token"],
            "category": cat_name,
            "center_global": p_global.tolist(),
            "center_sensor": p_sensor.tolist(),
            "size_wlh": [w, l, h],
            "yaw_rad": float(yaw_rad),
            "yaw_deg": float(yaw_deg),
            "gt_pts": a["num_lidar_pts"],
            "measured_inside_pts": num_inside,
            "rot_mat": r_box_sensor,
            "corners_3d": corners_3d
        })
        
    print(f"\n============================================================")
    print(f"SAMPLE {idx+1}/{len(selected_samples)}: {sc_name} | {os.path.basename(sd['filename'])}")
    print(f"============================================================")
    print(f"Total annotations in scene: {len(boxes)}")
    
    # Filter observable boxes in sensor range (e.g. within 45m radius, > 0 points)
    boxes_in_roi = [b for b in boxes if np.hypot(b["center_sensor"][0], b["center_sensor"][1]) <= 45.0 and b["gt_pts"] > 0]
    print(f"Observable boxes in 45m ROI: {len(boxes_in_roi)}")
    
    print("\nSample Ground-Truth 3D Bounding Boxes (Sensor Coordinates):")
    print(f"{'Category':<24} | {'Center [X, Y, Z] (m)':<24} | {'Size [W, L, H] (m)':<20} | {'Yaw':<8} | {'Dist':<6} | {'Pts'}")
    print("-" * 96)
    for b in sorted(boxes_in_roi, key=lambda x: np.hypot(x["center_sensor"][0], x["center_sensor"][1]))[:8]:
        c = b["center_sensor"]
        s = b["size_wlh"]
        dist = np.hypot(c[0], c[1])
        c_str = f"[{c[0]:+5.1f}, {c[1]:+5.1f}, {c[2]:+5.1f}]"
        s_str = f"[{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}]"
        print(f"{b['category']:<24} | {c_str:<24} | {s_str:<20} | {b['yaw_deg']:+5.0f}°  | {dist:4.1f}m | GT:{b['gt_pts']:<3} (Keyframe:{b['measured_inside_pts']})")

    # Generate Visualization 1: Wide-Area BEV with 3D Ground-Truth Bounding Boxes & Heading Vectors
    fig = plt.figure(figsize=(20, 16), facecolor="#0e1117")
    fig.suptitle(f"AutoBus Stage 3: Raw LiDAR + 3D Ground-Truth Bounding Box Alignment\nScene: {sc_name} | Keyframe Points: {len(pts_env):,} | Annotated Objects: {len(boxes_in_roi)}",
                 fontsize=15, color="#f0f6fc", fontweight="bold", y=0.98)
    
    # --- Top Left: BEV Wide View (-45m to 45m lateral, -30m to 50m longitudinal) ---
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#05070a")
    ax1.set_title("1. Top-Down BEV — Full Field of View with Ground-Truth 3D Boxes", color="#58a6ff", fontsize=12, fontweight="bold")
    
    # Scatter environmental points
    bev_mask = (pts_env[:, 0] >= -45) & (pts_env[:, 0] <= 45) & (pts_env[:, 1] >= -30) & (pts_env[:, 1] <= 50)
    sc1 = ax1.scatter(pts_env[bev_mask, 0], pts_env[bev_mask, 1], c=pts_env[bev_mask, 2], s=0.8, cmap="turbo", alpha=0.7, vmin=-2.2, vmax=2.5)
    
    # Range circles
    for cr in [10, 20, 30, 40, 50]:
        circle = plt.Circle((0, 0), cr, color="#484f58", fill=False, linestyle="--", linewidth=0.8, alpha=0.5)
        ax1.add_patch(circle)
        ax1.text(0, cr + 0.6, f"{cr}m", color="#8b949e", fontsize=8, ha="center")
        
    # Draw all boxes
    for b in boxes:
        dist = np.hypot(b["center_sensor"][0], b["center_sensor"][1])
        if dist > 50:
            continue
        c3d = b["corners_3d"] # (3, 8)
        # Bottom 4 corners for BEV polygon: 0, 1, 2, 3
        bev_corners = c3d[:2, [0, 1, 2, 3]].T
        color = get_class_color(b["category"])
        poly = Polygon(bev_corners, closed=True, fill=False, edgecolor=color, linewidth=1.5, alpha=0.9)
        ax1.add_patch(poly)
        
        # Heading vector arrow from center
        c = b["center_sensor"]
        v_head = b["rot_mat"] @ np.array([0, b["size_wlh"][1]/2.0, 0])
        ax1.arrow(c[0], c[1], v_head[0], v_head[1], color=color, width=0.15, head_width=0.6, head_length=0.6, length_includes_head=True, alpha=0.9)
        
        if dist < 28 and b["gt_pts"] > 0:
            short = b["category"].split(".")[-1]
            ax1.text(c[0], c[1], f"{short}\n{b['gt_pts']}p", color=color, fontsize=7, ha="center", va="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="#161b22", alpha=0.7, edgecolor=color, linewidth=0.5))
                     
    # Ego vehicle
    ego_rect = plt.Rectangle((-1.0, -2.5), 2.0, 4.5, edgecolor="#58a6ff", facecolor="#58a6ff", alpha=0.35, linewidth=2.0)
    ax1.add_patch(ego_rect)
    ax1.text(0, -3.2, "AutoBus", color="#58a6ff", fontsize=9, ha="center", fontweight="bold")
    ax1.set_xlim(-45, 45)
    ax1.set_ylim(-30, 50)
    ax1.set_xlabel("Lateral X (meters: +Right, -Left)", color="#8b949e", fontsize=10)
    ax1.set_ylabel("Longitudinal Y (meters: +Forward, -Rear)", color="#8b949e", fontsize=10)
    ax1.tick_params(colors="#8b949e")
    ax1.grid(True, color="#21262d", linestyle=":")
    cb1 = plt.colorbar(sc1, ax=ax1, pad=0.02)
    cb1.set_label("Z (m)", color="#8b949e")
    cb1.ax.tick_params(colors="#8b949e")

    # --- Top Right: Close-up Zoom (X in [-15, 15], Y in [-5, 30]) ---
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#05070a")
    ax2.set_title("2. Close-up Forward Sector — Precise Point-to-Box Alignment", color="#58a6ff", fontsize=12, fontweight="bold")
    
    close_mask = (pts_env[:, 0] >= -16) & (pts_env[:, 0] <= 16) & (pts_env[:, 1] >= -5) & (pts_env[:, 1] <= 32)
    sc2 = ax2.scatter(pts_env[close_mask, 0], pts_env[close_mask, 1], c=pts_env[close_mask, 2], s=2.2, cmap="turbo", alpha=0.85, vmin=-2.2, vmax=2.0)
    
    for cr in [5, 10, 15, 20, 25, 30]:
        circle = plt.Circle((0, 0), cr, color="#484f58", fill=False, linestyle="--", linewidth=0.8, alpha=0.5)
        ax2.add_patch(circle)
        
    for b in boxes:
        c = b["center_sensor"]
        if -16 <= c[0] <= 16 and -5 <= c[1] <= 32:
            c3d = b["corners_3d"]
            bev_corners = c3d[:2, [0, 1, 2, 3]].T
            color = get_class_color(b["category"])
            poly = Polygon(bev_corners, closed=True, fill=True, facecolor=color, alpha=0.25, edgecolor=color, linewidth=2.0)
            ax2.add_patch(poly)
            # Heading arrow
            v_head = b["rot_mat"] @ np.array([0, b["size_wlh"][1]/2.0, 0])
            ax2.arrow(c[0], c[1], v_head[0], v_head[1], color=color, width=0.18, head_width=0.6, head_length=0.6, length_includes_head=True)
            short = b["category"].split(".")[-1]
            ax2.text(c[0], c[1], f"{short}\n[{b['size_wlh'][1]:.1f}x{b['size_wlh'][0]:.1f}m]\n{b['gt_pts']} pts", 
                     color="#ffffff", fontsize=8, ha="center", va="center", fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85, edgecolor="none"))
                     
    ax2.add_patch(plt.Rectangle((-1.0, -2.5), 2.0, 4.5, edgecolor="#58a6ff", facecolor="#58a6ff", alpha=0.35, linewidth=2.0))
    ax2.set_xlim(-16, 16)
    ax2.set_ylim(-5, 32)
    ax2.set_xlabel("Lateral X (meters)", color="#8b949e", fontsize=10)
    ax2.set_ylabel("Longitudinal Y (meters)", color="#8b949e", fontsize=10)
    ax2.tick_params(colors="#8b949e")
    ax2.grid(True, color="#21262d", linestyle=":")
    cb2 = plt.colorbar(sc2, ax=ax2, pad=0.02)
    cb2.set_label("Z (m)", color="#8b949e")
    cb2.ax.tick_params(colors="#8b949e")

    # --- Bottom Left: Front Elevation View (X vs Z) with 3D Boxes projected ---
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#05070a")
    ax3.set_title("3. Front Elevation View (X vs Z) — Vertical Box Height & Ground Fit", color="#58a6ff", fontsize=12, fontweight="bold")
    
    front_mask = (pts_env[:, 1] >= 1.0) & (pts_env[:, 1] <= 32.0) & (pts_env[:, 0] >= -20) & (pts_env[:, 0] <= 20) & (pts_env[:, 2] >= -3.0) & (pts_env[:, 2] <= 3.5)
    sc3 = ax3.scatter(pts_env[front_mask, 0], pts_env[front_mask, 2], c=pts_env[front_mask, 1], s=1.2, cmap="viridis_r", alpha=0.75)
    ax3.axhline(-1.84, color="#f85149", linestyle="--", linewidth=1.5, label="Road Plane (Z ≈ -1.84m)")
    
    # Project 3D boxes onto X-Z plane
    for b in boxes:
        c = b["center_sensor"]
        if 1.0 <= c[1] <= 32.0 and -20 <= c[0] <= 20:
            c3d = b["corners_3d"] # (3, 8)
            # Front face corners: 3, 2, 6, 7
            # Or bounding box on X-Z:
            x_min, x_max = c3d[0, :].min(), c3d[0, :].max()
            z_min, z_max = c3d[2, :].min(), c3d[2, :].max()
            color = get_class_color(b["category"])
            rect = plt.Rectangle((x_min, z_min), x_max - x_min, z_max - z_min, fill=False, edgecolor=color, linewidth=1.8, linestyle="-")
            ax3.add_patch(rect)
            short = b["category"].split(".")[-1]
            ax3.text((x_min + x_max)/2.0, z_max + 0.15, short, color=color, fontsize=7.5, ha="center", fontweight="bold")
            
    ax3.set_xlim(-20, 20)
    ax3.set_ylim(-3.0, 3.5)
    ax3.set_xlabel("Lateral X (meters)", color="#8b949e", fontsize=10)
    ax3.set_ylabel("Vertical Height Z (meters)", color="#8b949e", fontsize=10)
    ax3.tick_params(colors="#8b949e")
    ax3.grid(True, color="#21262d", linestyle=":")
    ax3.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)
    cb3 = plt.colorbar(sc3, ax=ax3, pad=0.02)
    cb3.set_label("Distance Ahead Y (m)", color="#8b949e")
    cb3.ax.tick_params(colors="#8b949e")

    # --- Bottom Right: Isometric 3D Wireframe of Objects Ahead ---
    ax4 = fig.add_subplot(2, 2, 4, projection="3d", facecolor="#05070a")
    ax4.set_title("4. 3D Wireframe Bounding Boxes & Point Cloud Clusters", color="#58a6ff", fontsize=12, fontweight="bold")
    
    # Points in close range for 3D
    pts_3d_mask = (pts_env[:, 0] >= -15) & (pts_env[:, 0] <= 15) & (pts_env[:, 1] >= 0) & (pts_env[:, 1] <= 30) & (pts_env[:, 2] >= -2.5) & (pts_env[:, 2] <= 2.5)
    p3 = pts_env[pts_3d_mask][::2] # stride 2
    sc4 = ax4.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c=p3[:, 2], cmap="turbo", s=1.0, alpha=0.5, vmin=-2.2, vmax=2.0)
    
    # Draw 3D wireframe boxes
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0], # bottom
        [4, 5], [5, 6], [6, 7], [7, 4], # top
        [0, 4], [1, 5], [2, 6], [3, 7]  # vertical pillars
    ]
    
    for b in boxes:
        c = b["center_sensor"]
        if -15 <= c[0] <= 15 and 0 <= c[1] <= 30:
            c3d = b["corners_3d"] # (3, 8)
            color = get_class_color(b["category"])
            for e in edges:
                ax4.plot([c3d[0, e[0]], c3d[0, e[1]]],
                         [c3d[1, e[0]], c3d[1, e[1]]],
                         [c3d[2, e[0]], c3d[2, e[1]]],
                         color=color, linewidth=1.5, alpha=0.9)
                         
    ax4.view_init(elev=28, azim=-60)
    ax4.set_xlim(-15, 15)
    ax4.set_ylim(0, 30)
    ax4.set_zlim(-2.5, 2.5)
    ax4.set_xlabel("X (m)", color="#8b949e", labelpad=5)
    ax4.set_ylabel("Y (m)", color="#8b949e", labelpad=5)
    ax4.set_zlabel("Z (m)", color="#8b949e", labelpad=5)
    ax4.tick_params(colors="#8b949e")
    ax4.xaxis.pane.fill = False
    ax4.yaxis.pane.fill = False
    ax4.zaxis.pane.fill = False
    ax4.xaxis.pane.set_edgecolor("#21262d")
    ax4.yaxis.pane.set_edgecolor("#21262d")
    ax4.zaxis.pane.set_edgecolor("#21262d")
    ax4.grid(True, color="#21262d", linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_name = f"gt_alignment_{sc_name}.png"
    plt.savefig(os.path.join(output_dir, out_name), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.savefig(os.path.join(artifact_dir, out_name), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_name}")
    
    sample_inspection_results.append({
        "scene": sc_name,
        "sample_token": s_token,
        "total_annotations": len(boxes),
        "observable_in_45m": len(boxes_in_roi),
        "image_file": out_name
    })

# Save JSON metadata of Stage 3
with open(os.path.join(output_dir, "stage3_gt_inspection_summary.json"), "w") as f:
    json.dump({
        "coordinate_conventions": {
            "annotation_raw_frame": "Global World Coordinates (UTM)",
            "sensor_frame": "LIDAR_TOP (Right-handed: +X Right, +Y Forward, +Z Upward)",
            "transformation": "p_sensor = R_sensor.T @ (R_ego.T @ (p_global - t_ego) - t_sensor)"
        },
        "samples_evaluated": sample_inspection_results
    }, f, indent=2)

print("\nSTAGE 3 GROUND-TRUTH INSPECTION COMPLETED SUCCESSFULLY.")
