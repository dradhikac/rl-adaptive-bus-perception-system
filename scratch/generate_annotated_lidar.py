import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def quat_to_rot_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def get_box_corners_2d(center, size, rot_quat):
    # center: [x, y, z], size: [w, l, h], rot_quat: [w, x, y, z]
    w, l, h = size
    r_mat = quat_to_rot_mat(rot_quat)
    # 2D corners in box local frame (X lateral, Y longitudinal)
    x_c = [-w/2, w/2, w/2, -w/2]
    y_c = [-l/2, -l/2, l/2, l/2]
    z_c = [0, 0, 0, 0]
    local_corners = np.vstack([x_c, y_c, z_c])
    rotated_corners = r_mat @ local_corners
    world_corners = rotated_corners + np.array(center)[:, None]
    return world_corners[:2, :] # 2x4

dataset_root = r"datasets\nuscenes"
meta_dir = os.path.join(dataset_root, "v1.0-mini")
output_dir = r"visualizations\stage2_lidar"
artifact_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\1c0087c6-dac4-4253-a522-0bdbc5b00eb8"

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
with open(os.path.join(meta_dir, "scene.json"), "r") as f:
    scenes = {sc["token"]: sc["name"] for sc in json.load(f)}

# Select Sample 0 (scene-0061)
sample_0 = samples[0]
sc_name = scenes.get(sample_0["scene_token"], "scene-0061")
sd_0 = next(sd for sd in sample_data if sd["sample_token"] == sample_0["token"] and "LIDAR_TOP" in sd["filename"] and sd["is_key_frame"])

pts_path = os.path.join(dataset_root, sd_0["filename"].replace("/", os.sep))
pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)

ego_pose = ego_poses[sd_0["ego_pose_token"]]
cal_sensor = calibrated_sensors[sd_0["calibrated_sensor_token"]]

r_sensor = quat_to_rot_mat(cal_sensor["rotation"])
t_sensor = np.array(cal_sensor["translation"])
r_ego = quat_to_rot_mat(ego_pose["rotation"])
t_ego = np.array(ego_pose["translation"])

# Filter out self-reflections (r < 1.0m)
r_pts = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
pts_env = pts[r_pts >= 1.0]

# Transform annotations into sensor frame
sample_anns = [a for a in annotations if a["sample_token"] == sample_0["token"]]
sensor_boxes = []

for a in sample_anns:
    p_global = np.array(a["translation"])
    p_ego = r_ego.T @ (p_global - t_ego)
    p_sensor = r_sensor.T @ (p_ego - t_sensor)
    
    # Also rotate box orientation to sensor frame
    r_box_global = quat_to_rot_mat(a["rotation"])
    r_box_sensor = r_sensor.T @ r_ego.T @ r_box_global
    
    cat_name = categories.get(instances.get(a["instance_token"], ""), "unknown")
    sensor_boxes.append({
        "category": cat_name,
        "center": p_sensor,
        "size": a["size"], # [w, l, h]
        "rot_mat": r_box_sensor,
        "num_pts": a["num_lidar_pts"]
    })

# Compute 2D corners in sensor frame
def get_sensor_corners(box):
    w, l, h = box["size"]
    x_c = [-w/2, w/2, w/2, -w/2]
    y_c = [-l/2, -l/2, l/2, l/2]
    z_c = [0, 0, 0, 0]
    local_corners = np.vstack([x_c, y_c, z_c])
    rotated = box["rot_mat"] @ local_corners
    world = rotated + box["center"][:, None]
    return world[:2, :].T # 4x2

# Plot 1: Detailed BEV Object Ground Truth Alignment Verification
fig = plt.figure(figsize=(20, 10), facecolor="#0e1117")

# Panel A: Wide-area BEV with Ground Truth Boxes
ax1 = fig.add_subplot(1, 2, 1, facecolor="#05070a")
ax1.set_title(f"A. BEV Point Cloud & Ground Truth 3D Boxes Overlay\n(Scene: {sc_name} | {len(pts_env):,} Env Points | {len(sensor_boxes)} Ground Truth Objects)", 
              color="#58a6ff", fontsize=13, fontweight="bold", pad=12)

# Scatter LiDAR points
bev_mask = (pts_env[:, 0] >= -45) & (pts_env[:, 0] <= 45) & (pts_env[:, 1] >= -35) & (pts_env[:, 1] <= 55)
sc1 = ax1.scatter(pts_env[bev_mask, 0], pts_env[bev_mask, 1], c=pts_env[bev_mask, 2], s=0.9, cmap="turbo", alpha=0.75, vmin=-2.2, vmax=2.5)

# Range circles
for circle_r in [10, 20, 30, 40, 50]:
    circle = plt.Circle((0, 0), circle_r, color="#484f58", fill=False, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.add_patch(circle)
    ax1.text(0, circle_r + 0.6, f"{circle_r}m", color="#8b949e", fontsize=8, ha="center")

# Draw Ground Truth Boxes
class_colors = {
    "vehicle": "#f85149", # Red
    "human": "#3fb950",   # Green
    "movable_object": "#d29922", # Yellow/Orange
    "cycle": "#a371f7"    # Purple
}

for box in sensor_boxes:
    corners = get_sensor_corners(box)
    cat = box["category"]
    color = "#8b949e"
    for prefix, c in class_colors.items():
        if prefix in cat:
            color = c
            break
    poly = Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.5, alpha=0.9)
    ax1.add_patch(poly)
    # If in range, draw small label
    dist = np.hypot(box["center"][0], box["center"][1])
    if dist < 25 and box["num_pts"] > 0:
        short_name = cat.split(".")[-1]
        ax1.text(box["center"][0], box["center"][1], f"{short_name} ({box['num_pts']}p)", 
                 color=color, fontsize=7.5, ha="center", va="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="#161b22", alpha=0.65, edgecolor=color, linewidth=0.5))

# Draw Ego Bus
bus_rect = plt.Rectangle((-1.0, -2.5), 2.0, 4.5, edgecolor="#58a6ff", facecolor="#58a6ff", alpha=0.3, linewidth=2.0)
ax1.add_patch(bus_rect)
ax1.text(0, -3.2, "AutoBus Ego", color="#58a6ff", fontsize=9, ha="center", fontweight="bold")

ax1.set_xlim(-45, 45)
ax1.set_ylim(-35, 55)
ax1.set_xlabel("Lateral X (meters: +Right, -Left)", color="#8b949e", fontsize=11)
ax1.set_ylabel("Longitudinal Y (meters: +Forward, -Rear)", color="#8b949e", fontsize=11)
ax1.tick_params(colors="#8b949e")
ax1.grid(True, color="#21262d", linestyle=":", alpha=0.6)
cb1 = plt.colorbar(sc1, ax=ax1, pad=0.02)
cb1.set_label("Height Z (m)", color="#8b949e")
cb1.ax.tick_params(colors="#8b949e")

# Panel B: Close-up Region in Front of Bus (X in [-15, 15], Y in [0, 30])
ax2 = fig.add_subplot(1, 2, 2, facecolor="#05070a")
ax2.set_title("B. High-Resolution Forward Sector (0m to 30m Ahead)\n(Physical Alignment of Point Clouds with 3D Ground Truth)", 
              color="#58a6ff", fontsize=13, fontweight="bold", pad=12)

close_mask = (pts_env[:, 0] >= -16) & (pts_env[:, 0] <= 16) & (pts_env[:, 1] >= -2) & (pts_env[:, 1] <= 32)
sc2 = ax2.scatter(pts_env[close_mask, 0], pts_env[close_mask, 1], c=pts_env[close_mask, 2], s=2.5, cmap="turbo", alpha=0.85, vmin=-2.2, vmax=2.0)

for circle_r in [5, 10, 15, 20, 25, 30]:
    circle = plt.Circle((0, 0), circle_r, color="#484f58", fill=False, linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.add_patch(circle)
    ax2.text(0, circle_r + 0.3, f"{circle_r}m", color="#8b949e", fontsize=8, ha="center")

for box in sensor_boxes:
    c = box["center"]
    if -16 <= c[0] <= 16 and -2 <= c[1] <= 32:
        corners = get_sensor_corners(box)
        cat = box["category"]
        color = "#8b949e"
        for prefix, col in class_colors.items():
            if prefix in cat:
                color = col
                break
        poly = Polygon(corners, closed=True, fill=True, facecolor=color, alpha=0.2, edgecolor=color, linewidth=2.0)
        ax2.add_patch(poly)
        short_name = cat.split(".")[-1]
        ax2.text(c[0], c[1], f"{short_name}\n({box['num_pts']} pts)", color="#ffffff", fontsize=8, ha="center", va="center", fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8, edgecolor="none"))

# Bus Front
ax2.add_patch(plt.Rectangle((-1.0, -2.5), 2.0, 4.5, edgecolor="#58a6ff", facecolor="#58a6ff", alpha=0.4, linewidth=2.0))
ax2.text(0, 0.5, "Bus Front", color="#58a6ff", fontsize=9, ha="center", fontweight="bold")

ax2.set_xlim(-16, 16)
ax2.set_ylim(-2, 32)
ax2.set_xlabel("Lateral X (meters)", color="#8b949e", fontsize=11)
ax2.set_ylabel("Longitudinal Y (meters)", color="#8b949e", fontsize=11)
ax2.tick_params(colors="#8b949e")
ax2.grid(True, color="#21262d", linestyle=":", alpha=0.6)
cb2 = plt.colorbar(sc2, ax=ax2, pad=0.02)
cb2.set_label("Height Z (m)", color="#8b949e")
cb2.ax.tick_params(colors="#8b949e")

plt.tight_layout()
out_fig1 = "lidar_groundtruth_alignment_bev.png"
plt.savefig(os.path.join(output_dir, out_fig1), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig(os.path.join(artifact_dir, out_fig1), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print(f"Saved: {out_fig1}")

# Plot 2: Point Cloud Anatomy & Road Plane Segmentation Diagnostic
fig2 = plt.figure(figsize=(18, 12), facecolor="#0e1117")
fig2.suptitle(f"AutoBus LiDAR Point Cloud Physical Diagnostics\n(Coordinate Systems, Road Surface & Beam Structure)", 
              fontsize=15, color="#f0f6fc", fontweight="bold", y=0.98)

# Subplot 1: Self-reflections vs Environmental Points
ax2_1 = fig2.add_subplot(2, 2, 1, facecolor="#05070a")
ax2_1.set_title("1. Ego-Mount Reflections (r < 1m) vs Environmental Points", color="#58a6ff", fontsize=12, fontweight="bold")
self_mask = r_pts < 1.0
ax2_1.scatter(pts[self_mask, 0], pts[self_mask, 1], c="#f85149", s=2.0, alpha=0.8, label=f"Self-Reflections ({self_mask.sum():,} pts / {self_mask.mean()*100:.1f}%)")
ax2_1.scatter(pts[~self_mask, 0], pts[~self_mask, 1], c="#3fb950", s=0.5, alpha=0.3, label=f"Environment ({(~self_mask).sum():,} pts / {(~self_mask).mean()*100:.1f}%)")
ax2_1.set_xlim(-15, 15)
ax2_1.set_ylim(-15, 15)
ax2_1.set_xlabel("X (m)", color="#8b949e")
ax2_1.set_ylabel("Y (m)", color="#8b949e")
ax2_1.tick_params(colors="#8b949e")
ax2_1.grid(True, color="#21262d", linestyle=":")
ax2_1.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)

# Subplot 2: Height Distribution & Road Surface Detection
ax2_2 = fig2.add_subplot(2, 2, 2, facecolor="#05070a")
ax2_2.set_title("2. Vertical Height Histogram (Environmental Points, r ≥ 1m)", color="#58a6ff", fontsize=12, fontweight="bold")
z_env = pts_env[:, 2]
z_filtered = z_env[(z_env >= -3.0) & (z_env <= 4.0)]
n_counts, bins, patches = ax2_2.hist(z_filtered, bins=70, color="#58a6ff", edgecolor="#1f6feb", alpha=0.8)
# Find peak
peak_z = (bins[np.argmax(n_counts)] + bins[np.argmax(n_counts)+1]) / 2.0
ax2_2.axvline(peak_z, color="#f85149", linestyle="--", linewidth=2.0, label=f"Detected Road Ground Peak ({peak_z:.2f}m)")
ax2_2.axvline(-1.84, color="#e3b341", linestyle=":", linewidth=2.0, label="Nominal Sensor Mount Height (-1.84m)")
ax2_2.set_xlabel("Vertical Coordinate Z (meters relative to sensor origin)", color="#8b949e")
ax2_2.set_ylabel("Point Count", color="#8b949e")
ax2_2.tick_params(colors="#8b949e")
ax2_2.grid(True, color="#21262d", linestyle=":")
ax2_2.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)

# Subplot 3: Radial Point Density vs Distance
ax2_3 = fig2.add_subplot(2, 2, 3, facecolor="#05070a")
ax2_3.set_title("3. Point Density Decay vs Radial Range (r)", color="#58a6ff", fontsize=12, fontweight="bold")
r_env = np.sqrt(pts_env[:, 0]**2 + pts_env[:, 1]**2)
ax2_3.hist(r_env[r_env <= 60], bins=60, color="#bc8cff", edgecolor="#8957e5", alpha=0.8)
ax2_3.axvline(10, color="#8b949e", linestyle="--", linewidth=1.0)
ax2_3.axvline(20, color="#8b949e", linestyle="--", linewidth=1.0)
ax2_3.axvline(30, color="#8b949e", linestyle="--", linewidth=1.0)
ax2_3.set_xlabel("Distance from Bus Sensor r (meters)", color="#8b949e")
ax2_3.set_ylabel("LiDAR Points per 1-meter Bin", color="#8b949e")
ax2_3.tick_params(colors="#8b949e")
ax2_3.grid(True, color="#21262d", linestyle=":")

# Subplot 4: Laser Ring Structure (32 Beams Elevation Angles)
ax2_4 = fig2.add_subplot(2, 2, 4, facecolor="#05070a")
ax2_4.set_title("4. 32-Beam Laser Ring Elevation Distribution", color="#58a6ff", fontsize=12, fontweight="bold")
rings = pts_env[:, 4].astype(int)
unique_rings, ring_counts = np.unique(rings, return_counts=True)
ax2_4.bar(unique_rings, ring_counts, color="#39c5bb", edgecolor="#178880", alpha=0.85)
ax2_4.set_xlabel("Laser Ring Index (0 to 31)", color="#8b949e")
ax2_4.set_ylabel("Points per Beam Ring", color="#8b949e")
ax2_4.tick_params(colors="#8b949e")
ax2_4.grid(True, color="#21262d", linestyle=":")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out_fig2 = "lidar_diagnostics_anatomy.png"
plt.savefig(os.path.join(output_dir, out_fig2), dpi=180, facecolor=fig2.get_facecolor(), edgecolor="none")
plt.savefig(os.path.join(artifact_dir, out_fig2), dpi=180, facecolor=fig2.get_facecolor(), edgecolor="none")
plt.close(fig2)
print(f"Saved: {out_fig2}")

print("ALL ENHANCED VISUALIZATIONS GENERATED SUCCESSFULLY.")
