import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# Setup paths
dataset_root = r"datasets\nuscenes"
samples_dir = os.path.join(dataset_root, "samples", "LIDAR_TOP")
meta_dir = os.path.join(dataset_root, "v1.0-mini")
output_dir = r"visualizations\stage2_lidar"
artifact_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\1c0087c6-dac4-4253-a522-0bdbc5b00eb8"

os.makedirs(output_dir, exist_ok=True)
os.makedirs(artifact_dir, exist_ok=True)

print("============================================================")
print("STAGE 2: RAW LIDAR POINT CLOUD INSPECTION & VALIDATION")
print("============================================================")

# 1. Exhaustive integrity check across all 404 keyframe point cloud files
sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith(".pcd.bin")])
print(f"Total keyframe files found: {len(sample_files)}")

corrupted_files = []
nan_files = []
inf_files = []
empty_files = []
point_counts = []
min_coords = np.array([float("inf")] * 3)
max_coords = np.array([float("-inf")] * 3)
all_intensities = []

for sf in sample_files:
    fp = os.path.join(samples_dir, sf)
    try:
        data = np.fromfile(fp, dtype=np.float32)
        if len(data) == 0:
            empty_files.append(sf)
            continue
        if len(data) % 5 != 0:
            corrupted_files.append((sf, f"Size {len(data)} not divisible by 5 float32"))
            continue
        pts = data.reshape(-1, 5)
        
        # Check NaNs and Infs
        if np.isnan(pts).any():
            nan_files.append(sf)
        if np.isinf(pts).any():
            inf_files.append(sf)
            
        point_counts.append(len(pts))
        min_coords = np.minimum(min_coords, pts[:, :3].min(axis=0))
        max_coords = np.maximum(max_coords, pts[:, :3].max(axis=0))
    except Exception as e:
        corrupted_files.append((sf, str(e)))

print(f"\n--- DATA INTEGRITY SUMMARY ---")
print(f"Files tested: {len(sample_files)}")
print(f"Corrupted files: {len(corrupted_files)}")
print(f"NaN files: {len(nan_files)}")
print(f"Inf files: {len(inf_files)}")
print(f"Empty files: {len(empty_files)}")
print(f"Min points: {min(point_counts):,}")
print(f"Max points: {max(point_counts):,}")
print(f"Mean points: {np.mean(point_counts):,.1f}")
print(f"Median points: {np.median(point_counts):,.1f}")
print(f"Overall X bounds (lateral):      [{min_coords[0]:.2f}, {max_coords[0]:.2f}] m")
print(f"Overall Y bounds (longitudinal): [{min_coords[1]:.2f}, {max_coords[1]:.2f}] m")
print(f"Overall Z bounds (vertical):     [{min_coords[2]:.2f}, {max_coords[2]:.2f}] m")

# 2. Load metadata to link samples to scenes and timestamps
def load_json(name):
    with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as f:
        return json.load(f)

scenes = load_json("scene.json")
samples = load_json("sample.json")
sample_data = load_json("sample_data.json")
calibrated_sensors = load_json("calibrated_sensor.json")
sensors = load_json("sensor.json")
ego_poses = load_json("ego_pose.json")

# Map LIDAR calibrated sensor
lidar_sensor_token = [s["token"] for s in sensors if "LIDAR" in s.get("channel", "")][0]
lidar_cal = [cs for cs in calibrated_sensors if cs["sensor_token"] == lidar_sensor_token][0]
print(f"\n--- SENSOR CALIBRATION (LIDAR_TOP relative to Ego Vehicle) ---")
print(f"Translation [x, y, z]: {lidar_cal['translation']} (meters)")
print(f"Rotation Quaternion [w, x, y, z]: {lidar_cal['rotation']}")

# Select 6 diverse representative samples across different scenes
# Build mapping from sample to scene
sample_to_scene = {}
for sc in scenes:
    first_token = sc["first_sample_token"]
    last_token = sc["last_sample_token"]
    # traverse
    curr = first_token
    while curr:
        sample_to_scene[curr] = sc["name"]
        curr_sample = next((s for s in samples if s["token"] == curr), None)
        curr = curr_sample["next"] if curr_sample else None

# Filter sample_data for LIDAR_TOP keyframes
lidar_sd = [sd for sd in sample_data if sd.get("fileformat") == "pcd" and "LIDAR_TOP" in sd.get("filename", "") and sd.get("is_key_frame")]
print(f"Total keyframe sample_data records: {len(lidar_sd)}")

# Select samples from 6 distinct scenes (e.g. scene-0061, scene-0103, scene-0655, scene-0757, scene-0921, scene-1094)
scene_sample_map = {}
for sd in lidar_sd:
    sc_name = sample_to_scene.get(sd["sample_token"], "unknown")
    if sc_name not in scene_sample_map:
        scene_sample_map[sc_name] = []
    scene_sample_map[sc_name].append(sd)

selected_samples = []
for sc_name in sorted(scene_sample_map.keys())[:6]:
    # pick middle sample in the scene
    sds = scene_sample_map[sc_name]
    mid_sd = sds[len(sds) // 2]
    selected_samples.append((sc_name, mid_sd))

print(f"\nSelected {len(selected_samples)} representative samples across different scenes:")
for sc_name, sd in selected_samples:
    print(f"  Scene {sc_name}: filename={os.path.basename(sd['filename'])}, timestamp={sd['timestamp']}")

# 3. Deep Analysis & Visualization on Representative Samples
saved_visualizations = []

for idx, (sc_name, sd) in enumerate(selected_samples):
    rel_path = sd["filename"]
    # Handle windows vs posix path
    rel_path = rel_path.replace("/", os.sep)
    full_path = os.path.join(dataset_root, rel_path)
    
    pts = np.fromfile(full_path, dtype=np.float32).reshape(-1, 5)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    intensity = pts[:, 3]
    ring = pts[:, 4]
    
    r = np.sqrt(x**2 + y**2)
    
    # Statistical checks
    # Points in front (y > 0) vs back (y < 0)
    pct_front = (y > 0).mean() * 100
    pct_back = (y < 0).mean() * 100
    pct_left = (x < 0).mean() * 100
    pct_right = (x > 0).mean() * 100
    
    # Ground plane estimation: road points typically have z between -1.9m and -1.4m
    # Find mode of height distribution between -2.5m and 0.0m
    z_road_cand = z[(z > -2.5) & (z < 0.0)]
    hist, bin_edges = np.histogram(z_road_cand, bins=50)
    ground_z_est = (bin_edges[np.argmax(hist)] + bin_edges[np.argmax(hist) + 1]) / 2.0
    ground_pts_mask = (z >= ground_z_est - 0.25) & (z <= ground_z_est + 0.25)
    pct_ground = ground_pts_mask.mean() * 100
    
    # Distance ring statistics
    pts_0_10 = (r < 10).sum()
    pts_10_20 = ((r >= 10) & (r < 20)).sum()
    pts_20_30 = ((r >= 20) & (r < 30)).sum()
    pts_30_50 = ((r >= 30) & (r < 50)).sum()
    pts_50_plus = (r >= 50).sum()
    
    print(f"\n============================================================")
    print(f"SAMPLE {idx+1}/{len(selected_samples)}: {sc_name} | {os.path.basename(rel_path)}")
    print(f"============================================================")
    print(f"Total points: {len(pts):,}")
    print(f"X range: [{x.min():.2f}, {x.max():.2f}] m | Y range: [{y.min():.2f}, {y.max():.2f}] m | Z range: [{z.min():.2f}, {z.max():.2f}] m")
    print(f"Intensity range: [{intensity.min():.1f}, {intensity.max():.1f}] (mean={intensity.mean():.1f}, median={np.median(intensity):.1f})")
    print(f"Rings present: {np.unique(ring).astype(int).tolist()} ({len(np.unique(ring))} distinct laser beams)")
    print(f"Orientation distribution: Front (Y>0) = {pct_front:.1f}%, Back (Y<0) = {pct_back:.1f}%, Left (X<0) = {pct_left:.1f}%, Right (X>0) = {pct_right:.1f}%")
    print(f"Estimated Road Surface Height: {ground_z_est:.2f} m (Ground points ~{pct_ground:.1f}%)")
    print(f"Range breakdown: 0-10m: {pts_0_10:,} ({pts_0_10/len(pts)*100:.1f}%), 10-20m: {pts_10_20:,} ({pts_10_20/len(pts)*100:.1f}%), 20-30m: {pts_20_30:,} ({pts_20_30/len(pts)*100:.1f}%), 30-50m: {pts_30_50:,} ({pts_30_50/len(pts)*100:.1f}%), >50m: {pts_50_plus:,} ({pts_50_plus/len(pts)*100:.1f}%)")
    
    # 4. Generate Comprehensive 4-Panel Visualization
    fig = plt.figure(figsize=(20, 16), facecolor="#0e1117")
    fig.suptitle(f"AutoBus LiDAR Perception — Raw Point Cloud Inspection\nScene: {sc_name} | Frame: {os.path.basename(rel_path)} | Points: {len(pts):,}", 
                 fontsize=16, color="#f0f6fc", fontweight="bold", y=0.98)
    
    # Filter points for visualization range: X in [-40, 40], Y in [-30, 50], Z in [-3.5, 5]
    bev_mask = (x >= -45) & (x <= 45) & (y >= -35) & (y <= 55) & (z >= -3.5) & (z <= 5.0)
    x_sub = x[bev_mask]
    y_sub = y[bev_mask]
    z_sub = z[bev_mask]
    i_sub = intensity[bev_mask]
    
    # --- Subplot 1: Top-down Bird's Eye View (BEV) colored by Height (Z) ---
    ax1 = fig.add_subplot(2, 2, 1, facecolor="#05070a")
    ax1.set_title("1. Top-Down Bird's Eye View (BEV) — Height Colored", color="#58a6ff", fontsize=13, fontweight="bold", pad=10)
    sc1 = ax1.scatter(x_sub, y_sub, c=z_sub, s=0.7, cmap="turbo", alpha=0.75, vmin=-2.5, vmax=3.0)
    # Range circles
    for circle_r in [10, 20, 30, 40, 50]:
        circle = plt.Circle((0, 0), circle_r, color="#484f58", fill=False, linestyle="--", linewidth=0.8, alpha=0.6)
        ax1.add_patch(circle)
        ax1.text(0, circle_r + 0.5, f"{circle_r}m", color="#8b949e", fontsize=8, ha="center")
    # Bus/Ego Vehicle Box at origin (approx 2m width x 4.5m length)
    ego_rect = plt.Rectangle((-1.0, -2.5), 2.0, 4.5, edgecolor="#f85149", facecolor="#f85149", alpha=0.35, linewidth=1.5, label="Bus/Ego Vehicle")
    ax1.add_patch(ego_rect)
    ax1.plot([0, 0], [0, 4.0], color="#f85149", linewidth=2.0)  # Forward direction indicator
    ax1.text(0, -3.2, "Ego Bus", color="#f85149", fontsize=9, ha="center", fontweight="bold")
    ax1.set_xlim(-45, 45)
    ax1.set_ylim(-35, 55)
    ax1.set_xlabel("Lateral X (meters: +Right, -Left)", color="#8b949e", fontsize=10)
    ax1.set_ylabel("Longitudinal Y (meters: +Forward, -Backward)", color="#8b949e", fontsize=10)
    ax1.tick_params(colors="#8b949e")
    ax1.grid(True, color="#21262d", linestyle=":", alpha=0.6)
    cb1 = plt.colorbar(sc1, ax=ax1, pad=0.02)
    cb1.set_label("Height Z (m)", color="#8b949e")
    cb1.ax.tick_params(colors="#8b949e")
    
    # --- Subplot 2: Top-down BEV colored by LiDAR Intensity (Reflectivity) ---
    ax2 = fig.add_subplot(2, 2, 2, facecolor="#05070a")
    ax2.set_title("2. Top-Down BEV — Intensity (Surface Reflectivity)", color="#58a6ff", fontsize=13, fontweight="bold", pad=10)
    # Clip intensity to 99th percentile for contrast
    i_norm = np.clip(i_sub, 0, np.percentile(i_sub, 99))
    sc2 = ax2.scatter(x_sub, y_sub, c=i_norm, s=0.7, cmap="magma", alpha=0.75)
    for circle_r in [10, 20, 30, 40, 50]:
        circle = plt.Circle((0, 0), circle_r, color="#484f58", fill=False, linestyle="--", linewidth=0.8, alpha=0.6)
        ax2.add_patch(circle)
    ego_rect2 = plt.Rectangle((-1.0, -2.5), 2.0, 4.5, edgecolor="#3fb950", facecolor="#3fb950", alpha=0.35, linewidth=1.5)
    ax2.add_patch(ego_rect2)
    ax2.set_xlim(-45, 45)
    ax2.set_ylim(-35, 55)
    ax2.set_xlabel("Lateral X (meters: +Right, -Left)", color="#8b949e", fontsize=10)
    ax2.set_ylabel("Longitudinal Y (meters: +Forward, -Backward)", color="#8b949e", fontsize=10)
    ax2.tick_params(colors="#8b949e")
    ax2.grid(True, color="#21262d", linestyle=":", alpha=0.6)
    cb2 = plt.colorbar(sc2, ax=ax2, pad=0.02)
    cb2.set_label("LiDAR Intensity", color="#8b949e")
    cb2.ax.tick_params(colors="#8b949e")

    # --- Subplot 3: Front View (X vs Z) / Longitudinal Profile (Y vs Z) ---
    # We will show Front Elevation View: X (lateral) vs Z (height), points with Y in [1, 40]
    front_mask = (y >= 1.0) & (y <= 35.0) & (x >= -25) & (x <= 25) & (z >= -3.0) & (z <= 4.0)
    ax3 = fig.add_subplot(2, 2, 3, facecolor="#05070a")
    ax3.set_title("3. Front Elevation View (X vs Z) — Objects Ahead (Y ∈ [1m, 35m])", color="#58a6ff", fontsize=13, fontweight="bold", pad=10)
    sc3 = ax3.scatter(x[front_mask], z[front_mask], c=y[front_mask], s=0.8, cmap="viridis_r", alpha=0.7)
    # Ground reference line
    ax3.axhline(ground_z_est, color="#f85149", linestyle="--", linewidth=1.2, label=f"Road Plane (~{ground_z_est:.2f}m)")
    ax3.axvline(0, color="#8b949e", linestyle=":", linewidth=0.8, alpha=0.7)
    ax3.set_xlim(-25, 25)
    ax3.set_ylim(-3.0, 4.0)
    ax3.set_xlabel("Lateral X (meters)", color="#8b949e", fontsize=10)
    ax3.set_ylabel("Vertical Height Z (meters)", color="#8b949e", fontsize=10)
    ax3.tick_params(colors="#8b949e")
    ax3.grid(True, color="#21262d", linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)
    cb3 = plt.colorbar(sc3, ax=ax3, pad=0.02)
    cb3.set_label("Distance Ahead Y (m)", color="#8b949e")
    cb3.ax.tick_params(colors="#8b949e")

    # --- Subplot 4: Side Elevation Profile (Y vs Z) & Density Distribution ---
    # Side view: Y (forward) vs Z (height) for lateral corridor X in [-10, 10]
    corridor_mask = (np.abs(x) <= 10.0) & (y >= -20) & (y <= 45) & (z >= -3.0) & (z <= 4.0)
    ax4 = fig.add_subplot(2, 2, 4, facecolor="#05070a")
    ax4.set_title("4. Longitudinal Side Profile (Y vs Z) — Center Corridor (|X| ≤ 10m)", color="#58a6ff", fontsize=13, fontweight="bold", pad=10)
    sc4 = ax4.scatter(y[corridor_mask], z[corridor_mask], c=intensity[corridor_mask], s=0.8, cmap="plasma", alpha=0.7)
    ax4.axhline(ground_z_est, color="#f85149", linestyle="--", linewidth=1.2, label=f"Road Plane (~{ground_z_est:.2f}m)")
    ax4.axvline(0, color="#3fb950", linestyle="-", linewidth=1.5, label="LiDAR Sensor Origin")
    ax4.set_xlim(-20, 45)
    ax4.set_ylim(-3.0, 4.0)
    ax4.set_xlabel("Longitudinal Y (meters: Forward →)", color="#8b949e", fontsize=10)
    ax4.set_ylabel("Vertical Height Z (meters)", color="#8b949e", fontsize=10)
    ax4.tick_params(colors="#8b949e")
    ax4.grid(True, color="#21262d", linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)
    cb4 = plt.colorbar(sc4, ax=ax4, pad=0.02)
    cb4.set_label("Intensity", color="#8b949e")
    cb4.ax.tick_params(colors="#8b949e")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save image
    out_filename = f"raw_lidar_sample_{idx+1}_{sc_name}.png"
    out_path = os.path.join(output_dir, out_filename)
    art_path = os.path.join(artifact_dir, out_filename)
    plt.savefig(out_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.savefig(art_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved visualization: {out_filename}")
    saved_visualizations.append((out_filename, sc_name, len(pts), ground_z_est))

# 5. Generate a high-resolution 3D Perspective Projection View for Sample 1
# Let's create an isometric/3D view
sample_1_sd = selected_samples[0][1]
sample_1_path = os.path.join(dataset_root, sample_1_sd["filename"].replace("/", os.sep))
pts1 = np.fromfile(sample_1_path, dtype=np.float32).reshape(-1, 5)

# Downsample slightly for clear 3D rendering (e.g. 15,000 points)
stride = max(1, len(pts1) // 18000)
pts_3d = pts1[::stride]
x3, y3, z3, i3 = pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], pts_3d[:, 3]
mask_3d = (x3 >= -35) & (x3 <= 35) & (y3 >= -15) & (y3 <= 45) & (z3 >= -3.0) & (z3 <= 4.0)
x3, y3, z3, i3 = x3[mask_3d], y3[mask_3d], z3[mask_3d], i3[mask_3d]

fig3d = plt.figure(figsize=(16, 12), facecolor="#0e1117")
ax3d = fig3d.add_subplot(1, 1, 1, projection="3d", facecolor="#05070a")
fig3d.suptitle(f"AutoBus 3D LiDAR Point Cloud Perspective View\nScene: {selected_samples[0][0]} | Sample: {os.path.basename(sample_1_sd['filename'])}", 
               fontsize=15, color="#f0f6fc", fontweight="bold")

sc3d = ax3d.scatter(x3, y3, z3, c=z3, cmap="turbo", s=1.2, alpha=0.6, vmin=-2.5, vmax=3.0)
ax3d.set_xlabel("Lateral X (m)", color="#8b949e", labelpad=10)
ax3d.set_ylabel("Longitudinal Y (m)", color="#8b949e", labelpad=10)
ax3d.set_zlabel("Height Z (m)", color="#8b949e", labelpad=10)
ax3d.tick_params(colors="#8b949e")
ax3d.view_init(elev=32, azim=-55)
ax3d.set_xlim(-35, 35)
ax3d.set_ylim(-15, 45)
ax3d.set_zlim(-3, 4)
# Style 3d panes
ax3d.xaxis.pane.fill = False
ax3d.yaxis.pane.fill = False
ax3d.zaxis.pane.fill = False
ax3d.xaxis.pane.set_edgecolor("#21262d")
ax3d.yaxis.pane.set_edgecolor("#21262d")
ax3d.zaxis.pane.set_edgecolor("#21262d")
ax3d.grid(True, color="#21262d", linestyle=":", alpha=0.5)

cb3d = plt.colorbar(sc3d, ax=ax3d, pad=0.05, shrink=0.65)
cb3d.set_label("Height Z (meters)", color="#8b949e")
cb3d.ax.tick_params(colors="#8b949e")

out_3d_name = "raw_lidar_3d_perspective.png"
out_3d_path = os.path.join(output_dir, out_3d_name)
art_3d_path = os.path.join(artifact_dir, out_3d_name)
plt.savefig(out_3d_path, dpi=180, facecolor=fig3d.get_facecolor(), edgecolor="none")
plt.savefig(art_3d_path, dpi=180, facecolor=fig3d.get_facecolor(), edgecolor="none")
plt.close(fig3d)
print(f"Saved 3D visualization: {out_3d_name}")

# Save JSON metadata of Stage 2 verification
summary_data = {
    "total_samples_verified": len(sample_files),
    "corrupted_files_count": len(corrupted_files),
    "nan_files_count": len(nan_files),
    "inf_files_count": len(inf_files),
    "empty_files_count": len(empty_files),
    "point_count_stats": {
        "min": int(min(point_counts)),
        "max": int(max(point_counts)),
        "mean": float(np.mean(point_counts)),
        "median": float(np.median(point_counts)),
        "std": float(np.std(point_counts)),
    },
    "coordinate_bounds": {
        "x": [float(min_coords[0]), float(max_coords[0])],
        "y": [float(min_coords[1]), float(max_coords[1])],
        "z": [float(min_coords[2]), float(max_coords[2])],
    },
    "lidar_extrinsics": {
        "translation": lidar_cal["translation"],
        "rotation": lidar_cal["rotation"]
    },
    "visualizations_generated": [item[0] for item in saved_visualizations] + [out_3d_name]
}

with open(os.path.join(output_dir, "stage2_inspection_summary.json"), "w") as f:
    json.dump(summary_data, f, indent=2)

print("\nSTAGE 2 INSPECTION SCRIPT FINISHED SUCCESSFULLY.")
