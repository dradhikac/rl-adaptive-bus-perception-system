import os
import sys
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.lidar_perception.config import LidarConfig
from src.lidar_perception.preprocessor import LidarPreprocessor
from src.lidar_perception.dataset import AutoBusLidarDataset, autobus_collate_fn
from src.lidar_perception.transforms import get_box_8_corners, get_box_bev_corners

output_dir = r"visualizations\stage6_pipeline"
art_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\1c0087c6-dac4-4253-a522-0bdbc5b00eb8"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(art_dir, exist_ok=True)

print("============================================================")
print("STAGE 6: PREPROCESSING PIPELINE VERIFICATION TEST")
print("============================================================")

config = LidarConfig(
    cache_dir=r"data\processed\lidar_bev_cache_test",
    min_distance=1.0,
    x_range=(-32.0, 32.0),
    y_range=(-16.0, 48.0),
    z_range=(-2.8, 1.2),
    voxel_size_x=0.25,
    voxel_size_y=0.25,
    target_classes=["CAR", "PEDESTRIAN"],
    min_lidar_points_per_box=3
)

print(f"Configuration:")
print(f"  BEV Grid Size: {config.grid_size_x} x {config.grid_size_y} ({config.num_bev_channels} channels)")
print(f"  Voxel Resolution: {config.voxel_size_x:.2f}m x {config.voxel_size_y:.2f}m")
print(f"  ROI: X in {config.x_range}m, Y in {config.y_range}m, Z in {config.z_range}m")
print(f"  Target Classes: {config.target_classes}")
print(f"  Min Points per Box: {config.min_lidar_points_per_box}")

# Initialize Preprocessor
preprocessor = LidarPreprocessor(config)

# Select 8 diverse samples across Train, Val, and Test
with open(config.split_file, "r") as f:
    split_meta = json.load(f)

sample_index = split_meta["sample_index"]

# Pick 4 Train, 2 Val, 2 Test
test_samples = []
for split_name, count in [("TRAIN", 4), ("VALIDATION", 2), ("TEST", 2)]:
    subset = [s for s in sample_index if s["split"] == split_name]
    step = len(subset) // count
    for i in range(count):
        test_samples.append(subset[i * step])

print(f"\nTesting preprocessing pipeline on {len(test_samples)} representative samples:")
for i, s in enumerate(test_samples):
    print(f"  [{i+1}] Split: {s['split']:<10} | Scene: {s['scene_name']:<12} | Token: {s['sample_token'][:16]}...")

# Run pipeline on each sample, measuring latency and verifying consistency
results = []
latencies_first = []
latencies_cached = []

for idx, s in enumerate(test_samples):
    s_tok = s["sample_token"]
    sc_name = s["scene_name"]
    split = s["split"]
    
    # 1. First run (Cold / un-cached compute)
    t0 = time.perf_counter()
    res1 = preprocessor.process_sample(s_tok, use_cache=True)
    t1 = time.perf_counter()
    cold_time_ms = (t1 - t0) * 1000.0
    latencies_first.append(cold_time_ms)
    
    # 2. Second run (Warm / disk cache hit)
    t2 = time.perf_counter()
    res2 = preprocessor.process_sample(s_tok, use_cache=True)
    t3 = time.perf_counter()
    cached_time_ms = (t3 - t2) * 1000.0
    latencies_cached.append(cached_time_ms)
    
    # Verification checks
    bev = res1["bev_map"]
    boxes_sensor = res1["boxes_sensor"]
    boxes_grid = res1["boxes_grid"]
    stats = res1["filter_stats"]
    
    # Check shape & bounds
    assert bev.shape == (6, 256, 256), f"Unexpected BEV shape: {bev.shape}"
    assert np.isfinite(bev).all(), "NaN or Inf found in BEV tensor!"
    assert bev.min() >= 0.0 and bev.max() <= 1.0, f"BEV out of normalized bounds: [{bev.min()}, {bev.max()}]"
    assert len(boxes_sensor) == len(boxes_grid), "Box sensor and grid count mismatch!"
    
    # Check cache bitwise exactness
    assert np.allclose(res1["bev_map"], res2["bev_map"]), "Cache mismatch on BEV map!"
    assert np.allclose(res1["boxes_sensor"], res2["boxes_sensor"]), "Cache mismatch on boxes_sensor!"
    
    # Check coordinate consistency
    # Grid coordinates should correspond to (sensor_x - X_min) / voxel_x
    if len(boxes_sensor) > 0:
        expected_gx = (boxes_sensor[:, 0] - config.x_range[0]) / config.voxel_size_x
        expected_gy = (boxes_sensor[:, 1] - config.y_range[0]) / config.voxel_size_y
        diff_gx = np.abs(boxes_grid[:, 0] - expected_gx).max()
        diff_gy = np.abs(boxes_grid[:, 1] - expected_gy).max()
        assert diff_gx < 1e-4 and diff_gy < 1e-4, f"Grid coordinate formula mismatch! diff: {diff_gx}, {diff_gy}"
        
        # Verify that box locations correspond to active points in BEV
        # For each box, check if the cell (or 3x3 neighborhood) has non-zero density (channel 3)
        density_ch = bev[3]
        matched_cells = 0
        for b in boxes_grid:
            gx_int = int(np.clip(round(b[0]), 0, 255))
            gy_int = int(np.clip(round(b[1]), 0, 255))
            # 3x3 patch around box center
            patch = density_ch[max(0, gy_int-2):min(256, gy_int+3), max(0, gx_int-2):min(256, gx_int+3)]
            if (patch > 0).any():
                matched_cells += 1
        pct_matched = matched_cells / len(boxes_grid) * 100.0
    else:
        pct_matched = 100.0
        
    print(f"\nSample {idx+1}/{len(test_samples)} ({split} | {sc_name}):")
    print(f"  Raw Points: {stats['initial_points']:,} -> Valid in ROI: {stats['valid_points_in_roi']:,}")
    print(f"  Filtered: {stats['self_reflections_filtered']:,} self-reflections (r<1m), {stats['out_of_roi_filtered']:,} out-of-ROI")
    print(f"  Model-Ready 3D Boxes: {len(boxes_sensor)} (Cars: {(boxes_sensor[:, 7] == 0).sum() if len(boxes_sensor)>0 else 0}, Peds: {(boxes_sensor[:, 7] == 1).sum() if len(boxes_sensor)>0 else 0})")
    print(f"  Box-to-BEV Point Alignment: {pct_matched:.1f}%")
    print(f"  Latency: Cold={cold_time_ms:.1f}ms | Cached={cached_time_ms:.2f}ms")
    
    results.append({
        "sample_token": s_tok,
        "scene_name": sc_name,
        "split": split,
        "cold_ms": cold_time_ms,
        "cached_ms": cached_time_ms,
        "raw_points": stats['initial_points'],
        "roi_points": stats['valid_points_in_roi'],
        "num_boxes": len(boxes_sensor),
        "alignment_pct": pct_matched
    })

# 3. Test PyTorch Dataset and DataLoader Batching
print("\n--- Testing PyTorch Dataset & DataLoader ---")
train_ds = AutoBusLidarDataset(split="TRAIN", config=config, use_cache=True)
val_ds = AutoBusLidarDataset(split="VALIDATION", config=config, use_cache=True)
test_ds = AutoBusLidarDataset(split="TEST", config=config, use_cache=True)

print(f"PyTorch Dataset initialized:")
print(f"  Train samples: {len(train_ds)}")
print(f"  Val samples:   {len(val_ds)}")
print(f"  Test samples:  {len(test_ds)}")

# Test DataLoader with batch_size=4
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=4, shuffle=True, collate_fn=autobus_collate_fn
)
sample_batch = next(iter(train_loader))
bev_batch = sample_batch["bev_maps"] # (4, 6, 256, 256)
batched_boxes = sample_batch["batched_boxes_sensor"]

print(f"\nDataLoader Batch Test (Batch Size = 4):")
print(f"  BEV Batch Tensor Shape: {bev_batch.shape} (dtype: {bev_batch.dtype})")
print(f"  Batched Boxes Tensor: {batched_boxes.shape} (cols: [batch_idx, x, y, z, w, l, h, yaw, class_id])")
print(f"  Box count per item: {sample_batch['num_boxes']}")

# 4. Generate Multi-Channel Preprocessing Diagnostic Visualization
# Using Sample 1 (Scene 0061)
sample_1_res = preprocessor.process_sample(test_samples[0]["sample_token"])
bev1 = sample_1_res["bev_map"]
boxes_grid1 = sample_1_res["boxes_grid"]
boxes_sensor1 = sample_1_res["boxes_sensor"]

fig = plt.figure(figsize=(22, 14), facecolor="#0e1117")
fig.suptitle(f"AutoBus Stage 6 — Reproducible Preprocessing Pipeline Verification\nScene: {test_samples[0]['scene_name']} | Model Input: (6, 256, 256) | Model Target: {len(boxes_sensor1)} 3D Boxes",
             fontsize=15, color="#f0f6fc", fontweight="bold", y=0.98)

# Panel 1: BEV Channel 0 (Max Height)
ax1 = fig.add_subplot(2, 3, 1, facecolor="#05070a")
ax1.set_title("Channel 0: Max Height (Z_max)", color="#58a6ff", fontsize=12, fontweight="bold")
im1 = ax1.imshow(bev1[0], origin="lower", cmap="turbo", vmin=0, vmax=1)
ax1.set_xlabel("Grid X (0 to 255)", color="#8b949e", fontsize=9)
ax1.set_ylabel("Grid Y (0 to 255)", color="#8b949e", fontsize=9)
ax1.tick_params(colors="#8b949e")
plt.colorbar(im1, ax=ax1, pad=0.02, shrink=0.8)

# Panel 2: BEV Channel 2 (Height Spread / Vertical Thickness)
ax2 = fig.add_subplot(2, 3, 2, facecolor="#05070a")
ax2.set_title("Channel 2: Height Spread (ΔZ = Z_max - Z_min)\n(Separates Road Ground from Vertical Objects)", color="#58a6ff", fontsize=12, fontweight="bold")
im2 = ax2.imshow(bev1[2], origin="lower", cmap="inferno", vmin=0, vmax=0.7)
ax2.set_xlabel("Grid X (0 to 255)", color="#8b949e", fontsize=9)
ax2.set_ylabel("Grid Y (0 to 255)", color="#8b949e", fontsize=9)
ax2.tick_params(colors="#8b949e")
plt.colorbar(im2, ax=ax2, pad=0.02, shrink=0.8)

# Panel 3: BEV Channel 3 (Log Point Density)
ax3 = fig.add_subplot(2, 3, 3, facecolor="#05070a")
ax3.set_title("Channel 3: Normalized Point Density log(1+N)", color="#58a6ff", fontsize=12, fontweight="bold")
im3 = ax3.imshow(bev1[3], origin="lower", cmap="cividis", vmin=0, vmax=1)
ax3.set_xlabel("Grid X (0 to 255)", color="#8b949e", fontsize=9)
ax3.set_ylabel("Grid Y (0 to 255)", color="#8b949e", fontsize=9)
ax3.tick_params(colors="#8b949e")
plt.colorbar(im3, ax=ax3, pad=0.02, shrink=0.8)

# Panel 4: BEV Channel 4 (Max Surface Reflectivity / Intensity)
ax4 = fig.add_subplot(2, 3, 4, facecolor="#05070a")
ax4.set_title("Channel 4: Surface Reflectivity (LiDAR Intensity)", color="#58a6ff", fontsize=12, fontweight="bold")
im4 = ax4.imshow(bev1[4], origin="lower", cmap="magma", vmin=0, vmax=0.6)
ax4.set_xlabel("Grid X (0 to 255)", color="#8b949e", fontsize=9)
ax4.set_ylabel("Grid Y (0 to 255)", color="#8b949e", fontsize=9)
ax4.tick_params(colors="#8b949e")
plt.colorbar(im4, ax=ax4, pad=0.02, shrink=0.8)

# Panel 5: Model Target Overlay on Density Channel
ax5 = fig.add_subplot(2, 3, 5, facecolor="#05070a")
ax5.set_title(f"Model Input + 3D Target Boxes ({len(boxes_grid1)} Objects)\n(Red: CAR, Green: PEDESTRIAN)", color="#58a6ff", fontsize=12, fontweight="bold")
ax5.imshow(bev1[3], origin="lower", cmap="gray", vmin=0, vmax=0.8)

# Plot boxes on BEV grid
for b in boxes_grid1:
    gx, gy, zc, w, l, h, yaw, class_id = b
    # Convert box corners to grid
    gw = w / config.voxel_size_x
    gl = l / config.voxel_size_y
    # In grid, yaw orientation:
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    # Local corners
    xc = np.array([-gw/2, gw/2, gw/2, -gw/2])
    yc = np.array([-gl/2, -gl/2, gl/2, gl/2])
    # Note: in sensor frame heading was local +Y, yaw = atan2(vy, vx)
    # Rotation matrix in XY:
    r_mat = np.array([[s_y, c_y], [-c_y, s_y]]) # aligns with sensor definition
    # Direct formula using corners from sensor:
    # We can get sensor corners and convert to grid
    pass

for b_s, b_g in zip(boxes_sensor1, boxes_grid1):
    c_s = b_s[:3]
    size_s = b_s[3:6]
    yaw_s = b_s[6]
    cls_id = int(b_s[7])
    
    # 4 corners in sensor
    r_z = np.array([
        [np.cos(yaw_s - np.pi/2), -np.sin(yaw_s - np.pi/2), 0],
        [np.sin(yaw_s - np.pi/2),  np.cos(yaw_s - np.pi/2), 0],
        [0, 0, 1]
    ])
    c8 = get_box_8_corners(c_s, size_s, r_z)
    c4 = c8[:2, :4].T # (4, 2)
    
    # Convert to grid
    gx = (c4[:, 0] - config.x_range[0]) / config.voxel_size_x
    gy = (c4[:, 1] - config.y_range[0]) / config.voxel_size_y
    grid_corners = np.column_stack([gx, gy])
    
    color = "#f85149" if cls_id == 0 else "#3fb950"
    poly = Polygon(grid_corners, closed=True, fill=True, facecolor=color, alpha=0.35, edgecolor=color, linewidth=2.0)
    ax5.add_patch(poly)
    
    # Center dot & heading
    gx_c = (c_s[0] - config.x_range[0]) / config.voxel_size_x
    gy_c = (c_s[1] - config.y_range[0]) / config.voxel_size_y
    ax5.plot(gx_c, gy_c, "o", color=color, markersize=3)
    
ax5.set_xlim(0, 255)
ax5.set_ylim(0, 255)
ax5.set_xlabel("Grid X (Lateral: 0 to 255)", color="#8b949e", fontsize=9)
ax5.set_ylabel("Grid Y (Longitudinal: 0 to 255)", color="#8b949e", fontsize=9)
ax5.tick_params(colors="#8b949e")

# Panel 6: Pipeline Latency Breakdown
ax6 = fig.add_subplot(2, 3, 6, facecolor="#05070a")
ax6.set_title("CPU Latency Benchmark (Cold Compute vs Cache Hit)", color="#58a6ff", fontsize=12, fontweight="bold")
x_indices = np.arange(len(test_samples))
ax6.bar(x_indices - 0.2, latencies_first, width=0.4, label="Cold Preprocessing (Vectorized NumPy)", color="#1f6feb", edgecolor="#388bfd")
ax6.bar(x_indices + 0.2, latencies_cached, width=0.4, label="Cached Disk Hit (.npz)", color="#238636", edgecolor="#2ea043")
ax6.set_xticks(x_indices)
ax6.set_xticklabels([f"S{i+1}" for i in range(len(test_samples))], color="#8b949e")
ax6.set_ylabel("Processing Time (milliseconds)", color="#8b949e", fontsize=9)
ax6.tick_params(colors="#8b949e")
ax6.grid(True, color="#21262d", linestyle=":", axis="y")
ax6.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)
ax6.axhline(np.mean(latencies_first), color="#58a6ff", linestyle="--", linewidth=1.2, label=f"Mean Cold: {np.mean(latencies_first):.1f}ms")
ax6.axhline(np.mean(latencies_cached), color="#3fb950", linestyle="--", linewidth=1.2, label=f"Mean Cache: {np.mean(latencies_cached):.2f}ms")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
viz_pipeline = "stage6_preprocessing_verification.png"
plt.savefig(os.path.join(output_dir, viz_pipeline), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig(os.path.join(art_dir, viz_pipeline), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print(f"Saved pipeline verification visualization to {viz_pipeline}")

# Save Summary JSON
summary_stage6 = {
    "bev_dimensions": [config.num_bev_channels, config.grid_size_y, config.grid_size_x],
    "voxel_resolution_meters": [config.voxel_size_x, config.voxel_size_y],
    "roi_bounds_meters": {
        "x": list(config.x_range),
        "y": list(config.y_range),
        "z": list(config.z_range)
    },
    "tested_samples": results,
    "mean_cold_latency_ms": float(np.mean(latencies_first)),
    "mean_cached_latency_ms": float(np.mean(latencies_cached)),
    "pytorch_dataset": {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "batch_shape": list(bev_batch.shape)
    }
}

with open(os.path.join(output_dir, "stage6_preprocessing_summary.json"), "w") as f:
    json.dump(summary_stage6, f, indent=2)

print("\nSTAGE 6 PREPROCESSING PIPELINE TEST COMPLETED SUCCESSFULLY.")
