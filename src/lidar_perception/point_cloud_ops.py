"""
AutoBus LiDAR Point Cloud Operations & BEV Feature Generation
High-performance, vectorized CPU preprocessing.
"""

import os
import numpy as np
from typing import Tuple, Dict, Any
from .config import LidarConfig, DEFAULT_CONFIG

def load_point_cloud(file_path: str) -> np.ndarray:
    """
    Load raw binary LiDAR point cloud from disk.
    nuScenes format: float32, [x, y, z, intensity, ring_index] (5 floats = 20 bytes/point)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"LiDAR file not found: {file_path}")
        
    data = np.fromfile(file_path, dtype=np.float32)
    if len(data) % 5 == 0:
        pts = data.reshape(-1, 5)
    elif len(data) % 4 == 0:
        pts = data.reshape(-1, 4)
    else:
        pts = data.reshape(-1, 3)
    return pts

def validate_and_filter_points(
    pts: np.ndarray, 
    config: LidarConfig = DEFAULT_CONFIG
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Validate and clean raw LiDAR points:
    1. Filter non-finite points (NaN, Inf)
    2. Remove ego-vehicle self-reflections (r < min_distance, typically 1.0m)
    3. Filter points outside 3D Region of Interest (ROI)
    
    Returns:
      filtered_points: (M, 5) array of valid environmental points
      stats: dictionary with filtering counts
    """
    initial_count = len(pts)
    
    # 1. Non-finite check
    valid_finite = np.isfinite(pts[:, :3]).all(axis=1)
    pts = pts[valid_finite]
    nan_count = initial_count - len(pts)
    
    # 2. Self-reflection filter (r < min_distance)
    # Reflects off vehicle roof rack, mounting frame, and optical housing
    r_xy = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    non_self_mask = r_xy >= config.min_distance
    pts = pts[non_self_mask]
    self_reflection_count = len(valid_finite) - len(pts) - nan_count
    
    # 3. Spatial 3D Bounding Box ROI filter
    roi_mask = (
        (pts[:, 0] >= config.x_range[0]) & (pts[:, 0] <= config.x_range[1]) &
        (pts[:, 1] >= config.y_range[0]) & (pts[:, 1] <= config.y_range[1]) &
        (pts[:, 2] >= config.z_range[0]) & (pts[:, 2] <= config.z_range[1])
    )
    pts_roi = pts[roi_mask]
    out_of_roi_count = len(pts) - len(pts_roi)
    
    stats = {
        "initial_points": initial_count,
        "nan_points": nan_count,
        "self_reflections_filtered": self_reflection_count,
        "out_of_roi_filtered": out_of_roi_count,
        "valid_points_in_roi": len(pts_roi)
    }
    
    return pts_roi, stats

def generate_bev_pseudo_image(
    pts: np.ndarray, 
    config: LidarConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """
    Convert 3D point cloud into a 6-channel Bird's-Eye-View (BEV) pseudo-image.
    Output shape: (6, H, W) where H = grid_size_y, W = grid_size_x.
    
    Channels:
      Channel 0: Normalized Maximum Height (Z_max)
      Channel 1: Normalized Mean Height (Z_mean)
      Channel 2: Height Spread (Z_max - Z_min)
      Channel 3: Log Point Density log(1 + count) / log(1 + max_expected)
      Channel 4: Normalized Maximum Intensity
      Channel 5: Normalized Mean Intensity
    """
    H = config.grid_size_y # 256
    W = config.grid_size_x # 256
    bev = np.zeros((config.num_bev_channels, H, W), dtype=np.float32)
    
    if len(pts) == 0:
        return bev
        
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    intensity = pts[:, 3] if pts.shape[1] >= 4 else np.zeros_like(x)
    
    # Compute discrete grid indices
    # gx in [0, W-1], gy in [0, H-1]
    gx = np.floor((x - config.x_range[0]) / config.voxel_size_x).astype(np.int32)
    gy = np.floor((y - config.y_range[0]) / config.voxel_size_y).astype(np.int32)
    
    # Clip to valid boundaries
    gx = np.clip(gx, 0, W - 1)
    gy = np.clip(gy, 0, H - 1)
    
    # 1D flattened cell index: flat_idx = gy * W + gx
    flat_idx = gy * W + gx
    
    # Sort points by flat_idx for high-speed segment aggregation
    sort_order = np.argsort(flat_idx)
    flat_idx = flat_idx[sort_order]
    z = z[sort_order]
    intensity = intensity[sort_order]
    
    # Find unique non-empty cells and their boundaries
    unique_cells, split_indices, counts = np.unique(flat_idx, return_index=True, return_counts=True)
    
    # Vectorized reduction per cell
    # Segment boundaries: [split_indices[i] : split_indices[i] + counts[i]]
    # Using np.maximum.reduceat and np.add.reduceat for extreme speed
    z_max = np.maximum.reduceat(z, split_indices)
    z_min = np.minimum.reduceat(z, split_indices)
    z_sum = np.add.reduceat(z, split_indices)
    z_mean = z_sum / counts
    
    i_max = np.maximum.reduceat(intensity, split_indices)
    i_sum = np.add.reduceat(intensity, split_indices)
    i_mean = i_sum / counts
    
    # Normalize features
    z_span = config.z_range[1] - config.z_range[0]
    z_max_norm = np.clip((z_max - config.z_range[0]) / z_span, 0.0, 1.0)
    z_mean_norm = np.clip((z_mean - config.z_range[0]) / z_span, 0.0, 1.0)
    z_diff_norm = np.clip((z_max - z_min) / z_span, 0.0, 1.0)
    
    # Density normalized (saturating at count=64)
    density_norm = np.clip(np.log1p(counts) / np.log1p(64.0), 0.0, 1.0)
    
    # Intensity normalized (scale 0-255)
    i_max_norm = np.clip(i_max / 255.0, 0.0, 1.0)
    i_mean_norm = np.clip(i_mean / 255.0, 0.0, 1.0)
    
    # Populate BEV channels using flat indices
    u_gy = unique_cells // W
    u_gx = unique_cells % W
    
    bev[0, u_gy, u_gx] = z_max_norm
    bev[1, u_gy, u_gx] = z_mean_norm
    bev[2, u_gy, u_gx] = z_diff_norm
    bev[3, u_gy, u_gx] = density_norm
    bev[4, u_gy, u_gx] = i_max_norm
    bev[5, u_gy, u_gx] = i_mean_norm
    
    return bev
