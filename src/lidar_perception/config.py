"""
AutoBus LiDAR Perception Configuration
CPU-Optimized 3D Object Detection Pipeline
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class LidarConfig:
    # Dataset Paths
    dataset_root: str = r"datasets\nuscenes"
    meta_dir: str = r"datasets\nuscenes\v1.0-mini"
    split_file: str = r"datasets\splits\controlled_404_split.json"
    cache_dir: str = r"data\processed\lidar_bev_cache"
    
    # Point Cloud Filtering
    min_distance: float = 1.0  # Remove ego vehicle roof/bracket self-reflections (r < 1.0m)
    max_distance: float = 50.0 # Maximum operational sensor distance
    
    # 3D Region of Interest (ROI) in Sensor Coordinates (meters)
    # [X_min, X_max]: Lateral coverage (left to right)
    # [Y_min, Y_max]: Longitudinal coverage (rear to front)
    # [Z_min, Z_max]: Vertical coverage (road plane up into overhead)
    x_range: Tuple[float, float] = (-32.0, 32.0)   # 64.0m span (lateral)
    y_range: Tuple[float, float] = (-16.0, 48.0)   # 64.0m span (16m rear, 48m forward)
    z_range: Tuple[float, float] = (-2.8, 1.2)     # 4.0m span (ground at ~-1.84m)
    
    # BEV Grid Resolution
    voxel_size_x: float = 0.25  # meters per pixel along X
    voxel_size_y: float = 0.25  # meters per pixel along Y
    
    # Target Classes
    # Class mapping: CAR (0), PEDESTRIAN (1)
    target_classes: List[str] = field(default_factory=lambda: ["CAR", "PEDESTRIAN"])
    min_lidar_points_per_box: int = 3 # Filter unobservable annotations (<3 points)
    
    @property
    def num_classes(self) -> int:
        return len(self.target_classes)
    
    # Derived grid dimensions
    @property
    def grid_size_x(self) -> int:
        return int(round((self.x_range[1] - self.x_range[0]) / self.voxel_size_x)) # 256
        
    @property
    def grid_size_y(self) -> int:
        return int(round((self.y_range[1] - self.y_range[0]) / self.voxel_size_y)) # 256
        
    @property
    def num_bev_channels(self) -> int:
        # Channels:
        # 0: Max Height (Z_max)
        # 1: Mean Height (Z_mean)
        # 2: Height Spread (Z_max - Z_min)
        # 3: Log Density (log(1 + count))
        # 4: Max Intensity
        # 5: Mean Intensity
        return 6

DEFAULT_CONFIG = LidarConfig()
