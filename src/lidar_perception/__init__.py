"""
AutoBus LiDAR 3D Perception Package
"""

from .config import LidarConfig, DEFAULT_CONFIG
from .preprocessor import LidarPreprocessor
from .dataset import AutoBusLidarDataset, autobus_collate_fn
from .transforms import (
    quat_to_rot_mat, 
    global_to_sensor, 
    extract_yaw, 
    get_box_8_corners, 
    get_box_bev_corners,
    sensor_to_bev_grid_coords
)
from .point_cloud_ops import (
    load_point_cloud,
    validate_and_filter_points,
    generate_bev_pseudo_image
)
from .detector import AutoBusLidar3DDetector
from .decision_arbiter import arbitrate_lidar_decision

__all__ = [
    "LidarConfig",
    "DEFAULT_CONFIG",
    "LidarPreprocessor",
    "AutoBusLidarDataset",
    "autobus_collate_fn",
    "quat_to_rot_mat",
    "global_to_sensor",
    "extract_yaw",
    "get_box_8_corners",
    "get_box_bev_corners",
    "sensor_to_bev_grid_coords",
    "load_point_cloud",
    "validate_and_filter_points",
    "generate_bev_pseudo_image",
    "AutoBusLidar3DDetector",
    "arbitrate_lidar_decision"
]
