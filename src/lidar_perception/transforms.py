"""
AutoBus Coordinate Transformations & Geometry Utilities
Ensures 100% mathematical consistency across Global, Ego, Sensor, and BEV Grid frames.
"""

import numpy as np
from typing import Tuple, List, Dict, Any

def quat_to_rot_mat(q: List[float]) -> np.ndarray:
    """
    Convert a quaternion [w, x, y, z] to a 3x3 orthonormal rotation matrix.
    """
    w, x, y, z = q
    # Normalize to avoid numerical drift
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm > 0:
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        
    return np.array([
        [1.0 - 2.0*y*y - 2.0*z*z, 2.0*x*y - 2.0*z*w,       2.0*x*z + 2.0*y*w],
        [2.0*x*y + 2.0*z*w,       1.0 - 2.0*x*x - 2.0*z*z, 2.0*y*z - 2.0*x*w],
        [2.0*x*z - 2.0*y*w,       2.0*y*z + 2.0*x*w,       1.0 - 2.0*x*x - 2.0*y*y]
    ], dtype=np.float32)

def global_to_sensor(
    p_global: np.ndarray,
    r_box_global: np.ndarray,
    r_ego: np.ndarray,
    t_ego: np.ndarray,
    r_sensor: np.ndarray,
    t_sensor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform 3D position and orientation from Global UTM frame to LiDAR sensor frame:
      p_ego = R_ego^T * (p_global - t_ego)
      p_sensor = R_sensor^T * (p_ego - t_sensor)
      R_box_sensor = R_sensor^T * R_ego^T * R_box_global
    """
    p_ego = r_ego.T @ (p_global - t_ego)
    p_sensor = r_sensor.T @ (p_ego - t_sensor)
    r_box_sensor = r_sensor.T @ r_ego.T @ r_box_global
    return p_sensor, r_box_sensor

def extract_yaw(rot_mat: np.ndarray) -> float:
    """
    Extract the yaw heading angle (radians) in the sensor XY plane.
    In nuScenes box convention, the object heading points along local +Y:
      v_heading = rot_mat @ [0, 1, 0]^T
      yaw = atan2(v_y, v_x)
    """
    v_heading = rot_mat @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return float(np.arctan2(v_heading[1], v_heading[0]))

def get_box_8_corners(center: np.ndarray, size: List[float], rot_mat: np.ndarray) -> np.ndarray:
    """
    Compute 8 3D corners of a bounding box in sensor frame.
    center: [x, y, z]
    size: [w, l, h] (width along local X, length along local Y, height along local Z)
    rot_mat: 3x3 orientation matrix
    Returns: (3, 8) numpy array of 3D corner coordinates
    """
    w, l, h = size
    # 8 local corners
    x_c = [-w/2,  w/2,  w/2, -w/2, -w/2,  w/2,  w/2, -w/2]
    y_c = [-l/2, -l/2,  l/2,  l/2, -l/2, -l/2,  l/2,  l/2]
    z_c = [-h/2, -h/2, -h/2, -h/2,  h/2,  h/2,  h/2,  h/2]
    local_corners = np.vstack([x_c, y_c, z_c]) # (3, 8)
    return rot_mat @ local_corners + np.array(center, dtype=np.float32)[:, None]

def get_box_bev_corners(center: np.ndarray, size: List[float], rot_mat: np.ndarray) -> np.ndarray:
    """
    Compute 4 2D corners of a bounding box on the BEV plane.
    Returns: (4, 2) numpy array [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    """
    c8 = get_box_8_corners(center, size, rot_mat)
    # Bottom 4 corners: 0, 1, 2, 3
    return c8[:2, :4].T

def sensor_to_bev_grid_coords(
    x: float, 
    y: float, 
    x_range: Tuple[float, float], 
    y_range: Tuple[float, float], 
    voxel_x: float, 
    voxel_y: float
) -> Tuple[float, float]:
    """
    Convert physical metric sensor coordinates (x, y) in meters
    to continuous pixel coordinates (gx, gy) in the BEV grid.
    """
    gx = (x - x_range[0]) / voxel_x
    gy = (y - y_range[0]) / voxel_y
    return float(gx), float(gy)
