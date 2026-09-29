"""
AutoBus LiDAR Perception — Downstream Interface Contract Specification
========================================================================================
Defines the formal data contracts, type annotations, and transition schemas
connecting the standalone LiDAR 3D object detector to future autonomous subsystems:

LiDAR Detection
       ↓
  3D Tracking (Kalman / EKF / Hungarian Matching)
       ↓
Distance / Velocity Estimation (Numerical differentiation & state estimation)
       ↓
Blind Spot Analysis (Spatial zone classification relative to bus body)
       ↓
Risk Assessment (TTC, closing velocity, clearance violation)
       ↓
Camera + LiDAR Fusion (Cross-modal geometric alignment & IoU matching)
       ↓
RL Decision Engine (State vector encoding)
       ↓
Decision Fusion (MOVE / SLOW_DOWN / STOP / CHANGE_LEFT / CHANGE_RIGHT)

IMPORTANT ARCHITECTURAL RULE:
Do NOT prematurely integrate these modules until the standalone LiDAR detector is locked.
This module defines only the formal contracts and data schemas.
========================================================================================
"""

from typing import TypedDict, List, Dict, Optional, Literal, Tuple
from dataclasses import dataclass, field
import math
import numpy as np

# --------------------------------------------------------------------------------------
# 1. STANDALONE LIDAR 3D DETECTION SCHEMA (STAGE 12 OUTPUT)
# --------------------------------------------------------------------------------------

class Center3D(TypedDict):
    x: float  # Lateral position (+X = Right, -X = Left) in meters
    y: float  # Longitudinal position (+Y = Forward, -Y = Behind) in meters
    z: float  # Vertical position (+Z = Up, road at ~ -1.84m) in meters

class Dimensions3D(TypedDict):
    length: float  # Longitudinal extent (front-to-back, along yaw) in meters
    width: float   # Lateral extent (side-to-side, perpendicular to yaw) in meters
    height: float  # Vertical extent (ground-to-roof) in meters

class LidarDetection3D(TypedDict):
    """
    Standard output record from AutoBusLidar3DDetector.
    """
    class_name: Literal["car", "pedestrian"] # Class name (0: car, 1: pedestrian)
    confidence: float                        # Heatmap peak probability [0.0, 1.0]
    center: Center3D                         # Center coordinates (meters)
    dimensions: Dimensions3D                 # Metric box dimensions (meters)
    yaw: float                               # Heading in radians [-pi, +pi] CCW from +X
    distance: float                          # 3D Euclidean range from sensor origin (m)

# --------------------------------------------------------------------------------------
# 2. FUTURE SUBSYSTEM INTERFACE SCHEMAS (DOWNSTREAM TRANSITIONS)
# --------------------------------------------------------------------------------------

class Velocity3D(TypedDict):
    vx: float  # Lateral relative velocity (m/s)
    vy: float  # Longitudinal relative velocity (m/s)
    vz: float  # Vertical relative velocity (m/s)

class TrackedLidarObstacle3D(TypedDict):
    """
    Target schema for future '3D Tracking' & 'Distance / Velocity' modules.
    State vector: x_k = [x, y, z, vx, vy, vz, yaw, w, l, h]^T
    """
    track_id: int                            # Persistent identifier across frames
    class_name: Literal["car", "pedestrian"] # Filtered class label
    confidence: float                        # Track confidence score
    center: Center3D                         # Smoothed 3D center
    dimensions: Dimensions3D                 # Smoothed dimensions
    yaw: float                               # Velocity-aligned yaw (resolves 180° ambiguity)
    distance: float                          # Metric range
    velocity: Velocity3D                     # Relative velocity vector (m/s)
    speed_kmh: float                         # Absolute speed magnitude (km/h)
    age_frames: int                          # Total frames tracked
    time_since_update_sec: float             # Time elapsed since last LiDAR measurement

class BlindSpotZone(TypedDict):
    """
    Target schema for future 'Blind Spot Analysis' module.
    Classifies obstacle relative to the AutoBus physical footprint:
    Bus dimensions: Width ~ 2.5m (X in [-1.25, +1.25]), Length ~ 10.5m (Y in [-8.5, +2.0])
    """
    zone_name: Literal[
        "FRONT_CLEARANCE",       # Y in [2.0, 30.0], X in [-1.5, +1.5]
        "LEFT_BLIND_SPOT",       # Y in [-8.0, 2.0], X in [-4.0, -1.25]
        "RIGHT_BLIND_SPOT",      # Y in [-8.0, 2.0], X in [+1.25, +4.0]
        "REAR_TAIL_COLLISION",   # Y in [-16.0, -8.0], X in [-2.0, +2.0]
        "PERIPHERY"              # Outside critical proximity corridors
    ]
    lateral_clearance_m: float   # Distance to bus side body
    longitudinal_clearance_m: float # Distance to bus front/rear bumpers
    is_in_blind_spot: bool       # True if in left or right blind spot

class RiskAssessmentFeatures(TypedDict):
    """
    Target schema for future 'Risk Assessment' module.
    """
    obstacle_id: int
    time_to_collision_sec: Optional[float] # TTC = distance / closing_velocity (None if opening)
    closing_velocity_mps: float            # Positive if closing in on bus
    predicted_path_crosses_bus: bool       # Trajectory intersection flag
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    risk_score: float                      # Normalized risk in [0.0, 1.0]

class CameraLidarFusedObstacle(TypedDict):
    """
    Target schema for future 'Camera + LiDAR Fusion' module.
    """
    fusion_id: int
    camera_detected: bool
    lidar_detected: bool
    camera_bbox_2d: Optional[List[float]] # [ymin, xmin, ymax, xmax] in camera frame
    lidar_bbox_3d: LidarDetection3D       # Exact 3D bounding box
    cross_modal_iou: float                # 2D projection IoU overlap
    sensor_agreement: float               # Confidence agreement metric

class RLPerceptionState(TypedDict):
    """
    Target schema for future 'RL Decision Engine' input state vector.
    """
    closest_front_vehicle_dist: float
    closest_front_vehicle_ttc: float
    left_blind_spot_occupied: bool
    right_blind_spot_occupied: bool
    pedestrian_crosswalk_warning: bool
    min_overall_distance: float
    recommended_action_mask: List[int] # [MOVE, SLOW_DOWN, STOP, CHANGE_LEFT, CHANGE_RIGHT]
