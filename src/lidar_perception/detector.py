"""
AutoBus LiDAR 3D Object Detector Inference Interface
Clean, Production-Ready Prediction Pipeline for AutoBus Autonomous Transit System.

COORDINATE SYSTEM DEFINITION (Empirically Validated from Dataset):
========================================================================================
Reference Frame: Top LiDAR Sensor Frame (`LIDAR_TOP`)
Origin (0, 0, 0): Center of LiDAR optical sensor on vehicle roof rack.
Road Surface:     z ≈ -1.84 meters below the sensor origin.

Axes:
  • +X : LATERAL axis, pointing to the RIGHT of the vehicle.
         (-X is to the LEFT).
  • +Y : LONGITUDINAL axis, pointing FORWARD in the vehicle's direction of travel.
         (-Y is BEHIND the vehicle).
  • +Z : VERTICAL axis, pointing UPWARD toward the sky.
         (-Z is DOWNWARD toward the road surface).

Orientation (Yaw):
  • Measured in radians within [-π, +π], counter-clockwise (CCW) rotation around +Z axis.
  • yaw =  0.000 rad : Facing along +X (Right)
  • yaw = +1.571 rad (+π/2) : Facing along +Y (Directly Forward)
  • yaw = ±3.142 rad (±π)   : Facing along -X (Left)
  • yaw = -1.571 rad (-π/2) : Facing along -Y (Directly Backward)

Dimensions:
  • length : Metric size along the object's local longitudinal axis (front-to-back).
  • width  : Metric size along the object's local lateral axis (side-to-side).
  • height : Metric size along the object's vertical axis (ground-to-roof).

Distance:
  • Metric 3D Euclidean distance from the LiDAR sensor origin: sqrt(x^2 + y^2 + z^2).
========================================================================================
"""

import os
import math
import json
from typing import Union, List, Dict, Any, Optional
import numpy as np
import torch

from .config import LidarConfig, DEFAULT_CONFIG
from .model import TinyCenterNet3D, decode_predictions
from .point_cloud_ops import (
    load_point_cloud,
    validate_and_filter_points,
    generate_bev_pseudo_image
)

class AutoBusLidar3DDetector:
    """
    High-level, production-ready 3D LiDAR Object Detector for the AutoBus Perception System.
    Executes native CPU inference on raw LiDAR point clouds and outputs structured 3D bounding boxes.
    """
    
    # Class ID to human-readable label mapping
    CLASS_NAMES = {
        0: "car",
        1: "pedestrian"
    }
    
    def __init__(
        self,
        checkpoint_path: str = r"models/checkpoints_exp3_aug/best_model.pth",
        config: LidarConfig = DEFAULT_CONFIG,
        device: str = "cpu",
        score_thresh: float = 0.40,
        max_objects: int = 50
    ):
        """
        Initialize detector with model weights and configuration.
        
        Args:
            checkpoint_path: Path to PyTorch model weights (.pth)
            config: LidarConfig instance specifying spatial range and voxel resolution
            device: Execution device ('cpu')
            score_thresh: Confidence score threshold for detections (default: 0.40, optimal from Stage 11)
            max_objects: Maximum number of objects decoded per scene
        """
        self.device = torch.device(device)
        self.config = config
        self.score_thresh = score_thresh
        self.max_objects = max_objects
        
        # Verify checkpoint
        if not os.path.exists(checkpoint_path):
            # Fallback to baseline checkpoint if augmented is unavailable
            alt_path = r"models/checkpoints/best_model.pth"
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
            else:
                raise FileNotFoundError(f"Detector checkpoint not found at {checkpoint_path}")
                
        self.checkpoint_path = checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Instantiate architecture
        base_channels = checkpoint.get("base_channels", 24)
        self.model = TinyCenterNet3D(
            in_channels=config.num_bev_channels,
            num_classes=config.num_classes,
            base_channels=base_channels
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
    def predict(
        self,
        point_cloud: Union[str, np.ndarray],
        score_thresh: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate structured 3D bounding box predictions for a single LiDAR point cloud.
        
        Args:
            point_cloud: Either a filepath to a binary .pcd.bin / .bin LiDAR file,
                         OR a NumPy array of shape (N, 3), (N, 4), or (N, 5) with [x, y, z, ...].
            score_thresh: Optional override for the detection confidence threshold.
            
        Returns:
            A list of structured prediction dictionaries:
            [
              {
                "class": "car",
                "confidence": 0.87,
                "center": {
                  "x": 4.12,
                  "y": 12.45,
                  "z": -0.85
                },
                "dimensions": {
                  "length": 4.52,
                  "width": 1.95,
                  "height": 1.58
                },
                "yaw": 1.54,
                "distance": 13.14
              }, ...
            ]
        """
        thresh = score_thresh if score_thresh is not None else self.score_thresh
        
        # 1. Load point cloud if filepath provided
        if isinstance(point_cloud, str):
            pts = load_point_cloud(point_cloud)
        elif isinstance(point_cloud, np.ndarray):
            pts = point_cloud
        else:
            raise TypeError(f"Expected filepath (str) or NumPy array, got {type(point_cloud)}")
            
        # 2. Preprocess: validate points, filter self-reflections and out-of-ROI points
        pts_filtered, _ = validate_and_filter_points(pts, config=self.config)
        
        # 3. Generate 6-channel BEV pseudo-image
        bev_image = generate_bev_pseudo_image(pts_filtered, config=self.config)
        
        # 4. Prepare tensor input (1, C, H, W)
        bev_tensor = torch.from_numpy(bev_image).unsqueeze(0).to(self.device)
        
        # 5. Model forward pass
        with torch.no_grad():
            preds = self.model(bev_tensor)
            
            # 6. Decode 3D bounding boxes
            # decoded array: (K, 9) [x, y, z, w, l, h, yaw, score, class_id]
            decoded_boxes = decode_predictions(
                preds,
                config=self.config,
                score_thresh=thresh,
                max_objects=self.max_objects
            )[0]
            
        # 7. Format into structured output schema
        structured_predictions = []
        for box in decoded_boxes:
            x, y, z, w, l, h, yaw, score, class_id = box.tolist()
            
            class_name = self.CLASS_NAMES.get(int(class_id), f"class_{int(class_id)}")
            dist_3d = math.sqrt(x**2 + y**2 + z**2)
            
            structured_predictions.append({
                "class": class_name,
                "confidence": round(float(score), 4),
                "center": {
                    "x": round(float(x), 2),
                    "y": round(float(y), 2),
                    "z": round(float(z), 2)
                },
                "dimensions": {
                    "length": round(float(l), 2),
                    "width": round(float(w), 2),
                    "height": round(float(h), 2)
                },
                "yaw": round(float(yaw), 4),
                "distance": round(float(dist_3d), 2)
            })
            
        # Sort by distance (nearest critical objects first)
        structured_predictions.sort(key=lambda item: item["distance"])
        return structured_predictions

    def predict_json(
        self,
        point_cloud: Union[str, np.ndarray],
        score_thresh: Optional[float] = None,
        indent: int = 2
    ) -> str:
        """
        Generate structured predictions and return them as a formatted JSON string.
        """
        preds = self.predict(point_cloud, score_thresh=score_thresh)
        return json.dumps(preds, indent=indent)
