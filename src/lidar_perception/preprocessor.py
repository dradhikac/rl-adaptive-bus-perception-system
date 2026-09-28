"""
AutoBus LiDAR Data Preprocessor
Orchestrates raw point cloud loading, filtering, BEV projection, label transformation, and caching.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from .config import LidarConfig, DEFAULT_CONFIG
from .transforms import quat_to_rot_mat, global_to_sensor, extract_yaw, sensor_to_bev_grid_coords
from .point_cloud_ops import load_point_cloud, validate_and_filter_points, generate_bev_pseudo_image

class LidarPreprocessor:
    def __init__(self, config: LidarConfig = DEFAULT_CONFIG):
        self.config = config
        os.makedirs(self.config.cache_dir, exist_ok=True)
        
        # Load nuScenes metadata once
        self._load_metadata()
        
    def _load_metadata(self):
        meta_dir = self.config.meta_dir
        def load_json(name):
            path = os.path.join(meta_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Metadata file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
                
        self.samples = {s["token"]: s for s in load_json("sample.json")}
        self.scenes = {sc["token"]: sc for sc in load_json("scene.json")}
        self.sample_data = load_json("sample_data.json")
        self.calibrated_sensors = {cs["token"]: cs for cs in load_json("calibrated_sensor.json")}
        self.ego_poses = {ep["token"]: ep for ep in load_json("ego_pose.json")}
        self.annotations = load_json("sample_annotation.json")
        self.instances = {inst["token"]: inst["category_token"] for inst in load_json("instance.json")}
        self.categories = {cat["token"]: cat["name"] for cat in load_json("category.json")}
        
        # Build index from sample_token -> annotations
        self.sample_to_anns = {}
        for a in self.annotations:
            s_tok = a["sample_token"]
            if s_tok not in self.sample_to_anns:
                self.sample_to_anns[s_tok] = []
            self.sample_to_anns[s_tok].append(a)
            
        # Build index from sample_token -> LIDAR_TOP sample_data record
        self.sample_to_lidar_sd = {}
        for sd in self.sample_data:
            if "LIDAR_TOP" in sd.get("filename", "") and sd.get("is_key_frame"):
                self.sample_to_lidar_sd[sd["sample_token"]] = sd

    def map_category_to_class_id(self, cat_name: str) -> Optional[int]:
        """
        Map fine-grained nuScenes category to target class ID:
          CAR: 0
          PEDESTRIAN: 1
        Returns None for ignored classes.
        """
        if "vehicle.car" in cat_name:
            if "CAR" in self.config.target_classes:
                return self.config.target_classes.index("CAR")
        elif "human.pedestrian" in cat_name:
            if "PEDESTRIAN" in self.config.target_classes:
                return self.config.target_classes.index("PEDESTRIAN")
        return None

    def process_sample(
        self, 
        sample_token: str, 
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single sample:
        1. Check disk cache if use_cache is True
        2. Load & filter raw LiDAR point cloud
        3. Project into 6-channel BEV pseudo-image
        4. Load & transform ground-truth 3D boxes to sensor coordinates
        5. Filter boxes by ROI and minimum points
        6. Return model-ready dictionary
        """
        cache_path = os.path.join(self.config.cache_dir, f"{sample_token}.npz")
        if use_cache and os.path.exists(cache_path):
            with np.load(cache_path, allow_pickle=True) as data:
                return {
                    "sample_token": str(data["sample_token"]),
                    "bev_map": data["bev_map"],
                    "boxes_sensor": data["boxes_sensor"],
                    "boxes_grid": data["boxes_grid"],
                    "num_boxes": len(data["boxes_sensor"]),
                    "from_cache": True
                }

        # Retrieve LIDAR sample_data record
        sd = self.sample_to_lidar_sd.get(sample_token)
        if not sd:
            raise KeyError(f"No LIDAR_TOP keyframe found for sample token: {sample_token}")
            
        # 1. Load point cloud
        rel_path = sd["filename"].replace("/", os.sep)
        pts_path = os.path.join(self.config.dataset_root, rel_path)
        pts_raw = load_point_cloud(pts_path)
        
        # 2. Filter points & generate BEV pseudo-image
        pts_filtered, filter_stats = validate_and_filter_points(pts_raw, self.config)
        bev_map = generate_bev_pseudo_image(pts_filtered, self.config)
        
        # 3. Coordinate transformation matrices
        cal_sensor = self.calibrated_sensors[sd["calibrated_sensor_token"]]
        ego_pose = self.ego_poses[sd["ego_pose_token"]]
        
        r_sensor = quat_to_rot_mat(cal_sensor["rotation"])
        t_sensor = np.array(cal_sensor["translation"], dtype=np.float32)
        r_ego = quat_to_rot_mat(ego_pose["rotation"])
        t_ego = np.array(ego_pose["translation"], dtype=np.float32)
        
        # 4. Load & transform annotations
        raw_anns = self.sample_to_anns.get(sample_token, [])
        valid_boxes_sensor = []
        valid_boxes_grid = []
        
        for a in raw_anns:
            cat_name = self.categories.get(self.instances.get(a["instance_token"], ""), "unknown")
            class_id = self.map_category_to_class_id(cat_name)
            if class_id is None:
                continue
                
            # Filter unobservable targets (< min_points)
            if a.get("num_lidar_pts", 0) < self.config.min_lidar_points_per_box:
                continue
                
            # Global to sensor transform
            p_global = np.array(a["translation"], dtype=np.float32)
            r_box_global = quat_to_rot_mat(a["rotation"])
            p_sensor, r_box_sensor = global_to_sensor(p_global, r_box_global, r_ego, t_ego, r_sensor, t_sensor)
            
            # Check if center lies inside ROI
            xc, yc, zc = p_sensor
            if not (self.config.x_range[0] <= xc <= self.config.x_range[1] and
                    self.config.y_range[0] <= yc <= self.config.y_range[1] and
                    self.config.z_range[0] <= zc <= self.config.z_range[1]):
                continue
                
            w, l, h = a["size"] # [width, length, height]
            yaw = extract_yaw(r_box_sensor)
            
            # Box in sensor coordinates: [x, y, z, w, l, h, yaw, class_id]
            box_sensor = [xc, yc, zc, w, l, h, yaw, float(class_id)]
            valid_boxes_sensor.append(box_sensor)
            
            # Box in BEV grid continuous coordinates: [gx, gy, z, w, l, h, yaw, class_id]
            gx, gy = sensor_to_bev_grid_coords(
                xc, yc, 
                self.config.x_range, self.config.y_range, 
                self.config.voxel_size_x, self.config.voxel_size_y
            )
            box_grid = [gx, gy, zc, w, l, h, yaw, float(class_id)]
            valid_boxes_grid.append(box_grid)

        boxes_sensor_np = np.array(valid_boxes_sensor, dtype=np.float32) if valid_boxes_sensor else np.zeros((0, 8), dtype=np.float32)
        boxes_grid_np = np.array(valid_boxes_grid, dtype=np.float32) if valid_boxes_grid else np.zeros((0, 8), dtype=np.float32)

        result = {
            "sample_token": sample_token,
            "bev_map": bev_map,
            "boxes_sensor": boxes_sensor_np,
            "boxes_grid": boxes_grid_np,
            "num_boxes": len(boxes_sensor_np),
            "filter_stats": filter_stats,
            "from_cache": False
        }

        # Save to disk cache
        if use_cache:
            np.savez_compressed(
                cache_path,
                sample_token=sample_token,
                bev_map=bev_map,
                boxes_sensor=boxes_sensor_np,
                boxes_grid=boxes_grid_np
            )

        return result
