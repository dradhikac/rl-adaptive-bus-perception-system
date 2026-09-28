"""
AutoBus LiDAR PyTorch Dataset & DataLoader Interface
Reproducible, CPU-optimized data feeding pipeline.
"""

import os
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List, Any, Optional
from .config import LidarConfig, DEFAULT_CONFIG
from .preprocessor import LidarPreprocessor

class AutoBusLidarDataset(Dataset):
    def __init__(
        self, 
        split: str = "TRAIN", 
        config: LidarConfig = DEFAULT_CONFIG,
        use_cache: bool = True,
        augment: bool = False
    ):
        """
        PyTorch Dataset for AutoBus LiDAR 3D Perception.
        split: "TRAIN", "VALIDATION", or "TEST"
        """
        self.split = split.upper()
        self.config = config
        self.use_cache = use_cache
        self.augment = augment and (self.split == "TRAIN")
        
        # Load controlled split metadata
        if not os.path.exists(config.split_file):
            raise FileNotFoundError(f"Split file not found: {config.split_file}")
            
        with open(config.split_file, "r", encoding="utf-8") as f:
            split_data = json.load(f)
            
        all_samples = split_data.get("sample_index", [])
        self.samples = [s for s in all_samples if s["split"].upper() == self.split]
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}' in {config.split_file}")
            
        # Initialize preprocessor
        self.preprocessor = LidarPreprocessor(config)
        
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        sample_token = item["sample_token"]
        
        processed = self.preprocessor.process_sample(sample_token, use_cache=self.use_cache)
        
        # Convert to PyTorch tensors
        # bev_map: (6, H, W)
        bev_tensor = torch.from_numpy(processed["bev_map"].copy()).float()
        boxes_sensor_tensor = torch.from_numpy(processed["boxes_sensor"].copy()).float()
        boxes_grid_tensor = torch.from_numpy(processed["boxes_grid"].copy()).float()
        
        # Stochastic bilateral horizontal flip augmentation (p=0.5)
        if self.augment and (torch.rand(1).item() < 0.5):
            # 1. Flip BEV map along width dimension (dim 2)
            bev_tensor = torch.flip(bev_tensor, dims=[2])
            
            # 2. Flip boxes if any exist
            if len(boxes_sensor_tensor) > 0:
                # Invert metric X in sensor frame
                boxes_sensor_tensor[:, 0] = -boxes_sensor_tensor[:, 0]
                
                # Invert discrete X in BEV grid continuous frame
                boxes_grid_tensor[:, 0] = float(self.config.grid_size_x - 1) - boxes_grid_tensor[:, 0]
                
                # Invert heading yaw: theta' = atan2(sin(theta), -cos(theta))
                yaw_s = boxes_sensor_tensor[:, 6]
                new_yaw_s = torch.atan2(torch.sin(yaw_s), -torch.cos(yaw_s))
                boxes_sensor_tensor[:, 6] = new_yaw_s
                
                yaw_g = boxes_grid_tensor[:, 6]
                new_yaw_g = torch.atan2(torch.sin(yaw_g), -torch.cos(yaw_g))
                boxes_grid_tensor[:, 6] = new_yaw_g
        
        return {
            "sample_token": sample_token,
            "scene_name": item["scene_name"],
            "timestamp": item["timestamp"],
            "bev_map": bev_tensor,
            "boxes_sensor": boxes_sensor_tensor,
            "boxes_grid": boxes_grid_tensor,
            "num_boxes": processed["num_boxes"]
        }

def autobus_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for variable number of 3D bounding boxes per frame.
    """
    bev_maps = torch.stack([item["bev_map"] for item in batch], dim=0) # (B, 6, H, W)
    sample_tokens = [item["sample_token"] for item in batch]
    scene_names = [item["scene_name"] for item in batch]
    timestamps = [item["timestamp"] for item in batch]
    num_boxes = [item["num_boxes"] for item in batch]
    
    # Store boxes as list of tensors per sample
    boxes_sensor_list = [item["boxes_sensor"] for item in batch]
    boxes_grid_list = [item["boxes_grid"] for item in batch]
    
    # Also create batched boxes with batch index column: [batch_idx, x, y, z, w, l, h, yaw, class_id]
    batched_boxes_sensor = []
    batched_boxes_grid = []
    for b_idx, item in enumerate(batch):
        b_sensor = item["boxes_sensor"]
        b_grid = item["boxes_grid"]
        if len(b_sensor) > 0:
            idx_col = torch.full((len(b_sensor), 1), b_idx, dtype=torch.float32)
            batched_boxes_sensor.append(torch.cat([idx_col, b_sensor], dim=1))
            batched_boxes_grid.append(torch.cat([idx_col, b_grid], dim=1))
            
    if batched_boxes_sensor:
        batched_boxes_sensor = torch.cat(batched_boxes_sensor, dim=0)
        batched_boxes_grid = torch.cat(batched_boxes_grid, dim=0)
    else:
        batched_boxes_sensor = torch.zeros((0, 9), dtype=torch.float32)
        batched_boxes_grid = torch.zeros((0, 9), dtype=torch.float32)
        
    return {
        "bev_maps": bev_maps,
        "sample_tokens": sample_tokens,
        "scene_names": scene_names,
        "timestamps": timestamps,
        "num_boxes": num_boxes,
        "boxes_sensor_list": boxes_sensor_list,
        "boxes_grid_list": boxes_grid_list,
        "batched_boxes_sensor": batched_boxes_sensor,
        "batched_boxes_grid": batched_boxes_grid
    }
