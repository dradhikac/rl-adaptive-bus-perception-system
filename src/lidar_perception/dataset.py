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
        use_cache: bool = True
    ):
        """
        PyTorch Dataset for AutoBus LiDAR 3D Perception.
        split: "TRAIN", "VALIDATION", or "TEST"
        """
        self.split = split.upper()
        self.config = config
        self.use_cache = use_cache
        
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
        bev_tensor = torch.from_numpy(processed["bev_map"]).float()
        boxes_sensor_tensor = torch.from_numpy(processed["boxes_sensor"]).float()
        boxes_grid_tensor = torch.from_numpy(processed["boxes_grid"]).float()
        
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
