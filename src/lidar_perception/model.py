"""
TinyCenterNet3D Model Architecture & 3D Bounding Box Decoder
Ultra-lightweight, CPU-optimized CenterPoint-style 3D LiDAR Detector.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from .config import LidarConfig, DEFAULT_CONFIG

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class TinyCenterNet3D(nn.Module):
    """
    Lightweight CPU 3D Object Detector.
    Operates on 6-channel BEV pseudo-images.
    Outputs:
      - heatmap: (B, num_classes, 64, 64)
      - offset:  (B, 2, 64, 64)
      - z_elev:  (B, 1, 64, 64)
      - dim_3d:  (B, 3, 64, 64)
      - rot_3d:  (B, 2, 64, 64)
    """
    def __init__(self, in_channels=6, num_classes=2, base_channels=24):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        C = base_channels # 24
        
        # 1. Stem (256x256)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        
        # 2. Residual Stages
        self.stage1 = ResidualBlock(C, C, stride=1)      # 256x256
        self.stage2 = ResidualBlock(C, C*2, stride=2)    # 128x128
        self.stage3 = ResidualBlock(C*2, C*4, stride=2)  # 64x64
        self.stage4 = ResidualBlock(C*4, C*4, stride=1)  # 64x64
        
        # 3. Lightweight Neck (48 channels at 64x64)
        self.neck = nn.Sequential(
            nn.Conv2d(C*4, C*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C*2),
            nn.ReLU(inplace=True)
        )
        
        head_in = C*2 # 48
        
        # 4. Multi-Task Detection Heads
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, num_classes, kernel_size=1)
        )
        # Focal loss prior initialization (bias = -2.19)
        self.heatmap_head[-1].bias.data.fill_(-2.19)
        
        self.offset_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, 2, kernel_size=1)
        )
        
        self.z_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, 1, kernel_size=1)
        )
        
        self.dim_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, 3, kernel_size=1)
        )
        
        self.rot_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, 2, kernel_size=1)
        )

    def forward(self, x):
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.neck(feat)
        
        return {
            "heatmap": self.heatmap_head(feat),
            "offset": self.offset_head(feat),
            "z_elev": self.z_head(feat),
            "dim_3d": self.dim_head(feat),
            "rot_3d": self.rot_head(feat)
        }

def decode_predictions(
    preds: Dict[str, torch.Tensor],
    config: LidarConfig = DEFAULT_CONFIG,
    score_thresh: float = 0.25,
    max_objects: int = 50,
    downsample_stride: int = 4
) -> List[np.ndarray]:
    """
    Decodes CenterPoint model predictions into 3D bounding boxes in LiDAR sensor coordinates.
    Returns: List of arrays (one per sample in batch), where each array has shape (N, 9):
      [x, y, z, w, l, h, yaw, score, class_id]
    """
    heatmap = torch.sigmoid(preds["heatmap"]) # (B, num_classes, H', W')
    offset = preds["offset"]                  # (B, 2, H', W')
    z_elev = preds["z_elev"]                  # (B, 1, H', W')
    dim_3d = torch.exp(preds["dim_3d"])       # (B, 3, H', W')
    rot_3d = preds["rot_3d"]                  # (B, 2, H', W')
    
    # 3x3 max pooling for non-maximum suppression (NMS) on heatmap
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (heatmap == hmax) & (heatmap >= score_thresh)
    
    B, num_classes, H_p, W_p = heatmap.shape
    batch_boxes = []
    
    for b in range(B):
        boxes = []
        for cls_id in range(num_classes):
            mask = keep[b, cls_id]
            if not mask.any():
                continue
                
            scores = heatmap[b, cls_id][mask]
            y_inds, x_inds = torch.where(mask)
            
            # Sort by score and keep top max_objects
            if len(scores) > max_objects:
                top_scores, top_idx = torch.topk(scores, max_objects)
                scores = top_scores
                y_inds = y_inds[top_idx]
                x_inds = x_inds[top_idx]
                
            for i in range(len(scores)):
                ix = x_inds[i].item()
                iy = y_inds[i].item()
                score = scores[i].item()
                
                # Sub-pixel grid position
                dx = offset[b, 0, iy, ix].item()
                dy = offset[b, 1, iy, ix].item()
                gx_p = ix + dx
                gy_p = iy + dy
                
                # Convert from downsampled grid to full BEV grid
                gx = gx_p * downsample_stride
                gy = gy_p * downsample_stride
                
                # Convert from BEV grid to sensor metric coordinates (x, y)
                x = gx * config.voxel_size_x + config.x_range[0]
                y = gy * config.voxel_size_y + config.y_range[0]
                z = z_elev[b, 0, iy, ix].item()
                
                w = dim_3d[b, 0, iy, ix].item()
                l = dim_3d[b, 1, iy, ix].item()
                h = dim_3d[b, 2, iy, ix].item()
                
                # Yaw from [sin(yaw), cos(yaw)]
                sin_yaw = rot_3d[b, 0, iy, ix].item()
                cos_yaw = rot_3d[b, 1, iy, ix].item()
                yaw = math.atan2(sin_yaw, cos_yaw)
                
                boxes.append([x, y, z, w, l, h, yaw, score, float(cls_id)])
                
        if boxes:
            batch_boxes.append(np.array(boxes, dtype=np.float32))
        else:
            batch_boxes.append(np.zeros((0, 9), dtype=np.float32))
            
    return batch_boxes
