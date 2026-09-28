"""
CenterNet 3D Loss Functions & Gaussian Target Generator
Implements Modified Focal Loss for heatmaps and masked L1 losses for 3D bounding box regression.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    
    x, y = int(center[0]), int(center[1])
    H, W = heatmap.shape[0], heatmap.shape[1]
    
    left, right = min(x, radius), min(W - x, radius + 1)
    top, bottom = min(y, radius), min(H - y, radius + 1)
    
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian_radius(det_size, min_overlap=0.5):
    height, width = det_size
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

class ModifiedFocalLoss(nn.Module):
    """
    Focal loss for heatmaps (CenterNet / CornerNet).
    Penalty-reduced pixelwise focal loss.
    """
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        pred: (B, C, H, W) raw logits or sigmoid probabilities
        gt:   (B, C, H, W) Gaussian heatmap ground truth in [0, 1]
        """
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1.0 - 1e-4)
        
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

class CenterNet3DLoss(nn.Module):
    """
    Complete Multi-Task Loss for 3D Object Detection.
    Combines Heatmap Focal Loss + Masked L1 Losses for Offset, Z, Size, and Yaw.
    """
    def __init__(
        self,
        downsample_stride=4,
        hm_weight=1.0,
        off_weight=1.0,
        z_weight=1.0,
        size_weight=0.5,
        rot_weight=0.5
    ):
        super().__init__()
        self.downsample_stride = downsample_stride
        self.hm_weight = hm_weight
        self.off_weight = off_weight
        self.z_weight = z_weight
        self.size_weight = size_weight
        self.rot_weight = rot_weight
        
        self.focal_loss = ModifiedFocalLoss()
        self.l1_loss = nn.L1Loss(reduction="sum")

    def forward(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]):
        """
        preds: output dictionary from TinyCenterNet3D:
          - heatmap: (B, num_classes, H', W')
          - offset:  (B, 2, H', W')
          - z_elev:  (B, 1, H', W')
          - dim_3d:  (B, 3, H', W')
          - rot_3d:  (B, 2, H', W')
        batch: dictionary from AutoBusLidarDataset / collate_fn containing:
          - boxes_grid_list: list of (K, 8) tensors [gx, gy, zc, w, l, h, yaw, class_id]
        """
        device = preds["heatmap"].device
        B, num_classes, H_p, W_p = preds["heatmap"].shape
        
        # Build ground-truth heatmap and regression target masks on downsampled grid
        gt_heatmap = np.zeros((B, num_classes, H_p, W_p), dtype=np.float32)
        
        mask = torch.zeros((B, H_p, W_p), dtype=torch.bool, device=device)
        gt_offset = torch.zeros((B, 2, H_p, W_p), dtype=torch.float32, device=device)
        gt_z = torch.zeros((B, 1, H_p, W_p), dtype=torch.float32, device=device)
        gt_dim = torch.zeros((B, 3, H_p, W_p), dtype=torch.float32, device=device)
        gt_rot = torch.zeros((B, 2, H_p, W_p), dtype=torch.float32, device=device)
        
        total_targets = 0
        
        boxes_list = batch["boxes_grid_list"]
        for b_idx in range(B):
            boxes = boxes_list[b_idx]
            if len(boxes) == 0:
                continue
                
            for box in boxes:
                gx, gy, zc, w, l, h, yaw, class_id = box.tolist()
                cls_id = int(class_id)
                if cls_id >= num_classes or cls_id < 0:
                    continue
                    
                # Downsample grid coords by stride (4x)
                gx_p = gx / self.downsample_stride
                gy_p = gy / self.downsample_stride
                
                ix = int(math.floor(gx_p))
                iy = int(math.floor(gy_p))
                
                if 0 <= ix < W_p and 0 <= iy < H_p:
                    # Draw Gaussian on heatmap
                    # Object size in downsampled grid pixels
                    gw_p = (w / 0.25) / self.downsample_stride
                    gl_p = (l / 0.25) / self.downsample_stride
                    radius = max(1, int(gaussian_radius((gl_p, gw_p), min_overlap=0.5)))
                    draw_umich_gaussian(gt_heatmap[b_idx, cls_id], (ix, iy), radius)
                    
                    # Regression targets at center location
                    mask[b_idx, iy, ix] = True
                    gt_offset[b_idx, 0, iy, ix] = gx_p - ix
                    gt_offset[b_idx, 1, iy, ix] = gy_p - iy
                    gt_z[b_idx, 0, iy, ix] = zc
                    gt_dim[b_idx, 0, iy, ix] = math.log(max(w, 0.1))
                    gt_dim[b_idx, 1, iy, ix] = math.log(max(l, 0.1))
                    gt_dim[b_idx, 2, iy, ix] = math.log(max(h, 0.1))
                    gt_rot[b_idx, 0, iy, ix] = math.sin(yaw)
                    gt_rot[b_idx, 1, iy, ix] = math.cos(yaw)
                    total_targets += 1

        gt_heatmap_tensor = torch.from_numpy(gt_heatmap).to(device)
        
        # 1. Heatmap loss
        loss_hm = self.focal_loss(preds["heatmap"], gt_heatmap_tensor)
        
        # 2. Regression losses (masked to object centers)
        num_pos = max(float(total_targets), 1.0)
        
        if total_targets > 0:
            pred_off = preds["offset"].permute(0, 2, 3, 1)[mask]
            targ_off = gt_offset.permute(0, 2, 3, 1)[mask]
            loss_off = self.l1_loss(pred_off, targ_off) / num_pos
            
            pred_z = preds["z_elev"].permute(0, 2, 3, 1)[mask]
            targ_z = gt_z.permute(0, 2, 3, 1)[mask]
            loss_z = self.l1_loss(pred_z, targ_z) / num_pos
            
            pred_dim = preds["dim_3d"].permute(0, 2, 3, 1)[mask]
            targ_dim = gt_dim.permute(0, 2, 3, 1)[mask]
            loss_dim = self.l1_loss(pred_dim, targ_dim) / num_pos
            
            pred_rot = preds["rot_3d"].permute(0, 2, 3, 1)[mask]
            targ_rot = gt_rot.permute(0, 2, 3, 1)[mask]
            loss_rot = self.l1_loss(pred_rot, targ_rot) / num_pos
        else:
            loss_off = torch.tensor(0.0, device=device)
            loss_z = torch.tensor(0.0, device=device)
            loss_dim = torch.tensor(0.0, device=device)
            loss_rot = torch.tensor(0.0, device=device)
            
        total_loss = (
            self.hm_weight * loss_hm +
            self.off_weight * loss_off +
            self.z_weight * loss_z +
            self.size_weight * loss_dim +
            self.rot_weight * loss_rot
        )
        
        return {
            "total_loss": total_loss,
            "loss_hm": loss_hm,
            "loss_off": loss_off,
            "loss_z": loss_z,
            "loss_dim": loss_dim,
            "loss_rot": loss_rot,
            "num_targets": total_targets
        }
