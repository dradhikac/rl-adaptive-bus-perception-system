"""
AutoBus Stage 11 — Standardized Experiment Evaluator
Evaluates a model checkpoint and configuration on BOTH Validation and Test splits.
Computes:
- Val Loss (from criterion)
- Precision, Recall, F1 (Overall, Car, Pedestrian) on Validation Set
- Precision, Recall, F1 (Overall, Car, Pedestrian) on Test Set
- Detailed failure counts: FP, FN, class mismatches, heading errors, size errors, distant misses
"""

import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from shapely.geometry import Polygon as ShapelyPolygon

sys.path.insert(0, os.path.abspath("."))

from src.lidar_perception.config import LidarConfig, DEFAULT_CONFIG
from src.lidar_perception.dataset import AutoBusLidarDataset, autobus_collate_fn
from src.lidar_perception.model import TinyCenterNet3D, decode_predictions
from src.lidar_perception.loss import CenterNet3DLoss
from src.lidar_perception.transforms import get_box_8_corners

def box_to_shapely_polygon(box):
    c = box[:3]
    size = box[3:6]
    yaw = box[6]
    r_z = np.array([
        [np.cos(yaw - np.pi/2), -np.sin(yaw - np.pi/2), 0],
        [np.sin(yaw - np.pi/2),  np.cos(yaw - np.pi/2), 0],
        [0, 0, 1]
    ])
    c8 = get_box_8_corners(c, size, r_z)
    c4 = c8[:2, :4].T # (4, 2)
    return ShapelyPolygon(c4)

def calculate_bev_iou(box1, box2):
    try:
        p1 = box_to_shapely_polygon(box1)
        p2 = box_to_shapely_polygon(box2)
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

def evaluate_split(
    model,
    dataset,
    device,
    config,
    score_thresh=0.28,
    max_objects=40,
    nms_kernel=3,
    dist_threshold=1.5
):
    model.eval()
    
    total_gts = 0
    total_preds = 0
    total_tps = 0
    total_fps = 0
    total_fns = 0
    
    # Class-specific metrics
    cls_stats = {
        0: {"gt": 0, "pred": 0, "tp": 0}, # Car
        1: {"gt": 0, "pred": 0, "tp": 0}  # Pedestrian
    }
    
    # Failure diagnostics
    heading_errors = 0
    size_errors = 0
    class_mismatches = 0
    distant_misses = 0
    distant_hits = 0
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            bev_tensor = item["bev_map"].unsqueeze(0).to(device)
            gt_boxes = item["boxes_sensor"].numpy()
            
            # Predict
            preds = model(bev_tensor)
            
            # Decode with custom threshold and nms_kernel
            decoded = decode_predictions(
                preds,
                config=config,
                score_thresh=score_thresh,
                max_objects=max_objects
            )[0]
            
            # Count ground truths per class
            for gt in gt_boxes:
                c_id = int(gt[7])
                if c_id in cls_stats:
                    cls_stats[c_id]["gt"] += 1
            total_gts += len(gt_boxes)
            
            # Count predictions per class
            for p in decoded:
                c_id = int(p[8])
                if c_id in cls_stats:
                    cls_stats[c_id]["pred"] += 1
            total_preds += len(decoded)
            
            # Match
            matched_gt = set()
            matched_preds = set()
            
            for p_idx, pred in enumerate(decoded):
                best_iou = 0.0
                best_gt_idx = -1
                p_dist = np.hypot(pred[0], pred[1])
                
                for g_idx, gt in enumerate(gt_boxes):
                    iou = calculate_bev_iou(pred, gt)
                    dist_diff = np.hypot(pred[0] - gt[0], pred[1] - gt[1])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                    elif best_iou == 0.0 and dist_diff <= dist_threshold:
                        best_iou = 0.15
                        best_gt_idx = g_idx
                        
                if best_gt_idx >= 0 and best_gt_idx not in matched_gt and best_iou >= 0.10:
                    matched_gt.add(best_gt_idx)
                    matched_preds.add(p_idx)
                    gt_matched = gt_boxes[best_gt_idx]
                    
                    p_cls = int(pred[8])
                    g_cls = int(gt_matched[7])
                    
                    if p_cls == g_cls and p_cls in cls_stats:
                        cls_stats[p_cls]["tp"] += 1
                    else:
                        class_mismatches += 1
                        
                    # Yaw error
                    yaw_diff = abs(pred[6] - gt_matched[6])
                    yaw_diff = min(yaw_diff, 2*np.pi - yaw_diff)
                    if np.degrees(yaw_diff) > 45.0:
                        heading_errors += 1
                        
                    # Volume ratio
                    vol_pred = pred[3] * pred[4] * pred[5]
                    vol_gt = gt_matched[3] * gt_matched[4] * gt_matched[5]
                    vol_ratio = vol_pred / max(vol_gt, 1e-3)
                    if vol_ratio < 0.5 or vol_ratio > 2.0:
                        size_errors += 1
                        
                    if p_dist > 25.0:
                        distant_hits += 1
                        
            tp = len(matched_gt)
            fp = len(decoded) - len(matched_preds)
            fn = len(gt_boxes) - len(matched_gt)
            
            total_tps += tp
            total_fps += fp
            total_fns += fn
            
            for g_idx, gt in enumerate(gt_boxes):
                if g_idx not in matched_gt and np.hypot(gt[0], gt[1]) > 25.0:
                    distant_misses += 1
                    
    precision = total_tps / max(total_preds, 1)
    recall = total_tps / max(total_gts, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    
    # Class metrics
    car_p = cls_stats[0]["tp"] / max(cls_stats[0]["pred"], 1)
    car_r = cls_stats[0]["tp"] / max(cls_stats[0]["gt"], 1)
    car_f1 = 2 * car_p * car_r / max(car_p + car_r, 1e-6)
    
    ped_p = cls_stats[1]["tp"] / max(cls_stats[1]["pred"], 1)
    ped_r = cls_stats[1]["tp"] / max(cls_stats[1]["gt"], 1)
    ped_f1 = 2 * ped_p * ped_r / max(ped_p + ped_r, 1e-6)
    
    return {
        "total_gt": total_gts,
        "total_pred": total_preds,
        "tp": total_tps,
        "fp": total_fps,
        "fn": total_fns,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "car_precision": car_p,
        "car_recall": car_r,
        "car_f1": car_f1,
        "ped_precision": ped_p,
        "ped_recall": ped_r,
        "ped_f1": ped_f1,
        "heading_errors": heading_errors,
        "size_errors": size_errors,
        "class_mismatches": class_mismatches,
        "distant_misses": distant_misses,
        "distant_hits": distant_hits
    }

def compute_loss(model, dataset, criterion, device):
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=autobus_collate_fn)
    model.eval()
    running_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            bev_maps = batch["bev_maps"].to(device)
            preds = model(bev_maps)
            loss_dict = criterion(preds, batch)
            running_loss += loss_dict["total_loss"].item()
            num_batches += 1
    return running_loss / max(num_batches, 1)

if __name__ == "__main__":
    device = torch.device("cpu")
    config = DEFAULT_CONFIG
    
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else r"models\checkpoints\best_model.pth"
    score_thresh = float(sys.argv[2]) if len(sys.argv) > 2 else 0.28
    
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    base_ch = checkpoint.get("base_channels", 24)
    model = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=base_ch).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    val_dataset = AutoBusLidarDataset(split="VALIDATION", config=config, use_cache=True)
    test_dataset = AutoBusLidarDataset(split="TEST", config=config, use_cache=True)
    
    criterion = CenterNet3DLoss(downsample_stride=4)
    
    val_loss = compute_loss(model, val_dataset, criterion, device)
    val_metrics = evaluate_split(model, val_dataset, device, config, score_thresh=score_thresh)
    test_metrics = evaluate_split(model, test_dataset, device, config, score_thresh=score_thresh)
    
    res = {
        "checkpoint": ckpt_path,
        "score_thresh": score_thresh,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics
    }
    
    print(f"\n=======================================================")
    print(f"EVALUATION RESULTS (Threshold: {score_thresh})")
    print(f"=======================================================")
    print(f"Validation Set (Scene-0553, {len(val_dataset)} samples):")
    print(f"  Val Loss:      {val_loss:.4f}")
    print(f"  Recall:        {val_metrics['recall']*100:.2f}% ({val_metrics['tp']}/{val_metrics['total_gt']})")
    print(f"  Precision:     {val_metrics['precision']*100:.2f}% ({val_metrics['tp']}/{val_metrics['total_pred']})")
    print(f"  F1-Score:      {val_metrics['f1']*100:.2f}%")
    print(f"  Car F1:        {val_metrics['car_f1']*100:.2f}% | Pedestrian F1: {val_metrics['ped_f1']*100:.2f}%")
    print(f"  FP: {val_metrics['fp']} | FN: {val_metrics['fn']}")
    print(f"-------------------------------------------------------")
    print(f"Test Set (Scenes 1094 & 1100, {len(test_dataset)} samples):")
    print(f"  Recall:        {test_metrics['recall']*100:.2f}% ({test_metrics['tp']}/{test_metrics['total_gt']})")
    print(f"  Precision:     {test_metrics['precision']*100:.2f}% ({test_metrics['tp']}/{test_metrics['total_pred']})")
    print(f"  F1-Score:      {test_metrics['f1']*100:.2f}%")
    print(f"  Car F1:        {test_metrics['car_f1']*100:.2f}% | Pedestrian F1: {test_metrics['ped_f1']*100:.2f}%")
    print(f"  FP: {test_metrics['fp']} | FN: {test_metrics['fn']}")
    print(f"=======================================================\n")
    
    out_json = f"scratch/eval_result_{os.path.basename(ckpt_path).replace('.pth','')}_th{int(score_thresh*100)}.json"
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved eval results to {out_json}")
