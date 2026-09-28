"""
AutoBus Stage 11 — Experiment: Metric Box / Center-Distance NMS
Evaluates adding 3D metric non-maximum suppression (suppressing duplicate peaks within 2.0m)
to address multi-peak split boxes on long vehicles.
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src.lidar_perception.config import DEFAULT_CONFIG
from src.lidar_perception.dataset import AutoBusLidarDataset
from src.lidar_perception.model import TinyCenterNet3D, decode_predictions
from scratch.run_stage11_evaluator import calculate_bev_iou

device = torch.device("cpu")
config = DEFAULT_CONFIG

def apply_metric_nms(boxes, dist_thresh=2.2, iou_thresh=0.20):
    """
    boxes: (N, 9) [x, y, z, w, l, h, yaw, score, class_id]
    """
    if len(boxes) <= 1:
        return boxes
        
    # Sort descending by confidence score
    order = np.argsort(-boxes[:, 7])
    sorted_boxes = boxes[order]
    
    keep = []
    suppressed = np.zeros(len(sorted_boxes), dtype=bool)
    
    for i in range(len(sorted_boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        b_i = sorted_boxes[i]
        
        for j in range(i + 1, len(sorted_boxes)):
            if suppressed[j]:
                continue
            b_j = sorted_boxes[j]
            
            # Same class check
            if int(b_i[8]) == int(b_j[8]):
                # Center distance
                d = np.hypot(b_i[0] - b_j[0], b_i[1] - b_j[1])
                iou = calculate_bev_iou(b_i, b_j)
                
                # If centers are within dist_thresh or significant IoU, suppress lower score duplicate
                if d < dist_thresh or iou > iou_thresh:
                    suppressed[j] = True
                    
    return sorted_boxes[keep]

def evaluate_with_metric_nms(model, dataset, score_thresh=0.40, use_metric_nms=True):
    total_gts = 0
    total_preds = 0
    total_tps = 0
    cls_stats = {0: {"gt": 0, "pred": 0, "tp": 0}, 1: {"gt": 0, "pred": 0, "tp": 0}}
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            item = dataset[idx]
            bev_tensor = item["bev_map"].unsqueeze(0).to(device)
            gt_boxes = item["boxes_sensor"].numpy()
            
            preds = model(bev_tensor)
            decoded = decode_predictions(preds, config=config, score_thresh=score_thresh, max_objects=50)[0]
            
            if use_metric_nms:
                decoded = apply_metric_nms(decoded, dist_thresh=2.2, iou_thresh=0.20)
                
            for gt in gt_boxes:
                c_id = int(gt[7])
                if c_id in cls_stats:
                    cls_stats[c_id]["gt"] += 1
            total_gts += len(gt_boxes)
            
            for p in decoded:
                c_id = int(p[8])
                if c_id in cls_stats:
                    cls_stats[c_id]["pred"] += 1
            total_preds += len(decoded)
            
            matched_gt = set()
            matched_preds = set()
            
            for p_idx, pred in enumerate(decoded):
                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, gt in enumerate(gt_boxes):
                    iou = calculate_bev_iou(pred, gt)
                    dist_diff = np.hypot(pred[0] - gt[0], pred[1] - gt[1])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx
                    elif best_iou == 0.0 and dist_diff <= 1.5:
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
                        
            total_tps += len(matched_gt)
            
    p = total_tps / max(total_preds, 1)
    r = total_tps / max(total_gts, 1)
    f1 = 2 * p * r / max(p + r, 1e-6)
    
    car_p = cls_stats[0]["tp"] / max(cls_stats[0]["pred"], 1)
    car_r = cls_stats[0]["tp"] / max(cls_stats[0]["gt"], 1)
    car_f1 = 2 * car_p * car_r / max(car_p + car_r, 1e-6)
    
    ped_p = cls_stats[1]["tp"] / max(cls_stats[1]["pred"], 1)
    ped_r = cls_stats[1]["tp"] / max(cls_stats[1]["gt"], 1)
    ped_f1 = 2 * ped_p * ped_r / max(ped_p + ped_r, 1e-6)
    
    fp = total_preds - total_tps
    fn = total_gts - total_tps
    
    return {
        "recall": r, "precision": p, "f1": f1,
        "car_f1": car_f1, "ped_f1": ped_f1,
        "car_p": car_p, "car_r": car_r,
        "ped_p": ped_p, "ped_r": ped_r,
        "tp": total_tps, "fp": fp, "fn": fn,
        "total_gt": total_gts, "total_pred": total_preds
    }

# Test on Exp 3 model at th=0.40
ckpt_path = r"models\checkpoints_exp3_aug\best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
model = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=24).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_dataset = AutoBusLidarDataset(split="VALIDATION", config=config, use_cache=True)
test_dataset = AutoBusLidarDataset(split="TEST", config=config, use_cache=True)

print("Evaluating EXP 3 WITHOUT Metric NMS (score_thresh=0.40):")
val_raw = evaluate_with_metric_nms(model, val_dataset, score_thresh=0.40, use_metric_nms=False)
test_raw = evaluate_with_metric_nms(model, test_dataset, score_thresh=0.40, use_metric_nms=False)
print(f"Val F1:  {val_raw['f1']*100:.2f}% (P: {val_raw['precision']*100:.2f}%, R: {val_raw['recall']*100:.2f}%, FP: {val_raw['fp']})")
print(f"Test F1: {test_raw['f1']*100:.2f}% (P: {test_raw['precision']*100:.2f}%, R: {test_raw['recall']*100:.2f}%, FP: {test_raw['fp']})")

print("\nEvaluating EXP 3 WITH Metric NMS (dist_thresh=2.2m, score_thresh=0.40):")
val_nms = evaluate_with_metric_nms(model, val_dataset, score_thresh=0.40, use_metric_nms=True)
test_nms = evaluate_with_metric_nms(model, test_dataset, score_thresh=0.40, use_metric_nms=True)
print(f"Val F1:  {val_nms['f1']*100:.2f}% (P: {val_nms['precision']*100:.2f}%, R: {val_nms['recall']*100:.2f}%, FP: {val_nms['fp']})")
print(f"Test F1: {test_nms['f1']*100:.2f}% (P: {test_nms['precision']*100:.2f}%, R: {test_nms['recall']*100:.2f}%, FP: {test_nms['fp']})")

res_dict = {
    "without_metric_nms": {"val": val_raw, "test": test_raw},
    "with_metric_nms": {"val": val_nms, "test": test_nms}
}
with open("scratch/exp5_metric_nms_results.json", "w") as f:
    json.dump(res_dict, f, indent=2)
print("Saved NMS results to scratch/exp5_metric_nms_results.json")
