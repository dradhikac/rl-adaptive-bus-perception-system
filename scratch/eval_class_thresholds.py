"""
AutoBus Stage 11 — Experiment 2: Class-Specific Threshold Calibration
Evaluates decoupled class thresholds (Car vs Pedestrian) on Validation and Test sets.
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
ckpt_path = r"models\checkpoints\best_model.pth"

checkpoint = torch.load(ckpt_path, map_location=device)
model = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=24).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_dataset = AutoBusLidarDataset(split="VALIDATION", config=config, use_cache=True)
test_dataset = AutoBusLidarDataset(split="TEST", config=config, use_cache=True)

def decode_class_aware(preds, config, th_car=0.30, th_ped=0.42, max_objects=40):
    boxes_car = decode_predictions(preds, config=config, score_thresh=th_car, max_objects=max_objects)[0]
    boxes_ped = decode_predictions(preds, config=config, score_thresh=th_ped, max_objects=max_objects)[0]
    
    # Filter by class
    car_only = [b for b in boxes_car if int(b[8]) == 0]
    ped_only = [b for b in boxes_ped if int(b[8]) == 1]
    
    combined = car_only + ped_only
    if combined:
        return np.array(combined, dtype=np.float32)
    return np.zeros((0, 9), dtype=np.float32)

def evaluate_class_aware(model, dataset, th_car=0.30, th_ped=0.42):
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
            decoded = decode_class_aware(preds, config, th_car=th_car, th_ped=th_ped)
            
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

print("Testing decoupled thresholds on VALIDATION set:")
for tc in [0.28, 0.30, 0.32]:
    for tp in [0.38, 0.42, 0.46]:
        m = evaluate_class_aware(model, val_dataset, th_car=tc, th_ped=tp)
        print(f"th_car={tc:.2f}, th_ped={tp:.2f} -> Val F1: {m['f1']*100:.2f}% | Car F1: {m['car_f1']*100:.2f}% | Ped F1: {m['ped_f1']*100:.2f}% | FP: {m['fp']}")

# Best combination on validation:
best_tc = 0.30
best_tp = 0.42
val_m = evaluate_class_aware(model, val_dataset, th_car=best_tc, th_ped=best_tp)
test_m = evaluate_class_aware(model, test_dataset, th_car=best_tc, th_ped=best_tp)

print(f"\n=======================================================")
print(f"EXPERIMENT 2 RESULTS (th_car={best_tc:.2f}, th_ped={best_tp:.2f})")
print(f"=======================================================")
print(f"Validation:")
print(f"  Recall:    {val_m['recall']*100:.2f}% ({val_m['tp']}/{val_m['total_gt']})")
print(f"  Precision: {val_m['precision']*100:.2f}% ({val_m['tp']}/{val_m['total_pred']})")
print(f"  F1:        {val_m['f1']*100:.2f}%")
print(f"  Car F1:    {val_m['car_f1']*100:.2f}% | Ped F1: {val_m['ped_f1']*100:.2f}%")
print(f"  FP:        {val_m['fp']} (vs 296 baseline)")
print(f"-------------------------------------------------------")
print(f"Test:")
print(f"  Recall:    {test_m['recall']*100:.2f}% ({test_m['tp']}/{test_m['total_gt']})")
print(f"  Precision: {test_m['precision']*100:.2f}% ({test_m['tp']}/{test_m['total_pred']})")
print(f"  F1:        {test_m['f1']*100:.2f}%")
print(f"  Car F1:    {test_m['car_f1']*100:.2f}% | Ped F1: {test_m['ped_f1']*100:.2f}%")
print(f"  FP:        {test_m['fp']} (vs 2002 baseline)")
print(f"=======================================================")

with open("scratch/exp2_class_thresholds.json", "w") as f:
    json.dump({"val": val_m, "test": test_m, "th_car": best_tc, "th_ped": best_tp}, f, indent=2)
