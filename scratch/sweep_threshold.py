"""
AutoBus Stage 11 — Experiment 1: Confidence Threshold Sweep
Evaluates the baseline model checkpoint across candidate confidence thresholds on the VALIDATION set.
Selects the optimal threshold that maximizes Validation F1 score, then evaluates on the TEST set.
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))

from src.lidar_perception.config import DEFAULT_CONFIG
from src.lidar_perception.dataset import AutoBusLidarDataset
from src.lidar_perception.model import TinyCenterNet3D
from scratch.run_stage11_evaluator import evaluate_split, compute_loss
from src.lidar_perception.loss import CenterNet3DLoss

device = torch.device("cpu")
config = DEFAULT_CONFIG
ckpt_path = r"models\checkpoints\best_model.pth"

print(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)
model = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=24).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_dataset = AutoBusLidarDataset(split="VALIDATION", config=config, use_cache=True)
test_dataset = AutoBusLidarDataset(split="TEST", config=config, use_cache=True)

candidate_thresholds = [0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52]

print("==========================================================================================")
print("EXPERIMENT 1: CONFIDENCE THRESHOLD TUNING ON VALIDATION SET")
print("==========================================================================================")
print(f"{'Thresh':<8} | {'Val Rec':<9} | {'Val Prec':<9} | {'Val F1':<8} | {'Car F1':<8} | {'Ped F1':<8} | {'Val FP':<7} | {'Val FN':<7}")
print("------------------------------------------------------------------------------------------")

val_results = {}
for th in candidate_thresholds:
    m = evaluate_split(model, val_dataset, device, config, score_thresh=th)
    val_results[th] = m
    print(f"{th:<8.2f} | {m['recall']*100:<8.2f}% | {m['precision']*100:<8.2f}% | {m['f1']*100:<7.2f}% | {m['car_f1']*100:<7.2f}% | {m['ped_f1']*100:<7.2f}% | {m['fp']:<7} | {m['fn']:<7}")

best_th = max(candidate_thresholds, key=lambda t: val_results[t]["f1"])
print("==========================================================================================")
print(f"Optimal Validation Threshold selected: {best_th:.2f} (Val F1: {val_results[best_th]['f1']*100:.2f}%)")
print("==========================================================================================")

# Evaluate best_th on TEST set
print(f"\nEvaluating selected threshold ({best_th:.2f}) on UNSEEN TEST SET:")
test_res = evaluate_split(model, test_dataset, device, config, score_thresh=best_th)
print(f"Test Recall:    {test_res['recall']*100:.2f}% ({test_res['tp']}/{test_res['total_gt']})")
print(f"Test Precision: {test_res['precision']*100:.2f}% ({test_res['tp']}/{test_res['total_pred']})")
print(f"Test F1-Score:  {test_res['f1']*100:.2f}%")
print(f"Test Car F1:    {test_res['car_f1']*100:.2f}% | Pedestrian F1: {test_res['ped_f1']*100:.2f}%")
print(f"Test FP:        {test_res['fp']} (reduced from 2002)")
print(f"Test FN:        {test_res['fn']}")

out_file = "scratch/exp1_threshold_tuning.json"
with open(out_file, "w") as f:
    json.dump({
        "candidate_thresholds": candidate_thresholds,
        "validation_sweep": {str(k): v for k, v in val_results.items()},
        "best_threshold": best_th,
        "val_at_best": val_results[best_th],
        "test_at_best": test_res
    }, f, indent=2)
print(f"Saved results to {out_file}")
