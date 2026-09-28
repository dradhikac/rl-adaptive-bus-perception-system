"""
Evaluate threshold sweep on Experiment 3 (Augmented) model
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
from scratch.run_stage11_evaluator import evaluate_split

device = torch.device("cpu")
config = DEFAULT_CONFIG
ckpt_path = r"models\checkpoints_exp3_aug\best_model.pth"

checkpoint = torch.load(ckpt_path, map_location=device)
model = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=24).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

val_dataset = AutoBusLidarDataset(split="VALIDATION", config=config, use_cache=True)
test_dataset = AutoBusLidarDataset(split="TEST", config=config, use_cache=True)

candidate_thresholds = [0.28, 0.32, 0.36, 0.40, 0.44]

print("==========================================================================================")
print("EXP 3 THRESHOLD SWEEP ON VALIDATION AND TEST SETS")
print("==========================================================================================")

for th in candidate_thresholds:
    vm = evaluate_split(model, val_dataset, device, config, score_thresh=th)
    tm = evaluate_split(model, test_dataset, device, config, score_thresh=th)
    print(f"Thresh {th:.2f} | Val F1: {vm['f1']*100:.2f}% (P: {vm['precision']*100:.2f}%, R: {vm['recall']*100:.2f}%, FP: {vm['fp']}) | Test F1: {tm['f1']*100:.2f}% (P: {tm['precision']*100:.2f}%, R: {tm['recall']*100:.2f}%, FP: {tm['fp']})")
