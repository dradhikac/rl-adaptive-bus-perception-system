"""
AutoBus LiDAR 3D Perception Baseline Training Script
CPU-Optimized CenterPoint-Lite Training Pipeline
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure root in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.lidar_perception.config import LidarConfig, DEFAULT_CONFIG
from src.lidar_perception.dataset import AutoBusLidarDataset, autobus_collate_fn
from src.lidar_perception.model import TinyCenterNet3D, decode_predictions
from src.lidar_perception.loss import CenterNet3DLoss

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_losses = {
        "total": 0.0, "hm": 0.0, "off": 0.0, "z": 0.0, "dim": 0.0, "rot": 0.0
    }
    num_batches = 0
    total_targets = 0
    
    for batch in loader:
        bev_maps = batch["bev_maps"].to(device)
        
        optimizer.zero_grad()
        preds = model(bev_maps)
        loss_dict = criterion(preds, batch)
        
        loss = loss_dict["total_loss"]
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        
        running_losses["total"] += loss.item()
        running_losses["hm"] += loss_dict["loss_hm"].item()
        running_losses["off"] += loss_dict["loss_off"].item()
        running_losses["z"] += loss_dict["loss_z"].item()
        running_losses["dim"] += loss_dict["loss_dim"].item()
        running_losses["rot"] += loss_dict["loss_rot"].item()
        total_targets += loss_dict["num_targets"]
        num_batches += 1
        
    return {k: v / max(num_batches, 1) for k, v in running_losses.items()}, total_targets

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_losses = {
        "total": 0.0, "hm": 0.0, "off": 0.0, "z": 0.0, "dim": 0.0, "rot": 0.0
    }
    num_batches = 0
    total_targets = 0
    total_predictions = 0
    
    for batch in loader:
        bev_maps = batch["bev_maps"].to(device)
        preds = model(bev_maps)
        loss_dict = criterion(preds, batch)
        
        running_losses["total"] += loss_dict["total_loss"].item()
        running_losses["hm"] += loss_dict["loss_hm"].item()
        running_losses["off"] += loss_dict["loss_off"].item()
        running_losses["z"] += loss_dict["loss_z"].item()
        running_losses["dim"] += loss_dict["loss_dim"].item()
        running_losses["rot"] += loss_dict["loss_rot"].item()
        total_targets += loss_dict["num_targets"]
        num_batches += 1
        
        # Count decoded predictions
        decoded = decode_predictions(preds, score_thresh=0.25)
        total_predictions += sum(len(b) for b in decoded)
        
    avg_losses = {k: v / max(num_batches, 1) for k, v in running_losses.items()}
    return avg_losses, total_targets, total_predictions

def main():
    parser = argparse.ArgumentParser(description="AutoBus 3D Detection Baseline Training")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints", help="Output checkpoint dir")
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    art_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\1c0087c6-dac4-4253-a522-0bdbc5b00eb8"
    os.makedirs(art_dir, exist_ok=True)
    
    print("============================================================")
    print("STAGE 8 — BASELINE 3D DETECTION MODEL TRAINING")
    print("============================================================")
    print(f"Model: TinyCenterNet3D-Lite (0.49M params, CenterPoint-style)")
    print(f"Input representation: 6-channel BEV Pseudo-Image (256 x 256, 0.25m resolution)")
    print(f"Classes: ['CAR', 'PEDESTRIAN']")
    print(f"Train samples: 283 samples (7 scenes)")
    print(f"Validation samples: 41 samples (1 scene)")
    print(f"Test samples: 80 samples (2 scenes)")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Expected approximate training time: ~18-20 minutes on CPU (~74s / epoch)")
    print(f"Checkpoint location: {os.path.abspath(args.checkpoint_dir)}")
    print("============================================================\n")
    
    device = torch.device("cpu")
    config = DEFAULT_CONFIG
    
    # Datasets
    print("Initializing datasets & preprocessors...")
    train_dataset = AutoBusLidarDataset(split="TRAIN", config=config, use_cache=True)
    val_dataset = AutoBusLidarDataset(split="VALIDATION", config=config, use_cache=True)
    test_dataset = AutoBusLidarDataset(split="TEST", config=config, use_cache=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=autobus_collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=autobus_collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=autobus_collate_fn, num_workers=0
    )
    
    # Model, Loss, Optimizer
    model = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=24).to(device)
    criterion = CenterNet3DLoss(downsample_stride=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    history = {
        "train_loss": [],
        "train_hm": [],
        "val_loss": [],
        "val_hm": [],
        "val_preds_count": [],
        "lr": [],
        "epoch_times": []
    }
    
    best_val_loss = float("inf")
    start_total_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        
        train_losses, tr_targets = train_epoch(model, train_loader, criterion, optimizer, device)
        val_losses, val_targets, val_preds = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - t0
        
        history["train_loss"].append(train_losses["total"])
        history["train_hm"].append(train_losses["hm"])
        history["val_loss"].append(val_losses["total"])
        history["val_hm"].append(val_losses["hm"])
        history["val_preds_count"].append(val_preds)
        history["lr"].append(current_lr)
        history["epoch_times"].append(epoch_time)
        
        is_best = val_losses["total"] < best_val_loss
        if is_best:
            best_val_loss = val_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_losses["total"],
                "config": config.__dict__
            }, os.path.join(args.checkpoint_dir, "best_model.pth"))
            
        # Always save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_losses["total"],
            "config": config.__dict__
        }, os.path.join(args.checkpoint_dir, "latest_model.pth"))
        
        best_marker = " [BEST]" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_losses['total']:.4f} (HM: {train_losses['hm']:.4f}) | "
              f"Val Loss: {val_losses['total']:.4f} (HM: {val_losses['hm']:.4f}, Preds: {val_preds}){best_marker}")

    total_training_sec = time.time() - start_total_time
    print(f"\nTraining completed in {total_training_sec/60.0:.2f} minutes.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Evaluate best model on TEST split (unseen 80 samples)
    print("\n--- Evaluating Best Model on Unseen TEST Split (80 samples) ---")
    best_ckpt = torch.load(os.path.join(args.checkpoint_dir, "best_model.pth"))
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_losses, test_targets, test_preds = evaluate(model, test_loader, criterion, device)
    print(f"Test Set Loss: {test_losses['total']:.4f} | HM Loss: {test_losses['hm']:.4f}")
    print(f"Test Targets Available: {test_targets} | Detected Predictions (score >= 0.25): {test_preds}")
    
    # Save training metrics JSON
    metrics_data = {
        "model_name": "TinyCenterNet3D-Lite",
        "epochs_trained": args.epochs,
        "batch_size": args.batch_size,
        "total_training_time_seconds": total_training_sec,
        "best_epoch": int(np.argmin(history["val_loss"]) + 1),
        "best_val_loss": best_val_loss,
        "final_test_loss": test_losses["total"],
        "final_test_hm_loss": test_losses["hm"],
        "test_targets_count": test_targets,
        "test_predictions_count": test_preds,
        "history": history
    }
    with open(os.path.join(args.checkpoint_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    # Generate and save loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor="#0e1117")
    fig.suptitle("AutoBus Stage 8 — Baseline 3D Detection Model Training Curves\n(TinyCenterNet3D-Lite on CPU | 283 Train / 41 Val / 80 Test)",
                 fontsize=14, color="#f0f6fc", fontweight="bold")
                 
    epochs_range = range(1, args.epochs + 1)
    
    ax1.set_facecolor("#05070a")
    ax1.set_title("1. Total Multi-Task Loss (Train vs Validation)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax1.plot(epochs_range, history["train_loss"], label="Train Total Loss", color="#1f6feb", linewidth=2.0)
    ax1.plot(epochs_range, history["val_loss"], label="Val Total Loss", color="#d29922", linewidth=2.0)
    ax1.scatter([np.argmin(history['val_loss'])+1], [best_val_loss], color="#3fb950", s=80, zorder=5, label=f"Best Val: {best_val_loss:.3f}")
    ax1.set_xlabel("Epoch", color="#8b949e")
    ax1.set_ylabel("Total Loss", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    ax1.grid(True, color="#21262d", linestyle=":")
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")
    
    ax2.set_facecolor("#05070a")
    ax2.set_title("2. Center Heatmap Focal Loss (Object Detection Confidence)", color="#58a6ff", fontsize=12, fontweight="bold")
    ax2.plot(epochs_range, history["train_hm"], label="Train Heatmap Loss", color="#3fb950", linewidth=2.0)
    ax2.plot(epochs_range, history["val_hm"], label="Val Heatmap Loss", color="#f85149", linewidth=2.0)
    ax2.set_xlabel("Epoch", color="#8b949e")
    ax2.set_ylabel("Focal Loss", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    ax2.grid(True, color="#21262d", linestyle=":")
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc")
    
    plt.tight_layout()
    loss_png = "stage8_training_loss_curves.png"
    plt.savefig(os.path.join(args.checkpoint_dir, loss_png), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.savefig(os.path.join(art_dir, loss_png), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Saved loss curves to {loss_png}")

if __name__ == "__main__":
    main()
