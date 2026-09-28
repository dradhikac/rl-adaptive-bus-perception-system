import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

print("============================================================")
print("STAGE 7: CPU 3D DETECTION MODEL FEASIBILITY & BENCHMARK")
print("============================================================")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CPU threads: {torch.get_num_threads()}")

# -------------------------------------------------------------
# 1. Candidate Architecture: TinyCenterNet3D (Center-Based BEV)
# -------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.downsample(x)

class TinyCenterNet3D(nn.Module):
    """
    Lightweight CPU-optimized CenterPoint-style 3D Object Detector.
    Input: (B, 6, 256, 256) BEV pseudo-image
    Output: 
      - heatmap: (B, num_classes, 64, 64)
      - offset:  (B, 2, 64, 64)  [dx, dy]
      - z_elev:  (B, 1, 64, 64)  [zc]
      - dim_3d:  (B, 3, 64, 64)  [log(w), log(l), log(h)]
      - rot_3d:  (B, 2, 64, 64)  [sin(yaw), cos(yaw)]
    """
    def __init__(self, in_channels=6, num_classes=2, base_channels=32):
        super().__init__()
        C = base_channels # 32
        
        # Encoder (Downsample 256x256 -> 128x128 -> 64x64)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True)
        )
        
        self.stage1 = ConvBlock(C, C, stride=1)      # 256x256
        self.stage2 = ConvBlock(C, C*2, stride=2)    # 128x128
        self.stage3 = ConvBlock(C*2, C*4, stride=2)  # 64x64
        self.stage4 = ConvBlock(C*4, C*4, stride=1)  # 64x64
        
        # Neck / Feature Aggregation (FPN-lite)
        self.neck = nn.Sequential(
            nn.Conv2d(C*4, C*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C*2),
            nn.ReLU(inplace=True)
        )
        
        # Multi-task Detection Heads (at 64x64 resolution, 4x stride)
        head_in = C*2 # 64
        
        # 1. Classification Heatmap
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, num_classes, kernel_size=1)
        )
        # Initialize heatmap bias for focal loss stability
        self.heatmap_head[-1].bias.data.fill_(-2.19)
        
        # 2. Sub-pixel Offset Head [dx, dy]
        self.offset_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, 2, kernel_size=1)
        )
        
        # 3. Z Elevation Head [zc]
        self.z_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, 1, kernel_size=1)
        )
        
        # 4. 3D Dimensions Head [log(w), log(l), log(h)]
        self.dim_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, 3, kernel_size=1)
        )
        
        # 5. Continuous Yaw Orientation Head [sin(yaw), cos(yaw)]
        self.rot_head = nn.Sequential(
            nn.Conv2d(head_in, head_in, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_in, 2, kernel_size=1)
        )

    def forward(self, x):
        # Backbone
        feat = self.stem(x)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)
        feat = self.stage4(feat)
        feat = self.neck(feat)
        
        # Heads
        heatmap = self.heatmap_head(feat)
        offset = self.offset_head(feat)
        z_elev = self.z_head(feat)
        dim_3d = self.dim_head(feat)
        rot_3d = self.rot_head(feat)
        
        return {
            "heatmap": heatmap,
            "offset": offset,
            "z_elev": z_elev,
            "dim_3d": dim_3d,
            "rot_3d": rot_3d
        }

# Instantiate model
model = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=32)
model.eval()

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
param_size_mb = total_params * 4 / (1024 * 1024)

print(f"\nModel Parameter Statistics:")
print(f"  Total Parameters: {total_params:,} ({total_params/1e6:.2f} M)")
print(f"  Trainable Parameters: {trainable_params:,}")
print(f"  Model Memory Footprint (Weights): {param_size_mb:.2f} MB")

# Benchmark Forward Pass Latency on CPU
x_single = torch.randn(1, 6, 256, 256)
x_batch = torch.randn(4, 6, 256, 256)

# Warmup
for _ in range(5):
    _ = model(x_single)

# Single sample inference benchmark (10 iterations)
times_single = []
for _ in range(15):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x_single)
    times_single.append((time.perf_counter() - t0) * 1000.0)

# Batch size 4 forward pass benchmark
times_batch = []
for _ in range(10):
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x_batch)
    times_batch.append((time.perf_counter() - t0) * 1000.0)

print(f"\nCPU Forward Pass Latency (Intel CPU):")
print(f"  Single Sample (B=1): {np.mean(times_single):.1f} ms (std: {np.std(times_single):.1f} ms)")
print(f"  FPS (Frames/Sec):     {1000.0 / np.mean(times_single):.1f} FPS")
print(f"  Batch Size 4 (B=4):  {np.mean(times_batch):.1f} ms ({np.mean(times_batch)/4.0:.1f} ms/sample)")

# Benchmark Training Step (Forward + Backward on CPU)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
x_train = torch.randn(4, 6, 256, 256)

times_train = []
for _ in range(10):
    t0 = time.perf_counter()
    optimizer.zero_grad()
    preds = model(x_train)
    # Dummy loss
    loss = (
        preds["heatmap"].sum() + 
        preds["offset"].sum() + 
        preds["z_elev"].sum() + 
        preds["dim_3d"].sum() + 
        preds["rot_3d"].sum()
    )
    loss.backward()
    optimizer.step()
    times_train.append((time.perf_counter() - t0) * 1000.0)

step_time_ms = np.mean(times_train)
print(f"\nTraining Step Latency (B=4, Forward + Backward + Optimizer Step):")
print(f"  Step Time: {step_time_ms:.1f} ms")
print(f"  Steps per Second: {1000.0 / step_time_ms:.1f} steps/s")

# Calculate estimated training time for 283 samples (71 batches of size 4 per epoch)
batches_per_epoch = int(np.ceil(283 / 4))
epoch_time_sec = batches_per_epoch * (step_time_ms / 1000.0)
print(f"\nEstimated Training Time (283 Samples, Batch Size 4):")
print(f"  Batches per Epoch: {batches_per_epoch}")
print(f"  Estimated Time per Epoch: {epoch_time_sec:.1f} seconds")
print(f"  Estimated Time for 20 Epochs: {epoch_time_sec * 20 / 60.0:.1f} minutes")
print(f"  Estimated Time for 40 Epochs: {epoch_time_sec * 40 / 60.0:.1f} minutes")

print("\nSTAGE 7 BENCHMARK COMPLETED SUCCESSFULLY.")
