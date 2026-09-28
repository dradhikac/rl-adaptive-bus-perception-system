import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, ".")
from scratch.benchmark_models_cpu import TinyCenterNet3D

model_nano = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=24)
total_params = sum(p.numel() for p in model_nano.parameters())
print(f"Nano Model (base_channels=24): {total_params:,} params ({total_params/1e6:.2f} M, {total_params*4/1024/1024:.2f} MB)")

x = torch.randn(4, 6, 256, 256)
model_nano.train()
opt = torch.optim.AdamW(model_nano.parameters(), lr=1e-3)

times = []
for _ in range(5):
    t0 = time.perf_counter()
    opt.zero_grad()
    out = model_nano(x)
    loss = sum(v.sum() for v in out.values())
    loss.backward()
    opt.step()
    times.append((time.perf_counter() - t0) * 1000.0)

mean_step_ms = np.mean(times)
epoch_sec = 71 * (mean_step_ms / 1000.0)
print(f"Batch=4 Step Time: {mean_step_ms:.1f} ms | Epoch Time: {epoch_sec:.1f}s ({epoch_sec/60.0:.2f} min) | 15 Epochs: {epoch_sec*15/60.0:.1f} min")
