import os
import numpy as np

fp = r"datasets\nuscenes\samples\LIDAR_TOP\n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402939697511.pcd.bin"
pts = np.fromfile(fp, dtype=np.float32).reshape(-1, 5)

mask_zero = np.abs(pts[:, 2]) < 0.05
print(f"Points with |Z| < 0.05m: {mask_zero.sum()} / {len(pts)}")
pts_z0 = pts[mask_zero]
r_z0 = np.sqrt(pts_z0[:, 0]**2 + pts_z0[:, 1]**2)
print(f"Range of points near Z=0: min={r_z0.min():.2f}m, max={r_z0.max():.2f}m, median={np.median(r_z0):.2f}m")
print(f"Rings of points near Z=0: {np.unique(pts_z0[:, 4]).astype(int)}")
