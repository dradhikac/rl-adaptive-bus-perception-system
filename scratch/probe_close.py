import os
import numpy as np

fp = r"datasets\nuscenes\samples\LIDAR_TOP\n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402939697511.pcd.bin"
pts = np.fromfile(fp, dtype=np.float32).reshape(-1, 5)

r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
print("Points with distance r < 0.1m:", (r < 0.1).sum())
print("Points with distance r < 0.5m:", (r < 0.5).sum())
print("Points with distance r < 1.0m:", (r < 1.0).sum())

# What are the coordinates of points with r < 0.5m?
close_pts = pts[r < 0.5]
print("Close points sample (first 10):")
print(close_pts[:10])
