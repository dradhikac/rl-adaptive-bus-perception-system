import os
import numpy as np

fp = r"datasets\nuscenes\samples\LIDAR_TOP\n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402939697511.pcd.bin"
pts = np.fromfile(fp, dtype=np.float32).reshape(-1, 5)

print("Total points:", len(pts))
print("Z min, max, mean, median:", pts[:, 2].min(), pts[:, 2].max(), pts[:, 2].mean(), np.median(pts[:, 2]))

# Let's see percentiles of Z
for p in [0.1, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]:
    print(f"Z {p:4.1f}%: {np.percentile(pts[:, 2], p):.3f} m")

# Let's check points at different radial distances
r = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
for r_min, r_max in [(2, 5), (5, 10), (10, 20), (20, 30)]:
    mask = (r >= r_min) & (r < r_max)
    sub_z = pts[mask, 2]
    print(f"Range [{r_min}, {r_max}]m: {mask.sum()} points, Z median: {np.median(sub_z):.3f}m, Z 10%: {np.percentile(sub_z, 10):.3f}m, Z min: {sub_z.min():.3f}m")
