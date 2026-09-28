import numpy as np
import math

def flip_box(box_sensor, box_grid, grid_width=256):
    # box_sensor: [xc, yc, zc, w, l, h, yaw, class_id]
    # box_grid: [gx, gy, zc, w, l, h, yaw, class_id]
    s_flipped = box_sensor.copy()
    g_flipped = box_grid.copy()
    
    # 1. Metric sensor coords: x -> -x
    s_flipped[0] = -s_flipped[0]
    
    # 2. Grid coords: gx -> (grid_width - 1) - gx
    g_flipped[0] = float(grid_width - 1) - g_flipped[0]
    
    # 3. Heading yaw
    yaw = s_flipped[6]
    sin_yaw = math.sin(yaw)
    cos_yaw = -math.cos(yaw) # x-axis inverted
    new_yaw = math.atan2(sin_yaw, cos_yaw)
    s_flipped[6] = new_yaw
    g_flipped[6] = new_yaw
    
    return s_flipped, g_flipped

# Test cases
test_cases = [
    (0.0, math.pi),            # right -> left
    (math.pi/2, math.pi/2),    # forward -> forward
    (-math.pi/2, -math.pi/2),  # backward -> backward
    (math.pi/4, 3*math.pi/4)   # forward-right -> forward-left
]

for orig, expected in test_cases:
    b_s = np.array([5.0, 10.0, 0.0, 2.0, 4.5, 1.6, orig, 0.0])
    b_g = np.array([148.0, 168.0, 0.0, 2.0, 4.5, 1.6, orig, 0.0])
    res_s, res_g = flip_box(b_s, b_g)
    yaw_err = abs(math.atan2(math.sin(res_s[6] - expected), math.cos(res_s[6] - expected)))
    print(f"Orig yaw: {orig:.3f} -> Flipped yaw: {res_s[6]:.3f} (Expected: {expected:.3f}), err: {yaw_err:.4e}")
    assert yaw_err < 1e-5
    assert res_s[0] == -5.0
    assert abs(res_g[0] - (255.0 - 148.0)) < 1e-5
print("All flip unit tests passed!")
