import os
import sys
import json
import glob
import numpy as np

dataset_root = r"datasets\nuscenes"

print("============================================================")
print("1. DIRECTORY STRUCTURE & FILE INVENTORY")
print("============================================================")
print(f"Dataset root: {os.path.abspath(dataset_root)}")

root_contents = sorted(os.listdir(dataset_root))
print(f"Root items: {root_contents}")

# Check samples and sweeps
samples_dir = os.path.join(dataset_root, "samples")
sweeps_dir = os.path.join(dataset_root, "sweeps")
lidar_samples_dir = os.path.join(samples_dir, "LIDAR_TOP")
lidar_sweeps_dir = os.path.join(sweeps_dir, "LIDAR_TOP")

print(f"samples/ exists: {os.path.isdir(samples_dir)}")
if os.path.isdir(samples_dir):
    print(f"  samples/ sensor subfolders: {sorted(os.listdir(samples_dir))}")

print(f"sweeps/ exists: {os.path.isdir(sweeps_dir)}")
if os.path.isdir(sweeps_dir):
    print(f"  sweeps/ sensor subfolders: {sorted(os.listdir(sweeps_dir))}")

# Check metadata folder
meta_candidates = [d for d in root_contents if os.path.isdir(os.path.join(dataset_root, d)) and ("v1" in d or "meta" in d or "json" in d)]
print(f"Metadata directory candidates: {meta_candidates}")
meta_dir = os.path.join(dataset_root, meta_candidates[0]) if meta_candidates else None
if meta_dir:
    meta_files = sorted(os.listdir(meta_dir))
    print(f"Metadata directory ({os.path.basename(meta_dir)}): {len(meta_files)} files")
    for f in meta_files:
        fp = os.path.join(meta_dir, f)
        print(f"  - {f:<26} (size: {os.path.getsize(fp):>10,d} bytes)")

print("\n============================================================")
print("2. LIDAR_TOP FILE FORMAT & SAMPLES PROBE")
print("============================================================")
sample_files = sorted(os.listdir(lidar_samples_dir)) if os.path.isdir(lidar_samples_dir) else []
sweep_files = sorted(os.listdir(lidar_sweeps_dir)) if os.path.isdir(lidar_sweeps_dir) else []

print(f"Keyframe LiDAR samples (samples/LIDAR_TOP): {len(sample_files)} files")
print(f"Sweep LiDAR point clouds (sweeps/LIDAR_TOP) : {len(sweep_files)} files")

if sample_files:
    # Extensions and naming convention
    extensions = set(os.path.splitext(f)[1] for f in sample_files)
    print(f"LiDAR file extensions: {extensions}")
    print(f"First 3 sample filenames:")
    for f in sample_files[:3]:
        print(f"  {f}")

    # Inspect file sizes
    sizes = [os.path.getsize(os.path.join(lidar_samples_dir, f)) for f in sample_files]
    print(f"File size stats (bytes): min={min(sizes):,}, max={max(sizes):,}, mean={np.mean(sizes):,.1f}")

    # Inspect binary format of first few samples
    print("\nPoint Cloud Data Format Inspection:")
    first_file = os.path.join(lidar_samples_dir, sample_files[0])
    raw_bytes = os.path.getsize(first_file)
    print(f"Inspecting: {sample_files[0]}")
    print(f"Raw file byte size: {raw_bytes:,} bytes")

    # Try different float32 column structures (e.g. 4 float32 = 16 bytes: x,y,z,intensity or 5 float32 = 20 bytes: x,y,z,intensity,ring_index)
    for num_cols in [3, 4, 5, 6]:
        bytes_per_point = num_cols * 4
        if raw_bytes % bytes_per_point == 0:
            num_pts = raw_bytes // bytes_per_point
            print(f"  -> Valid divisor: {num_cols} float32 per point ({bytes_per_point} bytes/pt) => {num_pts:,} points")

    # Load point clouds and compute point count statistics across all keyframe samples
    # In nuScenes standard: float32 binary format, 5 dimensions: (x, y, z, intensity, ring_index)
    # Let's verify float32 array
    pts_counts = []
    coord_min = [float("inf")] * 3
    coord_max = [float("-inf")] * 3
    intensities = []

    for sf in sample_files:
        sfp = os.path.join(lidar_samples_dir, sf)
        data = np.fromfile(sfp, dtype=np.float32)
        # determine shape
        if len(data) % 5 == 0:
            pts = data.reshape(-1, 5)
        elif len(data) % 4 == 0:
            pts = data.reshape(-1, 4)
        else:
            pts = data.reshape(-1, 3)

        pts_counts.append(len(pts))
        for c in range(3):
            coord_min[c] = min(coord_min[c], float(pts[:, c].min()))
            coord_max[c] = max(coord_max[c], float(pts[:, c].max()))
        if pts.shape[1] >= 4:
            intensities.extend(pts[:50, 3].tolist())

    pts_counts = np.array(pts_counts)
    print(f"\nPoint Count Statistics across all {len(sample_files)} keyframes:")
    print(f"  Min points per sample   : {pts_counts.min():,}")
    print(f"  Max points per sample   : {pts_counts.max():,}")
    print(f"  Mean points per sample  : {pts_counts.mean():,.1f}")
    print(f"  Median points per sample: {np.median(pts_counts):,.1f}")
    print(f"  Std points per sample   : {pts_counts.std():,.1f}")
    print(f"Coordinate bounding range (Sensor Frame):")
    print(f"  X: [{coord_min[0]:.2f}, {coord_max[0]:.2f}] m  (lateral / forward depending on frame convention)")
    print(f"  Y: [{coord_min[1]:.2f}, {coord_max[1]:.2f}] m")
    print(f"  Z: [{coord_min[2]:.2f}, {coord_max[2]:.2f}] m (vertical)")
    if intensities:
        print(f"Sample Intensity range: [{min(intensities):.2f}, {max(intensities):.2f}]")

print("\n============================================================")
print("3. METADATA & 3D ANNOTATIONS PROBE")
print("============================================================")
if meta_dir:
    def load_json(filename):
        fp = os.path.join(meta_dir, filename)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    # Check sensors & calibration
    sensors = load_json("sensor.json")
    calibrated_sensors = load_json("calibrated_sensor.json")
    ego_poses = load_json("ego_pose.json")
    scenes = load_json("scene.json")
    samples = load_json("sample.json")
    sample_data = load_json("sample_data.json")
    categories = load_json("category.json")
    sample_annotations = load_json("sample_annotation.json")

    print(f"Scenes loaded: {len(scenes)}")
    print(f"Samples (keyframes) loaded: {len(samples)}")
    print(f"Sample data entries: {len(sample_data)}")
    print(f"Categories loaded: {len(categories)}")
    print(f"Sample annotations (3D bounding boxes) loaded: {len(sample_annotations)}")
    print(f"Calibrated sensors: {len(calibrated_sensors)}")
    print(f"Ego poses: {len(ego_poses)}")

    # Check LIDAR sensor and calibration
    lidar_sensors = [s for s in sensors if "LIDAR" in s.get("channel", "")]
    print(f"\nLiDAR Sensors defined in sensor.json: {len(lidar_sensors)}")
    for ls in lidar_sensors:
        print(f"  Sensor token: {ls.get('token')}, channel: {ls.get('channel')}, modality: {ls.get('modality')}")

    lidar_cals = [cs for cs in calibrated_sensors if any(cs.get("sensor_token") == ls.get("token") for ls in lidar_sensors)]
    print(f"LiDAR Calibration records found: {len(lidar_cals)}")
    if lidar_cals:
        cal = lidar_cals[0]
        print(f"  Extrinsics (Translation [x,y,z]): {cal.get('translation')}")
        print(f"  Extrinsics (Rotation [w,x,y,z]):    {cal.get('rotation')}")
        print(f"  Intrinsics / camera_intrinsic:      {cal.get('camera_intrinsic')} (None for LiDAR as expected)")

    # Check timestamps and frame IDs
    lidar_data = [sd for sd in sample_data if sd.get("fileformat") == "pcd" and "LIDAR_TOP" in sd.get("filename", "")]
    print(f"\nLiDAR sample_data entries: {len(lidar_data)}")
    if lidar_data:
        sd = lidar_data[0]
        print(f"  Sample data record fields: {list(sd.keys())}")
        print(f"  Filename: {sd.get('filename')}")
        print(f"  Timestamp: {sd.get('timestamp')}")
        print(f"  is_key_frame: {sd.get('is_key_frame')}")
        print(f"  sample_token: {sd.get('sample_token')}")

    # Check 3D annotations structure
    print("\n3D Bounding Box Structure Analysis:")
    if sample_annotations:
        ann = sample_annotations[0]
        print(f"  Annotation fields: {list(ann.keys())}")
        print(f"  translation (center x,y,z): {ann.get('translation')}")
        print(f"  size (w, l, h):            {ann.get('size')}")
        print(f"  rotation (quaternion):     {ann.get('rotation')}")
        print(f"  num_lidar_pts:             {ann.get('num_lidar_pts')}")
        print(f"  num_radar_pts:             {ann.get('num_radar_pts')}")
        print(f"  category_name (via token): {[c.get('name') for c in categories if c.get('token') == ann.get('category_token')]}")

        # Class breakdown via instance_token -> category_token
        instances = load_json("instance.json")
        inst_to_cat = {inst["token"]: inst["category_token"] for inst in instances}
        token_to_cat = {c.get("token"): c.get("name") for c in categories}
        
        cat_counts = {}
        pts_in_box = []
        for a in sample_annotations:
            cat_token = inst_to_cat.get(a.get("instance_token"))
            cat_name = token_to_cat.get(cat_token, "unknown")
            cat_counts[cat_name] = cat_counts.get(cat_name, 0) + 1
            if a.get("num_lidar_pts") is not None:
                pts_in_box.append(a.get("num_lidar_pts"))

        print("\nAvailable 3D Object Classes (sorted by frequency):")
        for cat, cnt in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {cat:<35}: {cnt:>5} boxes ({cnt/len(sample_annotations)*100:>5.1f}%)")

        if pts_in_box:
            pts_in_box = np.array(pts_in_box)
            print(f"\nGround Truth Points per 3D Bounding Box:")
            print(f"  Min: {pts_in_box.min()}, Max: {pts_in_box.max()}, Median: {np.median(pts_in_box):.1f}, Mean: {pts_in_box.mean():.1f}")
            print(f"  Boxes with > 0 LiDAR points: {(pts_in_box > 0).sum()} / {len(pts_in_box)} ({(pts_in_box > 0).mean()*100:.1f}%)")
            print(f"  Boxes with >= 5 LiDAR points: {(pts_in_box >= 5).sum()} / {len(pts_in_box)} ({(pts_in_box >= 5).mean()*100:.1f}%)")

    # Check Train / Val Split
    print("\nTraining / Validation Split Convention:")
    print(f"Number of scenes: {len(scenes)}")
    for sc in scenes:
        print(f"  Scene: {sc.get('name'):<12} ({sc.get('nbr_samples')} samples) - {sc.get('description')}")

    # Standard nuScenes mini split check:
    # nuScenes-mini officially has 10 scenes:
    # Train: 8 scenes (scene-0061, scene-0103, scene-0655, scene-0757, scene-0796, scene-0916, scene-1077, scene-1094)
    # Val: 2 scenes (scene-0553, scene-0921)
    val_scenes_std = {"scene-0553", "scene-0921"}
    scene_names = set(sc.get("name") for sc in scenes)
    print(f"Standard nuScenes-mini val scenes present: {val_scenes_std.issubset(scene_names)}")
    train_samples_count = sum(sc.get("nbr_samples") for sc in scenes if sc.get("name") not in val_scenes_std)
    val_samples_count = sum(sc.get("nbr_samples") for sc in scenes if sc.get("name") in val_scenes_std)
    print(f"Standard mini split: {train_samples_count} train samples (~80%) across 8 scenes | {val_samples_count} val samples (~20%) across 2 scenes")
    print(f"Total keyframes: {train_samples_count + val_samples_count}")

print("\n============================================================")
print("DATASET INSPECTION COMPLETED")
print("============================================================")
