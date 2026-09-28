import os
import json
from collections import defaultdict
import numpy as np

dataset_root = r"datasets\nuscenes"
meta_dir = os.path.join(dataset_root, "v1.0-mini")

def load_json(name):
    with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as f:
        return json.load(f)

scenes = {sc["token"]: sc for sc in load_json("scene.json")}
samples = load_json("sample.json")
annotations = load_json("sample_annotation.json")
instances = {inst["token"]: inst["category_token"] for inst in load_json("instance.json")}
categories = {cat["token"]: cat["name"] for cat in load_json("category.json")}
sample_data = load_json("sample_data.json")

def get_canonical_class(cat_name):
    if "vehicle.car" in cat_name:
        return "CAR"
    elif "human.pedestrian" in cat_name:
        return "PEDESTRIAN"
    elif "vehicle.truck" in cat_name:
        return "TRUCK"
    elif "vehicle.bus" in cat_name:
        return "BUS"
    elif "vehicle.motorcycle" in cat_name:
        return "MOTORCYCLE"
    elif "vehicle.bicycle" in cat_name:
        return "BICYCLE"
    elif "movable_object.barrier" in cat_name:
        return "BARRIER"
    elif "movable_object.trafficcone" in cat_name:
        return "TRAFFIC_CONE"
    elif "construction" in cat_name:
        return "CONSTR_VEHICLE"
    return "OTHER"

sample_anns = defaultdict(list)
for a in annotations:
    sample_anns[a["sample_token"]].append(a)

scene_name_to_samples = defaultdict(list)
for s in samples:
    sc = scenes[s["scene_token"]]
    scene_name_to_samples[sc["name"]].append(s)

# Let's evaluate Split Option 1:
# Train: 7 scenes (283 samples, 70.0%)
# Val:   1 scene  (41 samples,  10.1%)
# Test:  2 scenes (80 samples,  19.8%)
split_opt1 = {
    "TRAIN": ["scene-0061", "scene-0103", "scene-0655", "scene-0757", "scene-0796", "scene-0916", "scene-1077"],
    "VAL":   ["scene-0553"],
    "TEST":  ["scene-1094", "scene-1100"]
}

# Let's evaluate Split Option 2:
# Train: 7 scenes (282 samples, 69.8%)
# Val:   2 scenes (81 samples,  20.0%)
# Test:  1 scene  (41 samples,  10.1%)
split_opt2 = {
    "TRAIN": ["scene-0061", "scene-0103", "scene-0655", "scene-0757", "scene-0796", "scene-0916", "scene-1077"],
    "VAL":   ["scene-1094", "scene-1100"],
    "TEST":  ["scene-0553"]
}

# Let's evaluate Split Option 3:
# Balanced mix:
# Train: 7 scenes (283 samples, 70.0%)
# Val:   scene-0796 (40 samples, 9.9%) or scene-0553 (41 samples, 10.1%)
# Test:  scene-0757 (41 samples) + scene-1100 (40 samples) = 81 samples (20.0%)
split_opt3 = {
    "TRAIN": ["scene-0061", "scene-0103", "scene-0553", "scene-0655", "scene-0796", "scene-0916", "scene-1077"],
    "VAL":   ["scene-0757"], # Boston daytime intersection
    "TEST":  ["scene-1094", "scene-1100"] # Singapore night + rain
}

for opt_idx, split_def in enumerate([split_opt1, split_opt2, split_opt3], 1):
    print(f"\n============================================================")
    print(f"SPLIT OPTION {opt_idx}")
    print(f"============================================================")
    
    for split_name, sc_list in split_def.items():
        split_samples = []
        for sc in sc_list:
            split_samples.extend(scene_name_to_samples[sc])
            
        cls_counts = defaultdict(int)
        trainable_counts = defaultdict(int)
        
        for sm in split_samples:
            for a in sample_anns[sm["token"]]:
                cat_name = categories.get(instances.get(a["instance_token"], ""), "unknown")
                cls = get_canonical_class(cat_name)
                cls_counts[cls] += 1
                if a.get("num_lidar_pts", 0) >= 3:
                    trainable_counts[cls] += 1
                    
        pct = len(split_samples) / 404 * 100
        print(f"\n--- {split_name} SPLIT ({len(split_samples)} samples, {pct:.1f}%) ---")
        print(f"Scenes included: {sc_list}")
        print(f"{'Class':<15} | {'Total Boxes':>12} | {'Trainable (pts>=3)':>18}")
        print("-" * 50)
        for c in ["CAR", "PEDESTRIAN", "TRUCK", "BUS", "BARRIER", "TRAFFIC_CONE", "MOTORCYCLE", "BICYCLE", "CONSTR_VEHICLE", "OTHER"]:
            print(f"{c:<15} | {cls_counts[c]:>12} | {trainable_counts[c]:>18}")
