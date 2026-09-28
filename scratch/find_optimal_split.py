import os
import json
from collections import defaultdict
import itertools
import numpy as np

dataset_root = r"datasets\nuscenes"
meta_dir = os.path.join(dataset_root, "v1.0-mini")

def load_json(name):
    with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as f:
        return json.load(f)

scenes = load_json("scene.json")
samples = load_json("sample.json")
annotations = load_json("sample_annotation.json")
instances = {inst["token"]: inst["category_token"] for inst in load_json("instance.json")}
categories = {cat["token"]: cat["name"] for cat in load_json("category.json")}
logs = {log["token"]: log for log in load_json("log.json")}

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

scene_samples = defaultdict(list)
for s in samples:
    scene_samples[s["scene_token"]].append(s)

scene_info = {}
for sc in scenes:
    sc_token = sc["token"]
    sc_name = sc["name"]
    nbr = sc["nbr_samples"]
    desc = sc["description"]
    log = logs.get(sc["log_token"], {})
    loc = log.get("location", "unknown")
    is_night = "night" in desc.lower() or "1077" in sc_name or "1094" in sc_name or "1100" in sc_name
    is_rain = "rain" in desc.lower()
    
    sc_classes = defaultdict(int)
    for sm in scene_samples[sc_token]:
        for a in sample_anns[sm["token"]]:
            cat_name = categories.get(instances.get(a["instance_token"], ""), "unknown")
            sc_classes[get_canonical_class(cat_name)] += 1
            
    scene_info[sc_name] = {
        "token": sc_token,
        "samples": nbr,
        "location": loc,
        "is_night": is_night,
        "is_rain": is_rain,
        "desc": desc,
        "classes": sc_classes
    }

print("Total scenes:", len(scene_info))

# Let's test scene allocations for Train / Val / Test
# Total samples: 404
# Target: ~70% Train (approx 280 samples = 7 scenes)
# Target: ~15% Val (approx 60 samples = 1 or 2 scenes)
# Target: ~15% Test (approx 60 samples = 1 or 2 scenes)
# Note: 7 + 1 + 2 = 10 scenes, or 7 + 2 + 1 = 10 scenes, or 6 + 2 + 2 = 10 scenes (6 scenes = 243, 2 scenes = 81, 2 scenes = 80 -> exactly 60% / 20% / 20%)

# Let's evaluate candidate splits where:
# 1. Train has at least 1 night scene, at least 2 Boston, at least 2 Singapore
# 2. Val has both Cars and Pedestrians, at least 1 Boston/Singapore
# 3. Test has both Cars, Pedestrians, and diverse conditions
# 4. Zero data leakage (all scenes disjoint)

candidate_configs = [
    # Config A: 7 Train (283 samples, 70.0%), 1 Val (41 samples, 10.1%), 2 Test (80 samples, 19.8%)
    # Config B: 7 Train (282 samples, 69.8%), 2 Val (81 samples, 20.0%), 1 Test (41 samples, 10.1%)
    # Config C: 6 Train (243 samples, 60.1%), 2 Val (81 samples, 20.0%), 2 Test (80 samples, 19.8%)
]

all_scene_names = sorted(list(scene_info.keys()))

best_splits = []

for n_train, n_val, n_test in [(7, 1, 2), (7, 2, 1), (6, 2, 2)]:
    for train_scenes in itertools.combinations(all_scene_names, n_train):
        rem = [s for s in all_scene_names if s not in train_scenes]
        for val_scenes in itertools.combinations(rem, n_val):
            test_scenes = [s for s in rem if s not in val_scenes]
            
            # Check requirements
            # Train must have day and night
            train_night = sum(scene_info[s]["is_night"] for s in train_scenes)
            val_night = sum(scene_info[s]["is_night"] for s in val_scenes)
            test_night = sum(scene_info[s]["is_night"] for s in test_scenes)
            
            # Must have pedestrians in all splits
            train_peds = sum(scene_info[s]["classes"]["PEDESTRIAN"] for s in train_scenes)
            val_peds = sum(scene_info[s]["classes"]["PEDESTRIAN"] for s in val_scenes)
            test_peds = sum(scene_info[s]["classes"]["PEDESTRIAN"] for s in test_scenes)
            
            # Must have cars in all splits
            train_cars = sum(scene_info[s]["classes"]["CAR"] for s in train_scenes)
            val_cars = sum(scene_info[s]["classes"]["CAR"] for s in val_scenes)
            test_cars = sum(scene_info[s]["classes"]["CAR"] for s in test_scenes)
            
            # Must have trucks/buses in all splits
            train_trucks = sum(scene_info[s]["classes"]["TRUCK"] + scene_info[s]["classes"]["BUS"] for s in train_scenes)
            val_trucks = sum(scene_info[s]["classes"]["TRUCK"] + scene_info[s]["classes"]["BUS"] for s in val_scenes)
            test_trucks = sum(scene_info[s]["classes"]["TRUCK"] + scene_info[s]["classes"]["BUS"] for s in test_scenes)
            
            if val_peds < 80 or test_peds < 80:
                continue
            if val_cars < 400 or test_cars < 400:
                continue
            if val_trucks < 20 or test_trucks < 20:
                continue
            if train_night < 1 or (val_night == 0 and test_night == 0):
                continue
                
            n_tr_samples = sum(scene_info[s]["samples"] for s in train_scenes)
            n_va_samples = sum(scene_info[s]["samples"] for s in val_scenes)
            n_te_samples = sum(scene_info[s]["samples"] for s in test_scenes)
            
            # Score: balance of percentages close to 70 / 15 / 15
            pct_tr = n_tr_samples / 404.0
            pct_va = n_va_samples / 404.0
            pct_te = n_te_samples / 404.0
            
            score = abs(pct_tr - 0.70) + abs(pct_va - 0.15) + abs(pct_te - 0.15)
            best_splits.append({
                "score": score,
                "train_scenes": list(train_scenes),
                "val_scenes": list(val_scenes),
                "test_scenes": list(test_scenes),
                "n_train": n_tr_samples,
                "n_val": n_va_samples,
                "n_test": n_te_samples,
                "pct": (pct_tr*100, pct_va*100, pct_te*100),
                "has_night_in_test": test_night > 0,
                "has_night_in_val": val_night > 0,
                "val_peds": val_peds,
                "test_peds": test_peds,
                "val_cars": val_cars,
                "test_cars": test_cars
            })

best_splits.sort(key=lambda x: x["score"])
print(f"Found {len(best_splits)} valid candidate scene splits.")
print("\nTop 5 Candidate Splits:")
for i, bs in enumerate(best_splits[:5]):
    print(f"\nOption {i+1} (Score: {bs['score']:.3f}):")
    print(f"  Train ({bs['n_train']} samples, {bs['pct'][0]:.1f}%): {bs['train_scenes']}")
    print(f"  Val   ({bs['n_val']} samples, {bs['pct'][1]:.1f}%): {bs['val_scenes']} | Cars: {bs['val_cars']}, Peds: {bs['val_peds']}")
    print(f"  Test  ({bs['n_test']} samples, {bs['pct'][2]:.1f}%): {bs['test_scenes']} | Cars: {bs['test_cars']}, Peds: {bs['test_peds']}, Night: {bs['has_night_in_test']}")
