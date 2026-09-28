import os
import json
from collections import defaultdict
import numpy as np

dataset_root = r"datasets\nuscenes"
meta_dir = os.path.join(dataset_root, "v1.0-mini")

def load_json(name):
    with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as f:
        return json.load(f)

scenes = load_json("scene.json")
samples = load_json("sample.json")
sample_data = load_json("sample_data.json")
annotations = load_json("sample_annotation.json")
instances = {inst["token"]: inst["category_token"] for inst in load_json("instance.json")}
categories = {cat["token"]: cat["name"] for cat in load_json("category.json")}
logs = {log["token"]: log for log in load_json("log.json")}

print("============================================================")
print("SCENE INVENTORY AND METADATA INSPECTION")
print("============================================================")

sample_anns = defaultdict(list)
for a in annotations:
    sample_anns[a["sample_token"]].append(a)

scene_samples = defaultdict(list)
for s in samples:
    scene_samples[s["scene_token"]].append(s)

# nuScenes category to canonical class mapping
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

for sc in scenes:
    sc_token = sc["token"]
    sc_name = sc["name"]
    nbr = sc["nbr_samples"]
    desc = sc["description"]
    log = logs.get(sc["log_token"], {})
    location = log.get("location", "unknown")
    
    # Analyze annotations in this scene
    sc_samples = scene_samples[sc_token]
    sc_ann_counts = defaultdict(int)
    sc_trainable_counts = defaultdict(int)
    total_pts_in_boxes = 0
    
    for sm in sc_samples:
        anns = sample_anns[sm["token"]]
        for a in anns:
            cat_name = categories.get(instances.get(a["instance_token"], ""), "unknown")
            cls = get_canonical_class(cat_name)
            sc_ann_counts[cls] += 1
            if a.get("num_lidar_pts", 0) >= 3:
                sc_trainable_counts[cls] += 1
                
    cars = sc_ann_counts["CAR"]
    peds = sc_ann_counts["PEDESTRIAN"]
    trucks = sc_ann_counts["TRUCK"]
    buses = sc_ann_counts["BUS"]
    barriers = sc_ann_counts["BARRIER"]
    cones = sc_ann_counts["TRAFFIC_CONE"]
    cycles = sc_ann_counts["MOTORCYCLE"] + sc_ann_counts["BICYCLE"]
    
    print(f"\nScene: {sc_name} | Samples: {nbr} | Location: {location}")
    print(f"  Description: {desc}")
    print(f"  Class counts: Cars={cars}, Peds={peds}, Trucks={trucks}, Buses={buses}, Barriers={barriers}, Cones={cones}, Cycles={cycles}")
    print(f"  Trainable (pts>=3): Cars={sc_trainable_counts['CAR']}, Peds={sc_trainable_counts['PEDESTRIAN']}")
