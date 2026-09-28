import os
import json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dataset_root = r"datasets\nuscenes"
meta_dir = os.path.join(dataset_root, "v1.0-mini")
splits_dir = r"datasets\splits"
viz_dir = r"visualizations\stage5_dataset"
art_dir = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\1c0087c6-dac4-4253-a522-0bdbc5b00eb8"

os.makedirs(splits_dir, exist_ok=True)
os.makedirs(viz_dir, exist_ok=True)
os.makedirs(art_dir, exist_ok=True)

def load_json(name):
    with open(os.path.join(meta_dir, name), "r", encoding="utf-8") as f:
        return json.load(f)

scenes = {sc["token"]: sc for sc in load_json("scene.json")}
samples = load_json("sample.json")
annotations = load_json("sample_annotation.json")
instances = {inst["token"]: inst["category_token"] for inst in load_json("instance.json")}
categories = {cat["token"]: cat["name"] for cat in load_json("category.json")}
sample_data = load_json("sample_data.json")

# Map sample_token to LIDAR_TOP filename
lidar_sd = {
    sd["sample_token"]: sd["filename"]
    for sd in sample_data
    if "LIDAR_TOP" in sd["filename"] and sd.get("is_key_frame")
}

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

# Defined Scene Partitions
split_definition = {
    "TRAIN": ["scene-0061", "scene-0103", "scene-0655", "scene-0757", "scene-0796", "scene-0916", "scene-1077"],
    "VALIDATION": ["scene-0553"],
    "TEST": ["scene-1094", "scene-1100"]
}

split_samples_records = []
split_stats = {}

for split_tag, sc_list in split_definition.items():
    cur_samples = []
    cls_counts = defaultdict(int)
    trainable_counts = defaultdict(int)
    
    for sc_name in sc_list:
        for sm in scene_name_to_samples[sc_name]:
            s_tok = sm["token"]
            cur_samples.append({
                "sample_token": s_tok,
                "scene_name": sc_name,
                "split": split_tag,
                "timestamp": sm["timestamp"],
                "lidar_filename": lidar_sd.get(s_tok, "")
            })
            
            for a in sample_anns[s_tok]:
                cat_name = categories.get(instances.get(a["instance_token"], ""), "unknown")
                c_cls = get_canonical_class(cat_name)
                cls_counts[c_cls] += 1
                if a.get("num_lidar_pts", 0) >= 3:
                    trainable_counts[c_cls] += 1
                    
    split_samples_records.extend(cur_samples)
    split_stats[split_tag] = {
        "num_samples": len(cur_samples),
        "percentage": len(cur_samples) / len(samples) * 100,
        "scenes": sc_list,
        "class_counts_total": dict(cls_counts),
        "class_counts_trainable_pts_gte_3": dict(trainable_counts)
    }

# Save Controlled Split JSON
split_payload = {
    "total_samples": len(samples),
    "split_policy": "Strict Scene-Level Non-Overlapping (Zero Temporal Data Leakage)",
    "splits": split_stats,
    "sample_index": split_samples_records
}

json_path = os.path.join(splits_dir, "controlled_404_split.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(split_payload, f, indent=2)

print(f"Saved controlled split to {json_path}")

# Generate Visualization of Splits
classes = ["CAR", "PEDESTRIAN", "TRUCK", "BUS", "BARRIER", "TRAFFIC_CONE", "MOTORCYCLE", "BICYCLE"]
y_pos = np.arange(len(classes))
bar_height = 0.26

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0e1117")
fig.suptitle("AutoBus Stage 5 — Controlled Dataset Splits (Zero-Leakage Scene Partition)\nTrain: 283 samples (70.0%) | Validation: 41 samples (10.1%) | Test: 80 samples (19.8%)",
             fontsize=14, color="#f0f6fc", fontweight="bold", y=0.98)

# Subplot 1: Total annotations across splits
ax1.set_facecolor("#05070a")
ax1.set_title("1. Total Annotations per Class across Splits", color="#58a6ff", fontsize=12, fontweight="bold", pad=10)

tr_vals = [split_stats["TRAIN"]["class_counts_total"].get(c, 0) for c in classes]
va_vals = [split_stats["VALIDATION"]["class_counts_total"].get(c, 0) for c in classes]
te_vals = [split_stats["TEST"]["class_counts_total"].get(c, 0) for c in classes]

ax1.barh(y_pos - bar_height, tr_vals, bar_height, label="TRAIN (7 Scenes, 283 samples)", color="#1f6feb", edgecolor="#388bfd")
ax1.barh(y_pos, va_vals, bar_height, label="VALIDATION (1 Scene, 41 samples)", color="#d29922", edgecolor="#e3b341")
ax1.barh(y_pos + bar_height, te_vals, bar_height, label="TEST (2 Scenes, 80 samples)", color="#238636", edgecolor="#2ea043")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(classes, color="#f0f6fc", fontsize=10, fontweight="bold")
ax1.invert_yaxis()
ax1.set_xlabel("Number of 3D Bounding Boxes", color="#8b949e", fontsize=10)
ax1.tick_params(colors="#8b949e")
ax1.grid(True, color="#21262d", linestyle=":", alpha=0.6, axis="x")
ax1.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)

# Subplot 2: Trainable (Pts >= 3) per Class across Splits
ax2.set_facecolor("#05070a")
ax2.set_title("2. Trainable Instances (Points ≥ 3) per Class across Splits", color="#58a6ff", fontsize=12, fontweight="bold", pad=10)

tr_tr = [split_stats["TRAIN"]["class_counts_trainable_pts_gte_3"].get(c, 0) for c in classes]
va_tr = [split_stats["VALIDATION"]["class_counts_trainable_pts_gte_3"].get(c, 0) for c in classes]
te_tr = [split_stats["TEST"]["class_counts_trainable_pts_gte_3"].get(c, 0) for c in classes]

b1 = ax2.barh(y_pos - bar_height, tr_tr, bar_height, label="TRAIN (Trainable)", color="#1f6feb", edgecolor="#388bfd")
b2 = ax2.barh(y_pos, va_tr, bar_height, label="VALIDATION (Trainable)", color="#d29922", edgecolor="#e3b341")
b3 = ax2.barh(y_pos + bar_height, te_tr, bar_height, label="TEST (Trainable)", color="#238636", edgecolor="#2ea043")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(classes, color="#f0f6fc", fontsize=10, fontweight="bold")
ax2.invert_yaxis()
ax2.set_xlabel("Number of Observable 3D Bounding Boxes (Points ≥ 3)", color="#8b949e", fontsize=10)
ax2.tick_params(colors="#8b949e")
ax2.grid(True, color="#21262d", linestyle=":", alpha=0.6, axis="x")
ax2.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#f0f6fc", fontsize=9)

# Value annotations
for rect in b1:
    w = rect.get_width()
    if w > 0:
        ax2.text(w + 30, rect.get_y() + rect.get_height()/2, f"{int(w)}", ha="left", va="center", color="#58a6ff", fontsize=8)
for rect in b2:
    w = rect.get_width()
    if w > 0:
        ax2.text(w + 30, rect.get_y() + rect.get_height()/2, f"{int(w)}", ha="left", va="center", color="#e3b341", fontsize=8)
for rect in b3:
    w = rect.get_width()
    if w > 0:
        ax2.text(w + 30, rect.get_y() + rect.get_height()/2, f"{int(w)}", ha="left", va="center", color="#3fb950", fontsize=8)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
viz_png = "stage5_controlled_split_distribution.png"
plt.savefig(os.path.join(viz_dir, viz_png), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.savefig(os.path.join(art_dir, viz_png), dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print(f"Saved visualization to {viz_png}")

print("\nSTAGE 5 DATASET SPLITTING COMPLETED.")
