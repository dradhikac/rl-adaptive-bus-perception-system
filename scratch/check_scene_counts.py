import os
import json

meta_dir = r"datasets\nuscenes\v1.0-mini"
with open(os.path.join(meta_dir, "scene.json")) as f:
    scenes = json.load(f)

for sc in scenes:
    print(f"{sc['name']}: {sc['nbr_samples']} samples | {sc['description'][:60]}")
