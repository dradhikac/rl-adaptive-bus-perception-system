"""
AutoBus LiDAR 3D Object Detection CLI Tool
Run inference on a single LiDAR point cloud file (.pcd.bin / .bin) or demo sample.

Usage:
  python predict_lidar.py --input <path_to_point_cloud.pcd.bin>
  python predict_lidar.py --demo
  python predict_lidar.py --input <path_to_point_cloud.pcd.bin> --threshold 0.40 --output detections.json
"""

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.abspath("."))

from src.lidar_perception.detector import AutoBusLidar3DDetector

def main():
    parser = argparse.ArgumentParser(description="AutoBus LiDAR 3D Object Detector")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to binary LiDAR point cloud file (.pcd.bin)")
    parser.add_argument("--checkpoint", "-c", type=str, default=r"models/checkpoints_exp3_aug/best_model.pth", help="Model checkpoint path")
    parser.add_argument("--threshold", "-t", type=float, default=0.40, help="Confidence score threshold (default: 0.40)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output JSON path")
    parser.add_argument("--demo", action="store_true", help="Run detection on a sample from the unseen test set")
    args = parser.parse_args()
    
    # If demo or no input provided, select a sample from test set
    if args.demo or args.input is None:
        split_file = r"datasets/splits/controlled_404_split.json"
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                data = json.load(f)
            test_samples = [s for s in data["sample_index"] if s["split"] == "TEST"]
            if test_samples:
                # Select a rich test sample from scene-1094
                sample_token = test_samples[5]["sample_token"]
                # Look up file path
                sd_file = r"datasets/nuscenes/v1.0-mini/sample_data.json"
                with open(sd_file, "r") as f:
                    sd_list = json.load(f)
                sd_match = next((sd for sd in sd_list if sd["sample_token"] == sample_token and sd["filename"].endswith(".pcd.bin")), None)
                if sd_match:
                    args.input = os.path.join(r"datasets/nuscenes", sd_match["filename"])
                    print(f"[DEMO MODE] Selected unseen test sample: {args.input} (Scene: {test_samples[5]['scene_name']})")
                    
    if not args.input or not os.path.exists(args.input):
        print(f"Error: Input LiDAR file not found: {args.input}")
        sys.exit(1)
        
    print("=" * 70)
    print("AUTOBUS LiDAR 3D OBJECT DETECTOR (CPU INFERENCE)")
    print("=" * 70)
    print(f"Input LiDAR File : {args.input}")
    print(f"Model Checkpoint : {args.checkpoint}")
    print(f"Score Threshold  : {args.threshold}")
    print("-" * 70)
    
    detector = AutoBusLidar3DDetector(
        checkpoint_path=args.checkpoint,
        score_thresh=args.threshold
    )
    
    predictions = detector.predict(args.input)
    
    print(f"\nDetected {len(predictions)} 3D objects:\n")
    print(json.dumps(predictions, indent=2))
    print("-" * 70)
    print(f"Coordinate Reference: Top LiDAR Sensor Frame (Origin on vehicle roof rack)")
    print(f"  +X = Right, +Y = Forward, +Z = Up, Ground road surface at Z ~ -1.84m")
    print("=" * 70)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved structured detections to: {args.output}")

if __name__ == "__main__":
    main()
