#!/usr/bin/env python3
"""
scripts/extract_video_frames.py
================================================================================
Step 2 — Build a Video-Representative Training Set.

Usage:
  python scripts/extract_video_frames.py <video_path_or_folder> [--fps 2]
  python scripts/extract_video_frames.py datasets/self-collected/eval_clips/ --fps 2
  python scripts/extract_video_frames.py path/to/video.mp4 --fps 2

What this does:
  a. Pulls frames from local video files in data/raw/self-collected/ AND any
     provided path/folder at the given FPS.
  b. Auto-labels extracted frames using the CURRENT model (yolov8_bus_v1.pt)
     in YOLO label format (cls cx cy w h) into data/raw/video-frames/labels/.
  c. Saves review copies (frame + drawn boxes) to outputs/pseudo_label_review/
     so a human can scan and delete/fix obviously wrong labels.
  d. Merges with existing data/processed/ (adds images + labels to train split,
     split by source video NOT random shuffle so near-duplicate frames don't leak).
  e. Writes an updated data/processed/data_v2.yaml for training.

NOTE: Pseudo-labels are NOT treated as ground truth silently. The review folder
exists specifically for human verification before training.
================================================================================
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import random

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_FRAMES_DIR = REPO_ROOT / "data" / "raw" / "video-frames"
REVIEW_DIR = REPO_ROOT / "outputs" / "pseudo_label_review"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Classes from data.yaml — keep consistent
UNIFIED_CLASSES = [
    "person", "bicycle", "motorcycle", "autorickshaw",
    "car", "bus", "truck", "bus_stop_sign", "traffic_light", "pothole"
]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts", ".mts"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_video_files(source: Path) -> List[Path]:
    """Recursively find all video files under a directory or return a single file."""
    if source.is_file() and source.suffix.lower() in VIDEO_EXTENSIONS:
        return [source]
    if source.is_dir():
        found = []
        for ext in VIDEO_EXTENSIONS:
            found.extend(source.rglob(f"*{ext}"))
        return sorted(found)
    return []


def extract_frames_at_fps(
    video_path: Path,
    output_dir: Path,
    fps: float = 2.0,
    prefix: str = ""
) -> List[Path]:
    """
    Extract frames from video at target FPS using the same seek-based method
    as video_pipeline.py. Returns list of saved frame image paths.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    total_timesteps = max(1, int(math.ceil(duration * fps)))

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    stem = prefix + video_path.stem

    for step_idx in range(total_timesteps):
        t_sec = step_idx / fps
        target_frame_no = int(round(t_sec * src_fps))
        if target_frame_no >= total_frames:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_no)
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        fname = f"{stem}_f{target_frame_no:06d}.jpg"
        out_path = output_dir / fname
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved_paths.append(out_path)

    cap.release()
    return saved_paths


def yolo_auto_label(
    model,
    image_paths: List[Path],
    label_dir: Path,
    conf_threshold: float = 0.25,
    imgsz: int = 640,
) -> int:
    """
    Auto-label images using the YOLO model. Saves labels in YOLO format:
      class_id cx cy w h  (normalized 0–1)
    Returns number of images with at least 1 detection.
    """
    label_dir.mkdir(parents=True, exist_ok=True)
    labeled = 0

    for img_path in image_paths:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        h, w = bgr.shape[:2]
        results = model.predict(source=bgr, imgsz=imgsz, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes

        label_path = label_dir / (img_path.stem + ".txt")

        if boxes is None or len(boxes) == 0:
            # Write empty label file (background sample)
            label_path.write_text("", encoding="utf-8")
            continue

        lines = []
        for box in boxes:
            cls_id = int(box.cls[0].item())
            # Map to unified class list if needed (model names vs unified names)
            model_class_name = model.names.get(cls_id, "")
            if model_class_name in UNIFIED_CLASSES:
                unified_id = UNIFIED_CLASSES.index(model_class_name)
            else:
                # Map common COCO names to unified classes
                coco_to_unified = {
                    "person": "person", "bicycle": "bicycle",
                    "motorcycle": "motorcycle", "car": "car",
                    "bus": "bus", "truck": "truck",
                    "traffic light": "traffic_light",
                }
                mapped = coco_to_unified.get(model_class_name.lower())
                if mapped and mapped in UNIFIED_CLASSES:
                    unified_id = UNIFIED_CLASSES.index(mapped)
                else:
                    continue  # Skip unknown classes

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{unified_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            labeled += 1

    return labeled


def save_review_images(
    model,
    image_paths: List[Path],
    review_dir: Path,
    conf_threshold: float = 0.25,
    max_review: int = 100,
) -> None:
    """
    Save annotated images (frame + drawn boxes) to review_dir for human inspection.
    Saves up to max_review evenly-spaced samples.
    """
    review_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(image_paths) // max_review)
    saved = 0

    for i in range(0, len(image_paths), step):
        if saved >= max_review:
            break
        img_path = image_paths[i]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        results = model.predict(source=bgr, conf=conf_threshold, verbose=False)
        annotated = results[0].plot()
        out_path = review_dir / img_path.name
        cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        saved += 1

    print(f"[INFO] Saved {saved} review images to: {review_dir}")
    print(f"       ** HUMAN REVIEW REQUIRED ** — scan these and delete/fix wrong labels")
    print(f"       Corresponding labels are in: {VIDEO_FRAMES_DIR / 'labels'}")


def merge_into_processed(
    video_frames_img_dir: Path,
    video_frames_lbl_dir: Path,
    processed_dir: Path,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[int, int]:
    """
    Merges video-frame images+labels into data/processed/ train split.
    Split is by SOURCE VIDEO (not random shuffle) to avoid near-duplicate leakage.

    Returns (num_train_added, num_val_added).
    """
    all_img_files = sorted(video_frames_img_dir.glob("*.jpg"))
    if not all_img_files:
        print("[WARN] No extracted frames found to merge.")
        return 0, 0

    # Group by source video stem (everything before _f<frame_no>)
    from collections import defaultdict
    groups = defaultdict(list)
    for img_path in all_img_files:
        # stem looks like: PREFIX_clip_stem_f000000
        # Group key = everything before the last _fXXXXXX
        parts = img_path.stem.rsplit("_f", 1)
        group_key = parts[0] if len(parts) == 2 else img_path.stem
        groups[group_key].append(img_path)

    group_keys = sorted(groups.keys())
    random.seed(seed)
    random.shuffle(group_keys)

    n_val_groups = max(1, int(len(group_keys) * val_fraction))
    val_groups = set(group_keys[:n_val_groups])
    train_groups = set(group_keys[n_val_groups:])

    train_img_dir = processed_dir / "images" / "train"
    train_lbl_dir = processed_dir / "labels" / "train"
    val_img_dir = processed_dir / "images" / "val"
    val_lbl_dir = processed_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    num_train = 0
    num_val = 0

    for group_key, img_paths in groups.items():
        split = "val" if group_key in val_groups else "train"
        dest_img_dir = val_img_dir if split == "val" else train_img_dir
        dest_lbl_dir = val_lbl_dir if split == "val" else train_lbl_dir

        for img_path in img_paths:
            lbl_src = video_frames_lbl_dir / (img_path.stem + ".txt")
            dest_img = dest_img_dir / img_path.name
            dest_lbl = dest_lbl_dir / (img_path.stem + ".txt")

            shutil.copy2(img_path, dest_img)
            if lbl_src.is_file():
                shutil.copy2(lbl_src, dest_lbl)
            else:
                dest_lbl.write_text("", encoding="utf-8")

            if split == "train":
                num_train += 1
            else:
                num_val += 1

    return num_train, num_val


def write_data_v2_yaml(processed_dir: Path, output_path: Path) -> None:
    """Write data_v2.yaml pointing to merged train/val/test splits."""
    cfg = {
        "path": str(processed_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(UNIFIED_CLASSES),
        "names": {i: name for i, name in enumerate(UNIFIED_CLASSES)},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"[INFO] Wrote: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Extract video frames, auto-label with current model, merge with photo dataset."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=str(REPO_ROOT / "datasets" / "self-collected" / "eval_clips"),
        help="Path to video file or folder (default: datasets/self-collected/eval_clips/)"
    )
    parser.add_argument("--fps", type=float, default=2.0,
                        help="Frame extraction FPS (default: 2)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for auto-labeling (default: 0.25)")
    parser.add_argument("--model", default=None,
                        help="Model path (default: models/yolov8_bus_v1.pt or yolov8n.pt)")
    parser.add_argument("--no-merge", action="store_true",
                        help="Skip merging into data/processed/ (extract + label only)")
    parser.add_argument("--val-frac", type=float, default=0.10,
                        help="Fraction of video groups to use for validation (default: 0.10)")
    args = parser.parse_args()

    source_path = Path(args.source)
    video_files = find_video_files(source_path)

    # Also include any videos in data/raw/self-collected/
    raw_self_collected = REPO_ROOT / "data" / "raw" / "self-collected"
    if raw_self_collected.is_dir():
        extra = find_video_files(raw_self_collected)
        video_files = list({v.resolve(): v for v in video_files + extra}.values())

    if not video_files:
        print(f"[ERROR] No video files found at: {source_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[INFO] Found {len(video_files)} video file(s) to process:")
    for vf in video_files:
        print(f"  {vf}")

    # Load model
    from ultralytics import YOLO
    if args.model:
        model_path = args.model
    else:
        v1 = REPO_ROOT / "models" / "yolov8_bus_v1.pt"
        model_path = str(v1) if v1.is_file() else str(REPO_ROOT / "yolov8n.pt")
    print(f"\n[INFO] Auto-labeling model: {model_path}")
    model = YOLO(model_path)

    all_extracted: List[Path] = []
    frames_img_dir = VIDEO_FRAMES_DIR / "images"
    frames_lbl_dir = VIDEO_FRAMES_DIR / "labels"
    frames_img_dir.mkdir(parents=True, exist_ok=True)
    frames_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Step A: Extract frames from each video
    for vf in video_files:
        print(f"\n[EXTRACT] {vf.name} at {args.fps} FPS ...")
        extracted = extract_frames_at_fps(
            video_path=vf,
            output_dir=frames_img_dir,
            fps=args.fps,
            prefix="vid_"
        )
        print(f"  Extracted {len(extracted)} frames.")
        all_extracted.extend(extracted)

    print(f"\n[INFO] Total frames extracted: {len(all_extracted)}")

    # Step B: Auto-label with current model
    print(f"\n[AUTO-LABEL] Running pseudo-labeling on {len(all_extracted)} frames ...")
    print(f"             Labels saved to: {frames_lbl_dir}")
    labeled_count = yolo_auto_label(
        model=model,
        image_paths=all_extracted,
        label_dir=frames_lbl_dir,
        conf_threshold=args.conf,
        imgsz=640,
    )
    print(f"[INFO] {labeled_count}/{len(all_extracted)} frames have at least 1 detection.")

    # Step C: Save review images for human inspection
    print(f"\n[REVIEW] Generating pseudo-label review images for human inspection ...")
    save_review_images(
        model=model,
        image_paths=all_extracted,
        review_dir=REVIEW_DIR,
        conf_threshold=args.conf,
        max_review=min(100, len(all_extracted)),
    )

    if args.no_merge:
        print("\n[INFO] --no-merge flag set. Skipping merge into data/processed/.")
        print(f"[INFO] Frames at: {frames_img_dir}")
        print(f"[INFO] Labels at: {frames_lbl_dir}")
        print(f"[INFO] Review at: {REVIEW_DIR}")
        return

    # Step D: Merge into data/processed/ (split by source video, not shuffle)
    print(f"\n[MERGE] Merging video frames into data/processed/ ...")
    print(f"        Val fraction: {args.val_frac} (split by source video, no frame leakage)")
    n_train, n_val = merge_into_processed(
        video_frames_img_dir=frames_img_dir,
        video_frames_lbl_dir=frames_lbl_dir,
        processed_dir=PROCESSED_DIR,
        val_fraction=args.val_frac,
        seed=42,
    )
    print(f"[INFO] Added {n_train} frames to train, {n_val} frames to val.")

    # Step E: Write data_v2.yaml
    data_v2_path = PROCESSED_DIR / "data_v2.yaml"
    write_data_v2_yaml(PROCESSED_DIR, data_v2_path)

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)
    print(f"  Total frames extracted    : {len(all_extracted)}")
    print(f"  Frames with detections    : {labeled_count}")
    print(f"  Added to train split      : {n_train}")
    print(f"  Added to val split        : {n_val}")
    print(f"  Review images             : {REVIEW_DIR}")
    print(f"  Updated dataset config    : {data_v2_path}")
    print()
    print("  *** HUMAN REVIEW REQUIRED ***")
    print(f"  Open {REVIEW_DIR} and delete/fix any obviously wrong pseudo-labels.")
    print(f"  Corresponding .txt files are in {frames_lbl_dir}")
    print(f"  Then run: python training/train_detector_v2.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
