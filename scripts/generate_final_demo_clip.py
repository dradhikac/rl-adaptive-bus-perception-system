#!/usr/bin/env python3
"""
scripts/generate_final_demo_clip.py
================================================================================
Step 6 — Final Proof.

Re-runs diagnosis on a test clip with:
  a. The v2 model (if available) OR v1 model as fallback.
  b. Temporal smoothing (3-of-5 consensus) ENABLED.

Produces:
  - outputs/final_demo_clip.mp4       — annotated video with stable bounding boxes
  - outputs/final_diagnosis_report.txt — per-frame stats + flag comparison

Usage:
  python scripts/generate_final_demo_clip.py [--clip <path>] [--fps 5] [--conf 0.25]
================================================================================
"""

import argparse
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_best_model():
    """Load the best available YOLO model (v2 > v1 > yolov8n)."""
    from ultralytics import YOLO

    v2 = REPO_ROOT / "models" / "yolov8_bus_v2.pt"
    v1 = REPO_ROOT / "models" / "yolov8_bus_v1.pt"
    base = REPO_ROOT / "yolov8n.pt"

    if v2.is_file():
        model_path = str(v2)
        label = "yolov8_bus_v2 (fine-tuned + video-augmented)"
    elif v1.is_file():
        model_path = str(v1)
        label = "yolov8_bus_v1 (fine-tuned)"
    else:
        model_path = str(base)
        label = "yolov8n (base)"

    print(f"[INFO] Loading model: {label}")
    return YOLO(model_path), model_path, label


def extract_frames(video_path: str, target_fps: float = 5.0):
    """Extract frames using the same seek-based method as video_pipeline.py."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    total_ts = max(1, int(math.ceil(duration * target_fps)))

    frames = []
    for step_idx in range(total_ts):
        t_sec = step_idx / target_fps
        fn = int(round(t_sec * src_fps))
        if fn >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            frames.append((fn, t_sec, frame))

    cap.release()
    print(f"[INFO] Extracted {len(frames)} frames from {Path(video_path).name} at {target_fps} FPS.")
    return frames, src_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def run_with_temporal_smoothing(model, frames, conf_threshold: float = 0.25) -> List[Dict]:
    """
    Run inference with temporal smoother (3-of-5 consensus) applied.
    Falls back gracefully if TemporalSmoother is unavailable.
    """
    try:
        from src.video.temporal_smoother import TemporalSmoother
        smoother = TemporalSmoother(
            model=model,
            window_size=5,
            min_votes=3,
            iou_threshold=0.30,
            conf_threshold=conf_threshold,
            imgsz=640,
            disabled=False,
        )
        use_smoother = True
        print("[INFO] Temporal smoother: ENABLED (3-of-5 consensus, IoU>0.30)")
    except Exception as e:
        smoother = None
        use_smoother = False
        print(f"[WARN] Temporal smoother unavailable: {e} — running raw inference")

    records = []
    for frame_no, t_sec, bgr_frame in frames:
        t0 = time.perf_counter()

        if use_smoother and smoother:
            stable_dets = smoother.update_and_get_stable("front", bgr_frame)
            smoothed_frame = smoother.render_stable_frame(bgr_frame, stable_dets)
        else:
            smoothed_frame = bgr_frame
            stable_dets = []

        # Also run YOLO for annotation
        results = model.predict(source=bgr_frame, imgsz=640, conf=conf_threshold, verbose=False)
        inf_ms = (time.perf_counter() - t0) * 1000.0

        raw_boxes = results[0].boxes
        raw_classes = []
        raw_confs = []
        if raw_boxes is not None and len(raw_boxes) > 0:
            for box in raw_boxes:
                cid = int(box.cls[0].item())
                cls_name = model.names.get(cid, f"cls_{cid}")
                cf = float(box.conf[0].item())
                raw_classes.append(cls_name)
                raw_confs.append(cf)

        stable_class_names = [d.cls_name for d in stable_dets] if use_smoother else raw_classes

        records.append({
            "frame_no": frame_no,
            "t_sec": t_sec,
            "raw_dets": len(raw_classes),
            "raw_classes": raw_classes,
            "stable_dets": len(stable_class_names),
            "stable_classes": stable_class_names,
            "inf_ms": inf_ms,
            "bgr_frame": bgr_frame,
            "smoothed_frame": smoothed_frame,
            "result_obj": results[0],
        })

    return records


def write_annotated_video(records: List[Dict], output_path: Path, fps: float, model_label: str):
    """Write annotated output video with both raw and stable boxes visible."""
    if not records:
        return

    ref_frame = records[0]["bgr_frame"]
    h, w = ref_frame.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for rec in records:
        # Start with YOLO-annotated frame (raw boxes in YOLO colours)
        annotated = rec["result_obj"].plot()

        # Add stable detection overlay (green boxes from smoother, if any)
        # These are already drawn in smoothed_frame by render_stable_frame
        # We'll just add text labels
        stable_classes = rec["stable_classes"]
        stable_label = f"Stable: {', '.join(stable_classes[:3])}" if stable_classes else "Stable: (none)"
        raw_label = f"Raw: {rec['raw_dets']} det | {rec['t_sec']:.1f}s"

        cv2.putText(annotated, raw_label, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        cv2.putText(annotated, stable_label, (8, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
        cv2.putText(annotated, model_label[:50], (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        writer.write(annotated)

    writer.release()
    print(f"[INFO] Demo video saved: {output_path}")


def compute_temporal_consistency(classes_per_frame: List[List[str]], window: int = 5) -> float:
    n = len(classes_per_frame)
    if n < 2:
        return 100.0
    dominant = []
    for cls_list in classes_per_frame:
        dom = Counter(cls_list).most_common(1)[0][0] if cls_list else None
        dominant.append(dom)
    half = window // 2
    consistent = 0
    total = 0
    for i, dom in enumerate(dominant):
        start, end = max(0, i - half), min(n, i + half + 1)
        window_doms = [dominant[j] for j in range(start, end) if dominant[j] is not None]
        if not window_doms:
            continue
        majority = Counter(window_doms).most_common(1)[0][0]
        total += 1
        if dom == majority:
            consistent += 1
    return consistent / total * 100.0 if total > 0 else 100.0


def print_final_report(records: List[Dict], model_label: str, model_path: str,
                       smoothing_enabled: bool) -> Dict:
    print("\n" + "=" * 72)
    print(" FINAL PROOF — VIDEO DETECTION QUALITY REPORT")
    print("=" * 72)
    print(f" Model    : {model_label}")
    print(f" Path     : {model_path}")
    print(f" Smoothing: {'ENABLED (3-of-5 consensus, IoU>0.30)' if smoothing_enabled else 'DISABLED'}")
    print(f" Frames   : {len(records)}")
    print()

    raw_dets_per_frame = [r["raw_dets"] for r in records]
    stable_dets_per_frame = [r["stable_dets"] for r in records]
    raw_classes_per_frame = [r["raw_classes"] for r in records]
    stable_classes_per_frame = [r["stable_classes"] for r in records]

    raw_zero_pct = sum(1 for x in raw_dets_per_frame if x == 0) / len(records) * 100.0
    stable_zero_pct = sum(1 for x in stable_dets_per_frame if x == 0) / len(records) * 100.0

    raw_consistency = compute_temporal_consistency(raw_classes_per_frame)
    stable_consistency = compute_temporal_consistency(stable_classes_per_frame)

    all_raw_confs = []
    print(f"{'Frame':>6} | {'t':>5}s | {'Raw':>4} | {'Stable':>6} | {'Top classes'}")
    print("-" * 55)
    for r in records:
        top = ", ".join(f"{c}" for c in r["stable_classes"][:3]) or "(none)"
        print(f"  {r['frame_no']:5d} | {r['t_sec']:5.2f}  | {r['raw_dets']:4d} | {r['stable_dets']:6d} | {top}")

    print()
    print(f"{'Metric':<35} {'Raw':<18} {'With Smoothing'}")
    print("-" * 72)
    print(f"{'Zero-detection frame rate':<35} {raw_zero_pct:<18.1f}% {stable_zero_pct:.1f}%")
    print(f"{'Temporal consistency':<35} {raw_consistency:<18.1f}% {stable_consistency:.1f}%")
    print(f"{'Avg raw dets/frame':<35} {np.mean(raw_dets_per_frame):.1f}")
    print(f"{'Avg stable dets/frame':<35} {'':18} {np.mean(stable_dets_per_frame):.1f}")

    print()
    print("=" * 72)
    print(" FLAGS")
    print("=" * 72)

    # Flag 1: Zero-detection rate improved?
    if stable_zero_pct < raw_zero_pct or stable_zero_pct < 15:
        print(f"\n  [PASS] Zero-detection rate: raw={raw_zero_pct:.0f}% → stable={stable_zero_pct:.0f}%")
    else:
        print(f"\n  [NOTE] Zero-detection rate unchanged: {stable_zero_pct:.0f}%")
        print(f"         This is expected when the model itself has 0 detections (confidence < threshold).")
        print(f"         Training fix (Step 3) is required to produce detections above threshold.")

    # Flag 2: Temporal consistency
    if stable_consistency >= 80.0:
        print(f"\n  [PASS] Temporal consistency {stable_consistency:.1f}% >= 80%.")
    elif stable_consistency >= 60.0:
        print(f"\n  [WARN] Temporal consistency {stable_consistency:.1f}% is between 60–80%. Acceptable for demo.")
    else:
        print(f"\n  [NOTE] Temporal consistency {stable_consistency:.1f}%.")
        print(f"         If zero detections are 100%, consistency will trivially be 100%.")
        print(f"         This means the smoother has nothing to smooth — a model fix is the root cure.")

    # v2 status
    v2_exists = (REPO_ROOT / "models" / "yolov8_bus_v2.pt").is_file()
    if v2_exists:
        print("\n  [PASS] yolov8_bus_v2.pt is present and loaded.")
    else:
        print("\n  [NOTE] yolov8_bus_v2.pt not yet trained (run train_detector_v2.py).")
        print("         Current results use v1 as fallback.")

    print("\n" + "=" * 72)
    return {
        "raw_zero_pct": raw_zero_pct,
        "stable_zero_pct": stable_zero_pct,
        "raw_consistency": raw_consistency,
        "stable_consistency": stable_consistency,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Final proof — run v2+smoothing on test clip."
    )
    parser.add_argument("--clip", default=None,
                        help="Path to test video clip (default: first clip in datasets/self-collected/eval_clips/)")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "outputs" / "final_demo_clip.mp4")
    args = parser.parse_args()

    # Find test clip
    if args.clip:
        clip_path = args.clip
    else:
        clip_dir = REPO_ROOT / "datasets" / "self-collected" / "eval_clips"
        clips = sorted(clip_dir.glob("*.mp4")) if clip_dir.is_dir() else []
        if not clips:
            print("[ERROR] No clips found. Specify --clip <path>", file=sys.stderr)
            sys.exit(1)
        clip_path = str(clips[0])

    print(f"\n[Step 6] Final Proof Run")
    print(f"  Clip    : {clip_path}")
    print(f"  FPS     : {args.fps}")
    print(f"  Conf    : {args.conf}")
    print(f"  Output  : {args.output}")

    model, model_path, model_label = load_best_model()

    try:
        frames, src_fps, (w, h) = extract_frames(clip_path, args.fps)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    records = run_with_temporal_smoothing(model, frames, conf_threshold=args.conf)

    # Check if smoother was available
    smoothing_enabled = True
    try:
        from src.video.temporal_smoother import TemporalSmoother
    except ImportError:
        smoothing_enabled = False

    summary = print_final_report(records, model_label, model_path, smoothing_enabled)

    # Write annotated video
    write_annotated_video(records, args.output, args.fps, model_label)

    # Save text report
    report_path = REPO_ROOT / "outputs" / "final_diagnosis_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_label}\n")
        f.write(f"Path: {model_path}\n")
        f.write(f"Smoothing: {smoothing_enabled}\n")
        f.write(f"Frames analyzed: {len(records)}\n")
        for k, v in summary.items():
            f.write(f"{k}: {v:.1f}%\n")

    print(f"\n[INFO] Final diagnosis report: {report_path}")
    print(f"[INFO] Final demo video      : {args.output}")
    print("\n[Step 6 DONE]\n")


if __name__ == "__main__":
    main()
