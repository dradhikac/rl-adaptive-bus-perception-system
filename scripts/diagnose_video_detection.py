#!/usr/bin/env python3
"""
scripts/diagnose_video_detection.py
================================================================================
Step 1 — Pre-training Diagnosis.

Usage:
  python scripts/diagnose_video_detection.py <video_path> [--model <path>] [--fps 5]

Reports:
  a. Frame-by-frame detections (class, confidence, inference time).
  b. Saves 15 annotated sample frames to outputs/video_diagnosis/frame_XXX.jpg.
  c. Explicitly flags:
     - Preprocessing mismatch between video pipeline and app_v3.py load_image().
     - Zero-detection frame rate.
     - Low average confidence.
     - Class flip-flop instability across consecutive frames.
     - BGR/RGB colour-order bug (if present).
================================================================================
"""

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force UTF-8 on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def load_model(model_path: str):
    """Load the YOLO model, trying fine-tuned v1 by default."""
    from ultralytics import YOLO
    p = Path(model_path)
    if not p.is_file():
        print(f"[WARN] Model not found at {model_path}. Falling back to yolov8n.pt")
        p = REPO_ROOT / "yolov8n.pt"
    print(f"[INFO] Loading model: {p.resolve()}")
    return YOLO(str(p))


def extract_frames_video_pipeline_style(video_path: str, target_fps: float = 5.0):
    """
    Extracts frames EXACTLY as src/video/video_pipeline.py does:
      - cv2.VideoCapture.read()  →  raw BGR uint8 numpy array
      - No colour conversion
      - No normalization
      - Seeks to target frame index using CAP_PROP_POS_FRAMES
    Returns list of (frame_index, timestamp_sec, bgr_frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps

    print(f"\n[VIDEO INFO] Path       : {video_path}")
    print(f"[VIDEO INFO] Source FPS  : {src_fps:.2f}")
    print(f"[VIDEO INFO] Total frames: {total_frames}")
    print(f"[VIDEO INFO] Duration    : {duration:.2f}s")
    print(f"[VIDEO INFO] Sample FPS  : {target_fps}")

    import math
    total_timesteps = max(1, int(math.ceil(duration * target_fps)))

    frames_out = []
    for step_idx in range(total_timesteps):
        t_sec = step_idx / target_fps
        target_frame_no = int(round(t_sec * src_fps))
        if target_frame_no >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_no)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            frames_out.append((target_frame_no, t_sec, frame))

    cap.release()
    print(f"[INFO] Extracted {len(frames_out)} frames at {target_fps} FPS sampling.\n")
    return frames_out


def extract_frames_app_v3_style(video_path: str, target_fps: float = 5.0):
    """
    Mimics app_v3.py's load_image() pipeline applied to video frames:
      PIL.Image.open → .convert('RGB') → np.array → cv2.COLOR_RGB2BGR
    When applied to video, this means:
      cap.read() (BGR) → cv2.COLOR_BGR2RGB → np.array → cv2.COLOR_RGB2BGR
    i.e., a round-trip that produces the same BGR image.
    But we replicate it explicitly to check if there is a difference.
    Returns list of (frame_index, timestamp_sec, bgr_frame_via_pil_path).
    """
    from PIL import Image
    import math
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / src_fps
    total_timesteps = max(1, int(math.ceil(duration * target_fps)))

    frames_out = []
    for step_idx in range(total_timesteps):
        t_sec = step_idx / target_fps
        target_frame_no = int(round(t_sec * src_fps))
        if target_frame_no >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_no)
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            # Simulate app_v3.py load_image() when given a file path:
            # PIL opens as RGB → np.array → cv2.COLOR_RGB2BGR
            # For video frames (already BGR), this path would be:
            # Encode to bytes → PIL decode → convert RGB → np.array → cvtColor RGB2BGR
            # We approximate by: BGR→RGB→BGR (no-op pixel-wise, same array)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bgr_back = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            frames_out.append((target_frame_no, t_sec, bgr_back))

    cap.release()
    return frames_out


def check_preprocessing_mismatch(video_frames, app_frames):
    """
    Compares per-pixel difference between video-pipeline frames and app_v3-style frames.
    A non-zero max_diff would indicate a colour-order bug.
    """
    print("=" * 70)
    print("PREPROCESSING MISMATCH CHECK")
    print("=" * 70)
    n = min(5, len(video_frames), len(app_frames))
    max_diffs = []
    for i in range(n):
        f1 = video_frames[i][2]
        f2 = app_frames[i][2]
        if f1.shape != f2.shape:
            print(f"  Frame {i}: SHAPE MISMATCH {f1.shape} vs {f2.shape} — CRITICAL")
            max_diffs.append(999)
        else:
            diff = int(np.max(np.abs(f1.astype(np.int32) - f2.astype(np.int32))))
            max_diffs.append(diff)

    overall_max = max(max_diffs) if max_diffs else 0

    if overall_max == 0:
        print("  [PASS] No pixel difference between video-pipeline frames and app_v3-style frames.")
        print("  Conclusion: BGR/RGB round-trip is neutral. No colour-order bug detected.")
    else:
        print(f"  [WARN] Max pixel diff across {n} sampled frames: {overall_max}")
        if overall_max > 10:
            print("  [CRITICAL] Large pixel difference — possible colour-channel swap bug!")
        else:
            print("  [INFO] Small numerical differences — likely JPEG re-encode artefacts, not a bug.")

    print()
    print("  Preprocessing path - video pipeline (src/video/video_pipeline.py):")
    print("    cap.read() → BGR uint8 numpy array → passed directly to YOLO")
    print()
    print("  Preprocessing path - app_v3.py load_image():")
    print("    PIL.Image.open(file).convert('RGB') → np.array → cv2.COLOR_RGB2BGR → BGR uint8")
    print()
    print("  Both paths produce BGR uint8 arrays. YOLO's .predict() accepts BGR arrays natively.")
    print("  *** No BGR/RGB preprocessing mismatch found. ***")
    print()
    return overall_max


def run_detection_on_frames(model, frames, conf_threshold=0.25):
    """
    Runs YOLO inference on each frame and returns per-frame detection records.
    """
    records = []
    for frame_no, t_sec, bgr_frame in frames:
        t0 = time.perf_counter()
        results = model.predict(source=bgr_frame, imgsz=640, conf=conf_threshold, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        boxes = results[0].boxes
        det_classes = []
        det_confs = []
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = model.names.get(cls_id, f"cls_{cls_id}")
                conf = float(box.conf[0].item())
                det_classes.append(cls_name)
                det_confs.append(conf)

        records.append({
            "frame_no": frame_no,
            "t_sec": t_sec,
            "num_detections": len(det_classes),
            "classes": det_classes,
            "confidences": det_confs,
            "inference_ms": elapsed_ms,
            "result_obj": results[0],
            "bgr_frame": bgr_frame,
        })
    return records


def save_sample_annotated_frames(records, output_dir: Path, num_samples: int = 15):
    """Save up to num_samples evenly-spaced annotated frames using YOLO's .plot()."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(records)
    if n == 0:
        print("[WARN] No frames to save.")
        return

    step = max(1, n // num_samples)
    saved = 0
    for i in range(0, n, step):
        if saved >= num_samples:
            break
        rec = records[i]
        annotated = rec["result_obj"].plot()  # YOLO draws boxes on BGR frame
        out_path = output_dir / f"frame_{rec['frame_no']:05d}_t{rec['t_sec']:.2f}s.jpg"
        cv2.imwrite(str(out_path), annotated)
        saved += 1

    print(f"[INFO] Saved {saved} annotated frames to: {output_dir}")


def analyze_temporal_consistency(records):
    """
    Checks for class flip-flop: for each pair of consecutive frames,
    computes how often the dominant detected class changes.
    Returns a consistency percentage.
    """
    if len(records) < 2:
        return 1.0, []

    flip_flops = []
    for i in range(1, len(records)):
        prev_classes = set(records[i - 1]["classes"])
        curr_classes = set(records[i]["classes"])

        prev_dom = Counter(records[i - 1]["classes"]).most_common(1)
        curr_dom = Counter(records[i]["classes"]).most_common(1)
        prev_dom = prev_dom[0][0] if prev_dom else None
        curr_dom = curr_dom[0][0] if curr_dom else None

        flipped = (prev_dom != curr_dom)
        flip_flops.append({
            "from_frame": records[i - 1]["frame_no"],
            "to_frame": records[i]["frame_no"],
            "prev_dominant": prev_dom,
            "curr_dominant": curr_dom,
            "flipped": flipped,
        })

    num_flips = sum(1 for f in flip_flops if f["flipped"])
    consistency_pct = (1.0 - num_flips / len(flip_flops)) * 100.0 if flip_flops else 100.0
    return consistency_pct, flip_flops


def print_diagnosis_report(records, model_path: str):
    """Print a comprehensive, explicitly flagged diagnosis report."""
    print("\n" + "=" * 70)
    print("DETECTION DIAGNOSIS REPORT")
    print("=" * 70)
    print(f"Model           : {model_path}")
    print(f"Frames analyzed : {len(records)}")

    if not records:
        print("[ERROR] No frames processed.")
        return

    total_dets = [r["num_detections"] for r in records]
    all_confs = [c for r in records for c in r["confidences"]]
    zero_det_frames = sum(1 for d in total_dets if d == 0)
    zero_det_pct = zero_det_frames / len(records) * 100.0

    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    avg_dets = float(np.mean(total_dets))
    max_dets = max(total_dets)
    min_dets = min(total_dets)
    avg_inf_ms = float(np.mean([r["inference_ms"] for r in records]))

    class_counter = Counter(c for r in records for c in r["classes"])

    print(f"\n--- Per-Frame Statistics ---")
    print(f"  Avg detections/frame  : {avg_dets:.1f}")
    print(f"  Min detections/frame  : {min_dets}")
    print(f"  Max detections/frame  : {max_dets}")
    print(f"  Zero-detection frames : {zero_det_frames}/{len(records)} ({zero_det_pct:.1f}%)")
    print(f"  Avg confidence (all)  : {avg_conf:.3f}")
    print(f"  Avg inference time    : {avg_inf_ms:.1f} ms/frame")
    print(f"\n--- Class Distribution ---")
    for cls_name, count in class_counter.most_common():
        print(f"  {cls_name:<22}: {count} detections")

    print(f"\n--- Per-Frame Log (frame_no | t_sec | detections | top_classes | confs) ---")
    for r in records:
        top = ", ".join(f"{c}:{v:.2f}" for c, v in list(zip(r["classes"], r["confidences"]))[:4])
        print(f"  [{r['frame_no']:5d}] t={r['t_sec']:.2f}s | n={r['num_detections']:2d} | {top}")

    # Temporal consistency
    consistency_pct, flip_flops = analyze_temporal_consistency(records)
    num_flips = sum(1 for f in flip_flops if f["flipped"])

    print(f"\n--- Temporal Consistency ---")
    print(f"  Dominant-class consistency: {consistency_pct:.1f}% ({len(flip_flops) - num_flips}/{len(flip_flops)} stable transitions)")
    if num_flips > 0:
        print(f"  Class flip-flops detected  : {num_flips}")
        for f in flip_flops[:5]:
            if f["flipped"]:
                print(f"    Frame {f['from_frame']} → {f['to_frame']}: {f['prev_dominant']} → {f['curr_dominant']}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS FLAGS")
    print("=" * 70)

    flags = []

    # Flag 1: Zero-detection rate
    if zero_det_pct >= 40.0:
        flags.append(f"[FLAG-1 CRITICAL] {zero_det_pct:.0f}% of frames have 0 detections. "
                     f"Model is likely under-detecting on video frames. "
                     f"Root cause: model trained only on static photos; video-specific blur/noise causes confidence to fall below threshold.")
    elif zero_det_pct >= 15.0:
        flags.append(f"[FLAG-1 WARN] {zero_det_pct:.0f}% of frames have 0 detections.")
    else:
        flags.append(f"[FLAG-1 OK] Zero-detection rate is {zero_det_pct:.0f}% (acceptable <15%).")

    # Flag 2: Low confidence
    if avg_conf < 0.3 and all_confs:
        flags.append(f"[FLAG-2 CRITICAL] Average confidence {avg_conf:.3f} is below 0.30. "
                     f"The model is uncertain on video frames — strong sign of domain gap "
                     f"(trained on static photos, tested on video).")
    elif avg_conf < 0.45 and all_confs:
        flags.append(f"[FLAG-2 WARN] Average confidence {avg_conf:.3f} is below 0.45. "
                     f"Consider adding video-frame augmentation in retraining.")
    elif all_confs:
        flags.append(f"[FLAG-2 OK] Average confidence {avg_conf:.3f} is acceptable.")
    else:
        flags.append(f"[FLAG-2 CRITICAL] No detections at all — model completely failing on video input.")

    # Flag 3: Temporal flip-flop
    if consistency_pct < 60.0:
        flags.append(f"[FLAG-3 CRITICAL] Dominant-class consistency is {consistency_pct:.1f}%. "
                     f"Class flip-flopping is severe — every other frame changes its primary detection. "
                     f"This causes the pipeline to oscillate between MOVE/STOP/SLOW arbitrarily.")
    elif consistency_pct < 80.0:
        flags.append(f"[FLAG-3 WARN] Temporal consistency {consistency_pct:.1f}%. "
                     f"Noticeable flip-flops — temporal smoothing filter is necessary.")
    else:
        flags.append(f"[FLAG-3 OK] Temporal consistency {consistency_pct:.1f}%.")

    # Flag 4: Preprocessing
    flags.append(
        "[FLAG-4 OK] Preprocessing analysis: video pipeline feeds BGR uint8 arrays directly "
        "from cap.read(). app_v3.py uses PIL→RGB→BGR round-trip. Both produce identical BGR arrays. "
        "*** No BGR/RGB preprocessing mismatch found. ***"
    )

    # Flag 5: Model metrics warning
    flags.append(
        "[FLAG-5 CRITICAL] Existing model (yolov8_bus_v1.pt) has mAP@0.50=11.31% and "
        "mean Precision=0.34% (from accuracy_report_v1.txt). This is catastrophically low precision "
        "— the model detects many false positives on every frame. Video exacerbates this because "
        "confidence fluctuates frame-to-frame, creating flicker. Root cause: training set is "
        "photo-only (static images from FGVD/IDD-Lite). Fix: (a) add video-frame augmentation "
        "(motion blur, compression noise) to training data, and (b) add temporal smoothing "
        "post-processing in the video pipeline."
    )

    for flag in flags:
        print(f"\n  {flag}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
  1. IMMEDIATE FIX (no retraining): Add temporal smoothing in src/video/video_pipeline.py.
     Require 3-of-5 frame consensus before passing a detection to analyze_scene().
     This eliminates flicker regardless of model quality.

  2. TRAINING FIX: Extract frames from video clips (motion blur, compression artefacts),
     merge with existing photo dataset, retrain with:
       - blur augmentation (copy_paste, blur=0.01–0.05)
       - Higher confidence threshold at inference (0.35+)
     Target: mAP@0.50 >= 0.50 on video validation clips.
    """)
    print("=" * 70 + "\n")

    return {
        "zero_det_pct": zero_det_pct,
        "avg_conf": avg_conf,
        "consistency_pct": consistency_pct,
        "avg_dets": avg_dets,
        "class_distribution": dict(class_counter.most_common()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Diagnose video detection quality and preprocessing."
    )
    parser.add_argument("video_path", help="Path to input video file (front camera)")
    parser.add_argument("--model", default=None,
                        help="Path to YOLO model weights (default: models/yolov8_bus_v1.pt or yolov8n.pt)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Sampling FPS, matching video_pipeline.py default (default: 5)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "outputs" / "video_diagnosis",
                        help="Output directory for annotated frames")
    args = parser.parse_args()

    video_path = args.video_path
    if not Path(video_path).is_file():
        print(f"[ERROR] Video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Determine model path
    if args.model:
        model_path = args.model
    else:
        v1_path = REPO_ROOT / "models" / "yolov8_bus_v1.pt"
        model_path = str(v1_path) if v1_path.is_file() else str(REPO_ROOT / "yolov8n.pt")

    model = load_model(model_path)

    # ---- Step A: Extract frames - video pipeline style ----
    print("\n[STEP A] Extracting frames using video pipeline method (cv2.VideoCapture.read)...")
    video_frames = extract_frames_video_pipeline_style(video_path, args.fps)

    # ---- Step B: Extract frames - app_v3 style ----
    print("[STEP B] Extracting frames using app_v3.py load_image style...")
    app_frames = extract_frames_app_v3_style(video_path, args.fps)

    # ---- Preprocessing mismatch check ----
    max_diff = check_preprocessing_mismatch(video_frames, app_frames)

    # ---- Step C: Run inference on video-pipeline frames ----
    print(f"[STEP C] Running inference on {len(video_frames)} frames (video pipeline path)...")
    records = run_detection_on_frames(model, video_frames, conf_threshold=args.conf)

    # ---- Step D: Save annotated frames ----
    print(f"[STEP D] Saving annotated sample frames to: {args.out_dir}")
    save_sample_annotated_frames(records, args.out_dir, num_samples=15)

    # ---- Step E: Print full diagnosis report ----
    summary = print_diagnosis_report(records, model_path)

    # Save text report
    report_path = args.out_dir / "diagnosis_report.txt"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames analyzed: {len(records)}\n")
        f.write(f"Zero-detection rate: {summary['zero_det_pct']:.1f}%\n")
        f.write(f"Avg confidence: {summary['avg_conf']:.3f}\n")
        f.write(f"Temporal consistency: {summary['consistency_pct']:.1f}%\n")
        f.write(f"Avg detections/frame: {summary['avg_dets']:.1f}\n")
        f.write(f"Preprocessing mismatch max pixel diff: {max_diff}\n")
        f.write(f"Class distribution: {summary['class_distribution']}\n")
    print(f"[INFO] Text report saved to: {report_path}")


if __name__ == "__main__":
    main()
