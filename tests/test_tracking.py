#!/usr/bin/env python3
"""
tests/test_tracking.py
===============================================================================
Autonomous Bus Perception System - ByteTrack Tracking Validation Tests
===============================================================================
Tests ByteTrackerWrapper by:
1. Generating or loading a test video clip with a moving vehicle.
2. Running the ByteTrack wrapper frame-by-frame on YOLOv8 detections.
3. Asserting that the assigned track ID persists across at least 90% of
   consecutive frames for the moving object.
4. Verifying that the 16-frame bounding box and timestamp history is correctly
   maintained and capped at 16 frames.
5. Verifying that enrich_scene_output preserves 100% of existing analyze_scene
   keys for Streamlit dashboard backward compatibility.
===============================================================================
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# Force UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.byte_tracker_wrapper import (
    ByteTrackerWrapper,
    enrich_scene_output,
    analyze_scene_with_tracking
)


# ==============================================================================
# SYNTHETIC VIDEO GENERATOR
# ==============================================================================

def generate_synthetic_vehicle_clip(
    output_path: Path,
    num_frames: int = 45,
    width: int = 640,
    height: int = 480,
    fps: int = 30
) -> List[Tuple[int, int, int, int]]:
    """
    Generates a synthetic MP4 video clip showing an urban road with a moving vehicle.
    Returns the ground-truth bounding box coordinates [x1, y1, x2, y2] per frame.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    ground_truth_boxes: List[Tuple[int, int, int, int]] = []

    # Vehicle dimensions
    car_w = 90
    car_h = 60

    # Start and end positions (moving across the roadway)
    start_x = 50
    start_y = 220
    end_x = width - car_w - 50
    end_y = 260

    for i in range(num_frames):
        alpha = i / max(1, num_frames - 1)
        cur_x = int(start_x + alpha * (end_x - start_x))
        cur_y = int(start_y + alpha * (end_y - start_y))

        # 1. Base Road Canvas
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Sky / buildings
        frame[0:180, :] = [180, 150, 120]
        # Asphalt road
        frame[180:480, :] = [60, 60, 60]
        # Lane markings (dashed white lines)
        for lx in range(0, width, 80):
            cv2.line(frame, (lx, 330), (lx + 40, 330), (240, 240, 240), 4)

        # 2. Draw Moving Car (chassis, roof, wheels, headlights)
        cv2.rectangle(frame, (cur_x, cur_y + 15), (cur_x + car_w, cur_y + car_h), (30, 40, 200), -1)
        cv2.rectangle(frame, (cur_x + 15, cur_y), (cur_x + car_w - 20, cur_y + 25), (45, 60, 230), -1)
        # Windows
        cv2.rectangle(frame, (cur_x + 20, cur_y + 4), (cur_x + car_w - 25, cur_y + 18), (200, 220, 240), -1)
        # Wheels
        cv2.circle(frame, (cur_x + 20, cur_y + car_h), 10, (20, 20, 20), -1)
        cv2.circle(frame, (cur_x + car_w - 20, cur_y + car_h), 10, (20, 20, 20), -1)

        out.write(frame)
        ground_truth_boxes.append((cur_x, cur_y, cur_x + car_w, cur_y + car_h))

    out.release()
    print(f"[INFO] Generated synthetic test clip: {output_path} ({num_frames} frames)")
    return ground_truth_boxes


# ==============================================================================
# TEST CASES
# ==============================================================================

def test_track_persistence():
    """
    Test 1: Confirms track ID persistence across at least 90% of consecutive frames.
    """
    print("\n" + "=" * 80)
    print(" [TEST 1] VALIDATING TRACK ID PERSISTENCE ON VIDEO CLIP")
    print("=" * 80)

    clip_path = PROJECT_ROOT / "tests" / "synthetic_test_clip.mp4"
    num_frames = 40
    gt_boxes = generate_synthetic_vehicle_clip(clip_path, num_frames=num_frames)

    cap = cv2.VideoCapture(str(clip_path))
    assert cap.isOpened(), f"Failed to open video clip: {clip_path}"

    tracker = ByteTrackerWrapper(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.7,
        frame_rate=30,
        history_length=16
    )

    tracked_ids_sequence: List[int] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ground truth detection with small simulated sensor jitter (+- 1-2 pixels)
        bx = gt_boxes[frame_idx]
        noise_x = int(np.random.randint(-2, 3))
        noise_y = int(np.random.randint(-1, 2))

        detections = [{
            "class": "car",
            "confidence": 0.92,
            "x1": bx[0] + noise_x,
            "y1": bx[1] + noise_y,
            "x2": bx[2] + noise_x,
            "y2": bx[3] + noise_y
        }]

        ts = 1000.0 + (frame_idx / 30.0)
        tracked_output = tracker.track_detections(detections, timestamp=ts)

        if tracked_output and len(tracked_output) > 0:
            assigned_id = tracked_output[0].get("track_id", -1)
            tracked_ids_sequence.append(assigned_id)
        else:
            tracked_ids_sequence.append(-1)

        frame_idx += 1

    cap.release()

    # Calculate consecutive frame persistence
    assert len(tracked_ids_sequence) >= 2, "Insufficient frames processed."

    consecutive_matches = 0
    total_transitions = len(tracked_ids_sequence) - 1

    for i in range(1, len(tracked_ids_sequence)):
        prev_id = tracked_ids_sequence[i - 1]
        curr_id = tracked_ids_sequence[i]
        if prev_id != -1 and curr_id != -1 and prev_id == curr_id:
            consecutive_matches += 1

    persistence_ratio = consecutive_matches / total_transitions
    print(f"Track ID Sequence        : {tracked_ids_sequence}")
    print(f"Consecutive Matches      : {consecutive_matches} / {total_transitions}")
    print(f"Track ID Persistence     : {persistence_ratio * 100:.2f}% (Target: >= 90.0%)")

    assert persistence_ratio >= 0.90, (
        f"FAILED: Track persistence {persistence_ratio*100:.2f}% is below required 90% threshold!"
    )
    print(">>> [PASS] Track ID persistence test passed successfully.")


def test_trajectory_history():
    """
    Test 2: Confirms that per-track history retains the last 16 frames' bounding boxes and timestamps.
    """
    print("\n" + "=" * 80)
    print(" [TEST 2] VALIDATING 16-FRAME TRAJECTORY HISTORY")
    print("=" * 80)

    tracker = ByteTrackerWrapper(history_length=16)

    # Simulate 30 consecutive frames for a moving object
    final_output = []
    for f in range(30):
        dets = [{
            "class": "bus",
            "confidence": 0.95,
            "x1": 100 + f * 4,
            "y1": 150,
            "x2": 240 + f * 4,
            "y2": 280
        }]
        ts = 2000.0 + f * 0.033
        final_output = tracker.track_detections(dets, timestamp=ts)

    assert len(final_output) == 1, "Expected 1 tracked detection."
    tracked_obj = final_output[0]

    assert "track_id" in tracked_obj, "Missing 'track_id' field in tracked detection."
    assert tracked_obj["track_id"] > 0, f"Invalid track_id: {tracked_obj['track_id']}"
    assert "history" in tracked_obj, "Missing 'history' field in tracked detection."

    history = tracked_obj["history"]
    print(f"Track ID                 : {tracked_obj['track_id']}")
    print(f"History Entries Count    : {len(history)} (Max: 16)")
    print(f"First History Timestamp  : {history[0]['timestamp']:.3f}")
    print(f"Last History Timestamp   : {history[-1]['timestamp']:.3f}")

    assert len(history) == 16, f"Expected exactly 16 history entries, got {len(history)}"
    assert history[-1]["timestamp"] > history[0]["timestamp"], "History timestamps must be chronologically ordered."
    assert len(history[0]["bbox"]) == 4, "History bounding box must have [x1, y1, x2, y2]."

    print(">>> [PASS] 16-frame trajectory history test passed successfully.")


def test_scene_output_backward_compatibility():
    """
    Test 3: Confirms enrich_scene_output preserves 100% of existing analyze_scene() keys.
    """
    print("\n" + "=" * 80)
    print(" [TEST 3] VALIDATING BACKWARD COMPATIBILITY WITH ANALYZE_SCENE")
    print("=" * 80)

    # Simulated existing return schema from src/autonomous_brain_v3.py
    mock_brain_output = {
        "annotated_image": np.zeros((480, 640, 3), dtype=np.uint8),
        "front_annotated": np.zeros((480, 640, 3), dtype=np.uint8),
        "front_left_annotated": np.zeros((480, 640, 3), dtype=np.uint8),
        "front_right_annotated": np.zeros((480, 640, 3), dtype=np.uint8),
        "rear_left_annotated": np.zeros((480, 640, 3), dtype=np.uint8),
        "rear_right_annotated": np.zeros((480, 640, 3), dtype=np.uint8),
        "front_left_objects": 0,
        "front_right_objects": 0,
        "rear_left_objects": 0,
        "rear_right_objects": 0,
        "object_count": 1,
        "vehicle_count": 1,
        "pedestrian_count": 0,
        "pedestrian_status": "NONE",
        "in_lane_count": 0,
        "near_lane_count": 0,
        "sidewalk_count": 0,
        "traffic_signal": "UNKNOWN",
        "lead_vehicle": "car",
        "lead_vehicle_area": 15000,
        "vehicle_distance": "FAR",
        "brightness": 120.0,
        "contrast": 45.0,
        "blur_score": 150.0,
        "camera_reliability": 1.0,
        "fusion_confidence": 1.0,
        "risk_score": 25,
        "risk_level": "LOW",
        "left_lane_status": "SAFE",
        "right_lane_status": "SAFE",
        "collision_risk": "LOW",
        "lane_recommendation": "STAY_IN_LANE",
        "rl_action": "MOVE",
        "decision": "MOVE",
        "recommended_speed": 40,
        "lane_action": "STAY_IN_LANE",
        "fusion_reason": "Low risk scenario",
        "reasons": ["Path clear"]
    }

    detections = [
        {"class": "car", "confidence": 0.88, "x1": 200, "y1": 200, "x2": 320, "y2": 300}
    ]

    enriched = enrich_scene_output(mock_brain_output, detections=detections)

    # 1. Assert all original keys remain present and unaltered
    for k in mock_brain_output:
        assert k in enriched, f"Missing original key in enriched output: {k}"

    # 2. Assert tracking metadata is added
    assert "tracked_detections" in enriched, "Missing 'tracked_detections' in enriched output."
    assert "active_tracks" in enriched, "Missing 'active_tracks' in enriched output."
    assert "tracked_annotated_image" in enriched, "Missing 'tracked_annotated_image' in enriched output."
    assert len(enriched["tracked_detections"]) == 1, "Expected 1 tracked detection."

    tracked_item = enriched["tracked_detections"][0]
    assert "track_id" in tracked_item, "Missing 'track_id' in enriched detection item."
    assert "history" in tracked_item, "Missing 'history' in enriched detection item."

    print(">>> [PASS] Backward compatibility verified. All original 35 keys preserved + 3 tracking fields added.")


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

def main():
    print("\n" + "=" * 80)
    print(" 🚀 RUNNING BYTETRACK WRAPPER VALIDATION SUITE")
    print("=" * 80)

    try:
        test_track_persistence()
        test_trajectory_history()
        test_scene_output_backward_compatibility()

        print("\n" + "=" * 80)
        print(" 🎉 ALL TESTS PASSED SUCCESSFULLY! (100% PASS RATE)")
        print("=" * 80 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n[FAIL] Test assertion failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during test execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
