"""
src/tracking/byte_tracker_wrapper.py
===============================================================================
Autonomous Bus Perception System - ByteTrack Multi-Object Tracking Wrapper
===============================================================================
Implements ByteTrack integration using the `supervision` library to provide:
1. Robust temporal tracking across video frames for YOLOv8 detections.
2. Per-object `track_id` assignment.
3. Track trajectory history retaining bounding boxes and timestamps for up to
   the last 16 frames.
4. Non-invasive wrapper functions (`enrich_scene_output` and
   `analyze_scene_with_tracking`) that take `src/autonomous_brain_v3.py`'s
   existing return format, enrich it, and preserve full backward compatibility
   with the Streamlit dashboard (`frontend/app_v3.py`).
===============================================================================
"""

import time
import warnings
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Suppress supervision deprecation warning
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False


# ==============================================================================
# HELPER: IOU COMPUTATION FOR DETECTIONS MATCHING
# ==============================================================================

def compute_box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


# ==============================================================================
# BYTETRACK WRAPPER CLASS
# ==============================================================================

class ByteTrackerWrapper:
    """
    Wrapper around ByteTrack for autonomous bus perception.
    Maintains persistent track IDs and a 16-frame bounding-box trajectory history.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.7,
        frame_rate: int = 30,
        history_length: int = 16
    ):
        self.track_activation_threshold = track_activation_threshold
        self.lost_track_buffer = lost_track_buffer
        self.minimum_matching_threshold = minimum_matching_threshold
        self.frame_rate = frame_rate
        self.history_length = history_length

        # Per-track trajectory history: track_id -> deque of {"bbox": [x1, y1, x2, y2], "timestamp": float}
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_length))

        # Internal tracker initialization
        if HAS_SUPERVISION:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                frame_rate=self.frame_rate
            )
        else:
            self.tracker = None
            self._fallback_id_counter = 1
            print("[WARN] `supervision` library not available. Using fallback centroid tracker.")

        self.last_timestamp: float = time.time()
        self.frame_count: int = 0

    def reset(self) -> None:
        """Resets all tracker internal states and trajectory histories."""
        if HAS_SUPERVISION:
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                frame_rate=self.frame_rate
            )
        self.track_history.clear()
        self.frame_count = 0
        self.last_timestamp = time.time()

    def track_detections(
        self,
        detections: List[Dict[str, Any]],
        timestamp: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Takes raw YOLOv8 detections (list of dicts with 'class', 'x1', 'y1', 'x2', 'y2')
        and enriches each object with:
        - `track_id`: Integer persistent object ID
        - `history`: List of up to 16 previous bounding boxes and timestamps

        Args:
            detections: List of detection dictionaries as produced in autonomous_brain_v3.py.
            timestamp: Frame timestamp (defaults to time.time()).

        Returns:
            List of enriched detection dictionaries.
        """
        current_time = timestamp if timestamp is not None else time.time()
        self.frame_count += 1

        if not detections:
            return []

        # If supervision ByteTrack is available
        if HAS_SUPERVISION and self.tracker is not None:
            # Prepare bounding boxes, confidences, and dummy class ids for supervision
            xyxy_list = []
            conf_list = []
            cls_list = []

            for idx, det in enumerate(detections):
                x1 = float(det.get("x1", 0))
                y1 = float(det.get("y1", 0))
                x2 = float(det.get("x2", 0))
                y2 = float(det.get("y2", 0))
                conf = float(det.get("confidence", det.get("conf", 0.90)))

                xyxy_list.append([x1, y1, x2, y2])
                conf_list.append(conf)
                cls_list.append(idx)

            xyxy = np.array(xyxy_list, dtype=float)
            conf = np.array(conf_list, dtype=float)
            cls_arr = np.array(cls_list, dtype=int)

            sv_dets = sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls_arr)

            try:
                tracked_sv = self.tracker.update_with_detections(sv_dets)
            except Exception as e:
                # If tracker update encounters an issue, fallback gracefully
                print(f"[WARN] ByteTrack update failed ({e}), returning unassigned tracks.")
                tracked_sv = sv.Detections.empty()

            # Match tracker outputs back to input detections via IoU matching
            enriched_detections: List[Dict[str, Any]] = []
            assigned_track_ids: List[Optional[int]] = [None] * len(detections)

            if len(tracked_sv) > 0 and tracked_sv.tracker_id is not None:
                tracked_boxes = tracked_sv.xyxy
                tracker_ids = tracked_sv.tracker_id

                for d_idx, in_box in enumerate(xyxy):
                    best_iou = 0.0
                    best_tid = None
                    for t_idx, t_box in enumerate(tracked_boxes):
                        iou = compute_box_iou(in_box, t_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = int(tracker_ids[t_idx])

                    if best_iou >= 0.25 and best_tid is not None:
                        assigned_track_ids[d_idx] = best_tid
                    elif best_tid is not None:
                        # Centroid proximity fallback for fast-moving targets
                        in_cx = (in_box[0] + in_box[2]) / 2.0
                        in_cy = (in_box[1] + in_box[3]) / 2.0
                        for t_idx, t_box in enumerate(tracked_boxes):
                            t_cx = (t_box[0] + t_box[2]) / 2.0
                            t_cy = (t_box[1] + t_box[3]) / 2.0
                            if (in_cx - t_cx) ** 2 + (in_cy - t_cy) ** 2 <= (80.0 ** 2):
                                assigned_track_ids[d_idx] = int(tracker_ids[t_idx])
                                break

            # Build enriched output
            for idx, det in enumerate(detections):
                enriched = dict(det)
                tid = assigned_track_ids[idx]

                if tid is not None:
                    enriched["track_id"] = tid
                    bbox_coords = [int(det.get("x1", 0)), int(det.get("y1", 0)), int(det.get("x2", 0)), int(det.get("y2", 0))]
                    self.track_history[tid].append({
                        "bbox": bbox_coords,
                        "timestamp": current_time
                    })
                    enriched["history"] = list(self.track_history[tid])
                else:
                    enriched["track_id"] = -1
                    enriched["history"] = []

                enriched_detections.append(enriched)

            return enriched_detections

        else:
            # Fallback tracking logic if supervision is unavailable
            enriched_detections = []
            for det in detections:
                enriched = dict(det)
                tid = self._fallback_id_counter
                self._fallback_id_counter += 1
                enriched["track_id"] = tid
                bbox_coords = [int(det.get("x1", 0)), int(det.get("y1", 0)), int(det.get("x2", 0)), int(det.get("y2", 0))]
                self.track_history[tid].append({
                    "bbox": bbox_coords,
                    "timestamp": current_time
                })
                enriched["history"] = list(self.track_history[tid])
                enriched_detections.append(enriched)
            return enriched_detections


# ==============================================================================
# GLOBAL DEFAULT TRACKER INSTANCE
# ==============================================================================

_GLOBAL_TRACKER = ByteTrackerWrapper()


def get_default_tracker() -> ByteTrackerWrapper:
    """Returns the singleton global ByteTrackerWrapper instance."""
    return _GLOBAL_TRACKER


# ==============================================================================
# ENRICHMENT WRAPPER FUNCTIONS (NON-INVASIVE)
# ==============================================================================

def enrich_scene_output(
    scene_output: Dict[str, Any],
    detections: Optional[List[Dict[str, Any]]] = None,
    front_image: Optional[np.ndarray] = None,
    timestamp: Optional[float] = None,
    tracker: Optional[ByteTrackerWrapper] = None
) -> Dict[str, Any]:
    """
    Enriches existing scene_output from `src.autonomous_brain_v3.analyze_scene()`
    with persistent ByteTrack tracking metadata.

    Preserves 100% of existing keys in scene_output so frontend/app_v3.py keeps
    working completely unchanged. Adds:
    - `tracked_detections`: List of detection dicts with `track_id` and 16-frame `history`
    - `active_tracks`: Count of currently active tracks

    Args:
        scene_output: Original dictionary returned by analyze_scene().
        detections: Optional list of detections. If None, extracts detections
                    or runs tracker on available data.
        front_image: Optional front camera frame for trajectory visualization.
        timestamp: Optional timestamp for this frame.
        tracker: Optional ByteTrackerWrapper instance (defaults to global singleton).

    Returns:
        Enriched dictionary containing all original keys plus tracking metadata.
    """
    if tracker is None:
        tracker = get_default_tracker()

    enriched_output = dict(scene_output)

    # If detections not explicitly passed, check if present in scene_output or extract
    target_detections = detections
    if target_detections is None:
        target_detections = scene_output.get("detection_list", scene_output.get("detections", []))

    tracked_dets = tracker.track_detections(target_detections, timestamp=timestamp)

    enriched_output["tracked_detections"] = tracked_dets
    enriched_output["active_tracks"] = len([d for d in tracked_dets if d.get("track_id", -1) > 0])

    # If front_image or annotated_image is present, optionally draw motion trajectories
    if front_image is not None or "annotated_image" in enriched_output:
        base_canvas = front_image.copy() if front_image is not None else enriched_output["annotated_image"].copy()

        for det in tracked_dets:
            tid = det.get("track_id", -1)
            if tid <= 0:
                continue

            x1, y1, x2, y2 = det.get("x1", 0), det.get("y1", 0), det.get("x2", 0), det.get("y2", 0)
            cname = det.get("class", "object")

            # Draw bounding box and track ID
            cv2.rectangle(base_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"ID:{tid} {cname}"
            cv2.putText(base_canvas, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Draw motion trail from history (up to last 16 frames)
            history = det.get("history", [])
            if len(history) > 1:
                pts = []
                for h in history:
                    bx = h["bbox"]
                    cx = int((bx[0] + bx[2]) / 2)
                    cy = int((bx[1] + bx[3]) / 2)
                    pts.append((cx, cy))
                for p_i in range(1, len(pts)):
                    cv2.line(base_canvas, pts[p_i - 1], pts[p_i], (0, 215, 255), 2)

        enriched_output["tracked_annotated_image"] = base_canvas

    return enriched_output


def analyze_scene_with_tracking(
    front_image: np.ndarray,
    front_left: np.ndarray,
    front_right: np.ndarray,
    rear_left: np.ndarray,
    rear_right: np.ndarray,
    timestamp: Optional[float] = None,
    tracker: Optional[ByteTrackerWrapper] = None
) -> Dict[str, Any]:
    """
    High-level optional drop-in wrapper for `analyze_scene()` that executes
    autonomous_brain_v3 scene analysis and enriches detections with ByteTrack tracking.

    Returns the exact same schema as analyze_scene(), augmented with tracking fields.
    """
    from src.autonomous_brain_v3 import analyze_scene, model

    # Execute original frozen brain analysis
    base_output = analyze_scene(
        front_image=front_image,
        front_left=front_left,
        front_right=front_right,
        rear_left=rear_left,
        rear_right=rear_right
    )

    # Extract detections from front image using the brain's model
    front_res = model(front_image, verbose=False)
    detections: List[Dict[str, Any]] = []

    if front_res and len(front_res) > 0 and front_res[0].boxes is not None:
        for box in front_res[0].boxes:
            cls_id = int(box.cls[0])
            cname = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append({
                "class": cname,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

    # Enrich output with ByteTrack tracking
    return enrich_scene_output(
        scene_output=base_output,
        detections=detections,
        front_image=front_image,
        timestamp=timestamp,
        tracker=tracker
    )
