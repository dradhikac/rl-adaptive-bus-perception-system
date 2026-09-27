"""
src/video/temporal_smoother.py
================================================================================
Step 5 — Temporal Smoothing Filter (engineering safeguard).

Wraps the frame-level detection + analyze_scene() call with a sliding-window
majority-vote smoother that:

  a. Collects per-slot raw YOLO detections for the last N frames (default N=5).
  b. Applies 3-of-5 frame consensus: a detection is only forwarded to
     analyze_scene() if the same class appeared in at least 3 of the last 5
     frames at roughly the same location (IoU > 0.30).
  c. Builds a "smoothed" frame set for analyze_scene() by painting the
     consensus detections onto a clean copy of the latest frame instead of
     passing the raw (flickery) frame.

IMPORTANT:
  - analyze_scene() signature is unchanged (takes 5 BGR numpy arrays).
  - This module is ADDITIVE. It can be enabled or disabled by the pipeline
    via the USE_TEMPORAL_SMOOTHING flag or the constructor argument.
  - Frozen modules (decision_fusion_engine, risk_based_decision_engine, etc.)
    are not touched.
================================================================================
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("temporal_smoother")


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

def _box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Axis-aligned IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union_area = area_a + area_b - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


# ---------------------------------------------------------------------------
# Detection record
# ---------------------------------------------------------------------------

class _DetectionRecord:
    """Lightweight struct for one raw YOLO detection on a single frame."""

    __slots__ = ["cls_name", "cls_id", "conf", "x1", "y1", "x2", "y2"]

    def __init__(self, cls_name: str, cls_id: int, conf: float,
                 x1: int, y1: int, x2: int, y2: int):
        self.cls_name = cls_name
        self.cls_id = cls_id
        self.conf = conf
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


# ---------------------------------------------------------------------------
# Per-slot smoothing buffer
# ---------------------------------------------------------------------------

class _SlotSmoother:
    """
    Maintains a sliding window of raw detections for a single camera slot,
    and computes the consensus (3-of-N stable) detection set.
    """

    def __init__(self, window_size: int = 5, min_votes: int = 3, iou_threshold: float = 0.30):
        self.window_size = window_size
        self.min_votes = min_votes
        self.iou_threshold = iou_threshold
        # deque of List[_DetectionRecord]: per-frame detections for the last N frames
        self._history: deque = deque(maxlen=window_size)

    def update(self, detections: List[_DetectionRecord]) -> None:
        """Push the latest frame's detections into the history buffer."""
        self._history.append(detections)

    def get_stable_detections(self) -> List[_DetectionRecord]:
        """
        Returns the consensus (stable) detection set across the current window.

        Algorithm:
          For each detection in the LATEST frame, check if a detection of the
          same class overlaps (IoU >= iou_threshold) in at least min_votes frames
          in the current window. If yes, include it in the stable set.

        This means: a detection that appeared on a single frame and disappears
        in the next is FILTERED OUT. Only detections that persist across multiple
        frames pass through.
        """
        if not self._history:
            return []

        latest_frame = list(self._history)[-1]
        if not latest_frame:
            return []

        stable: List[_DetectionRecord] = []
        history_list = list(self._history)

        for det in latest_frame:
            vote_count = 0
            for past_frame_dets in history_list:
                # Check if any detection in this past frame matches by class AND IoU
                for past_det in past_frame_dets:
                    if past_det.cls_name == det.cls_name:
                        iou = _box_iou(det.xyxy, past_det.xyxy)
                        if iou >= self.iou_threshold:
                            vote_count += 1
                            break  # only count once per past frame

            if vote_count >= self.min_votes:
                stable.append(det)

        return stable

    def reset(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# Main smoother class integrated with YOLO model
# ---------------------------------------------------------------------------

class TemporalSmoother:
    """
    Wraps YOLO inference with temporal smoothing.

    Usage:
        smoother = TemporalSmoother(model, window_size=5, min_votes=3)

        # Per timestep:
        stable_detections = smoother.update_and_get_stable(slot_name, bgr_frame)
        smoothed_frame = smoother.render_stable_frame(bgr_frame, stable_detections)
        # Pass smoothed_frame to analyze_scene() instead of the raw frame.

    The smoother is DISABLED by default if disabled=True (pass-through mode),
    which means it returns the raw frame unchanged. This allows easy A/B testing.
    """

    def __init__(
        self,
        model: Any,  # ultralytics YOLO instance
        window_size: int = 5,
        min_votes: int = 3,
        iou_threshold: float = 0.30,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        disabled: bool = False,
    ):
        self.model = model
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.disabled = disabled
        self._slot_smoothers: Dict[str, _SlotSmoother] = defaultdict(
            lambda: _SlotSmoother(window_size, min_votes, iou_threshold)
        )

    def _detect_raw(self, bgr_frame: np.ndarray) -> List[_DetectionRecord]:
        """Run YOLO on a single BGR frame and return raw detections."""
        if bgr_frame is None or bgr_frame.size == 0:
            return []
        try:
            results = self.model.predict(
                source=bgr_frame,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                verbose=False,
            )
            detections = []
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, f"cls_{cls_id}")
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                detections.append(_DetectionRecord(cls_name, cls_id, conf, x1, y1, x2, y2))
            return detections
        except Exception as exc:
            logger.warning(f"YOLO inference error: {exc}")
            return []

    def update_and_get_stable(
        self, slot: str, bgr_frame: np.ndarray
    ) -> List[_DetectionRecord]:
        """
        Run YOLO on frame, update per-slot history, and return stable detections
        based on 3-of-5 consensus. In disabled mode, returns raw detections directly.
        """
        raw_dets = self._detect_raw(bgr_frame)

        if self.disabled:
            return raw_dets

        smoother = self._slot_smoothers[slot]
        smoother.update(raw_dets)
        stable = smoother.get_stable_detections()
        logger.debug(
            f"Slot [{slot}]: raw={len(raw_dets)} dets → stable={len(stable)} after smoothing"
        )
        return stable

    def render_stable_frame(
        self,
        bgr_frame: np.ndarray,
        stable_detections: List[_DetectionRecord],
    ) -> np.ndarray:
        """
        Returns a copy of bgr_frame with stable detection boxes drawn on it.
        This provides a visually clean frame to pass to analyze_scene().

        In disabled mode, returns the original frame unchanged.
        """
        if self.disabled or not stable_detections:
            return bgr_frame

        rendered = bgr_frame.copy()
        for det in stable_detections:
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            color = (0, 200, 100)  # green for stable boxes
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            label = f"{det.cls_name} {det.conf:.2f}"
            cv2.putText(rendered, label, (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return rendered

    def reset_slot(self, slot: str) -> None:
        """Reset the smoother history for a specific camera slot."""
        if slot in self._slot_smoothers:
            self._slot_smoothers[slot].reset()

    def reset_all(self) -> None:
        """Reset all per-slot histories (call between video jobs)."""
        for smoother in self._slot_smoothers.values():
            smoother.reset()
        self._slot_smoothers.clear()

    def get_raw_stats(self, slot: str) -> Dict[str, Any]:
        """Return debug stats for a slot smoother."""
        smoother = self._slot_smoothers.get(slot)
        if not smoother:
            return {"history_len": 0, "window_size": 5}
        return {
            "history_len": len(smoother._history),
            "window_size": smoother.window_size,
            "min_votes": smoother.min_votes,
            "total_frames_in_buffer": sum(len(f) for f in smoother._history),
        }
