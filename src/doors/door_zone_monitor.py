"""
src/doors/door_zone_monitor.py
===============================================================================
Autonomous Bus Perception System - 6th Camera Feed: Door Zone Safety Monitor
===============================================================================
Monitors the vehicle passenger door portal / vestibule zone using a dedicated
door camera feed. Implements an active safety state machine:

  DOOR_CLOSED
    ↓ (open requested / commanded)
  DOOR_OPENING
    ↓ (doors fully open)
  BOARDING/ALIGHTING
    ↓ (close requested / dwell elapsed)
  DOOR_CLOSE_PENDING
    ↓ (if occupied) ──────────┐
  DOOR_CLOSE_BLOCKED          │ (continuous safety re-check)
    ↓ (cleared)               │
  DOOR_CLOSING ←──────────────┘ (if person re-enters during closing)
    ↓ (fully closed & clear)
  READY_TO_DEPART

Features:
1. Door Safety ROI Monitoring: Tracks passengers entering/lingering in doorway.
2. ByteTrack Integration: Tracks passenger trajectories across the threshold
   to count boarding (inward) and alighting (outward) movements.
3. Continuous Safety Re-check: Even after DOOR_CLOSE_PENDING is reached,
   re-verifies zone clearance immediately before DOOR_CLOSING. If a person
   re-enters, instantly aborts to DOOR_CLOSE_BLOCKED.
4. Standalone operation: Does not alter arbitration logic in decision_fusion_engine.py.
===============================================================================
"""

import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO

from src.tracking.byte_tracker_wrapper import ByteTrackerWrapper


# ==============================================================================
# DOOR ZONE STATE ENUM
# ==============================================================================

class DoorZoneState(str, Enum):
    DOOR_CLOSED = "DOOR_CLOSED"
    DOOR_OPENING = "DOOR_OPENING"
    BOARDING_ALIGHTING = "BOARDING/ALIGHTING"
    DOOR_CLOSE_PENDING = "DOOR_CLOSE_PENDING"
    DOOR_CLOSE_BLOCKED = "DOOR_CLOSE_BLOCKED"
    DOOR_CLOSING = "DOOR_CLOSING"
    READY_TO_DEPART = "READY_TO_DEPART"


# ==============================================================================
# DOOR ZONE SAFETY MONITOR CLASS
# ==============================================================================

class DoorZoneMonitor:
    """
    Dedicated door zone monitor for the 6th camera feed.
    """

    def __init__(
        self,
        model_path: Optional[str] = "models/yolov8_door_v1.pt",
        door_roi_normalized: Tuple[float, float, float, float] = (0.25, 0.20, 0.75, 0.95),
        threshold_y_normalized: float = 0.55,
        clear_debounce_seconds: float = 0.40,
        door_actuation_seconds: float = 1.0,
        tracker: Optional[ByteTrackerWrapper] = None
    ):
        """
        Args:
            model_path: Path to YOLO door/passenger detector weights.
            door_roi_normalized: (x1_norm, y1_norm, x2_norm, y2_norm) of door safety zone.
            threshold_y_normalized: Horizontal line separating exterior (bottom) from interior (top).
            clear_debounce_seconds: Required continuous clear duration before allowing DOOR_CLOSING.
            door_actuation_seconds: Physical door movement transit duration.
            tracker: Optional ByteTrackerWrapper instance (instantiates local one if None).
        """
        self.door_roi_norm = door_roi_normalized
        self.threshold_y_norm = threshold_y_normalized
        self.clear_debounce_seconds = clear_debounce_seconds
        self.door_actuation_seconds = door_actuation_seconds

        # State machine initialization
        self.state: DoorZoneState = DoorZoneState.DOOR_CLOSED
        self.prev_state: DoorZoneState = DoorZoneState.DOOR_CLOSED
        self.current_reason: str = "INITIAL_CLOSED_STATE"
        self.state_enter_timestamp: float = time.time()
        self.last_clear_timestamp: float = time.time()

        # Passenger tracking & counting
        self.tracker = tracker if tracker is not None else ByteTrackerWrapper(history_length=16)
        self.boarding_count: int = 0
        self.alighting_count: int = 0
        self.boarded_track_ids: Set[int] = set()
        self.alighted_track_ids: Set[int] = set()

        # Detector model
        self.model: Optional[YOLO] = None
        self.model_path = model_path
        self._load_model(model_path)

    def _load_model(self, model_path: Optional[str]) -> None:
        """Loads detector weights if available, otherwise falls back gracefully."""
        if not model_path:
            return
        p = Path(model_path)
        if not p.is_file():
            # Try repo root
            p = Path(__file__).resolve().parent.parent.parent / model_path
        if p.is_file():
            try:
                self.model = YOLO(str(p))
            except Exception as e:
                print(f"[WARN] Failed to load door detector from {model_path}: {e}")
                self.model = None

    def reset(self, timestamp: Optional[float] = None) -> None:
        """Resets the state machine and tracking counters."""
        now = timestamp if timestamp is not None else time.time()
        self.state = DoorZoneState.DOOR_CLOSED
        self.prev_state = DoorZoneState.DOOR_CLOSED
        self.current_reason = "RESET_TO_CLOSED"
        self.state_enter_timestamp = now
        self.last_clear_timestamp = now
        self.boarding_count = 0
        self.alighting_count = 0
        self.boarded_track_ids.clear()
        self.alighted_track_ids.clear()
        self.tracker.reset()

    def transition_to(
        self,
        new_state: DoorZoneState,
        reason: str,
        timestamp: Optional[float] = None
    ) -> None:
        """Executes a state transition."""
        now = timestamp if timestamp is not None else time.time()
        self.prev_state = self.state
        self.state = new_state
        self.state_enter_timestamp = now
        self.current_reason = reason

    # --------------------------------------------------------------------------
    # ROI & OCCUPANCY EVALUATION
    # --------------------------------------------------------------------------
    def get_pixel_roi(self, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
        """Returns pixel coordinates (x1, y1, x2, y2) of the door safety zone."""
        x1 = int(self.door_roi_norm[0] * img_w)
        y1 = int(self.door_roi_norm[1] * img_h)
        x2 = int(self.door_roi_norm[2] * img_w)
        y2 = int(self.door_roi_norm[3] * img_h)
        return x1, y1, x2, y2

    def check_zone_occupancy(
        self,
        tracked_detections: List[Dict[str, Any]],
        img_w: int,
        img_h: int
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Checks if any tracked person overlaps the Door Safety ROI.
        Returns (is_occupied, occupants_list).
        """
        roi_x1, roi_y1, roi_x2, roi_y2 = self.get_pixel_roi(img_w, img_h)
        occupants: List[Dict[str, Any]] = []

        for det in tracked_detections:
            cname = str(det.get("class", "")).lower()
            if cname not in ("person", "passenger", "pedestrian"):
                continue

            x1 = float(det.get("x1", 0))
            y1 = float(det.get("y1", 0))
            x2 = float(det.get("x2", 0))
            y2 = float(det.get("y2", 0))

            # Compute box-ROI intersection
            inter_x1 = max(x1, roi_x1)
            inter_y1 = max(y1, roi_y1)
            inter_x2 = min(x2, roi_x2)
            inter_y2 = min(y2, roi_y2)

            inter_w = max(0.0, inter_x2 - inter_x1)
            inter_h = max(0.0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            box_area = max(1.0, (x2 - x1) * (y2 - y1))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Inside if center point is in ROI or >15% of bbox overlaps door zone
            if (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2) or (inter_area / box_area > 0.15):
                occupants.append(det)

        return (len(occupants) > 0), occupants

    # --------------------------------------------------------------------------
    # PASSENGER INGRESS / EGRESS CROSSING TRACKER
    # --------------------------------------------------------------------------
    def update_crossing_counts(
        self,
        tracked_detections: List[Dict[str, Any]],
        img_h: int
    ) -> None:
        """
        Evaluates passenger trajectory history to detect threshold crossings:
        - Inward (bottom to top): boarding
        - Outward (top to bottom): alighting
        """
        threshold_y = self.threshold_y_norm * img_h

        for det in tracked_detections:
            tid = det.get("track_id", -1)
            if tid <= 0:
                continue

            history = det.get("history", [])
            if len(history) < 2:
                continue

            first_bbox = history[0]["bbox"]
            last_bbox = history[-1]["bbox"]
            prev_bbox = history[-2]["bbox"]

            first_cy = (first_bbox[1] + first_bbox[3]) / 2.0
            last_cy = (last_bbox[1] + last_bbox[3]) / 2.0
            prev_cy = (prev_bbox[1] + prev_bbox[3]) / 2.0

            # Inward crossing (moving from bottom/exterior to top/interior)
            if (first_cy > threshold_y and last_cy <= threshold_y) or (prev_cy > threshold_y and last_cy <= threshold_y):
                if tid not in self.boarded_track_ids:
                    self.boarded_track_ids.add(tid)
                    self.boarding_count += 1

            # Outward crossing (moving from interior down to exterior)
            elif (first_cy <= threshold_y and last_cy > threshold_y) or (prev_cy <= threshold_y and last_cy > threshold_y):
                if tid not in self.alighted_track_ids:
                    self.alighted_track_ids.add(tid)
                    self.alighting_count += 1

    # --------------------------------------------------------------------------
    # MAIN PROCESSING ENTRYPOINT
    # --------------------------------------------------------------------------
    def process_frame(
        self,
        door_image: Optional[np.ndarray] = None,
        detections: Optional[List[Dict[str, Any]]] = None,
        open_requested: bool = False,
        close_requested: bool = False,
        doors_physically_open: bool = False,
        doors_physically_closed: bool = False,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a door camera frame and evaluates the door safety state machine.

        Args:
            door_image: Optional (H, W, 3) BGR door camera frame.
            detections: Optional pre-computed detections list. If None, runs self.model on door_image.
            open_requested: Command signal to open doors (e.g. from docking state machine).
            close_requested: Command signal to initiate door closure.
            doors_physically_open: Sensor feedback confirming doors are physically open.
            doors_physically_closed: Sensor feedback confirming doors are physically closed.
            timestamp: Frame timestamp (defaults to time.time()).

        Returns:
            Telemetry dict with state, occupancy, crossing counts, and annotated visualization.
        """
        now = timestamp if timestamp is not None else time.time()
        time_in_state = max(0.0, now - self.state_enter_timestamp)

        # Image dimensions
        if door_image is not None:
            img_h, img_w = door_image.shape[:2]
        else:
            img_w, img_h = 640, 480

        # 1. Obtain detections
        current_dets: List[Dict[str, Any]] = []
        if detections is not None:
            current_dets = list(detections)
        elif door_image is not None and self.model is not None:
            res = self.model(door_image, verbose=False)
            if res and len(res) > 0 and res[0].boxes is not None:
                for b in res[0].boxes:
                    cls_id = int(b.cls[0])
                    cname = self.model.names.get(cls_id, "person")
                    # Map passenger to person for tracking consistency
                    if cname in ("passenger", "5"):
                        cname = "person"
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    current_dets.append({
                        "class": cname,
                        "confidence": conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })

        # 2. Update multi-object tracking (ByteTrack)
        tracked_dets = self.tracker.track_detections(current_dets, timestamp=now)

        # 3. Update ingress/egress crossing counters
        self.update_crossing_counts(tracked_dets, img_h=img_h)

        # 4. Check door safety ROI occupancy
        is_occupied, occupants = self.check_zone_occupancy(tracked_dets, img_w=img_w, img_h=img_h)

        if not is_occupied:
            # Zone is clear; record last clear time
            pass
        else:
            self.last_clear_timestamp = now

        clear_dwell = max(0.0, now - self.last_clear_timestamp)

        # ----------------------------------------------------------------------
        # 5. STATE MACHINE LOGIC WITH CONTINUOUS RE-CHECK
        # ----------------------------------------------------------------------

        # STATE: DOOR_CLOSED
        if self.state == DoorZoneState.DOOR_CLOSED:
            if open_requested or doors_physically_open:
                self.transition_to(
                    DoorZoneState.DOOR_OPENING,
                    "DOOR_OPEN_COMMANDED",
                    timestamp=now
                )

        # STATE: DOOR_OPENING
        elif self.state == DoorZoneState.DOOR_OPENING:
            if doors_physically_open or time_in_state >= self.door_actuation_seconds:
                self.transition_to(
                    DoorZoneState.BOARDING_ALIGHTING,
                    "DOORS_FULLY_OPEN_PASSENGER_EXCHANGE_ACTIVE",
                    timestamp=now
                )

        # STATE: BOARDING/ALIGHTING
        elif self.state == DoorZoneState.BOARDING_ALIGHTING:
            if close_requested:
                self.transition_to(
                    DoorZoneState.DOOR_CLOSE_PENDING,
                    "DOOR_CLOSE_REQUESTED_INITIATING_CLEARANCE_CHECK",
                    timestamp=now
                )
                if is_occupied:
                    self.transition_to(
                        DoorZoneState.DOOR_CLOSE_BLOCKED,
                        f"DOOR_SAFETY_ZONE_OCCUPIED: {len(occupants)} passenger(s) in doorway threshold",
                        timestamp=now
                    )

        # STATE: DOOR_CLOSE_PENDING
        elif self.state == DoorZoneState.DOOR_CLOSE_PENDING:
            # Continuous re-check: if occupied, immediately block closure
            if is_occupied:
                self.transition_to(
                    DoorZoneState.DOOR_CLOSE_BLOCKED,
                    f"DOOR_SAFETY_ZONE_OCCUPIED: {len(occupants)} passenger(s) in doorway threshold",
                    timestamp=now
                )
            else:
                # Zone clear: require clear debounce dwell before allowing doors to close
                if clear_dwell >= self.clear_debounce_seconds or time_in_state >= 0.50:
                    # Final safety re-verification immediately before allowing DOOR_CLOSING
                    if not is_occupied:
                        self.transition_to(
                            DoorZoneState.DOOR_CLOSING,
                            "DOOR_ZONE_VERIFIED_CLEAR: Actuating door closure",
                            timestamp=now
                        )

        # STATE: DOOR_CLOSE_BLOCKED
        elif self.state == DoorZoneState.DOOR_CLOSE_BLOCKED:
            # CRITICAL SAFETY REQUIREMENT:
            # Never advance while someone is present in the door zone!
            if is_occupied:
                # Keep blocked
                self.current_reason = f"DOOR_SAFETY_ZONE_OCCUPIED: {len(occupants)} passenger(s) lingering in doorway"
            else:
                # Obstruction cleared! Revert to DOOR_CLOSE_PENDING for re-verification
                self.transition_to(
                    DoorZoneState.DOOR_CLOSE_PENDING,
                    "OBSTRUCTION_CLEARED: Re-verifying door zone clearance",
                    timestamp=now
                )

        # STATE: DOOR_CLOSING
        elif self.state == DoorZoneState.DOOR_CLOSING:
            # CONTINUOUS RE-CHECK: If someone re-enters or rushes the door, immediately abort!
            if is_occupied:
                self.transition_to(
                    DoorZoneState.DOOR_CLOSE_BLOCKED,
                    "EMERGENCY_REOPEN: Obstruction detected during door closure!",
                    timestamp=now
                )
            elif doors_physically_closed or time_in_state >= self.door_actuation_seconds:
                # Successfully closed and clear
                self.transition_to(
                    DoorZoneState.READY_TO_DEPART,
                    "DOORS_SECURELY_LOCKED_AND_ZONE_CLEAR",
                    timestamp=now
                )

        # STATE: READY_TO_DEPART
        elif self.state == DoorZoneState.READY_TO_DEPART:
            if open_requested:
                self.transition_to(
                    DoorZoneState.DOOR_OPENING,
                    "DOOR_REOPEN_REQUESTED",
                    timestamp=now
                )
            elif time_in_state >= 2.0:
                self.transition_to(
                    DoorZoneState.DOOR_CLOSED,
                    "DEPARTURE_CYCLE_COMPLETE",
                    timestamp=now
                )

        # ----------------------------------------------------------------------
        # 6. ANNOTATED VISUALIZATION
        # ----------------------------------------------------------------------
        annotated_frame = None
        if door_image is not None:
            annotated_frame = door_image.copy()
            rx1, ry1, rx2, ry2 = self.get_pixel_roi(img_w, img_h)

            # Draw Door Safety ROI: Red if occupied/blocked, Green if clear
            roi_color = (0, 0, 255) if is_occupied else (0, 255, 0)
            cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), roi_color, 2)
            cv2.putText(
                annotated_frame,
                f"DOOR SAFETY ROI [{'BLOCKED' if is_occupied else 'CLEAR'}]",
                (rx1 + 5, ry1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, roi_color, 2
            )

            # Draw threshold crossing boundary
            th_y = int(self.threshold_y_norm * img_h)
            cv2.line(annotated_frame, (rx1, th_y), (rx2, th_y), (255, 255, 0), 2)

            # Draw tracked passengers
            for det in tracked_dets:
                tid = det.get("track_id", -1)
                x1, y1, x2, y2 = det.get("x1", 0), det.get("y1", 0), det.get("x2", 0), det.get("y2", 0)
                box_color = (0, 0, 255) if det in occupants else (255, 200, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    annotated_frame,
                    f"ID:{tid}",
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2
                )

            # HUD Status Header
            hud_text = f"STATE: {self.state.value} | IN: {self.boarding_count} OUT: {self.alighting_count}"
            cv2.rectangle(annotated_frame, (0, 0), (img_w, 32), (30, 30, 30), -1)
            cv2.putText(
                annotated_frame,
                hud_text,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2
            )

        return {
            "state": self.state.value,
            "previous_state": self.prev_state.value,
            "state_reason": self.current_reason,
            "door_zone_occupied": bool(is_occupied),
            "occupant_count": len(occupants),
            "occupant_track_ids": [d.get("track_id", -1) for d in occupants],
            "boarding_count": self.boarding_count,
            "alighting_count": self.alighting_count,
            "time_in_state": round(float(time_in_state), 2),
            "can_close_doors": (not is_occupied and self.state in [DoorZoneState.DOOR_CLOSE_PENDING, DoorZoneState.DOOR_CLOSING]),
            "ready_to_depart": (self.state == DoorZoneState.READY_TO_DEPART),
            "door_roi": list(self.get_pixel_roi(img_w, img_h)),
            "annotated_frame": annotated_frame
        }


# ==============================================================================
# CONVENIENCE STANDALONE FUNCTION
# ==============================================================================

_GLOBAL_DOOR_MONITOR: Optional[DoorZoneMonitor] = None

def get_default_door_monitor() -> DoorZoneMonitor:
    global _GLOBAL_DOOR_MONITOR
    if _GLOBAL_DOOR_MONITOR is None:
        _GLOBAL_DOOR_MONITOR = DoorZoneMonitor()
    return _GLOBAL_DOOR_MONITOR
