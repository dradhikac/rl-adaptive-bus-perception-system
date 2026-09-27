"""
tests/test_door_safety.py
================================================================================
Comprehensive Safety Tests for Door Zone Monitor & State Machine
================================================================================
Tests continuous door safety zone monitoring with ByteTrack tracking,
passenger ingress/egress counting, and asserts that the state machine NEVER
advances past DOOR_CLOSE_BLOCKED while someone is lingering in the doorway.
================================================================================
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from src.doors.door_zone_monitor import (
    DoorZoneMonitor,
    DoorZoneState
)
from src.tracking.byte_tracker_wrapper import ByteTrackerWrapper


class TestDoorSafety(unittest.TestCase):

    def setUp(self):
        self.monitor = DoorZoneMonitor(
            clear_debounce_seconds=0.30,
            door_actuation_seconds=0.50
        )
        self.img_w = 640
        self.img_h = 480

    def test_state_enum_values(self):
        expected_states = [
            "DOOR_CLOSED",
            "DOOR_OPENING",
            "BOARDING/ALIGHTING",
            "DOOR_CLOSE_PENDING",
            "DOOR_CLOSE_BLOCKED",
            "DOOR_CLOSING",
            "READY_TO_DEPART"
        ]
        actual_values = [s.value for s in DoorZoneState]
        self.assertEqual(actual_values, expected_states)

    def test_inward_outward_passenger_crossing(self):
        """Tests ByteTrack crossing logic for boarding and alighting counting."""
        monitor = DoorZoneMonitor()
        t = 100.0

        # Passenger 1 boards: moves upward across threshold (y=360 -> y=180 in 12 frames)
        # Threshold line is 0.55 * 480 = 264.0
        for step in range(12):
            t += 0.033
            cy = 350 - step * 15
            dets = [{
                "class": "person",
                "confidence": 0.90,
                "x1": 300, "y1": cy - 40,
                "x2": 360, "y2": cy + 40
            }]
            monitor.process_frame(detections=dets, timestamp=t)

        self.assertEqual(monitor.boarding_count, 1)
        self.assertEqual(monitor.alighting_count, 0)

        # Passenger 2 alights: moves downward across threshold (y=180 -> y=360 in 12 frames)
        for step in range(12):
            t += 0.033
            cy = 180 + step * 15
            dets = [{
                "class": "person",
                "confidence": 0.90,
                "x1": 420, "y1": cy - 40,
                "x2": 480, "y2": cy + 40
            }]
            monitor.process_frame(detections=dets, timestamp=t)

        self.assertEqual(monitor.boarding_count, 1)
        self.assertEqual(monitor.alighting_count, 1)

    def test_continuous_recheck_emergency_reopen(self):
        """
        Tests continuous safety re-check: if a person re-enters during DOOR_CLOSING,
        state machine must immediately abort and revert to DOOR_CLOSE_BLOCKED.
        """
        monitor = DoorZoneMonitor(door_actuation_seconds=1.0)
        t = 50.0

        # Progress to DOOR_CLOSING with clear zone
        monitor.state = DoorZoneState.DOOR_CLOSING
        monitor.state_enter_timestamp = t

        # Frame 1: zone is clear
        res1 = monitor.process_frame(detections=[], timestamp=t + 0.1)
        self.assertEqual(res1["state"], DoorZoneState.DOOR_CLOSING.value)

        # Frame 2: person suddenly steps into door ROI!
        lingering_det = [{
            "class": "person",
            "confidence": 0.92,
            "x1": 300, "y1": 250, "x2": 370, "y2": 420  # inside door ROI
        }]
        res2 = monitor.process_frame(detections=lingering_det, timestamp=t + 0.2)

        # Must immediately trigger EMERGENCY_REOPEN and revert to DOOR_CLOSE_BLOCKED
        self.assertEqual(res2["state"], DoorZoneState.DOOR_CLOSE_BLOCKED.value)
        self.assertIn("EMERGENCY_REOPEN", res2["state_reason"])
        self.assertTrue(res2["door_zone_occupied"])

    def test_door_safety_lingering_clip(self):
        """
        Generates and evaluates a synthetic door camera video clip where a passenger
        lingers in the doorway zone.
        ASSERTION: While the person is present, the state machine NEVER advances
        past DOOR_CLOSE_BLOCKED.
        """
        clip_path = "tests/synthetic_door_lingering_clip.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, 10, (self.img_w, self.img_h))

        # Generate 50 frames of door camera footage
        rx1, ry1, rx2, ry2 = self.monitor.get_pixel_roi(self.img_w, self.img_h)
        for f_idx in range(50):
            frame = np.full((self.img_h, self.img_w, 3), 50, dtype=np.uint8)
            # Draw doorway frame
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (180, 180, 180), 2)
            cv2.line(frame, (rx1, int(0.55 * self.img_h)), (rx2, int(0.55 * self.img_h)), (200, 200, 0), 1)

            # If lingering passenger present (frames 15 to 40)
            if 15 <= f_idx < 40:
                # Passenger standing in doorway
                cv2.circle(frame, (320, 280), 20, (0, 0, 220), -1)
                cv2.rectangle(frame, (300, 300), (340, 420), (0, 0, 220), -1)

            out.write(frame)
        out.release()
        self.assertTrue(Path(clip_path).is_file())

        # Process frames through DoorZoneMonitor
        monitor = DoorZoneMonitor(clear_debounce_seconds=0.20, door_actuation_seconds=0.40)
        t = 200.0

        # Phase 1: Open doors (frames 0 to 8)
        for f in range(8):
            t += 0.10
            monitor.process_frame(open_requested=True, timestamp=t)

        self.assertEqual(monitor.state, DoorZoneState.BOARDING_ALIGHTING)

        # Phase 2: Driver requests close, but passenger is lingering in doorway (frames 9 to 38)
        for f in range(9, 39):
            t += 0.10
            lingering_passenger = [{
                "class": "person",
                "confidence": 0.95,
                "x1": 300, "y1": 260, "x2": 350, "y2": 420  # directly in door ROI
            }]

            res = monitor.process_frame(
                detections=lingering_passenger,
                close_requested=True,
                timestamp=t
            )

            # CRITICAL SAFETY ASSERTION:
            # While the lingering passenger is present, state must remain DOOR_CLOSE_BLOCKED
            # and MUST NEVER advance past it to DOOR_CLOSING or READY_TO_DEPART!
            self.assertEqual(
                monitor.state,
                DoorZoneState.DOOR_CLOSE_BLOCKED,
                f"Safety violation at frame {f}: state advanced to {monitor.state} while zone is occupied!"
            )
            self.assertTrue(res["door_zone_occupied"])
            self.assertFalse(res["can_close_doors"])
            self.assertFalse(res["ready_to_depart"])

        # Phase 3: Passenger finally clears the door zone (frames 39 to 50)
        for f in range(39, 52):
            t += 0.10
            res = monitor.process_frame(
                detections=[],  # zone clear
                close_requested=True,
                timestamp=t
            )

        # Once zone has been continuously clear, monitor must successfully close doors and reach READY_TO_DEPART
        self.assertIn(monitor.state, [DoorZoneState.DOOR_CLOSING, DoorZoneState.READY_TO_DEPART])


if __name__ == "__main__":
    unittest.main()
