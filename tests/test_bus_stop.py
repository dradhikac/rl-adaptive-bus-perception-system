"""
tests/test_bus_stop.py
================================================================================
Comprehensive Tests for Bus Stop Detector and Docking State Machine
================================================================================
"""

import json
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

from src.stops.bus_stop_detector import (
    BusStopDetector,
    detect_bus_stop,
    haversine_distance_meters
)
from src.stops.docking_state_machine import (
    DockingState,
    DockingStateMachine
)


class TestBusStopDetectionAndDocking(unittest.TestCase):

    def setUp(self):
        self.route_file = "configs/routes/route_42A.json"
        self.detector = BusStopDetector(route_config_path=self.route_file)
        self.state_machine = DockingStateMachine(min_dwell_seconds=1.0)

    def test_haversine_calculation(self):
        # Known coordinates: Central Station (17.4399, 78.4983) and near point
        dist = haversine_distance_meters(17.4399, 78.4983, 17.4400, 78.4983)
        self.assertGreater(dist, 5.0)
        self.assertLess(dist, 20.0)

    def test_sign_signal_detection(self):
        detections = [
            {"class": "car", "confidence": 0.90, "x1": 100, "y1": 200, "x2": 250, "y2": 320},
            {"class": "bus_stop_sign", "confidence": 0.92, "x1": 480, "y1": 150, "x2": 540, "y2": 260}
        ]
        score, best_box = self.detector.evaluate_sign_signal(detections)
        self.assertGreaterEqual(score, 0.80)
        self.assertIsNotNone(best_box)
        self.assertEqual(best_box["class"], "bus_stop_sign")

    def test_gps_geofence_signal(self):
        # STOP_001 is at lat 17.4399, lon 78.4983 (radius 35m, approach 120m)
        # Inside stop radius
        score_in, stop_in, dist_in = self.detector.evaluate_gps_signal(17.43991, 78.49831)
        self.assertEqual(score_in, 1.0)
        self.assertEqual(stop_in["stop_id"], "STOP_001")
        self.assertLess(dist_in, 35.0)

        # In approach corridor (~60m away)
        score_app, stop_app, dist_app = self.detector.evaluate_gps_signal(17.4404, 78.4983)
        self.assertGreater(score_app, 0.40)
        self.assertLess(score_app, 1.0)

        # Far away (>500m)
        score_far, _, dist_far = self.detector.evaluate_gps_signal(17.5000, 78.6000)
        self.assertEqual(score_far, 0.0)

    def test_crowd_cluster_heuristic(self):
        # 4 stationary people clustered on right-side curbside (x: 480-560, y: 300-400)
        detections = [
            {"class": "person", "x1": 480, "y1": 310, "x2": 510, "y2": 390, "motion_intent": "stationary"},
            {"class": "person", "x1": 500, "y1": 305, "x2": 530, "y2": 385, "motion_intent": "stationary"},
            {"class": "person", "x1": 520, "y1": 315, "x2": 550, "y2": 395, "motion_intent": "stationary"},
            {"class": "person", "x1": 535, "y1": 310, "x2": 565, "y2": 390, "motion_intent": "stationary"},
            # 1 person walking on road (crossing)
            {"class": "person", "x1": 200, "y1": 300, "x2": 240, "y2": 380, "motion_intent": "crossing"}
        ]
        score, count, peds = self.detector.evaluate_crowd_signal(detections)
        self.assertGreaterEqual(score, 0.75)
        self.assertEqual(count, 4)
        self.assertEqual(len(peds), 4)

    def test_fused_detection_confidence(self):
        # Test multi-signal fusion when all 3 signals are active
        detections = [
            {"class": "bus_stop_sign", "confidence": 0.90, "x1": 480, "y1": 150, "x2": 540, "y2": 260},
            {"class": "person", "x1": 480, "y1": 310, "x2": 510, "y2": 390, "motion_intent": "stationary"},
            {"class": "person", "x1": 500, "y1": 305, "x2": 530, "y2": 385, "motion_intent": "stationary"},
            {"class": "person", "x1": 520, "y1": 315, "x2": 550, "y2": 395, "motion_intent": "stationary"}
        ]
        lat, lon = 17.43991, 78.49831

        result = self.detector.detect_bus_stop(detections, latitude=lat, longitude=lon)
        self.assertTrue(result["is_approaching"])
        self.assertGreaterEqual(result["approach_confidence"], 0.85)
        self.assertIn("BUS_STOP_SIGN_DETECTED", result["trigger_reasons"][0])
        self.assertTrue(result["signals"]["sign_detected"])
        self.assertTrue(result["signals"]["gps_near_stop"])
        self.assertTrue(result["signals"]["crowd_cluster_detected"])

    def test_docking_sequence_clip(self):
        """
        Creates and processes a synthetic approach-to-stop sequence clip
        asserting that the docking state machine reaches STOPPED.
        """
        clip_path = "tests/synthetic_bus_stop_approach.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, 10, (640, 480))

        # Generate 40 synthetic frames showing bus approach to stop
        for frame_idx in range(40):
            frame = np.full((480, 640, 3), 40, dtype=np.uint8)
            # Draw road and curb
            cv2.line(frame, (320, 200), (100, 480), (255, 255, 255), 2)
            cv2.line(frame, (320, 200), (540, 480), (255, 255, 255), 2)
            # As bus approaches, draw curbside sign that grows
            if frame_idx > 10:
                scale = (frame_idx - 10) * 2
                cv2.rectangle(frame, (480, 250 - scale), (510 + scale, 310), (0, 200, 255), -1)
            out.write(frame)
        out.release()
        self.assertTrue(Path(clip_path).is_file())

        # Simulate vehicle telemetry across the approach
        states_encountered = []
        t = 1000.0

        for frame_idx in range(40):
            t += 0.10

            # Signal profile over time
            if frame_idx < 10:
                # Cruising at road speed
                speed = 35.0
                dets = []
                lat, lon = 17.4350, 78.4900  # far
            elif frame_idx < 22:
                # Approaching stop, slowing down
                speed = 35.0 - (frame_idx - 10) * 1.8
                dets = [
                    {"class": "bus_stop_sign", "confidence": 0.88, "x1": 480, "y1": 200, "x2": 520, "y2": 260},
                    {"class": "person", "x1": 500, "y1": 310, "x2": 525, "y2": 380, "motion_intent": "stationary"},
                    {"class": "person", "x1": 530, "y1": 310, "x2": 555, "y2": 380, "motion_intent": "stationary"},
                    {"class": "person", "x1": 560, "y1": 310, "x2": 585, "y2": 380, "motion_intent": "stationary"},
                ]
                lat, lon = 17.4402, 78.4983  # within approach zone
            elif frame_idx < 32:
                # Decelerating to halt right at curb
                speed = max(0.0, 14.0 - (frame_idx - 22) * 2.0)
                dets = [
                    {"class": "bus_stop_sign", "confidence": 0.95, "x1": 480, "y1": 150, "x2": 540, "y2": 270},
                    {"class": "person", "x1": 500, "y1": 300, "x2": 530, "y2": 390, "motion_intent": "stationary"},
                    {"class": "person", "x1": 535, "y1": 300, "x2": 565, "y2": 390, "motion_intent": "stationary"},
                    {"class": "person", "x1": 570, "y1": 300, "x2": 600, "y2": 390, "motion_intent": "stationary"},
                ]
                lat, lon = 17.43991, 78.49831  # directly at stop
            else:
                # At full halt
                speed = 0.0
                dets = [
                    {"class": "bus_stop_sign", "confidence": 0.95, "x1": 480, "y1": 150, "x2": 540, "y2": 270},
                    {"class": "person", "x1": 500, "y1": 300, "x2": 530, "y2": 390, "motion_intent": "stationary"},
                    {"class": "person", "x1": 535, "y1": 300, "x2": 565, "y2": 390, "motion_intent": "stationary"},
                    {"class": "person", "x1": 570, "y1": 300, "x2": 600, "y2": 390, "motion_intent": "stationary"},
                ]
                lat, lon = 17.43991, 78.49831

            det_res = self.detector.detect_bus_stop(dets, latitude=lat, longitude=lon)
            sm_res = self.state_machine.update(det_res, vehicle_speed_kmh=speed, timestamp=t)
            states_encountered.append(sm_res["state"])

        # Assert key state transitions occurred
        self.assertIn("DRIVE", states_encountered)
        self.assertIn("APPROACH_STOP", states_encountered)
        self.assertIn("STOPPING", states_encountered)
        self.assertIn("STOPPED", states_encountered)

        # Critical requirement: assert state machine reaches STOPPED
        self.assertEqual(states_encountered[-1], "STOPPED")
        self.assertIn("FULL_HALT_REACHED", sm_res["state_reason"])

    def test_full_docking_cycle(self):
        """
        Tests the entire cycle:
        DRIVE -> APPROACH_STOP -> STOPPING -> STOPPED -> BOARDING -> READY_TO_DEPART -> DRIVE
        """
        sm = DockingStateMachine(min_dwell_seconds=1.5)
        t = 100.0

        # 1. Start in DRIVE
        self.assertEqual(sm.state, DockingState.DRIVE)

        # 2. Trigger APPROACH_STOP
        res = sm.update({"is_approaching": True, "approach_confidence": 0.85, "trigger_reasons": ["BUS_STOP_SIGN"]}, vehicle_speed_kmh=35.0, timestamp=t)
        self.assertEqual(sm.state, DockingState.APPROACH_STOP)

        # 3. Decelerate to STOPPING
        t += 1.0
        res = sm.update({"is_approaching": True, "signals": {"sign_confidence": 0.80}}, vehicle_speed_kmh=12.0, timestamp=t)
        self.assertEqual(sm.state, DockingState.STOPPING)

        # 4. Halt to STOPPED
        t += 1.0
        res = sm.update({"is_approaching": True}, vehicle_speed_kmh=0.0, timestamp=t)
        self.assertEqual(sm.state, DockingState.STOPPED)

        # 5. Open doors to BOARDING
        t += 0.5
        res = sm.update({"is_approaching": True}, vehicle_speed_kmh=0.0, doors_open=True, timestamp=t)
        self.assertEqual(sm.state, DockingState.BOARDING)

        # 6. Dwell time elapsed, close doors to READY_TO_DEPART
        t += 2.0  # > min_dwell_seconds (1.5)
        res = sm.update({"is_approaching": True}, vehicle_speed_kmh=0.0, doors_open=False, timestamp=t)
        self.assertEqual(sm.state, DockingState.READY_TO_DEPART)

        # 7. Resume driving back to DRIVE
        t += 0.5
        res = sm.update({"is_approaching": False}, vehicle_speed_kmh=5.0, doors_open=False, timestamp=t)
        self.assertEqual(sm.state, DockingState.DRIVE)


if __name__ == "__main__":
    unittest.main()
