"""
tests/test_motion_intent.py
================================================================================
Comprehensive Unit & Integration Tests for Temporal Motion Intent Module
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

from src.pedestrian_intelligence import analyze_pedestrians
from src.temporal.motion_intent import (
    IDX_TO_INTENT,
    INTENT_CLASSES,
    INTENT_TO_IDX,
    MotionIntentGRU,
    classify_gru,
    classify_heuristic,
    classify_motion_intent,
    extract_trajectory_features
)


class TestMotionIntent(unittest.TestCase):

    def setUp(self):
        self.img_w = 640
        self.img_h = 480
        self.dt = 1.0 / 30.0

    def test_intent_classes_definition(self):
        expected_classes = ["approaching-road", "crossing", "stationary", "receding", "erratic"]
        self.assertEqual(INTENT_CLASSES, expected_classes)
        for c in expected_classes:
            self.assertIn(c, INTENT_TO_IDX)
            self.assertEqual(IDX_TO_INTENT[INTENT_TO_IDX[c]], c)

    def test_stationary_heuristic(self):
        # Standing still at bus stop with tiny jitter
        history = []
        cx, cy, w, h = 150.0, 360.0, 35.0, 80.0
        for i in range(16):
            history.append({
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "timestamp": i * self.dt
            })
            cx += 0.1
            cy += 0.05

        intent = classify_heuristic(history, self.img_w, self.img_h)
        self.assertEqual(intent, "stationary")

    def test_crossing_heuristic(self):
        # Moving laterally from sidewalk across lane (x=100 -> x=300)
        history = []
        cx, cy, w, h = 100.0, 350.0, 35.0, 80.0
        for i in range(16):
            history.append({
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "timestamp": i * self.dt
            })
            cx += 7.0  # ~210 px/s lateral

        intent = classify_heuristic(history, self.img_w, self.img_h)
        self.assertEqual(intent, "crossing")

    def test_approaching_road_heuristic(self):
        # Stepping off curb toward road center (x: 120 -> 220) and camera (y: 300 -> 350), area expanding
        history = []
        cx, cy, w, h = 120.0, 300.0, 30.0, 70.0
        for i in range(16):
            history.append({
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "timestamp": i * self.dt
            })
            cx += 4.0   # moving toward center (320)
            cy += 3.0   # moving down toward camera
            w *= 1.015  # expanding
            h *= 1.015

        intent = classify_heuristic(history, self.img_w, self.img_h)
        self.assertEqual(intent, "approaching-road")

    def test_receding_heuristic(self):
        # Walking away from road into sidewalk (x: 200 -> 100), moving up and shrinking
        history = []
        cx, cy, w, h = 200.0, 360.0, 45.0, 100.0
        for i in range(16):
            history.append({
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "timestamp": i * self.dt
            })
            cx -= 5.0   # moving away from center (320)
            cy -= 4.0   # moving toward horizon
            w *= 0.985  # shrinking
            h *= 0.985

        intent = classify_heuristic(history, self.img_w, self.img_h)
        self.assertEqual(intent, "receding")

    def test_erratic_heuristic(self):
        # Reversing direction repeatedly
        history = []
        cx, cy, w, h = 320.0, 350.0, 35.0, 80.0
        direction = 1.0
        for i in range(16):
            if i % 3 == 0:
                direction = -direction
            cx += direction * 15.0
            history.append({
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "timestamp": i * self.dt
            })

        intent = classify_heuristic(history, self.img_w, self.img_h)
        self.assertEqual(intent, "erratic")

    def test_gru_mode(self):
        # Test GRU inference via unified API
        history = []
        cx, cy, w, h = 150.0, 360.0, 35.0, 80.0
        for i in range(16):
            history.append({
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "timestamp": i * self.dt
            })

        pred_gru = classify_motion_intent(history, mode="gru", model_path="models/motion_intent_gru.pt")
        self.assertIn(pred_gru, INTENT_CLASSES)

    def test_unified_api_default_mode(self):
        # Default mode should be heuristic
        history = []
        cx, cy, w, h = 100.0, 350.0, 35.0, 80.0
        for i in range(16):
            history.append({
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "timestamp": i * self.dt
            })
            cx += 8.0

        intent_default = classify_motion_intent(history)
        intent_heuristic = classify_heuristic(history)
        self.assertEqual(intent_default, intent_heuristic)
        self.assertEqual(intent_default, "crossing")

    def test_pedestrian_intelligence_integration(self):
        # Test that analyze_pedestrians attaches motion_intent to detections and returns it
        ped_history = []
        cx, cy, w, h = 100.0, 350.0, 35.0, 80.0
        for i in range(16):
            ped_history.append({
                "bbox": [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)],
                "timestamp": i * self.dt
            })
            cx += 7.0

        detections = [
            {
                "class": "person",
                "x1": int(cx - w / 2),
                "y1": int(cy - h / 2),
                "x2": int(cx + w / 2),
                "y2": int(cy + h / 2),
                "history": ped_history
            }
        ]

        result = analyze_pedestrians(detections, self.img_w, self.img_h)

        # Existing fields preserved
        self.assertEqual(result["pedestrian_count"], 1)
        self.assertIn("in_lane_count", result)
        self.assertIn("near_lane_count", result)
        self.assertIn("sidewalk_count", result)
        self.assertIn("pedestrian_status", result)

        # New optional motion_intent signal
        self.assertIn("motion_intent", result)
        self.assertIn("pedestrian_intents", result)
        self.assertEqual(result["motion_intent"], "crossing")
        self.assertEqual(detections[0]["motion_intent"], "crossing")


if __name__ == "__main__":
    unittest.main()
