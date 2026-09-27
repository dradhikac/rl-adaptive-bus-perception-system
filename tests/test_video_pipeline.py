"""
tests/test_video_pipeline.py
===============================================================================
Comprehensive test suite for STEP 2 Video Ingestion Pipeline.

Validates:
1. Slot alias mapping and normalization.
2. Degraded-camera handling & reliability computation reuse.
3. Synchronous and asynchronous multi-camera frame sampling at configurable FPS.
4. JSON timeline structure, schema keys, and decision outputs.
5. In-memory VideoJobStore thread-safe lifecycle operations.
6. Flask REST endpoints: POST /api/demo/upload and GET /api/demo/status/<job_id>.
===============================================================================
"""

import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.video.video_pipeline import (
    VideoIngestionPipeline,
    VideoJobStore,
    compute_camera_reliability,
    create_degraded_dead_frame,
    normalize_slot_name,
    get_global_job_store,
    CANONICAL_SLOTS,
)
from web_flask.app import app


class TestVideoPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_clip_1 = REPO_ROOT / "tests" / "synthetic_test_clip.mp4"
        cls.test_clip_2 = REPO_ROOT / "datasets" / "self-collected" / "eval_clips" / "clip_03_urban_traffic.mp4"
        assert cls.test_clip_1.is_file(), f"Missing test clip: {cls.test_clip_1}"
        assert cls.test_clip_2.is_file(), f"Missing test clip: {cls.test_clip_2}"

        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_01_slot_normalization(self):
        """Validates canonical slot normalization and alias resolution."""
        self.assertEqual(normalize_slot_name("front"), "front")
        self.assertEqual(normalize_slot_name("front_image"), "front")
        self.assertEqual(normalize_slot_name("front_left"), "front_left")
        self.assertEqual(normalize_slot_name("camera_front_right"), "front_right")
        self.assertEqual(normalize_slot_name("back_left"), "rear_left")
        self.assertEqual(normalize_slot_name("rear_left"), "rear_left")
        self.assertEqual(normalize_slot_name("back_right"), "rear_right")
        self.assertEqual(normalize_slot_name("rear_right"), "rear_right")
        self.assertIsNone(normalize_slot_name("invalid_slot_name"))

    def test_02_degraded_camera_reliability(self):
        """Verifies degraded camera handling reuses sensor_reliability heuristics."""
        dead_frame = create_degraded_dead_frame(640, 480)
        self.assertEqual(dead_frame.shape, (480, 640, 3))

        # Missing feed score is 0.0
        missing_rel = compute_camera_reliability(None, is_missing=True)
        self.assertEqual(missing_rel, 0.0)

        # Blank black frame has brightness=0, contrast=0, blur=0, edges=0
        # 1.0 - 0.2*4 = 0.20
        dead_rel = compute_camera_reliability(dead_frame, is_missing=False)
        self.assertAlmostEqual(dead_rel, 0.20, places=2)

        # Natural image with features has higher reliability
        natural_img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        natural_rel = compute_camera_reliability(natural_img, is_missing=False)
        self.assertGreater(natural_rel, 0.20)

    def test_03_single_camera_ingestion_with_missing_feeds(self):
        """
        Validates pipeline accepts 1 uploaded video (front camera),
        fills the 4 missing camera feeds with degraded dead frames,
        and generates a time-aligned perception timeline.
        """
        pipeline = VideoIngestionPipeline(
            video_paths={"front": str(self.test_clip_1)},
            target_fps=5.0,
            max_duration_sec=1.0  # Limit to 1.0 second for fast unit test
        )

        self.assertEqual(pipeline.active_slots, ["front"])
        self.assertEqual(len(pipeline.missing_slots), 4)

        timeline, summary = pipeline.process()
        self.assertGreater(len(timeline), 0)
        self.assertLessEqual(len(timeline), 6)

        entry = timeline[0]
        # Required timeline keys
        expected_keys = [
            "timestamp",
            "frame_index",
            "decision",
            "speed",
            "lane_action",
            "risk_score",
            "risk_level",
            "explanation",
            "contributing_factors",
            "reliability_scores",
        ]
        for key in expected_keys:
            self.assertIn(key, entry, f"Missing timeline key: {key}")

        # Missing cameras in reliability scores must be 0.0
        self.assertEqual(entry["reliability_scores"]["front_left"], 0.0)
        self.assertEqual(entry["reliability_scores"]["rear_right"], 0.0)

        # Summary check
        self.assertEqual(summary["total_timesteps"], len(timeline))
        self.assertIn("dominant_decision", summary)
        self.assertIn("max_risk_score", summary)

    def test_04_multi_camera_synchronization(self):
        """Validates synchronization across multiple uploaded video feeds."""
        pipeline = VideoIngestionPipeline(
            video_paths={
                "front": str(self.test_clip_1),
                "back_left": str(self.test_clip_2),  # tests alias
            },
            target_fps=5.0,
            max_duration_sec=1.0
        )
        self.assertIn("front", pipeline.active_slots)
        self.assertIn("rear_left", pipeline.active_slots)

        timeline, summary = pipeline.process()
        self.assertGreater(len(timeline), 0)
        entry = timeline[0]
        # Both active feeds should have positive reliability
        self.assertGreater(entry["reliability_scores"]["front"], 0.0)
        self.assertGreater(entry["reliability_scores"]["rear_left"], 0.0)
        # Unprovided feed remains 0.0
        self.assertEqual(entry["reliability_scores"]["front_right"], 0.0)

    def test_05_job_store_lifecycle(self):
        """Tests VideoJobStore thread-safe lifecycle operations."""
        store = VideoJobStore()
        job_id = store.create_job(
            active_slots=["front"],
            missing_slots=["front_left", "front_right", "rear_left", "rear_right"],
            target_fps=5.0,
            video_paths={"front": str(self.test_clip_1)}
        )

        job = store.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["progress"], 0.0)

        # Update progress
        store.update_progress(job_id, current_frame=2, total_frames=10)
        job = store.get_job(job_id)
        self.assertEqual(job["status"], "processing")
        self.assertEqual(job["progress"], 20.0)

        # Complete job
        store.complete_job(job_id, timeline=[{"decision": "MOVE"}], summary={"total_timesteps": 1})
        job = store.get_job(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"], 100.0)
        self.assertEqual(len(job["timeline"]), 1)

    def test_06_flask_upload_and_status_endpoints(self):
        """Tests Flask POST /api/demo/upload and GET /api/demo/status/<job_id> endpoints."""
        # 1. Test upload with file path payload
        payload = {
            "front": str(self.test_clip_1),
            "target_fps": 5.0,
            "max_duration_sec": 0.6  # ~3 timesteps
        }
        res = self.client.post("/api/demo/upload", data=payload)
        self.assertEqual(res.status_code, 202)
        res_data = res.get_json()
        self.assertIn("job_id", res_data)
        job_id = res_data["job_id"]
        self.assertEqual(res_data["status"], "processing")
        self.assertIn("front", res_data["active_cameras"])

        # 2. Poll status endpoint until completion
        max_wait = 15.0
        start = time.time()
        completed = False
        final_data = {}

        while time.time() - start < max_wait:
            poll_res = self.client.get(f"/api/demo/status/{job_id}")
            self.assertEqual(poll_res.status_code, 200)
            final_data = poll_res.get_json()
            if final_data["status"] == "completed":
                completed = True
                break
            time.sleep(0.5)

        self.assertTrue(completed, f"Job did not complete within {max_wait}s. Status: {final_data.get('status')}")
        self.assertGreater(len(final_data["timeline"]), 0)
        self.assertIn("dominant_decision", final_data["summary"])

    def test_07_flask_upload_validation_errors(self):
        """Tests error responses for invalid upload inputs and non-existent jobs."""
        # No videos provided -> 400
        res = self.client.post("/api/demo/upload", data={})
        self.assertEqual(res.status_code, 400)
        self.assertIn("error", res.get_json())

        # Unknown job id -> 404
        res_404 = self.client.get("/api/demo/status/unknown-uuid-12345")
        self.assertEqual(res_404.status_code, 404)


if __name__ == "__main__":
    unittest.main()
