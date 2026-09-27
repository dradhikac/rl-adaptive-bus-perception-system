"""
tests/test_flask_api.py
===============================================================================
Smoke test suite for web_flask REST and WebSocket API.
Verifies:
1. GET /api/health returns 200 and system metadata.
2. POST /api/analyze returns 200 with standard perception keys from
   src/autonomous_brain_v3.py on a sample image.
3. POST /api/analyze with query flags enables Phase 4-7 enrichment fields.
4. POST /api/analyze accepts base64 JSON payload.
5. SocketIO client can connect to /ws/live namespace.
===============================================================================
"""

import base64
import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from web_flask.app import app, socketio


class TestFlaskAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()
        cls.sample_image_path = REPO_ROOT / "data" / "front.jpg"
        assert cls.sample_image_path.is_file(), f"Sample image missing: {cls.sample_image_path}"

    def test_00_landing_page(self):
        """GET / should return 200 and render the autonomous bus landing page HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AutoBus", response.data)
        self.assertIn(b"SMARTER BUSES", response.data)
        self.assertIn(b"hero-bus-video", response.data)

    def test_00b_demo_page(self):
        """GET /demo should return 200 and render the multi-camera Live Demo page HTML."""
        response = self.client.get("/demo")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AutoBus", response.data)
        self.assertIn(b"360", response.data)
        self.assertIn(b"btn-run-pipeline", response.data)
        self.assertIn(b"results-section", response.data)

    def test_00c_sample_cameras(self):
        """GET /api/sample-cameras should return 200 with camera preset paths."""
        response = self.client.get("/api/sample-cameras")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("front", data)
        self.assertIn("front_left", data)
        self.assertIn("front_right", data)
        self.assertIn("rear_left", data)
        self.assertIn("rear_right", data)

    def test_01_health_check(self):
        """GET /api/health should return 200 and system operational status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIsNotNone(data)
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("models", data)
        self.assertIn("phases_available", data)

    def test_02_analyze_multipart_file(self):
        """POST /api/analyze with multipart image file returns 200 and core perception keys."""
        with open(self.sample_image_path, "rb") as img_file:
            response = self.client.post(
                "/api/analyze",
                data={"file": (img_file, "front.jpg")},
                content_type="multipart/form-data"
            )

        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIsInstance(json_data, dict)

        # Expected core perception keys from unmodified src.autonomous_brain_v3.analyze_scene()
        expected_keys = [
            "object_count",
            "vehicle_count",
            "pedestrian_count",
            "pedestrian_status",
            "traffic_signal",
            "lead_vehicle",
            "lead_vehicle_area",
            "vehicle_distance",
            "brightness",
            "contrast",
            "blur_score",
            "camera_reliability",
            "fusion_confidence",
            "risk_score",
            "risk_level",
            "decision",
            "final_decision",
            "lane_action",
            "recommended_speed",
            "annotated_image"
        ]

        for key in expected_keys:
            self.assertIn(key, json_data, f"Missing expected key: '{key}' in response")

        # Basic type and sanity checks
        self.assertIsInstance(json_data["object_count"], int)
        self.assertIsInstance(json_data["vehicle_count"], int)
        self.assertIsInstance(json_data["risk_score"], (int, float))
        self.assertIn(json_data["risk_level"], ["LOW", "MODERATE", "CRITICAL"])
        self.assertIsInstance(json_data["annotated_image"], dict)
        self.assertEqual(json_data["annotated_image"].get("_type"), "image_ndarray")

    def test_03_analyze_with_enrichment_flags(self):
        """POST /api/analyze with query flags enables Phase 4-7 intelligence enrichments."""
        with open(self.sample_image_path, "rb") as img_file:
            response = self.client.post(
                "/api/analyze?enable_tracking=true&enable_motion_intent=true&enable_bus_stops=true&enable_door_monitor=true",
                data={"file": (img_file, "front.jpg")},
                content_type="multipart/form-data"
            )

        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()

        # Phase 4 & 5 enrichment keys
        self.assertIn("tracked_detections", json_data)
        self.assertIn("active_tracks", json_data)
        self.assertIsInstance(json_data["tracked_detections"], list)

        # Phase 6 enrichment keys
        self.assertIn("bus_stop_detection", json_data)
        self.assertIn("docking_state", json_data)
        self.assertIn("approach_confidence", json_data["bus_stop_detection"])
        self.assertIn("is_approaching", json_data["bus_stop_detection"])
        self.assertIn("state", json_data["docking_state"])

        # Phase 7 enrichment key
        self.assertIn("door_zone_monitor", json_data)
        self.assertIn("state", json_data["door_zone_monitor"])

    def test_04_analyze_base64_json_payload(self):
        """POST /api/analyze accepts base64-encoded frame in JSON body."""
        with open(self.sample_image_path, "rb") as img_file:
            b64_str = "data:image/jpeg;base64," + base64.b64encode(img_file.read()).decode("utf-8")

        response = self.client.post(
            "/api/analyze",
            json={"frame": b64_str},
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("object_count", json_data)
        self.assertIn("decision", json_data)

    def test_05_websocket_connection(self):
        """SocketIO client should connect to /ws/live and receive connection acknowledgment."""
        ws_client = socketio.test_client(app, namespace="/ws/live")
        self.assertTrue(ws_client.is_connected(namespace="/ws/live"))

        received = ws_client.get_received(namespace="/ws/live")
        self.assertGreater(len(received), 0)
        first_event = received[0]
        self.assertEqual(first_event["name"], "connection_status")
        self.assertEqual(first_event["args"][0]["status"], "connected")
        ws_client.disconnect(namespace="/ws/live")

    def test_06_video_demo_upload_and_status(self):
        """POST /api/demo/upload and GET /api/demo/status/<job_id> step 2 integration."""
        test_clip = REPO_ROOT / "tests" / "synthetic_test_clip.mp4"
        self.assertTrue(test_clip.is_file())

        # Test upload
        res = self.client.post("/api/demo/upload", data={
            "front": str(test_clip),
            "target_fps": 5.0,
            "max_duration_sec": 0.4
        })
        self.assertEqual(res.status_code, 202)
        data = res.get_json()
        self.assertIn("job_id", data)
        self.assertEqual(data["status"], "processing")
        job_id = data["job_id"]

        # Poll status
        max_wait = 15.0
        start = time.time()
        final_data = {}
        while time.time() - start < max_wait:
            poll = self.client.get(f"/api/demo/status/{job_id}")
            self.assertEqual(poll.status_code, 200)
            final_data = poll.get_json()
            if final_data.get("status") == "completed":
                break
            time.sleep(0.4)

        self.assertEqual(final_data.get("status"), "completed")
        self.assertGreater(len(final_data.get("timeline", [])), 0)
        first_step = final_data["timeline"][0]
        for key in ["timestamp", "frame_index", "decision", "speed", "lane_action", "risk_score", "risk_level", "explanation", "reliability_scores"]:
            self.assertIn(key, first_step)


if __name__ == "__main__":
    unittest.main(verbosity=2)

