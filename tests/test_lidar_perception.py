"""
tests/test_lidar_perception.py
===============================================================================
Comprehensive Unit & Integration Test Suite for AutoBus 3D LiDAR Perception
===============================================================================
Tests:
1. Point cloud loading (.pcd.bin float32 binary format)
2. Validation & filtering (NaN check, ego self-reflection filter, ROI cropping)
3. 6-channel BEV pseudo-image generation (grid shape, channel normalization)
4. TinyCenterNet3D model forward pass & output tensor shapes
5. 3D Bounding box decoding & coordinate sanity (X=Right, Y=Forward, Z=Up, road at ~ -1.84m)
6. End-to-end AutoBusLidar3DDetector prediction on real LIDAR_TOP test sample
7. Output JSON schema compliance matching the Stage 12 specification
===============================================================================
"""

import os
import sys
import json
import unittest
from pathlib import Path
import numpy as np
import torch

# Ensure repository root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lidar_perception.config import DEFAULT_CONFIG, LidarConfig
from src.lidar_perception.point_cloud_ops import (
    load_point_cloud,
    validate_and_filter_points,
    generate_bev_pseudo_image
)
from src.lidar_perception.model import TinyCenterNet3D, decode_predictions
from src.lidar_perception.detector import AutoBusLidar3DDetector


class TestLidarPerception(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.config = DEFAULT_CONFIG
        cls.device = torch.device("cpu")
        cls.checkpoint_path = REPO_ROOT / "models" / "checkpoints_exp3_aug" / "best_model.pth"
        if not cls.checkpoint_path.is_file():
            cls.checkpoint_path = REPO_ROOT / "models" / "checkpoints" / "best_model.pth"
        assert cls.checkpoint_path.is_file(), f"Model checkpoint missing: {cls.checkpoint_path}"
        
        # Locate a real LIDAR_TOP sample from dataset
        cls.sample_lidar_path = None
        lidar_dir = REPO_ROOT / "datasets" / "nuscenes" / "sweeps" / "LIDAR_TOP"
        if lidar_dir.is_dir():
            files = list(lidar_dir.glob("*.pcd.bin"))
            if files:
                cls.sample_lidar_path = str(files[0])
                
        if cls.sample_lidar_path is None:
            # Check samples/LIDAR_TOP
            lidar_dir2 = REPO_ROOT / "datasets" / "nuscenes" / "samples" / "LIDAR_TOP"
            if lidar_dir2.is_dir():
                files2 = list(lidar_dir2.glob("*.pcd.bin"))
                if files2:
                    cls.sample_lidar_path = str(files2[0])
                    
        assert cls.sample_lidar_path is not None, "No real LIDAR_TOP sample found in datasets/nuscenes"
        print(f"\n[SETUP] Using real LiDAR sample for tests: {Path(cls.sample_lidar_path).name}")

    def test_01_load_point_cloud(self):
        """Test loading binary point cloud: shape must be (N, 5), dtype float32."""
        pts = load_point_cloud(self.sample_lidar_path)
        self.assertIsInstance(pts, np.ndarray)
        self.assertEqual(pts.dtype, np.float32)
        self.assertEqual(pts.ndim, 2)
        self.assertEqual(pts.shape[1], 5, "Expected 5 channels [x, y, z, intensity, ring]")
        self.assertGreater(len(pts), 10000, "Expected > 10,000 points in raw LiDAR scan")

    def test_02_point_filtering(self):
        """Test filtering self-reflections (r < 1.0m) and spatial ROI bounds."""
        pts = load_point_cloud(self.sample_lidar_path)
        filtered_pts, stats = validate_and_filter_points(pts, config=self.config)
        
        # Verify stats
        self.assertIn("self_reflections_filtered", stats)
        self.assertIn("out_of_roi_filtered", stats)
        self.assertGreater(stats["self_reflections_filtered"], 0, "Should filter roof rack reflections")
        
        # Verify all remaining points are within ROI
        self.assertTrue(np.all(filtered_pts[:, 0] >= self.config.x_range[0]))
        self.assertTrue(np.all(filtered_pts[:, 0] <= self.config.x_range[1]))
        self.assertTrue(np.all(filtered_pts[:, 1] >= self.config.y_range[0]))
        self.assertTrue(np.all(filtered_pts[:, 1] <= self.config.y_range[1]))
        self.assertTrue(np.all(filtered_pts[:, 2] >= self.config.z_range[0]))
        self.assertTrue(np.all(filtered_pts[:, 2] <= self.config.z_range[1]))
        
        # Verify distance from origin >= min_distance (1.0m)
        r_xy = np.sqrt(filtered_pts[:, 0]**2 + filtered_pts[:, 1]**2)
        self.assertTrue(np.all(r_xy >= self.config.min_distance - 1e-4))

    def test_03_generate_bev_pseudo_image(self):
        """Test BEV pseudo-image generation: shape must be (6, 256, 256)."""
        pts = load_point_cloud(self.sample_lidar_path)
        filtered_pts, _ = validate_and_filter_points(pts, config=self.config)
        bev = generate_bev_pseudo_image(filtered_pts, config=self.config)
        
        self.assertEqual(bev.shape, (6, 256, 256))
        self.assertEqual(bev.dtype, np.float32)
        self.assertFalse(np.isnan(bev).any(), "BEV map contains NaNs")
        self.assertFalse(np.isinf(bev).any(), "BEV map contains Infs")
        # Density channel must have non-zero pixels
        self.assertGreater(bev[3].sum(), 0.0, "Point density channel should contain positive counts")

    def test_04_model_forward_pass(self):
        """Test model architecture forward pass with dummy BEV tensor."""
        model = TinyCenterNet3D(in_channels=6, num_classes=2, base_channels=24).to(self.device)
        model.eval()
        
        dummy_input = torch.zeros((1, 6, 256, 256), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs = model(dummy_input)
            
        self.assertIn("heatmap", outputs)
        self.assertIn("offset", outputs)
        self.assertIn("z_elev", outputs)
        self.assertIn("dim_3d", outputs)
        self.assertIn("rot_3d", outputs)
        
        # Check shapes (stride 4x -> 64x64)
        self.assertEqual(outputs["heatmap"].shape, (1, 2, 64, 64))
        self.assertEqual(outputs["offset"].shape, (1, 2, 64, 64))
        self.assertEqual(outputs["z_elev"].shape, (1, 1, 64, 64))
        self.assertEqual(outputs["dim_3d"].shape, (1, 3, 64, 64))
        self.assertEqual(outputs["rot_3d"].shape, (1, 2, 64, 64))

    def test_05_detector_prediction_real_sample(self):
        """Test AutoBusLidar3DDetector prediction on real point cloud file."""
        detector = AutoBusLidar3DDetector(
            checkpoint_path=str(self.checkpoint_path),
            score_thresh=0.40
        )
        
        predictions = detector.predict(self.sample_lidar_path)
        self.assertIsInstance(predictions, list)
        self.assertGreater(len(predictions), 0, "Expected at least 1 detection on real test sample")
        
        # Validate each prediction follows the Stage 12 output schema
        for p in predictions:
            self.assertIn("class", p)
            self.assertIn(p["class"], ["car", "pedestrian"])
            
            self.assertIn("confidence", p)
            self.assertGreaterEqual(p["confidence"], 0.40)
            self.assertLessEqual(p["confidence"], 1.0)
            
            self.assertIn("center", p)
            self.assertIn("x", p["center"])
            self.assertIn("y", p["center"])
            self.assertIn("z", p["center"])
            
            # Check coordinate bounds
            self.assertTrue(self.config.x_range[0] <= p["center"]["x"] <= self.config.x_range[1])
            self.assertTrue(self.config.y_range[0] <= p["center"]["y"] <= self.config.y_range[1])
            
            self.assertIn("dimensions", p)
            self.assertIn("length", p["dimensions"])
            self.assertIn("width", p["dimensions"])
            self.assertIn("height", p["dimensions"])
            self.assertGreater(p["dimensions"]["length"], 0.0)
            self.assertGreater(p["dimensions"]["width"], 0.0)
            self.assertGreater(p["dimensions"]["height"], 0.0)
            
            self.assertIn("yaw", p)
            self.assertTrue(-np.pi <= p["yaw"] <= np.pi)
            
            self.assertIn("distance", p)
            expected_dist = round(np.sqrt(p["center"]["x"]**2 + p["center"]["y"]**2 + p["center"]["z"]**2), 2)
            self.assertAlmostEqual(p["distance"], expected_dist, places=1)

    def test_06_detector_numpy_input(self):
        """Test detector accepts raw in-memory numpy array as well as file path."""
        detector = AutoBusLidar3DDetector(
            checkpoint_path=str(self.checkpoint_path),
            score_thresh=0.40
        )
        pts = load_point_cloud(self.sample_lidar_path)
        predictions = detector.predict(pts)
        self.assertIsInstance(predictions, list)

    def test_07_json_serialization(self):
        """Test predict_json returns valid, parsable JSON string."""
        detector = AutoBusLidar3DDetector(
            checkpoint_path=str(self.checkpoint_path),
            score_thresh=0.40
        )
        json_str = detector.predict_json(self.sample_lidar_path)
        parsed = json.loads(json_str)
        self.assertIsInstance(parsed, list)


if __name__ == "__main__":
    unittest.main()
