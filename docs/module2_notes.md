# Module 2 — nuScenes Sensor Fusion Module

## Goal

Build a Camera + LiDAR Sensor Fusion System for Autonomous Bus Perception.

---

## Progress

### STEP 1 — Dataset Setup ✅

Completed:

- Installed nuScenes SDK
- Installed Open3D
- Installed sensor fusion dependencies
- Downloaded nuScenes Mini dataset
- Verified dataset loading

Dataset Statistics:

- 10 scenes
- 404 samples
- 31206 sample_data entries
- 18538 annotations

---

### STEP 2 — Dataset Exploration ✅

Completed:

- Loaded first scene
- Accessed first sample
- Explored sensor hierarchy
- Listed all available sensors

Sensors Available:

- CAM_FRONT
- CAM_FRONT_LEFT
- CAM_FRONT_RIGHT
- CAM_BACK
- CAM_BACK_LEFT
- CAM_BACK_RIGHT
- LIDAR_TOP
- RADAR_FRONT
- RADAR_FRONT_LEFT
- RADAR_FRONT_RIGHT
- RADAR_BACK_LEFT
- RADAR_BACK_RIGHT

---

### STEP 3 — Camera Data Loading ✅

Completed:

- Loaded first camera frame
- Extracted image path
- Displayed camera image

Results:

Image Shape:
(900, 1600, 3)

---

### STEP 4 — LiDAR Point Cloud Loading ✅

Completed:

- Loaded first LiDAR scan
- Parsed .pcd.bin file
- Extracted point cloud
- Visualized top-view point cloud

Results:

Point Cloud Shape:
(34688, 5)

Total Points:
34688

Point Format:

[x, y, z, intensity, ring]

---

## Current Architecture

nuScenes
    ↓
Scene
    ↓
Sample
    ↓
Camera Image

    +

LiDAR Point Cloud

---

## Next Step

### STEP 5 — 3D Point Cloud Visualization ✅

Completed:

- Installed Open3D
- Visualized LiDAR point cloud
- Explored 3D environment

Results:

Point Cloud Shape:
(34688, 5)

XYZ Shape:
(34688, 3)

---

### STEP 6 — Sensor Calibration & Synchronization ✅

Completed:

- Loaded camera calibration
- Loaded LiDAR calibration
- Accessed sensor translations
- Accessed sensor rotations
- Understood coordinate systems

Key Concept:

Camera and LiDAR occupy different
physical positions and require
coordinate transformation before fusion.

### STEP 7 — Camera Projection Foundations ✅

Completed:

- Loaded camera intrinsic matrix
- Accessed camera parameters
- Understood 3D → 2D projection
- Created first projection example

Key Concepts:

- Camera intrinsics
- Focal length
- Image center
- Pixel projection

Foundation for Camera-LiDAR Fusion established.

### STEP 8 — First Camera + LiDAR Fusion Visualization ✅

Completed:

- Loaded camera image
- Loaded LiDAR point cloud
- Filtered forward-facing points
- Projected LiDAR points onto image
- Created first fusion visualization

Results:

- Camera image loaded
- LiDAR point cloud loaded
- Front-facing point extraction completed
- Overlay visualization completed

Key Concept:

Camera provides semantic information.
LiDAR provides spatial information.
Fusion combines both.

### STEP 9 — Calibration-Based Sensor Fusion ✅

Completed:

- Loaded LiDAR point cloud
- Loaded camera calibration
- Loaded LiDAR calibration
- Performed coordinate transformations
- Projected 3D LiDAR points into camera image
- Created real fusion visualization

Key Concepts:

LiDAR Frame
→ Vehicle Frame
→ Camera Frame
→ Image Plane

First true sensor fusion pipeline implemented.

### STEP 10 — Multi-Camera Perception System ✅

Completed:

- Loaded six camera streams
- Built 360° surround view
- Created multi-camera visualization

Cameras:

- CAM_FRONT
- CAM_FRONT_LEFT
- CAM_FRONT_RIGHT
- CAM_BACK
- CAM_BACK_LEFT
- CAM_BACK_RIGHT

Result:

360° environmental awareness achieved.

Key Concept:

Autonomous buses require perception beyond the front camera to monitor blind spots and surrounding traffic.

### STEP 11 — Sensor Reliability Estimation ✅

Completed:

- Camera reliability estimation
- LiDAR reliability estimation
- Environmental difficulty scoring
- Fusion confidence scoring

Metrics:

- Brightness
- Contrast
- Blur
- Edge Count
- LiDAR Point Count
- LiDAR Intensity

Outputs:

- Camera Reliability
- LiDAR Reliability
- Environment Difficulty
- Fusion Confidence

Key Concept:

Sensor fusion quality depends on the reliability of each individual sensor.

### STEP 12 — Adaptive Sensor Weighting ✅

Completed:

- Computed sensor reliability scores
- Converted reliability into dynamic weights
- Built adaptive weighting engine

Results:

Camera Reliability : 0.800
LiDAR Reliability  : 1.000

Camera Weight      : 0.444
LiDAR Weight       : 0.556

Key Concept:

More reliable sensors receive greater influence in the perception pipeline.

### STEP 13 — Multi-Camera Reliability Analysis ✅

Completed:

- Evaluated all six cameras
- Computed per-camera reliability
- Ranked cameras by quality
- Identified best and worst camera streams

Outputs:

- Camera reliability table
- Camera ranking
- Reliability report CSV

Key Concept:

Different cameras may experience different environmental conditions and should not be treated equally.

### STEP 15 — Fusion Dashboard ✅

Completed:

- Unified sensor dashboard
- Adaptive weighting display
- Global confidence calculation
- Environment difficulty estimation
- System status monitoring

Outputs:

- Fusion dashboard
- Sensor status report
- Confidence metrics

Key Concept:

The dashboard acts as the central perception monitor for the autonomous bus.