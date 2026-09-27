"""
src/temporal/motion_intent.py
===============================================================================
Autonomous Bus Perception System - Temporal Motion Intent Classifier
===============================================================================
Classifies tracked pedestrians and vehicles over an 8-16 frame trajectory window
into one of five operational motion intent categories:
  1. 'approaching-road' : Object moving towards the road center or approaching the bus.
  2. 'crossing'         : Object moving laterally across the lane / travel corridor.
  3. 'stationary'       : Minimal displacement, standing/waiting at curb or stopped.
  4. 'receding'         : Moving away from the road / moving away from the bus.
  5. 'erratic'          : High directional variance, rapid reversals, or zig-zag path.

Architecture:
- Unified function signature: `classify_motion_intent()`
- Mode 1: Rule-based heuristic classifier using velocity-vector kinematics,
          path tortuosity, road-center distance delta, and bounding box growth.
- Mode 2: Lightweight 1-2 layer PyTorch GRU neural sequence model running over
          normalized 8-D trajectory features.
===============================================================================
"""

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ==============================================================================
# CLASS LABELS & ENCODINGS
# ==============================================================================

INTENT_CLASSES = [
    "approaching-road",
    "crossing",
    "stationary",
    "receding",
    "erratic"
]

INTENT_TO_IDX = {name: idx for idx, name in enumerate(INTENT_CLASSES)}
IDX_TO_INTENT = {idx: name for idx, name in enumerate(INTENT_CLASSES)}

# Feature dimension per frame:
# [norm_cx, norm_cy, norm_w, norm_h, norm_vx, norm_vy, area_growth, dist_to_center]
FEATURE_DIM = 8
DEFAULT_SEQUENCE_LENGTH = 16


# ==============================================================================
# TRAJECTORY PARSING & FEATURE EXTRACTION
# ==============================================================================

def normalize_track_input(
    track_input: Any,
    image_width: int = 640,
    image_height: int = 480,
    fps: float = 30.0
) -> List[Dict[str, float]]:
    """
    Parses various input formats into a unified list of trajectory state dicts:
    [{'x1': float, 'y1': float, 'x2': float, 'y2': float, 't': float}, ...]
    """
    if track_input is None:
        return []

    # If passed a detection dict containing 'history'
    if isinstance(track_input, dict):
        if "history" in track_input and isinstance(track_input["history"], list):
            raw_list = track_input["history"]
        else:
            # Single detection dict
            raw_list = [track_input]
    elif isinstance(track_input, (list, tuple)):
        raw_list = list(track_input)
    else:
        return []

    if not raw_list:
        return []

    dt = 1.0 / max(1.0, float(fps))
    parsed_states: List[Dict[str, float]] = []

    for idx, item in enumerate(raw_list):
        if isinstance(item, dict):
            # Check for {"bbox": [x1, y1, x2, y2], "timestamp": t}
            if "bbox" in item and len(item["bbox"]) >= 4:
                b = item["bbox"]
                t = float(item.get("timestamp", idx * dt))
                parsed_states.append({
                    "x1": float(b[0]), "y1": float(b[1]),
                    "x2": float(b[2]), "y2": float(b[3]),
                    "t": t
                })
            elif "x1" in item and "y1" in item and "x2" in item and "y2" in item:
                t = float(item.get("timestamp", idx * dt))
                parsed_states.append({
                    "x1": float(item["x1"]), "y1": float(item["y1"]),
                    "x2": float(item["x2"]), "y2": float(item["y2"]),
                    "t": t
                })
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            parsed_states.append({
                "x1": float(item[0]), "y1": float(item[1]),
                "x2": float(item[2]), "y2": float(item[3]),
                "t": float(idx * dt)
            })

    # Sort chronologically by timestamp
    parsed_states.sort(key=lambda s: s["t"])
    return parsed_states


def extract_trajectory_features(
    track_input: Any,
    image_width: int = 640,
    image_height: int = 480,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    fps: float = 30.0
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extracts normalized temporal feature matrix (seq_len, FEATURE_DIM) and
    scalar kinematic summary metrics from track history.

    Features per timestep:
      0: norm_cx (center x / image_width)
      1: norm_cy (center y / image_height)
      2: norm_w  (width / image_width)
      3: norm_h  (height / image_height)
      4: norm_vx (frame velocity x / 100.0, clipped)
      5: norm_vy (frame velocity y / 100.0, clipped)
      6: area_growth (delta area / initial area)
      7: dist_to_center (normalized distance |cx - road_center| / (W/2))

    Returns:
      feature_matrix: np.ndarray of shape (seq_len, FEATURE_DIM)
      metrics: dict containing aggregated kinematics for heuristic classification
    """
    states = normalize_track_input(track_input, image_width, image_height, fps)
    W = max(1.0, float(image_width))
    H = max(1.0, float(image_height))
    road_cx = W / 2.0

    if not states:
        empty_features = np.zeros((seq_len, FEATURE_DIM), dtype=np.float32)
        empty_metrics = {
            "num_frames": 0, "net_disp": 0.0, "path_len": 0.0, "mean_speed": 0.0,
            "delta_x": 0.0, "delta_y": 0.0, "lateral_ratio": 0.0, "area_growth_ratio": 1.0,
            "delta_road_dist": 0.0, "tortuosity": 1.0, "heading_std": 0.0,
            "reversals": 0
        }
        return empty_features, empty_metrics

    # Take at most the most recent seq_len states
    if len(states) > seq_len:
        states = states[-seq_len:]

    num_pts = len(states)
    cx_arr = np.zeros(num_pts, dtype=np.float32)
    cy_arr = np.zeros(num_pts, dtype=np.float32)
    w_arr = np.zeros(num_pts, dtype=np.float32)
    h_arr = np.zeros(num_pts, dtype=np.float32)
    area_arr = np.zeros(num_pts, dtype=np.float32)
    t_arr = np.zeros(num_pts, dtype=np.float32)

    for i, s in enumerate(states):
        w = max(1.0, s["x2"] - s["x1"])
        h = max(1.0, s["y2"] - s["y1"])
        cx_arr[i] = (s["x1"] + s["x2"]) / 2.0
        cy_arr[i] = (s["y1"] + s["y2"]) / 2.0
        w_arr[i] = w
        h_arr[i] = h
        area_arr[i] = w * h
        t_arr[i] = s["t"]

    # Compute step deltas & velocities
    vx_arr = np.zeros(num_pts, dtype=np.float32)
    vy_arr = np.zeros(num_pts, dtype=np.float32)
    step_lens = np.zeros(num_pts, dtype=np.float32)
    headings = []

    for i in range(1, num_pts):
        dt = t_arr[i] - t_arr[i - 1]
        if dt <= 0:
            dt = 1.0 / max(1.0, float(fps))
        dx = cx_arr[i] - cx_arr[i - 1]
        dy = cy_arr[i] - cy_arr[i - 1]
        vx_arr[i] = dx / dt
        vy_arr[i] = dy / dt
        step_len = math.hypot(dx, dy)
        step_lens[i] = step_len
        if step_len > 1e-2:
            headings.append(math.atan2(dy, dx))

    # Replicate first frame velocity from second frame if available
    if num_pts > 1:
        vx_arr[0] = vx_arr[1]
        vy_arr[0] = vy_arr[1]

    # Overall kinematic summary metrics
    initial_area = max(1.0, area_arr[0])
    final_area = area_arr[-1]
    area_growth_ratio = float(final_area / initial_area)

    net_disp = float(math.hypot(cx_arr[-1] - cx_arr[0], cy_arr[-1] - cy_arr[0]))
    path_len = float(np.sum(step_lens))
    total_time = max(0.001, t_arr[-1] - t_arr[0]) if num_pts > 1 else (1.0 / fps)
    mean_speed = float(path_len / total_time)

    delta_x = float(cx_arr[-1] - cx_arr[0])
    delta_y = float(cy_arr[-1] - cy_arr[0])
    lateral_ratio = float(abs(delta_x) / max(1.0, abs(delta_y)))

    d_start_road = abs(cx_arr[0] - road_cx)
    d_end_road = abs(cx_arr[-1] - road_cx)
    delta_road_dist = float(d_end_road - d_start_road)  # negative = closing in on road center

    tortuosity = float(path_len / max(1.0, net_disp))

    heading_std = float(np.std(headings)) if len(headings) >= 2 else 0.0

    # Count lateral direction reversals
    reversals = 0
    if len(vx_arr) >= 3:
        for i in range(2, len(vx_arr)):
            if (vx_arr[i] * vx_arr[i - 1] < -30.0):
                reversals += 1

    metrics = {
        "num_frames": num_pts,
        "net_disp": net_disp,
        "path_len": path_len,
        "mean_speed": mean_speed,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "lateral_ratio": lateral_ratio,
        "area_growth_ratio": area_growth_ratio,
        "delta_road_dist": delta_road_dist,
        "tortuosity": tortuosity,
        "heading_std": heading_std,
        "reversals": reversals
    }

    # Build sequence matrix (num_pts, FEATURE_DIM)
    seq_matrix = np.zeros((num_pts, FEATURE_DIM), dtype=np.float32)
    seq_matrix[:, 0] = cx_arr / W
    seq_matrix[:, 1] = cy_arr / H
    seq_matrix[:, 2] = w_arr / W
    seq_matrix[:, 3] = h_arr / H
    seq_matrix[:, 4] = np.clip(vx_arr / 100.0, -3.0, 3.0)
    seq_matrix[:, 5] = np.clip(vy_arr / 100.0, -3.0, 3.0)
    seq_matrix[:, 6] = np.clip((area_arr - initial_area) / initial_area, -2.0, 3.0)
    seq_matrix[:, 7] = np.clip(np.abs(cx_arr - road_cx) / (W / 2.0), 0.0, 1.0)

    # Pad or slice to exact seq_len
    if num_pts < seq_len:
        # Prepend copies of the first state (left padding for GRU)
        pad_len = seq_len - num_pts
        padding = np.repeat(seq_matrix[0:1, :], pad_len, axis=0)
        feature_matrix = np.vstack([padding, seq_matrix])
    else:
        feature_matrix = seq_matrix[-seq_len:]

    return feature_matrix, metrics


# ==============================================================================
# MODE 1: HEURISTIC CLASSIFIER (DEFAULT, ZERO TRAINING REQUIRED)
# ==============================================================================

def classify_heuristic(
    track_input: Any,
    image_width: int = 640,
    image_height: int = 480,
    road_center_x: Optional[float] = None,
    fps: float = 30.0
) -> str:
    """
    Classifies motion intent using deterministic velocity-vector kinematics,
    bounding-box growth, and lateral/longitudinal path analysis.

    Returns one of:
      ['approaching-road', 'crossing', 'stationary', 'receding', 'erratic']
    """
    _, m = extract_trajectory_features(
        track_input,
        image_width=image_width,
        image_height=image_height,
        fps=fps
    )

    num_frames = m["num_frames"]
    if num_frames == 0:
        return "stationary"

    # Single-frame detection has no temporal motion
    if num_frames == 1:
        return "stationary"

    net_disp = m["net_disp"]
    path_len = m["path_len"]
    mean_speed = m["mean_speed"]
    delta_x = m["delta_x"]
    delta_y = m["delta_y"]
    lateral_ratio = m["lateral_ratio"]
    area_growth = m["area_growth_ratio"]
    delta_road = m["delta_road_dist"]
    tortuosity = m["tortuosity"]
    heading_std = m["heading_std"]
    reversals = m["reversals"]

    # 1. STATIONARY CHECK
    # Low net displacement and low mean velocity over observed frames
    if (net_disp < 14.0 and mean_speed < 24.0 and (0.84 <= area_growth <= 1.18)):
        return "stationary"
    if path_len < 16.0 and abs(delta_x) < 10.0 and abs(delta_y) < 10.0:
        return "stationary"

    # 2. ERRATIC CHECK
    # Significant path tortuosity, excessive heading variance, or frequent reversals
    if (tortuosity > 1.85 and path_len > 28.0) or \
       (heading_std > 1.15 and mean_speed > 22.0) or \
       (reversals >= 2 and mean_speed > 20.0):
        return "erratic"

    # 3. CROSSING CHECK
    # Dominant lateral movement across the camera plane
    if abs(delta_x) > 22.0 and lateral_ratio > 1.35 and abs(delta_x) > abs(delta_y):
        return "crossing"

    # 4. APPROACHING ROAD CHECK
    # Either getting laterally closer to road center or longitudinally approaching the bus
    if (delta_road < -10.0) or \
       (delta_y > 18.0 and area_growth > 1.10) or \
       (delta_road < -4.0 and delta_y > 8.0):
        return "approaching-road"

    # 5. RECEDING CHECK
    # Moving outward towards shoulder/sidewalk or moving away from bus (shrinking)
    if (delta_road > 12.0) or \
       (delta_y < -16.0 and area_growth < 0.90) or \
       (delta_road > 5.0 and delta_y < -6.0):
        return "receding"

    # Secondary arbitrations for subtle trajectories
    if delta_road < -3.0 or (area_growth > 1.05 and delta_y > 4.0):
        return "approaching-road"
    elif delta_road > 3.0 or (area_growth < 0.95 and delta_y < -4.0):
        return "receding"
    elif abs(delta_x) > 16.0:
        return "crossing"
    elif mean_speed < 30.0:
        return "stationary"

    return "stationary"


# ==============================================================================
# MODE 2: NEURAL SEQUENCE MODEL (GRU)
# ==============================================================================

class MotionIntentGRU(nn.Module):
    """
    Compact 1-2 layer Gated Recurrent Unit (GRU) for temporal motion intent.
    Input: (batch_size, seq_len, 8)
    Output: (batch_size, 5) logits for INTENT_CLASSES
    """
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_dim: int = 32,
        num_layers: int = 1,
        num_classes: int = len(INTENT_CLASSES),
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Linear(24, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.gru(x)
        # Use last timestep representation
        last_hidden = out[:, -1, :]
        logits = self.classifier(last_hidden)
        return logits


# Global cached GRU model instance
_CACHED_GRU_MODEL: Optional[MotionIntentGRU] = None
_CACHED_MODEL_PATH: Optional[str] = None


def load_gru_model(model_path: str = "models/motion_intent_gru.pt") -> Optional[MotionIntentGRU]:
    """Loads and caches the PyTorch GRU model weights."""
    global _CACHED_GRU_MODEL, _CACHED_MODEL_PATH

    resolved_path = Path(model_path)
    if not resolved_path.is_file():
        return None

    if _CACHED_GRU_MODEL is not None and _CACHED_MODEL_PATH == str(resolved_path):
        return _CACHED_GRU_MODEL

    try:
        model = MotionIntentGRU()
        state_dict = torch.load(resolved_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        _CACHED_GRU_MODEL = model
        _CACHED_MODEL_PATH = str(resolved_path)
        return _CACHED_GRU_MODEL
    except Exception as e:
        print(f"[WARN] Failed to load GRU model from {model_path}: {e}")
        return None


def classify_gru(
    track_input: Any,
    image_width: int = 640,
    image_height: int = 480,
    model_path: str = "models/motion_intent_gru.pt",
    fps: float = 30.0
) -> str:
    """
    Classifies motion intent using the trained PyTorch GRU model.
    Falls back gracefully to classify_heuristic if model is unavailable.
    """
    model = load_gru_model(model_path)
    if model is None:
        # Graceful fallback to heuristic
        return classify_heuristic(track_input, image_width, image_height, fps=fps)

    feat_matrix, _ = extract_trajectory_features(
        track_input,
        image_width=image_width,
        image_height=image_height,
        seq_len=DEFAULT_SEQUENCE_LENGTH,
        fps=fps
    )

    tensor_in = torch.tensor(feat_matrix, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 8)
    with torch.no_grad():
        logits = model(tensor_in)
        pred_idx = int(torch.argmax(logits, dim=1).item())

    if 0 <= pred_idx < len(INTENT_CLASSES):
        return IDX_TO_INTENT[pred_idx]
    return "stationary"


# ==============================================================================
# UNIFIED ENTRYPOINT FUNCTION
# ==============================================================================

def classify_motion_intent(
    track_history: Any,
    image_width: int = 640,
    image_height: int = 480,
    mode: str = "heuristic",
    model_path: Optional[str] = "models/motion_intent_gru.pt",
    road_center_x: Optional[float] = None,
    fps: float = 30.0
) -> str:
    """
    Unified public API for motion intent classification.

    Args:
        track_history: Trajectory history list of bounding boxes / dicts, or detection dict.
        image_width: Frame pixel width (default: 640).
        image_height: Frame pixel height (default: 480).
        mode: 'heuristic' (default, rule-based) or 'gru' (trained neural sequence model).
        model_path: Path to trained PyTorch weights for GRU mode.
        road_center_x: Lateral pixel coordinate of road center (default: image_width / 2).
        fps: Video frame rate (default: 30.0).

    Returns:
        One of: 'approaching-road', 'crossing', 'stationary', 'receding', 'erratic'
    """
    selected_mode = (mode or "heuristic").lower().strip()

    if selected_mode == "gru":
        target_path = model_path or "models/motion_intent_gru.pt"
        return classify_gru(
            track_input=track_history,
            image_width=image_width,
            image_height=image_height,
            model_path=target_path,
            fps=fps
        )
    else:
        # Default heuristic mode
        return classify_heuristic(
            track_input=track_history,
            image_width=image_width,
            image_height=image_height,
            road_center_x=road_center_x,
            fps=fps
        )
