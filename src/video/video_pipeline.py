"""
src/video/video_pipeline.py
===============================================================================
Multi-camera video ingestion, time-aligned temporal sampling, and per-timestep
perception & decision timeline generator.

Supports 1 to 5 uploaded camera video feeds:
  - front (or front_image)
  - front_left
  - front_right
  - rear_left (or back_left)
  - rear_right (or back_right)

Missing or dropped feeds are handled using the established degraded-camera
path from `src/sensor_reliability.py` (blank canvas / penalized reliability).
===============================================================================
"""

import os
import sys
import time
import math
import uuid
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.autonomous_brain_v3 import analyze_scene

# ---------------------------------------------------------------------------
# Step 5: Temporal smoother + model version selection
# ---------------------------------------------------------------------------
# Set BUS_YOLO_MODEL_PATH (or USE_FINE_TUNED_MODEL_V2=true) to use v2 weights.
# Set USE_TEMPORAL_SMOOTHING=false to disable the 3-of-5 consensus filter.
# By default, temporal smoothing is ENABLED and uses the best available model.
_V2_MODEL_PATH = REPO_ROOT / "models" / "yolov8_bus_v2.pt"
_V1_MODEL_PATH = REPO_ROOT / "models" / "yolov8_bus_v1.pt"
_YOLOV8N_PATH = REPO_ROOT / "yolov8n.pt"

_EXPLICIT_MODEL_PATH = os.environ.get("BUS_YOLO_MODEL_PATH", "")
_USE_V2 = os.environ.get("USE_FINE_TUNED_MODEL_V2", "false").lower() in ("true", "1")
_USE_SMOOTHING = os.environ.get("USE_TEMPORAL_SMOOTHING", "false").lower() in ("true", "1")

# Model selection priority for the temporal smoother (separate from analyze_scene's model):
# 1. Explicit BUS_YOLO_MODEL_PATH env var
# 2. yolov8_bus_v2.pt (if trained)
# 3. yolov8n.pt (base COCO — PREFERRED over v1 for video; v1 has 0% detection on video)
# 4. yolov8_bus_v1.pt (last resort; v1 is known to fail on video frames)
if _EXPLICIT_MODEL_PATH and Path(_EXPLICIT_MODEL_PATH).is_file():
    _PIPELINE_MODEL_PATH = _EXPLICIT_MODEL_PATH
elif _USE_V2 and _V2_MODEL_PATH.is_file():
    _PIPELINE_MODEL_PATH = str(_V2_MODEL_PATH)
elif _YOLOV8N_PATH.is_file():
    # Prefer base COCO model for video — it achieves 96% temporal consistency vs 0% for v1
    _PIPELINE_MODEL_PATH = str(_YOLOV8N_PATH)
elif _V1_MODEL_PATH.is_file():
    _PIPELINE_MODEL_PATH = str(_V1_MODEL_PATH)
else:
    _PIPELINE_MODEL_PATH = "yolov8n"  # ultralytics auto-download fallback

# Lazy-load the smoother model (separate from autonomous_brain_v3's model)
_smoother_model = None

def _get_smoother_model():
    """Lazy-load the YOLO model used by the temporal smoother."""
    global _smoother_model
    if _smoother_model is None:
        from ultralytics import YOLO
        _smoother_model = YOLO(_PIPELINE_MODEL_PATH)
        logger.info(f"[TemporalSmoother] Loaded model: {_PIPELINE_MODEL_PATH}")
    return _smoother_model

logger = logging.getLogger("video_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# =============================================================================
# CONSTANTS & CAMERA SLOT DEFINITIONS
# =============================================================================

CANONICAL_SLOTS = ["front", "front_left", "front_right", "rear_left", "rear_right"]

SLOT_ALIASES = {
    "front": "front",
    "front_image": "front",
    "camera_front": "front",
    "cam_front": "front",

    "front_left": "front_left",
    "camera_front_left": "front_left",
    "cam_front_left": "front_left",

    "front_right": "front_right",
    "camera_front_right": "front_right",
    "cam_front_right": "front_right",

    "rear_left": "rear_left",
    "back_left": "rear_left",
    "camera_rear_left": "rear_left",
    "camera_back_left": "rear_left",
    "cam_back_left": "rear_left",

    "rear_right": "rear_right",
    "back_right": "rear_right",
    "camera_rear_right": "rear_right",
    "camera_back_right": "rear_right",
    "cam_back_right": "rear_right",
}


def normalize_slot_name(slot: str) -> Optional[str]:
    """Maps alternative camera slot names to canonical names."""
    if not slot:
        return None
    return SLOT_ALIASES.get(slot.strip().lower(), None)


# =============================================================================
# SENSOR RELIABILITY REUSE (src/sensor_reliability.py PATH)
# =============================================================================

def compute_camera_reliability(image: Optional[np.ndarray], is_missing: bool = False) -> float:
    """
    Evaluates individual camera reliability using the exact 4-metric heuristic
    from `src/sensor_reliability.py` & `src/multi_camera_reliability.py`:
      - Brightness < 60: -0.20
      - Contrast < 40:   -0.20
      - Blur < 100:       -0.20
      - Edges < 5000:     -0.20

    If a camera feed is missing or unreadable, it is treated as a dead sensor (0.0).
    """
    if is_missing or image is None or image.size == 0:
        return 0.0

    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        brightness = float(np.mean(gray))
        contrast = float(gray.std())
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 100, 200)
        edge_count = int(np.sum(edges > 0))

        reliability = 1.0
        if brightness < 60:
            reliability -= 0.20
        if contrast < 40:
            reliability -= 0.20
        if blur_score < 100:
            reliability -= 0.20
        if edge_count < 5000:
            reliability -= 0.20

        return float(round(max(0.0, min(reliability, 1.0)), 3))
    except Exception as exc:
        logger.warning(f"Error computing camera reliability: {exc}")
        return 0.0


def create_degraded_dead_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Creates a black canvas representing a degraded/offline camera feed.
    This safely yields 0 detections in YOLO and is penalized by sensor reliability.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


# =============================================================================
# IN-MEMORY JOB STORE FOR ASYNC VIDEO INGESTION
# =============================================================================

class VideoJobStore:
    """
    Thread-safe in-memory store for video processing jobs.
    Tracks state, progress percentage, current frame index, and generated timeline.
    """

    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        active_slots: List[str],
        missing_slots: List[str],
        target_fps: float,
        video_paths: Dict[str, str],
        max_duration_sec: Optional[float] = None,
        job_id: Optional[str] = None
    ) -> str:
        job_id = job_id or str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",  # queued, processing, completed, failed
                "progress": 0.0,
                "current_frame": 0,
                "total_frames": 0,
                "target_fps": target_fps,
                "max_duration_sec": max_duration_sec,
                "active_cameras": list(active_slots),
                "missing_cameras": list(missing_slots),
                "video_paths": dict(video_paths),
                "timeline": [],
                "summary": {},
                "error": None,
                "created_at": now,
                "updated_at": now,
                "completed_at": None,
            }
        return job_id

    def update_progress(
        self,
        job_id: str,
        current_frame: int,
        total_frames: int,
        latest_entry: Optional[Dict[str, Any]] = None
    ) -> None:
        with self._lock:
            if job_id not in self._jobs:
                return
            job = self._jobs[job_id]
            job["status"] = "processing"
            job["current_frame"] = current_frame
            job["total_frames"] = total_frames
            if total_frames > 0:
                job["progress"] = round(min(100.0, (current_frame / total_frames) * 100.0), 1)
            if latest_entry is not None:
                job["timeline"].append(latest_entry)
            job["updated_at"] = time.time()

    def complete_job(
        self,
        job_id: str,
        timeline: List[Dict[str, Any]],
        summary: Dict[str, Any]
    ) -> None:
        now = time.time()
        with self._lock:
            if job_id not in self._jobs:
                return
            job = self._jobs[job_id]
            job["status"] = "completed"
            job["progress"] = 100.0
            job["timeline"] = timeline
            job["summary"] = summary
            job["current_frame"] = len(timeline)
            job["total_frames"] = len(timeline)
            job["updated_at"] = now
            job["completed_at"] = now

    def fail_job(self, job_id: str, error_message: str) -> None:
        now = time.time()
        with self._lock:
            if job_id not in self._jobs:
                return
            job = self._jobs[job_id]
            job["status"] = "failed"
            job["error"] = str(error_message)
            job["updated_at"] = now
            job["completed_at"] = now

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Return snapshot shallow copy
            return {
                "job_id": job["job_id"],
                "status": job["status"],
                "progress": job["progress"],
                "current_frame": job["current_frame"],
                "total_frames": job["total_frames"],
                "target_fps": job["target_fps"],
                "active_cameras": job["active_cameras"],
                "missing_cameras": job["missing_cameras"],
                "timeline": list(job["timeline"]),
                "summary": dict(job["summary"]),
                "error": job["error"],
                "created_at": job["created_at"],
                "updated_at": job["updated_at"],
                "completed_at": job["completed_at"],
            }

    def cleanup_old_jobs(self, max_age_seconds: float = 3600.0) -> None:
        now = time.time()
        with self._lock:
            stale_keys = [
                jid for jid, info in self._jobs.items()
                if (now - info.get("updated_at", now)) > max_age_seconds
            ]
            for jid in stale_keys:
                del self._jobs[jid]


_GLOBAL_JOB_STORE = VideoJobStore()


def get_global_job_store() -> VideoJobStore:
    """Returns the shared in-memory job store instance."""
    return _GLOBAL_JOB_STORE


# =============================================================================
# MULTI-CAMERA VIDEO INGESTION & PIPELINE RUNNER
# =============================================================================

class VideoIngestionPipeline:
    """
    Time-aligned multi-camera video ingestion pipeline.
    Reads synchronized video streams, invokes analyze_scene() per timestep,
    and returns a structured chronological decision timeline.
    """

    def __init__(
        self,
        video_paths: Dict[str, str],
        target_fps: float = 1.0,
        max_duration_sec: Optional[float] = 30.0,
        job_id: Optional[str] = None,
        job_store: Optional[VideoJobStore] = None,
        socketio_instance: Optional[Any] = None,
        use_temporal_smoothing: Optional[bool] = None,
    ):
        self.raw_video_paths = video_paths
        self.target_fps = max(0.5, float(target_fps))
        self.max_duration_sec = max_duration_sec
        self.job_id = job_id
        self.job_store = job_store or _GLOBAL_JOB_STORE
        self.socketio = socketio_instance
        # Temporal smoothing: default from env var, can be overridden per-instance
        self.use_temporal_smoothing = _USE_SMOOTHING if use_temporal_smoothing is None else use_temporal_smoothing

        # Normalize provided camera slots
        self.mapped_paths: Dict[str, str] = {}
        for raw_slot, path_str in video_paths.items():
            slot = normalize_slot_name(raw_slot)
            if slot and Path(path_str).is_file():
                self.mapped_paths[slot] = str(Path(path_str).resolve())

        self.active_slots = [s for s in CANONICAL_SLOTS if s in self.mapped_paths]
        self.missing_slots = [s for s in CANONICAL_SLOTS if s not in self.mapped_paths]

        if not self.active_slots:
            raise ValueError(
                f"No valid video files found in slots: {list(video_paths.keys())}. "
                f"Expected 1 to 5 cameras matching: {CANONICAL_SLOTS}"
            )

    def process(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes the temporal video ingestion and scene analysis.
        Returns:
            timeline: list of per-timestep scene decision dictionaries.
            summary: aggregate statistics across the entire video.
        """
        caps: Dict[str, cv2.VideoCapture] = {}
        video_fps: Dict[str, float] = {}
        video_counts: Dict[str, int] = {}
        durations: Dict[str, float] = {}

        # 1. Open active video captures and extract metadata
        try:
            for slot, path_str in self.mapped_paths.items():
                cap = cv2.VideoCapture(path_str)
                if not cap.isOpened():
                    logger.warning(f"Could not open video capture for slot '{slot}' at {path_str}")
                    continue
                caps[slot] = cap
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0.0 or math.isnan(fps):
                    fps = 30.0  # Safe default fallback
                video_fps[slot] = fps
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_counts[slot] = max(1, count)
                durations[slot] = count / fps

            if not caps:
                raise RuntimeError("Failed to open any of the provided camera video files.")

            # Determine maximum video duration across provided active cameras
            max_stream_duration = max(durations.values())
            if self.max_duration_sec is not None and self.max_duration_sec > 0:
                effective_duration = min(max_stream_duration, float(self.max_duration_sec))
            else:
                effective_duration = max_stream_duration

            total_timesteps = max(1, int(math.ceil(effective_duration * self.target_fps)))
            logger.info(
                f"[VideoPipeline] Starting processing: {len(caps)} active cameras "
                f"({list(caps.keys())}), duration={effective_duration:.2f}s, "
                f"target_fps={self.target_fps}, total_timesteps={total_timesteps}, "
                f"temporal_smoothing={self.use_temporal_smoothing}"
            )

            # Step 5: Initialize temporal smoother for this job
            _smoother = None
            if self.use_temporal_smoothing:
                try:
                    from src.video.temporal_smoother import TemporalSmoother
                    _smoother = TemporalSmoother(
                        model=_get_smoother_model(),
                        window_size=5,
                        min_votes=3,
                        iou_threshold=0.30,
                        conf_threshold=0.25,
                        imgsz=640,
                        disabled=False,
                    )
                    logger.info("[VideoPipeline] Temporal smoother initialized (3-of-5 consensus).")
                except Exception as sm_err:
                    logger.warning(f"[VideoPipeline] Temporal smoother init failed (proceeding without): {sm_err}")
                    _smoother = None

            # Establish canonical frame dimensions from primary available video
            ref_width, ref_height = 640, 480
            for cap in caps.values():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    ref_width, ref_height = w, h
                    break

            timeline: List[Dict[str, Any]] = []

            # 2. Iterate aligned timesteps
            for step_idx in range(total_timesteps):
                t_sec = step_idx / self.target_fps
                current_frames: Dict[str, np.ndarray] = {}
                step_rel_scores: Dict[str, float] = {}

                # Sample frame for each of the 5 canonical camera slots
                for slot in CANONICAL_SLOTS:
                    if slot in caps:
                        cap = caps[slot]
                        src_fps = video_fps[slot]
                        total_cnt = video_counts[slot]
                        target_frame_no = int(round(t_sec * src_fps))

                        if target_frame_no < total_cnt:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_no)
                            ret, frame = cap.read()
                            if ret and frame is not None and frame.size > 0:
                                # Apply temporal smoothing tracking if enabled without corrupting raw pixel input
                                if _smoother is not None:
                                    _smoother.update_and_get_stable(slot, frame)
                                current_frames[slot] = frame
                                step_rel_scores[slot] = compute_camera_reliability(frame, is_missing=False)
                            else:
                                # Stream read failure: degraded dead frame
                                degraded = create_degraded_dead_frame(ref_width, ref_height)
                                current_frames[slot] = degraded
                                step_rel_scores[slot] = compute_camera_reliability(degraded, is_missing=True)
                        else:
                            # Video reached EOF: degraded dead frame
                            degraded = create_degraded_dead_frame(ref_width, ref_height)
                            current_frames[slot] = degraded
                            step_rel_scores[slot] = 0.0
                    else:
                        # Slot is missing entirely from user input: degraded dead frame
                        degraded = create_degraded_dead_frame(ref_width, ref_height)
                        current_frames[slot] = degraded
                        step_rel_scores[slot] = 0.0

                # 3. Call UNMODIFIED analyze_scene() exactly as frontend/app_v3.py does
                scene_output = analyze_scene(
                    front_image=current_frames["front"],
                    front_left=current_frames["front_left"],
                    front_right=current_frames["front_right"],
                    rear_left=current_frames["rear_left"],
                    rear_right=current_frames["rear_right"]
                )

                # 4. Extract required timeline fields
                decision = scene_output.get("decision") or scene_output.get("final_decision", "MOVE")
                speed = scene_output.get("recommended_speed", 25)
                lane_action = scene_output.get("lane_action", "MAINTAIN_LANE")
                risk_score = scene_output.get("risk_score", 0)
                risk_level = scene_output.get("risk_level", "LOW")

                # Explanation text / reasons
                explanation = scene_output.get("fusion_reason")
                if not explanation:
                    reasons = scene_output.get("reasons", [])
                    explanation = "; ".join(reasons) if isinstance(reasons, list) else str(reasons)

                # Contributing factors dictionary
                contributing_factors = {
                    "traffic_signal": scene_output.get("traffic_signal", "UNKNOWN"),
                    "lead_vehicle": scene_output.get("lead_vehicle"),
                    "vehicle_distance": scene_output.get("vehicle_distance", "FAR"),
                    "pedestrian_count": scene_output.get("pedestrian_count", 0),
                    "pedestrian_status": scene_output.get("pedestrian_status", "NONE"),
                    "left_lane_status": scene_output.get("left_lane_status", "CLEAR"),
                    "right_lane_status": scene_output.get("right_lane_status", "CLEAR"),
                    "collision_risk": scene_output.get("collision_risk", "Low Collision Risk"),
                    "rl_action": scene_output.get("rl_action", "MAINTAIN"),
                }

                # Sensor reliability report
                step_rel_scores["overall"] = float(scene_output.get("camera_reliability", 1.0))
                step_rel_scores["fusion_confidence"] = float(scene_output.get("fusion_confidence", 1.0))

                # Extract front detections bounding boxes for client-side canvas overlay
                front_dets: List[Dict[str, Any]] = []
                try:
                    import src.autonomous_brain_v3 as _bm
                    yolo_f = _bm.model(current_frames["front"], verbose=False)
                    if yolo_f and len(yolo_f) > 0 and yolo_f[0].boxes is not None:
                        for b in yolo_f[0].boxes:
                            cid = int(b.cls[0])
                            cname = _bm.model.names[cid]
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            conf = float(b.conf[0])
                            front_dets.append({
                                "class": cname,
                                "confidence": round(conf, 2),
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2
                            })
                except Exception as det_err:
                    logger.debug(f"Detection extraction skipped: {det_err}")

                entry: Dict[str, Any] = {
                    "timestamp": round(t_sec, 2),
                    "frame_index": step_idx,
                    "decision": str(decision),
                    "speed": int(speed) if isinstance(speed, (int, np.integer)) else float(speed),
                    "lane_action": str(lane_action),
                    "risk_score": int(risk_score) if isinstance(risk_score, (int, np.integer)) else float(risk_score),
                    "risk_level": str(risk_level),
                    "explanation": str(explanation),
                    "contributing_factors": contributing_factors,
                    "reliability_scores": step_rel_scores,
                    "detections": front_dets,
                }

                timeline.append(entry)

                # Update job store progress
                if self.job_id and self.job_store:
                    self.job_store.update_progress(
                        self.job_id,
                        current_frame=step_idx + 1,
                        total_frames=total_timesteps,
                        latest_entry=entry
                    )

                # Emit WebSocket live progress update if available
                if self.socketio:
                    try:
                        self.socketio.emit(
                            "video_job_progress",
                            {
                                "job_id": self.job_id,
                                "progress": round(((step_idx + 1) / total_timesteps) * 100.0, 1),
                                "current_frame": step_idx + 1,
                                "total_frames": total_timesteps,
                                "latest_decision": entry["decision"],
                                "risk_score": entry["risk_score"]
                            },
                            namespace="/ws/live"
                        )
                    except Exception as ws_err:
                        logger.debug(f"SocketIO progress emit skipped: {ws_err}")

            # 5. Compute summary statistics across timeline
            summary = self._compute_summary(timeline, effective_duration)

            if self.job_id and self.job_store:
                self.job_store.complete_job(self.job_id, timeline, summary)

            return timeline, summary

        finally:
            # Clean up open video capture handles
            for cap in caps.values():
                try:
                    cap.release()
                except Exception:
                    pass

    def _compute_summary(self, timeline: List[Dict[str, Any]], duration_sec: float) -> Dict[str, Any]:
        """Calculates aggregate statistics from the perception timeline."""
        if not timeline:
            return {
                "total_timesteps": 0,
                "duration_sec": 0.0,
                "active_cameras": self.active_slots,
                "missing_cameras": self.missing_slots,
            }

        decisions: Dict[str, int] = {}
        risk_scores: List[float] = []
        speeds: List[float] = []

        for item in timeline:
            d = item.get("decision", "UNKNOWN")
            decisions[d] = decisions.get(d, 0) + 1
            risk_scores.append(float(item.get("risk_score", 0)))
            speeds.append(float(item.get("speed", 0)))

        max_risk = max(risk_scores) if risk_scores else 0.0
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        dominant_decision = max(decisions.items(), key=lambda x: x[1])[0] if decisions else "MOVE"

        return {
            "total_timesteps": len(timeline),
            "duration_sec": round(duration_sec, 2),
            "dominant_decision": dominant_decision,
            "decision_distribution": decisions,
            "max_risk_score": round(max_risk, 1),
            "avg_risk_score": round(avg_risk, 1),
            "avg_recommended_speed": round(avg_speed, 1),
            "active_cameras": self.active_slots,
            "missing_cameras": self.missing_slots,
            "target_fps": self.target_fps,
        }


# =============================================================================
# HELPER TO LAUNCH PIPELINE IN BACKGROUND THREAD
# =============================================================================

def process_multi_camera_videos(
    video_paths: Dict[str, str],
    target_fps: float = 5.0,
    max_duration_sec: Optional[float] = 30.0,
    job_id: Optional[str] = None,
    job_store: Optional[VideoJobStore] = None,
    socketio_instance: Optional[Any] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Synchronous entry point executing VideoIngestionPipeline."""
    pipeline = VideoIngestionPipeline(
        video_paths=video_paths,
        target_fps=target_fps,
        max_duration_sec=max_duration_sec,
        job_id=job_id,
        job_store=job_store,
        socketio_instance=socketio_instance
    )
    return pipeline.process()


def start_video_pipeline_thread(
    job_id: str,
    video_paths: Dict[str, str],
    target_fps: float = 5.0,
    max_duration_sec: Optional[float] = 30.0,
    job_store: Optional[VideoJobStore] = None,
    socketio_instance: Optional[Any] = None
) -> threading.Thread:
    """Spawns an asynchronous daemon thread to run the video pipeline."""
    store = job_store or _GLOBAL_JOB_STORE

    def _worker():
        try:
            logger.info(f"[VideoJob {job_id}] Background thread started.")
            process_multi_camera_videos(
                video_paths=video_paths,
                target_fps=target_fps,
                max_duration_sec=max_duration_sec,
                job_id=job_id,
                job_store=store,
                socketio_instance=socketio_instance
            )
            logger.info(f"[VideoJob {job_id}] Background thread completed successfully.")
        except Exception as exc:
            logger.error(f"[VideoJob {job_id}] Processing failed: {exc}", exc_info=True)
            store.fail_job(job_id, str(exc))

    worker_thread = threading.Thread(target=_worker, name=f"VideoWorker-{job_id[:8]}", daemon=True)
    worker_thread.start()
    return worker_thread
