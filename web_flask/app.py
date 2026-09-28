"""
web_flask/app.py
===============================================================================
Autonomous Bus Perception System - Flask + Flask-SocketIO API Server
===============================================================================
Exposes:
1. REST API:
   - `POST /api/analyze`: Accepts video frame or image file, executes the
     unmodified `src.autonomous_brain_v3.analyze_scene()` pipeline, and returns
     the standard perception JSON schema. Optionally enriches output with
     Phases 4-7 signals if requested via query parameters:
       - `?enable_tracking=true`     (Phase 4: ByteTrack multi-object tracking)
       - `?enable_motion_intent=true` (Phase 5: Temporal motion intent classification)
       - `?enable_bus_stops=true`     (Phase 6: Multi-signal bus stop & docking)
       - `?enable_door_monitor=true`  (Phase 7: Door zone safety monitor)
   - `GET /api/health`: System health check and model status.

2. WebSocket Stream:
   - `/ws/live`: Real-time streaming namespace for live decision and perception
     telemetry pushing.

Standalone: Does NOT modify or replace Streamlit dashboard (`frontend/app_v3.py`).
===============================================================================
"""

import base64
import io
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import existing unmodified autonomous brain
import src.autonomous_brain_v3 as brain_module
from src.autonomous_brain_v3 import analyze_scene, model

# Import Phase 4-7 optional modules
from src.tracking.byte_tracker_wrapper import ByteTrackerWrapper, get_default_tracker
from src.temporal.motion_intent import classify_motion_intent
from src.stops.bus_stop_detector import BusStopDetector, get_default_bus_stop_detector
from src.stops.docking_state_machine import DockingStateMachine, DockingState
from src.doors.door_zone_monitor import DoorZoneMonitor, DoorZoneState, get_default_door_monitor

# Import Step 2 Video Pipeline modules
from src.video.video_pipeline import (
    VideoIngestionPipeline,
    VideoJobStore,
    get_global_job_store,
    start_video_pipeline_thread,
    normalize_slot_name,
    CANONICAL_SLOTS,
)


# ==============================================================================
# FLASK & SOCKETIO INITIALIZATION
# ==============================================================================

def create_app() -> Tuple[Flask, SocketIO]:
    """Factory creating and configuring the Flask + SocketIO app."""
    app = Flask(__name__, static_folder="static", static_url_path="")
    app.config["SECRET_KEY"] = "bus_perception_secure_token_v3"
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    register_routes(app, socketio)
    return app, socketio


# ==============================================================================
# IMAGE DECODING & ENCODING HELPERS
# ==============================================================================

def decode_image_input(raw_input: Any) -> Optional[np.ndarray]:
    """Decodes file storage, bytes, or base64 string into BGR numpy array."""
    if raw_input is None:
        return None

    try:
        # FileStorage object from request.files
        if hasattr(raw_input, "read"):
            data = raw_input.read()
            if hasattr(raw_input, "seek"):
                raw_input.seek(0)
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            # Video file fallback
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                cap = cv2.VideoCapture(tmp_path)
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    return frame
            finally:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        # Raw bytes
        if isinstance(raw_input, bytes):
            nparr = np.frombuffer(raw_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Base64 string
        if isinstance(raw_input, str):
            if "," in raw_input:
                raw_input = raw_input.split(",", 1)[1]
            data = base64.b64decode(raw_input)
            nparr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Already a numpy array
        if isinstance(raw_input, np.ndarray):
            return raw_input
    except Exception as e:
        print(f"[WARN] Failed to decode image: {e}")
        return None

    return None


def load_default_camera_feed(name: str) -> np.ndarray:
    """Loads fallback sample image from data/ folder or generates neutral canvas."""
    target_path = REPO_ROOT / "data" / f"{name}.jpg"
    if target_path.is_file():
        img = cv2.imread(str(target_path))
        if img is not None:
            return img
    # Neutral canvas fallback
    return np.full((480, 640, 3), 70, dtype=np.uint8)


def serialize_for_json(obj: Any, include_base64_images: bool = False) -> Any:
    """Recursively serializes perception dict converting numpy objects for JSON."""
    if isinstance(obj, np.ndarray):
        if include_base64_images:
            success, buffer = cv2.imencode(".jpg", obj, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
        return {
            "_type": "image_ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype)
        }
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v, include_base64_images) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item, include_base64_images) for item in obj]
    elif hasattr(obj, "value"):  # Enums
        return obj.value
    return obj


# Global state tracking instances
_SHARED_DOCKING_SM = DockingStateMachine()


# ==============================================================================
# ROUTE DEFINITIONS & WEBSOCKET HANDLERS
# ==============================================================================

def register_routes(app: Flask, socketio: SocketIO) -> None:
    @app.route("/", methods=["GET"])
    def landing_page():
        """Serves the single-screen autonomous bus landing page."""
        return app.send_static_file("index.html")

    @app.route("/static/<path:filename>", methods=["GET"])
    def serve_static_fallback(filename: str):
        """Fallback for explicit /static/ URLs."""
        return app.send_static_file(filename)

    @app.route("/Create_a_photorealistic_premiu.mp4", methods=["GET"])
    def serve_hero_video_root():
        """Serves the hero video when requested from root."""
        return app.send_static_file("videos/bus_hero.mp4")

    @app.route("/demo", methods=["GET"])
    @app.route("/pages/live_demo", methods=["GET"])
    @app.route("/live_demo", methods=["GET"])
    def demo_page():
        """Serves the interactive multi-camera perception and decision demo page."""
        return app.send_static_file("demo.html")

    @app.route("/api/sample-videos", methods=["GET"])
    def sample_videos():
        """Returns available sample video clips in the repository for 1-click video demo."""
        return jsonify({
            "urban_traffic": "/datasets/self-collected/eval_clips/clip_03_urban_traffic.mp4",
            "bus_stop_approach": "/datasets/self-collected/eval_clips/clip_01_bus_stop_approach.mp4",
            "door_boarding": "/datasets/self-collected/eval_clips/clip_02_door_boarding.mp4",
            "synthetic_clip": "/tests/synthetic_test_clip.mp4",
        }), 200

    @app.route("/datasets/<path:filename>", methods=["GET"])
    def serve_dataset_file(filename: str):
        """Serves sample video/image files directly from datasets/ directory."""
        from flask import send_from_directory
        return send_from_directory(str(REPO_ROOT / "datasets"), filename)

    @app.route("/tests/<path:filename>", methods=["GET"])
    def serve_test_file(filename: str):
        """Serves test video files directly from tests/ directory."""
        from flask import send_from_directory
        return send_from_directory(str(REPO_ROOT / "tests"), filename)

    @app.route("/api/sample-cameras", methods=["GET"])
    def sample_cameras():
        """Returns the list of available sample camera files in data/ for presets."""
        return jsonify({
            "front": "/data/front.jpg",
            "front_left": "/data/front_left.jpg",
            "front_right": "/data/front_right.jpg",
            "rear_left": "/data/rear_left.jpg",
            "rear_right": "/data/rear_right.jpg"
        }), 200

    @app.route("/api/sample-images", methods=["GET"])
    def sample_images():
        """Returns sample multi-camera image presets for instant perception demo."""
        return jsonify({
            "red_light_crossing": {
                "name": "Red Light & Crossing Pedestrians",
                "front": "/data/red_light_sample.jpg"
            },
            "urban_street": {
                "name": "Urban Multi-Vehicle Scene",
                "front": "/data/front.jpg",
                "front_left": "/data/front_left.jpg",
                "front_right": "/data/front_right.jpg"
            },
            "full_surround": {
                "name": "360° 5-Camera Surround Coverage",
                "front": "/data/front.jpg",
                "front_left": "/data/front_left.jpg",
                "front_right": "/data/front_right.jpg",
                "rear_left": "/data/rear_left.jpg",
                "rear_right": "/data/rear_right.jpg"
            }
        }), 200

    @app.route("/data/<path:filename>", methods=["GET"])
    def serve_data_file(filename: str):
        """Serves sample image files directly from the data/ directory."""
        from flask import send_from_directory
        return send_from_directory(str(REPO_ROOT / "data"), filename)

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        fine_tuned_avail = (REPO_ROOT / "models" / "yolov8_bus_v1.pt").is_file()
        current_model_name = str(getattr(brain_module.model, "model_name", "yolov8n"))
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "models": {
                "yolo_detector": "active",
                "current_model": current_model_name,
                "fine_tuned_available": fine_tuned_avail,
                "stock_available": (REPO_ROOT / "yolov8n.pt").is_file()
            },
            "phases_available": {
                "phase_4_tracking": True,
                "phase_5_motion_intent": True,
                "phase_6_bus_stops": True,
                "phase_7_door_monitor": True
            }
        }), 200

    # =========================================================================
    # STEP 2 VIDEO INGESTION ENDPOINTS
    # =========================================================================

    @app.route("/api/demo/upload", methods=["POST"])
    def demo_upload_videos():
        """
        Accepts 1 to 5 uploaded video files mapped to camera slots:
          - front (or front_image)
          - front_left
          - front_right
          - rear_left (or back_left)
          - rear_right (or back_right)
        Samples frames at a configurable rate (default 5 fps) time-aligned.
        Missing feeds are handled using sensor_reliability degraded dead-frame path.
        Spawns background processing job and returns job_id for polling.
        """
        job_store = get_global_job_store()
        job_id = str(uuid.uuid4())
        upload_dir = REPO_ROOT / "data" / "temp_videos" / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        video_paths: Dict[str, str] = {}

        # 1. Multipart file upload check
        if request.files:
            for key in request.files:
                norm_slot = normalize_slot_name(key)
                if norm_slot:
                    file_obj = request.files[key]
                    if file_obj and file_obj.filename:
                        safe_ext = Path(file_obj.filename).suffix.lower() or ".mp4"
                        target_file = upload_dir / f"{norm_slot}{safe_ext}"
                        file_obj.save(str(target_file))
                        video_paths[norm_slot] = str(target_file)

        # 2. JSON or form-data check for local file paths
        data = request.get_json(silent=True) if request.is_json else request.form
        if data:
            for key, val in data.items():
                norm_slot = normalize_slot_name(key)
                if norm_slot and norm_slot not in video_paths and isinstance(val, str) and val.strip():
                    cand_path = Path(val.strip())
                    if not cand_path.is_file():
                        cand_path = REPO_ROOT / val.strip().lstrip("/\\")
                    if cand_path.is_file():
                        video_paths[norm_slot] = str(cand_path)

        if not video_paths:
            return jsonify({
                "error": "No valid video files provided. Upload 1 to 5 video files under slots: "
                         "front, front_left, front_right, rear_left, rear_right (or back_left, back_right)."
            }), 400

        # Target FPS: default to 1.0 FPS for fast, seconds-level analysis
        target_fps = request.args.get("target_fps", type=float)
        if target_fps is None and data:
            try:
                target_fps = float(data.get("target_fps", 1.0))
            except (ValueError, TypeError):
                target_fps = 1.0
        if target_fps is None or target_fps <= 0:
            target_fps = 1.0

        max_duration_sec = request.args.get("max_duration_sec", type=float)
        if max_duration_sec is None and data:
            try:
                raw_dur = data.get("max_duration_sec")
                if raw_dur is not None:
                    max_duration_sec = float(raw_dur)
            except (ValueError, TypeError):
                max_duration_sec = 60.0
        if max_duration_sec is None:
            max_duration_sec = 60.0

        # Model selection: default to stock yolov8n for production accuracy (recognizes traffic lights & red signals)
        requested_model = (request.args.get("model") or (data.get("model") if data else "") or "stock").lower()
        fine_tuned_path = REPO_ROOT / "models" / "yolov8_bus_v1.pt"
        stock_path = REPO_ROOT / "yolov8n.pt"
        if requested_model == "fine_tuned" and fine_tuned_path.is_file():
            if not hasattr(brain_module, "_fine_tuned_instance"):
                from ultralytics import YOLO
                brain_module._fine_tuned_instance = YOLO(str(fine_tuned_path))
            brain_module.model = brain_module._fine_tuned_instance
        else:
            if not hasattr(brain_module, "_stock_instance") and stock_path.is_file():
                from ultralytics import YOLO
                brain_module._stock_instance = YOLO(str(stock_path))
            if hasattr(brain_module, "_stock_instance"):
                brain_module.model = brain_module._stock_instance

        active_slots = [s for s in CANONICAL_SLOTS if s in video_paths]
        missing_slots = [s for s in CANONICAL_SLOTS if s not in video_paths]

        # Register and start background job
        job_store.create_job(
            active_slots=active_slots,
            missing_slots=missing_slots,
            target_fps=target_fps,
            video_paths=video_paths,
            max_duration_sec=max_duration_sec,
            job_id=job_id
        )

        start_video_pipeline_thread(
            job_id=job_id,
            video_paths=video_paths,
            target_fps=target_fps,
            max_duration_sec=max_duration_sec,
            job_store=job_store,
            socketio_instance=socketio
        )

        return jsonify({
            "job_id": job_id,
            "status": "processing",
            "target_fps": target_fps,
            "max_duration_sec": max_duration_sec,
            "active_cameras": active_slots,
            "missing_cameras": missing_slots,
            "message": "Video ingestion started. Poll /api/demo/status/<job_id> for timeline."
        }), 202

    @app.route("/api/demo/status/<job_id>", methods=["GET"])
    def demo_job_status(job_id: str):
        """
        Returns the current progress, state, and decision timeline when complete.
        """
        job_store = get_global_job_store()
        job_data = job_store.get_job(job_id)
        if not job_data:
            return jsonify({"error": f"Job ID '{job_id}' not found."}), 404
        return jsonify(job_data), 200

    @app.route("/api/demo/video/<job_id>/<slot>", methods=["GET"])
    def demo_get_video(job_id: str, slot: str):
        """
        Serves the video file corresponding to a job slot for HTML5 video playback.
        """
        norm_slot = normalize_slot_name(slot) or slot
        job_dir = REPO_ROOT / "data" / "temp_videos" / job_id
        for ext in [".mp4", ".mov", ".avi", ".webm"]:
            cand = job_dir / f"{norm_slot}{ext}"
            if cand.is_file():
                from flask import send_file
                return send_file(str(cand), mimetype="video/mp4")

        # Check job store if original path was provided
        job_store = get_global_job_store()
        job_data = job_store.get_job(job_id)
        if job_data:
            active_paths = job_data.get("video_paths", {})
            if norm_slot in active_paths:
                from flask import send_file
                p = Path(active_paths[norm_slot])
                if p.is_file():
                    return send_file(str(p), mimetype="video/mp4")

        return jsonify({"error": f"Video for slot '{slot}' not found."}), 404

    @app.route("/api/analyze", methods=["POST"])
    def analyze_api():
        """
        Main REST endpoint executing autonomous_brain_v3.analyze_scene().
        Supports image upload (form-data or json) and returns complete perception output.
        """
        now = time.time()

        # Parse query params for Phase 4-7 optional enrichments
        enable_tracking = request.args.get("enable_tracking", "false").lower() in ("true", "1")
        enable_motion_intent = request.args.get("enable_motion_intent", "false").lower() in ("true", "1")
        enable_bus_stops = request.args.get("enable_bus_stops", "false").lower() in ("true", "1")
        enable_door_monitor = request.args.get("enable_door_monitor", "false").lower() in ("true", "1")
        include_base64_images = request.args.get("include_images", "false").lower() in ("true", "1")

        # Dynamic model selection: default to stock yolov8n for production accuracy
        requested_model = (request.args.get("model") or "stock").lower()
        fine_tuned_path = REPO_ROOT / "models" / "yolov8_bus_v1.pt"
        stock_path = REPO_ROOT / "yolov8n.pt"
        if requested_model == "fine_tuned" and fine_tuned_path.is_file():
            if not hasattr(brain_module, "_fine_tuned_instance"):
                from ultralytics import YOLO
                brain_module._fine_tuned_instance = YOLO(str(fine_tuned_path))
            brain_module.model = brain_module._fine_tuned_instance
        else:
            if not hasattr(brain_module, "_stock_instance") and stock_path.is_file():
                from ultralytics import YOLO
                brain_module._stock_instance = YOLO(str(stock_path))
            if hasattr(brain_module, "_stock_instance"):
                brain_module.model = brain_module._stock_instance

        # 1. Retrieve camera images
        front_img = None
        front_left_img = None
        front_right_img = None
        rear_left_img = None
        rear_right_img = None
        door_img = None

        if request.files:
            # Check for file upload under 'file', 'image', 'frame', or 'front_image'
            for key in ["file", "frame", "image", "front_image", "front"]:
                if key in request.files:
                    front_img = decode_image_input(request.files[key])
                    break

            front_left_img = decode_image_input(request.files.get("front_left"))
            front_right_img = decode_image_input(request.files.get("front_right"))
            rear_left_img = decode_image_input(request.files.get("rear_left"))
            rear_right_img = decode_image_input(request.files.get("rear_right"))
            door_img = decode_image_input(request.files.get("door_image"))

        elif request.is_json:
            data = request.get_json(silent=True) or {}
            for key in ["file", "frame", "image", "front_image", "front"]:
                if key in data:
                    front_img = decode_image_input(data[key])
                    break
            front_left_img = decode_image_input(data.get("front_left"))
            front_right_img = decode_image_input(data.get("front_right"))
            rear_left_img = decode_image_input(data.get("rear_left"))
            rear_right_img = decode_image_input(data.get("rear_right"))
            door_img = decode_image_input(data.get("door_image"))

        # Fallback to sample data/ images if not supplied
        if front_img is None:
            front_img = load_default_camera_feed("front")
        if front_left_img is None:
            front_left_img = load_default_camera_feed("front_left")
        if front_right_img is None:
            front_right_img = load_default_camera_feed("front_right")
        if rear_left_img is None:
            rear_left_img = load_default_camera_feed("rear_left")
        if rear_right_img is None:
            rear_right_img = load_default_camera_feed("rear_right")

        # 2. Call the EXISTING UNMODIFIED analyze_scene() function
        raw_scene_output = analyze_scene(
            front_image=front_img,
            front_left=front_left_img,
            front_right=front_right_img,
            rear_left=rear_left_img,
            rear_right=rear_right_img
        )

        response_payload = dict(raw_scene_output)
        if "final_decision" not in response_payload and "decision" in response_payload:
            response_payload["final_decision"] = response_payload["decision"]

        model_name_str = str(getattr(brain_module.model, "model_name", "YOLOv8"))
        is_fine_tuned = "bus_v1" in model_name_str.lower()
        response_payload["active_model"] = {
            "name": "YOLOv8 Bus v1 (Fine-Tuned FGVD/IDD)" if is_fine_tuned else "YOLOv8n Base (80 COCO Classes)",
            "version": "v1.0-custom" if is_fine_tuned else "v8.0-stock",
            "is_fine_tuned": is_fine_tuned,
            "classes": len(brain_module.model.names) if hasattr(brain_module.model, "names") else 80
        }

        # 3. Optional Phase 4: ByteTrack Tracking Enrichment
        if enable_tracking or enable_motion_intent:
            tracker = get_default_tracker()
            # Extract front camera detections from YOLO model
            yolo_res = model(front_img, verbose=False)
            current_dets = []
            if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
                for b in yolo_res[0].boxes:
                    cid = int(b.cls[0])
                    cname = model.names[cid]
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])
                    current_dets.append({
                        "class": cname,
                        "confidence": conf,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2
                    })

            tracked = tracker.track_detections(current_dets, timestamp=now)

            # 4. Optional Phase 5: Motion Intent Classification
            if enable_motion_intent:
                h_img, w_img = front_img.shape[:2]
                for det in tracked:
                    hist = det.get("history", [det])
                    det["motion_intent"] = classify_motion_intent(
                        hist,
                        image_width=w_img,
                        image_height=h_img
                    )

            response_payload["tracked_detections"] = tracked
            response_payload["active_tracks"] = len([d for d in tracked if d.get("track_id", -1) > 0])

        # 5. Optional Phase 6: Bus Stop & Docking Intelligence
        if enable_bus_stops:
            bus_detector = get_default_bus_stop_detector()
            h_img, w_img = front_img.shape[:2]

            # Read optional GPS parameters
            lat = request.args.get("latitude", type=float)
            lon = request.args.get("longitude", type=float)
            veh_speed = request.args.get("vehicle_speed", default=25.0, type=float)

            dets_for_stops = response_payload.get("tracked_detections", [])
            if not dets_for_stops:
                # Run quick extraction if tracking was not requested
                yolo_res = model(front_img, verbose=False)
                if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
                    for b in yolo_res[0].boxes:
                        cid = int(b.cls[0])
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        dets_for_stops.append({
                            "class": model.names[cid],
                            "confidence": float(b.conf[0]),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2
                        })

            stop_result = bus_detector.detect_bus_stop(
                detections=dets_for_stops,
                latitude=lat,
                longitude=lon,
                image_width=w_img,
                image_height=h_img
            )
            docking_telemetry = _SHARED_DOCKING_SM.update(
                detector_result=stop_result,
                vehicle_speed_kmh=veh_speed,
                timestamp=now
            )

            response_payload["bus_stop_detection"] = stop_result
            response_payload["docking_state"] = docking_telemetry

        # 6. Optional Phase 7: Door Zone Safety Monitor
        if enable_door_monitor:
            door_monitor = get_default_door_monitor()
            close_req = request.args.get("close_requested", "false").lower() in ("true", "1")
            open_req = request.args.get("open_requested", "false").lower() in ("true", "1")

            door_telemetry = door_monitor.process_frame(
                door_image=door_img,
                open_requested=open_req,
                close_requested=close_req,
                timestamp=now
            )
            response_payload["door_zone_monitor"] = door_telemetry

        # 7. Generate rich visual annotations on front camera for instant UI display
        try:
            annotated_canvas = front_img.copy()
            sig = response_payload.get("traffic_signal", "UNKNOWN")
            dec = response_payload.get("final_decision", "MOVE")
            spd = response_payload.get("recommended_speed", 25)
            risk = response_payload.get("risk_score", 0)

            # Draw bounding boxes for detected objects
            det_res = brain_module.model(front_img, conf=0.18, verbose=False)
            if det_res and len(det_res) > 0 and det_res[0].boxes is not None:
                for b in det_res[0].boxes:
                    cid = int(b.cls[0])
                    cname = brain_module.model.names[cid]
                    bx1, by1, bx2, by2 = map(int, b.xyxy[0])
                    bconf = float(b.conf[0])

                    if cname in ("traffic light", "traffic_light"):
                        box_col = (0, 0, 255) if sig == "RED" else ((0, 255, 255) if sig == "YELLOW" else (0, 255, 0))
                        box_lbl = f"SIGNAL: {sig} ({bconf:.2f})"
                    elif cname == "person":
                        box_col = (0, 140, 255)
                        box_lbl = f"PEDESTRIAN ({bconf:.2f})"
                    elif cname in ("car", "bus", "truck", "motorcycle", "autorickshaw"):
                        box_col = (255, 180, 0)
                        box_lbl = f"{cname.upper()} ({bconf:.2f})"
                    else:
                        box_col = (180, 180, 180)
                        box_lbl = f"{cname} ({bconf:.2f})"

                    cv2.rectangle(annotated_canvas, (bx1, by1), (bx2, by2), box_col, 2)
                    lbl_w = len(box_lbl) * 8 + 8
                    cv2.rectangle(annotated_canvas, (bx1, max(0, by1 - 20)), (bx1 + lbl_w, by1), box_col, -1)
                    cv2.putText(annotated_canvas, box_lbl, (bx1 + 4, max(14, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw HUD status banner
            hud_bg = (15, 23, 42)
            cv2.rectangle(annotated_canvas, (0, 0), (annotated_canvas.shape[1], 44), hud_bg, -1)
            act_col = (0, 0, 255) if dec == "STOP" else ((0, 255, 0) if dec == "MOVE" else (0, 255, 255))
            hud_str = f"ACTION: {dec} | SPEED: {spd} km/h | SIGNAL: {sig} | RISK SCORE: {risk}"
            cv2.putText(annotated_canvas, hud_str, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, act_col, 2, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", annotated_canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])
            response_payload["annotated_front_image"] = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
        except Exception as anno_err:
            print(f"[WARN] Failed to generate visual annotation: {anno_err}")

        # Serialize numpy arrays safely for JSON response
        serialized_response = serialize_for_json(response_payload, include_base64_images)

        # Broadcast update over WebSocket stream
        try:
            socketio.emit("scene_update", {
                "timestamp": now,
                "decision": serialized_response.get("final_decision", "MAINTAIN"),
                "risk_score": serialized_response.get("risk_score", 0.0),
                "risk_level": serialized_response.get("risk_level", "LOW"),
                "pedestrian_status": serialized_response.get("pedestrian_status", "NONE"),
                "traffic_signal": serialized_response.get("traffic_signal", "UNKNOWN"),
                "docking_state": serialized_response.get("docking_state", {}).get("state", "DRIVE"),
                "door_state": serialized_response.get("door_zone_monitor", {}).get("state", "DOOR_CLOSED")
            }, namespace="/ws/live")
        except Exception:
            pass

        return jsonify(serialized_response), 200

    # ==========================================================================
    # WEBSOCKET EVENT HANDLERS (/ws/live)
    # ==========================================================================

    @socketio.on("connect", namespace="/ws/live")
    def handle_ws_connect():
        """Handles new client connection to live WebSocket feed."""
        emit("connection_status", {
            "status": "connected",
            "namespace": "/ws/live",
            "message": "Connected to Autonomous Bus Perception live telemetry stream.",
            "server_time": time.time()
        })

    @socketio.on("disconnect", namespace="/ws/live")
    def handle_ws_disconnect():
        """Handles client disconnection."""
        pass

    @socketio.on("stream_frame", namespace="/ws/live")
    def handle_ws_frame(data):
        """Processes streamed camera frame over WebSocket and emits real-time perception update."""
        try:
            frame_data = data.get("frame") if isinstance(data, dict) else data
            frame_bgr = decode_image_input(frame_data)
            if frame_bgr is not None:
                # Fast inference using brain model
                scene_res = analyze_scene(
                    front_image=frame_bgr,
                    front_left=load_default_camera_feed("front_left"),
                    front_right=load_default_camera_feed("front_right"),
                    rear_left=load_default_camera_feed("rear_left"),
                    rear_right=load_default_camera_feed("rear_right")
                )
                emit("scene_update", {
                    "timestamp": time.time(),
                    "final_decision": scene_res.get("final_decision"),
                    "risk_score": scene_res.get("risk_score"),
                    "risk_level": scene_res.get("risk_level"),
                    "pedestrian_status": scene_res.get("pedestrian_status"),
                    "object_count": scene_res.get("object_count")
                })
        except Exception as e:
            emit("error", {"error": str(e)})


# ==============================================================================
# MAIN STANDALONE ENTRYPOINT
# ==============================================================================

app, socketio = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 5000))
    print(f"\n[START] Launching Autonomous Bus Perception Flask Server on port {port}...")
    print(f"  - REST API Endpoint: POST http://localhost:{port}/api/analyze")
    print(f"  - WebSocket Stream : ws://localhost:{port}/ws/live")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
