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
from src.autonomous_brain_v3 import analyze_scene, model

# Import Phase 4-7 optional modules
from src.tracking.byte_tracker_wrapper import ByteTrackerWrapper, get_default_tracker
from src.temporal.motion_intent import classify_motion_intent
from src.stops.bus_stop_detector import BusStopDetector, get_default_bus_stop_detector
from src.stops.docking_state_machine import DockingStateMachine, DockingState
from src.doors.door_zone_monitor import DoorZoneMonitor, DoorZoneState, get_default_door_monitor


# ==============================================================================
# FLASK & SOCKETIO INITIALIZATION
# ==============================================================================

def create_app() -> Tuple[Flask, SocketIO]:
    """Factory creating and configuring the Flask + SocketIO app."""
    app = Flask(__name__, static_folder="static", static_url_path="/static")
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
            return img

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

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "models": {
                "yolo_detector": "active",
                "yolo_model_path": str(getattr(model, "model_name", "yolov8n"))
            },
            "phases_available": {
                "phase_4_tracking": True,
                "phase_5_motion_intent": True,
                "phase_6_bus_stops": True,
                "phase_7_door_monitor": True
            }
        }), 200

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
