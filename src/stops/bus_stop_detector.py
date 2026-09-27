"""
src/stops/bus_stop_detector.py
===============================================================================
Autonomous Bus Perception System - Multi-Signal Bus Stop Detector
===============================================================================
Fuses three independent modalities into a single continuous confidence score
[0.0, 1.0] indicating whether a bus stop is currently being approached:

1. Visual Detection Signal:
   - Scans for `bus_stop_sign` class detections (trained in YOLOv8 bus model).
   - Weights confidence and geometric scale/bounding-box area.

2. Geofence / GPS Signal:
   - Haversine distance lookup against predefined route stop waypoints
     loaded from configs/routes/*.json.
   - Activates within configurable stop radius and approach thresholds.

3. Curbside Crowd Cluster Heuristic:
   - Reuses existing pedestrian detections without modifying them.
   - Evaluates whether >=3 people are clustered along the road edge/curbside
     ROI and exhibit stationary / waiting motion profiles.
===============================================================================
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ==============================================================================
# HAVERSINE DISTANCE HELPER FOR GPS GEOFENCING
# ==============================================================================

def haversine_distance_meters(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """
    Calculates great-circle distance between two GPS coordinates in meters.
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return float(R * c)


# ==============================================================================
# BUS STOP DETECTOR CLASS
# ==============================================================================

class BusStopDetector:
    """
    Multi-signal fusion engine for detecting bus stops ahead.
    """

    def __init__(
        self,
        route_config_path: Optional[str] = "configs/routes/route_42A.json",
        approach_threshold: float = 0.50,
        sign_weight: float = 0.45,
        gps_weight: float = 0.35,
        crowd_weight: float = 0.20,
        default_cluster_radius_px: float = 160.0
    ):
        self.route_config_path = route_config_path
        self.approach_threshold = approach_threshold
        self.sign_weight = sign_weight
        self.gps_weight = gps_weight
        self.crowd_weight = crowd_weight
        self.cluster_radius_px = default_cluster_radius_px

        self.stops: List[Dict[str, Any]] = []
        if route_config_path:
            self.load_route(route_config_path)

    def load_route(self, route_path: str) -> None:
        """Loads known bus stop waypoints from a route JSON file."""
        p = Path(route_path)
        if not p.is_file():
            # If relative path not found, try from repo root
            p = Path(__file__).resolve().parent.parent.parent / route_path
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.stops = data.get("stops", [])
            except Exception as e:
                print(f"[WARN] Failed to load route from {route_path}: {e}")
                self.stops = []
        else:
            self.stops = []

    # --------------------------------------------------------------------------
    # SIGNAL 1: VISUAL DETECTION (bus_stop_sign)
    # --------------------------------------------------------------------------
    def evaluate_sign_signal(
        self,
        detections: List[Dict[str, Any]],
        image_width: int = 640,
        image_height: int = 480
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Evaluates visual detections for bus stop signage.
        Returns (signal_score [0.0, 1.0], best_sign_box).
        """
        best_score = 0.0
        best_box = None

        for det in detections:
            cls_name = str(det.get("class", "")).lower()
            if cls_name in ("bus_stop_sign", "bus_stop", "stop_sign"):
                conf = float(det.get("confidence", det.get("conf", 0.85)))
                x1 = float(det.get("x1", 0))
                y1 = float(det.get("y1", 0))
                x2 = float(det.get("x2", 0))
                y2 = float(det.get("y2", 0))

                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                area = w * h
                total_area = float(image_width * image_height)
                area_ratio = area / max(1.0, total_area)

                # Score increases with confidence and scale (nearness)
                scale_boost = min(1.2, 0.7 + (math.sqrt(area) / 40.0))
                sign_score = min(1.0, conf * scale_boost)

                if sign_score > best_score:
                    best_score = sign_score
                    best_box = det

        return float(best_score), best_box

    # --------------------------------------------------------------------------
    # SIGNAL 2: GEOFENCE / GPS CHECK
    # --------------------------------------------------------------------------
    def evaluate_gps_signal(
        self,
        latitude: Optional[float],
        longitude: Optional[float]
    ) -> Tuple[float, Optional[Dict[str, Any]], Optional[float]]:
        """
        Evaluates proximity to known stops on the active route.
        Returns (signal_score [0.0, 1.0], nearest_stop_dict, distance_meters).
        """
        if latitude is None or longitude is None or not self.stops:
            return 0.0, None, None

        nearest_stop = None
        min_dist = float("inf")

        for stop in self.stops:
            s_lat = float(stop.get("latitude", 0.0))
            s_lon = float(stop.get("longitude", 0.0))
            dist = haversine_distance_meters(latitude, longitude, s_lat, s_lon)
            if dist < min_dist:
                min_dist = dist
                nearest_stop = stop

        if nearest_stop is None:
            return 0.0, None, None

        stop_radius = float(nearest_stop.get("radius_meters", 35.0))
        approach_dist = float(nearest_stop.get("approach_distance_meters", 120.0))

        if min_dist <= stop_radius:
            gps_score = 1.0
        elif min_dist <= approach_dist:
            # Linear decay from 1.0 down to 0.1 over approach corridor
            span = max(1.0, approach_dist - stop_radius)
            decay = (approach_dist - min_dist) / span
            gps_score = 0.10 + 0.90 * decay
        else:
            gps_score = 0.0

        return float(min(1.0, max(0.0, gps_score))), nearest_stop, float(min_dist)

    # --------------------------------------------------------------------------
    # SIGNAL 3: CROWD CLUSTER HEURISTIC
    # --------------------------------------------------------------------------
    def evaluate_crowd_signal(
        self,
        detections: List[Dict[str, Any]],
        image_width: int = 640,
        image_height: int = 480,
        curbside_roi: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[float, int, List[Dict[str, Any]]]:
        """
        Evaluates pedestrian clusters along the curbside / road-edge ROI.
        Requires >=3 people clustered near road edge and not moving aggressively.

        Returns (signal_score [0.0, 1.0], cluster_size, clustered_pedestrians).
        """
        W = float(image_width)
        H = float(image_height)

        # Default curbside ROI: right side road shoulder (60% to 100% W, 30% to 95% H)
        if curbside_roi is None:
            roi_x1, roi_y1, roi_x2, roi_y2 = 0.55 * W, 0.25 * H, 1.0 * W, 0.95 * H
        else:
            roi_x1, roi_y1, roi_x2, roi_y2 = curbside_roi

        candidate_peds: List[Dict[str, Any]] = []

        for det in detections:
            cls_name = str(det.get("class", "")).lower()
            if cls_name != "person":
                continue

            x1 = float(det.get("x1", 0))
            y1 = float(det.get("y1", 0))
            x2 = float(det.get("x2", 0))
            y2 = float(det.get("y2", 0))
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Must fall inside defined curbside ROI
            if not (roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2):
                continue

            # Motion condition: must be stationary or waiting (not crossing or erratic)
            intent = det.get("motion_intent", "stationary").lower()
            if intent in ("crossing", "erratic"):
                continue

            candidate_peds.append({
                "det": det,
                "cx": cx,
                "cy": cy
            })

        if len(candidate_peds) < 3:
            return 0.0, len(candidate_peds), []

        # Find largest connected spatial cluster within cluster_radius_px
        cluster_radius = self.cluster_radius_px
        n = len(candidate_peds)
        adj = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                dist = math.hypot(
                    candidate_peds[i]["cx"] - candidate_peds[j]["cx"],
                    candidate_peds[i]["cy"] - candidate_peds[j]["cy"]
                )
                if dist <= cluster_radius:
                    adj[i].append(j)
                    adj[j].append(i)

        # Breadth-first search for connected components
        visited = [False] * n
        max_cluster: List[int] = []

        for i in range(n):
            if not visited[i]:
                comp = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    comp.append(curr)
                    for neighbor in adj[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                if len(comp) > len(max_cluster):
                    max_cluster = comp

        cluster_size = len(max_cluster)
        clustered_peds = [candidate_peds[idx]["det"] for idx in max_cluster]

        if cluster_size >= 3:
            # Score scales from 0.75 for 3 people up to 1.0 for 5+ people
            crowd_score = min(1.0, 0.75 + 0.08 * (cluster_size - 3))
            return float(crowd_score), cluster_size, clustered_peds
        else:
            return 0.0, cluster_size, []

    # --------------------------------------------------------------------------
    # MULTI-SIGNAL FUSION ENTRYPOINT
    # --------------------------------------------------------------------------
    def detect_bus_stop(
        self,
        detections: List[Dict[str, Any]],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        image_width: int = 640,
        image_height: int = 480,
        curbside_roi: Optional[Tuple[float, float, float, float]] = None
    ) -> Dict[str, Any]:
        """
        Fuses visual, GPS, and crowd signals into a single unified bus stop score.
        """
        # Signal 1: Bus stop signage
        sign_score, sign_box = self.evaluate_sign_signal(
            detections,
            image_width=image_width,
            image_height=image_height
        )

        # Signal 2: GPS Geofence
        gps_score, nearest_stop, dist_m = self.evaluate_gps_signal(
            latitude=latitude,
            longitude=longitude
        )

        # Signal 3: Curbside crowd cluster
        crowd_score, crowd_count, crowd_peds = self.evaluate_crowd_signal(
            detections,
            image_width=image_width,
            image_height=image_height,
            curbside_roi=curbside_roi
        )

        # Dynamic weighting based on GPS availability
        trigger_reasons: List[str] = []
        positive_signals_count = 0

        if sign_score >= 0.50:
            trigger_reasons.append("BUS_STOP_SIGN_DETECTED")
            positive_signals_count += 1

        gps_available = (latitude is not None and longitude is not None and len(self.stops) > 0)
        if gps_available:
            if gps_score >= 0.50:
                stop_name = nearest_stop.get("name", "Unknown") if nearest_stop else "Route Stop"
                trigger_reasons.append(f"GPS_GEOFENCE_APPROACH ({stop_name}, {dist_m:.0f}m)")
                positive_signals_count += 1
            w_sign = self.sign_weight
            w_gps = self.gps_weight
            w_crowd = self.crowd_weight
        else:
            # Rebalance weights between visual and crowd if GPS unavailable
            w_sign = 0.65
            w_gps = 0.0
            w_crowd = 0.35

        if crowd_score >= 0.50:
            trigger_reasons.append(f"CURBSIDE_CROWD_CLUSTER ({crowd_count} people waiting)")
            positive_signals_count += 1

        # Base weighted fusion
        fused_score = (w_sign * sign_score) + (w_gps * gps_score) + (w_crowd * crowd_score)

        # Multi-modal agreement boost
        if positive_signals_count >= 2:
            fused_score = min(1.0, fused_score + 0.18)
        if positive_signals_count == 3:
            fused_score = min(1.0, fused_score + 0.25)

        is_approaching = (fused_score >= self.approach_threshold)

        return {
            "approach_confidence": round(float(fused_score), 3),
            "is_approaching": bool(is_approaching),
            "trigger_reasons": trigger_reasons,
            "signals": {
                "sign_detected": (sign_score >= 0.50),
                "sign_confidence": round(float(sign_score), 3),
                "sign_bbox": sign_box,
                "gps_available": bool(gps_available),
                "gps_near_stop": (gps_score >= 0.50),
                "gps_distance_m": round(float(dist_m), 1) if dist_m is not None else None,
                "nearest_stop": nearest_stop,
                "crowd_cluster_detected": (crowd_score >= 0.50),
                "crowd_count": int(crowd_count),
                "crowd_score": round(float(crowd_score), 3)
            }
        }


# ==============================================================================
# CONVENIENCE STANDALONE FUNCTION
# ==============================================================================

_GLOBAL_DETECTOR: Optional[BusStopDetector] = None

def get_default_bus_stop_detector() -> BusStopDetector:
    global _GLOBAL_DETECTOR
    if _GLOBAL_DETECTOR is None:
        _GLOBAL_DETECTOR = BusStopDetector()
    return _GLOBAL_DETECTOR


def detect_bus_stop(
    detections: List[Dict[str, Any]],
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    image_width: int = 640,
    image_height: int = 480,
    detector: Optional[BusStopDetector] = None
) -> Dict[str, Any]:
    """
    Public functional interface for bus stop detection.
    """
    if detector is None:
        detector = get_default_bus_stop_detector()
    return detector.detect_bus_stop(
        detections=detections,
        latitude=latitude,
        longitude=longitude,
        image_width=image_width,
        image_height=image_height
    )
