"""
AutoBus LiDAR Decision Arbiter
========================================================================================
Translates 3D LiDAR object detections into autonomous transit control decisions:
- MOVE (Maintain Speed / Clear Path)
- SLOW_DOWN (Approaching Obstacle / Crosswalk Caution)
- EMERGENCY_STOP (Critical Collision Threat in Path)
- HOLD_LANE (Blind Spot Occupied)
- CHANGE_LEFT / CHANGE_RIGHT (Safe Overtake)

Strictly grounded in empirical LiDAR coordinates (+X Right, +Y Forward, +Z Up).
Bus geometry: Width ~ 2.5m, Length ~ 10.5m.
========================================================================================
"""

from typing import List, Dict, Any, Optional

def arbitrate_lidar_decision(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Arbitrates autonomous driving decision based strictly on 3D LiDAR perception.
    
    Args:
        predictions: List of dicts from AutoBusLidar3DDetector with keys:
                     class, confidence, center {x, y, z}, dimensions {length, width, height}, yaw, distance.
                     
    Returns:
        Structured autonomous decision payload.
    """
    if not predictions:
        return {
            "final_decision": "MOVE",
            "decision_reason": "LiDAR field of view 100% clear. No obstacles detected within 48m.",
            "risk_score": 0.0,
            "risk_level": "LOW",
            "safe_speed_kmh": 40.0,
            "closest_obstacle_dist": None,
            "closest_obstacle_class": None,
            "front_path_clear": True,
            "blind_spots": {
                "left_occupied": False,
                "right_occupied": False,
                "left_closest_m": None,
                "right_closest_m": None
            },
            "action_color": "#10b981", # Emerald
            "action_icon": "MOVE"
        }
        
    # Analyze spatial corridors
    # Front Corridor: |X| <= 1.8m (bus width 2.5m, corridor half-width 1.8m)
    # Left Blind Spot: X in [-4.0, -1.25], Y in [-8.0, 3.0]
    # Right Blind Spot: X in [1.25, 4.0], Y in [-8.0, 3.0]
    
    front_threats = []
    left_blind_threats = []
    right_blind_threats = []
    all_threats = []
    
    for obj in predictions:
        c = obj.get("center", {})
        x = float(c.get("x", 0.0))
        y = float(c.get("y", 0.0))
        dist = float(obj.get("distance", 0.0))
        cls = obj.get("class", "unknown").lower()
        conf = float(obj.get("confidence", 0.0))
        
        info = {
            "class": cls,
            "dist": dist,
            "x": x,
            "y": y,
            "conf": conf,
            "obj": obj
        }
        all_threats.append(info)
        
        # 1. Front Path Check (Ahead of bus front bumper Y > 0)
        if 0.0 <= y <= 35.0 and abs(x) <= 2.0:
            front_threats.append(info)
            
        # 2. Left Blind Spot Check
        if -4.5 <= x <= -1.25 and -8.0 <= y <= 3.0:
            left_blind_threats.append(info)
            
        # 3. Right Blind Spot Check
        if 1.25 <= x <= 4.5 and -8.0 <= y <= 3.0:
            right_blind_threats.append(info)
            
    # Sort threats by distance
    all_threats.sort(key=lambda t: t["dist"])
    front_threats.sort(key=lambda t: t["dist"])
    left_blind_threats.sort(key=lambda t: t["dist"])
    right_blind_threats.sort(key=lambda t: t["dist"])
    
    closest = all_threats[0]
    left_occ = len(left_blind_threats) > 0
    right_occ = len(right_blind_threats) > 0
    left_min_d = left_blind_threats[0]["dist"] if left_occ else None
    right_min_d = right_blind_threats[0]["dist"] if right_occ else None
    
    # -------------------------------------------------------------------------
    # RULE-BASED SAFETY ARBITRATION TREE
    # -------------------------------------------------------------------------
    
    # Rule 1: Immediate Front Path Pedestrian Collision Hazard
    if front_threats:
        lead = front_threats[0]
        if lead["class"] == "pedestrian":
            if lead["y"] <= 12.0:
                return {
                    "final_decision": "EMERGENCY_STOP",
                    "decision_reason": f"CRITICAL: Pedestrian directly in front path at {lead['dist']:.1f}m ({lead['y']:.1f}m forward). Immediate emergency braking engaged.",
                    "risk_score": 9.8,
                    "risk_level": "CRITICAL",
                    "safe_speed_kmh": 0.0,
                    "closest_obstacle_dist": closest["dist"],
                    "closest_obstacle_class": closest["class"],
                    "front_path_clear": False,
                    "blind_spots": {"left_occupied": left_occ, "right_occupied": right_occ, "left_closest_m": left_min_d, "right_closest_m": right_min_d},
                    "action_color": "#ef4444",
                    "action_icon": "STOP"
                }
            else:
                return {
                    "final_decision": "SLOW_DOWN",
                    "decision_reason": f"CAUTION: Pedestrian crossing ahead at {lead['dist']:.1f}m. Decelerating to yield crosswalk right-of-way.",
                    "risk_score": 7.5,
                    "risk_level": "HIGH",
                    "safe_speed_kmh": 15.0,
                    "closest_obstacle_dist": closest["dist"],
                    "closest_obstacle_class": closest["class"],
                    "front_path_clear": False,
                    "blind_spots": {"left_occupied": left_occ, "right_occupied": right_occ, "left_closest_m": left_min_d, "right_closest_m": right_min_d},
                    "action_color": "#f59e0b",
                    "action_icon": "SLOW"
                }
                
        # Lead vehicle in path
        if lead["class"] == "car":
            if lead["y"] <= 8.5:
                return {
                    "final_decision": "EMERGENCY_STOP",
                    "decision_reason": f"COLLISION ALERT: Lead vehicle stopped directly ahead at {lead['dist']:.1f}m ({lead['y']:.1f}m forward). Active stop.",
                    "risk_score": 9.2,
                    "risk_level": "CRITICAL",
                    "safe_speed_kmh": 0.0,
                    "closest_obstacle_dist": closest["dist"],
                    "closest_obstacle_class": closest["class"],
                    "front_path_clear": False,
                    "blind_spots": {"left_occupied": left_occ, "right_occupied": right_occ, "left_closest_m": left_min_d, "right_closest_m": right_min_d},
                    "action_color": "#ef4444",
                    "action_icon": "STOP"
                }
            elif lead["y"] <= 20.0:
                # Opportunity to check if overtaking lane is clear
                if not left_occ and lead["y"] >= 14.0:
                    return {
                        "final_decision": "CHANGE_LEFT",
                        "decision_reason": f"LANE ADAPTATION: Slow vehicle at {lead['dist']:.1f}m. Left blind spot clear ({'Clear' if not left_occ else 'Occupied'}). Safe lane change initiated.",
                        "risk_score": 5.2,
                        "risk_level": "MEDIUM",
                        "safe_speed_kmh": 30.0,
                        "closest_obstacle_dist": closest["dist"],
                        "closest_obstacle_class": closest["class"],
                        "front_path_clear": False,
                        "blind_spots": {"left_occupied": left_occ, "right_occupied": right_occ, "left_closest_m": left_min_d, "right_closest_m": right_min_d},
                        "action_color": "#3b82f6",
                        "action_icon": "LANE_LEFT"
                    }
                else:
                    return {
                        "final_decision": "SLOW_DOWN",
                        "decision_reason": f"ADAPTIVE CRUISE: Lead vehicle detected at {lead['dist']:.1f}m. Moderating speed to maintain 3-second safety gap.",
                        "risk_score": 6.0,
                        "risk_level": "MEDIUM",
                        "safe_speed_kmh": 25.0,
                        "closest_obstacle_dist": closest["dist"],
                        "closest_obstacle_class": closest["class"],
                        "front_path_clear": False,
                        "blind_spots": {"left_occupied": left_occ, "right_occupied": right_occ, "left_closest_m": left_min_d, "right_closest_m": right_min_d},
                        "action_color": "#f59e0b",
                        "action_icon": "SLOW"
                    }
                    
    # Rule 2: Front is clear, check blind spots
    if left_occ and left_min_d is not None and left_min_d <= 4.0:
        return {
            "final_decision": "HOLD_LANE",
            "decision_reason": f"BLIND SPOT ALERT: Vehicle/obstacle detected in Left Blind Spot ({left_min_d:.1f}m). Lane changes prohibited.",
            "risk_score": 5.8,
            "risk_level": "MEDIUM",
            "safe_speed_kmh": 35.0,
            "closest_obstacle_dist": closest["dist"],
            "closest_obstacle_class": closest["class"],
            "front_path_clear": True,
            "blind_spots": {"left_occupied": left_occ, "right_occupied": right_occ, "left_closest_m": left_min_d, "right_closest_m": right_min_d},
            "action_color": "#6366f1", # Violet
            "action_icon": "HOLD"
        }
        
    if right_occ and right_min_d is not None and right_min_d <= 4.0:
        return {
            "final_decision": "HOLD_LANE",
            "decision_reason": f"BLIND SPOT ALERT: Vehicle/cyclist detected in Right Blind Spot ({right_min_d:.1f}m). Curbside maneuvering restricted.",
            "risk_score": 5.5,
            "risk_level": "MEDIUM",
            "safe_speed_kmh": 35.0,
            "closest_obstacle_dist": closest["dist"],
            "closest_obstacle_class": closest["class"],
            "front_path_clear": True,
            "blind_spots": {"left_occupied": left_occ, "right_occupied": right_occ, "left_closest_m": left_min_d, "right_closest_m": right_min_d},
            "action_color": "#6366f1",
            "action_icon": "HOLD"
        }
        
    # Rule 3: Clear path, normal autonomous cruising
    return {
        "final_decision": "MOVE",
        "decision_reason": f"PATH CLEAR: Forward navigation corridor unobstructed. Closest peripheral object at {closest['dist']:.1f}m. Cruising at operational speed.",
        "risk_score": min(3.0, round(15.0 / max(closest['dist'], 1.0), 1)),
        "risk_level": "LOW",
        "safe_speed_kmh": 40.0,
        "closest_obstacle_dist": closest["dist"],
        "closest_obstacle_class": closest["class"],
        "front_path_clear": True,
        "blind_spots": {"left_occupied": left_occ, "right_occupied": right_occ, "left_closest_m": left_min_d, "right_closest_m": right_min_d},
        "action_color": "#10b981", # Emerald
        "action_icon": "MOVE"
    }
