"""
src/stops/docking_state_machine.py
===============================================================================
Autonomous Bus Perception System - Bus Stop Docking State Machine
===============================================================================
Manages autonomous bus stop approach, deceleration, halting, passenger dwell,
and departure progression through a formal 6-phase state machine:

  DRIVE
    ↓ (bus stop confidence threshold reached)
  APPROACH_STOP
    ↓ (vehicle decelerating, entering stop approach zone)
  STOPPING
    ↓ (speed reaches full halt at curbside)
  STOPPED
    ↓ (doors open / boarding initiated)
  BOARDING
    ↓ (passenger ingress/egress complete, doors closed, min dwell elapsed)
  READY_TO_DEPART
    ↓ (clearance verified, vehicle resumes forward motion)
  DRIVE

Outputs current state and explicit trigger reasons without modifying existing
arbitration formulas in decision_fusion_engine.py.
===============================================================================
"""

import time
from enum import Enum
from typing import Any, Dict, List, Optional


class DockingState(str, Enum):
    DRIVE = "DRIVE"
    APPROACH_STOP = "APPROACH_STOP"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    BOARDING = "BOARDING"
    READY_TO_DEPART = "READY_TO_DEPART"


class DockingStateMachine:
    """
    State machine orchestrating the bus stop docking workflow.
    """

    def __init__(
        self,
        min_dwell_seconds: float = 3.0,
        approach_speed_threshold_kmh: float = 18.0,
        stop_speed_threshold_kmh: float = 1.5,
        depart_speed_threshold_kmh: float = 3.0
    ):
        self.state: DockingState = DockingState.DRIVE
        self.prev_state: DockingState = DockingState.DRIVE
        self.min_dwell_seconds = min_dwell_seconds
        self.approach_speed_threshold = approach_speed_threshold_kmh
        self.stop_speed_threshold = stop_speed_threshold_kmh
        self.depart_speed_threshold = depart_speed_threshold_kmh

        self.state_enter_timestamp: float = time.time()
        self.current_reason: str = "INITIAL_CRUISING_STATE"
        self.active_signals: List[str] = []
        self.approach_confidence: float = 0.0

        # Halt persistence counter for debounce
        self._halt_counter: int = 0

    def reset(self, timestamp: Optional[float] = None) -> None:
        """Resets the state machine back to DRIVE."""
        now = timestamp if timestamp is not None else time.time()
        self.state = DockingState.DRIVE
        self.prev_state = DockingState.DRIVE
        self.state_enter_timestamp = now
        self.current_reason = "RESET_TO_DRIVE"
        self.active_signals = []
        self.approach_confidence = 0.0
        self._halt_counter = 0

    def transition_to(
        self,
        new_state: DockingState,
        reason: str,
        timestamp: Optional[float] = None
    ) -> None:
        """Performs a state transition and records reason and timestamp."""
        now = timestamp if timestamp is not None else time.time()
        self.prev_state = self.state
        self.state = new_state
        self.state_enter_timestamp = now
        self.current_reason = reason
        self._halt_counter = 0

    def update(
        self,
        detector_result: Dict[str, Any],
        vehicle_speed_kmh: float = 30.0,
        doors_open: bool = False,
        passenger_activity: bool = False,
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Processes a perception frame and advances the docking state machine.

        Args:
            detector_result: Output dict from BusStopDetector.detect_bus_stop()
            vehicle_speed_kmh: Current vehicle forward speed in km/h
            doors_open: Whether vehicle passenger doors are open
            passenger_activity: Sensor/camera detection of passenger ingress/egress
            timestamp: Frame timestamp (defaults to time.time())

        Returns:
            Dict containing current state, previous state, trigger reasons, and telemetry.
        """
        now = timestamp if timestamp is not None else time.time()
        time_in_state = max(0.0, now - self.state_enter_timestamp)

        self.approach_confidence = float(detector_result.get("approach_confidence", 0.0))
        self.active_signals = list(detector_result.get("trigger_reasons", []))
        is_approaching = bool(detector_result.get("is_approaching", False))

        gps_info = detector_result.get("signals", {}).get("nearest_stop")
        dist_m = detector_result.get("signals", {}).get("gps_distance_m")

        # ----------------------------------------------------------------------
        # STATE: DRIVE
        # ----------------------------------------------------------------------
        if self.state == DockingState.DRIVE:
            if is_approaching or self.approach_confidence >= 0.50:
                reasons_str = ", ".join(self.active_signals) if self.active_signals else "HIGH_CONFIDENCE"
                self.transition_to(
                    DockingState.APPROACH_STOP,
                    f"STOP_DETECTED: {reasons_str}",
                    timestamp=now
                )

        # ----------------------------------------------------------------------
        # STATE: APPROACH_STOP
        # ----------------------------------------------------------------------
        elif self.state == DockingState.APPROACH_STOP:
            # Check for deceleration into docking bay
            close_by_gps = (dist_m is not None and dist_m <= 25.0)
            close_by_sign = (detector_result.get("signals", {}).get("sign_confidence", 0.0) >= 0.70)
            speed_low = (vehicle_speed_kmh <= self.approach_speed_threshold)

            if speed_low and (close_by_gps or close_by_sign or is_approaching):
                self.transition_to(
                    DockingState.STOPPING,
                    f"DECELERATING_FOR_DOCK (speed={vehicle_speed_kmh:.1f} km/h)",
                    timestamp=now
                )
            elif not is_approaching and self.approach_confidence < 0.25 and vehicle_speed_kmh > 25.0 and time_in_state > 4.0:
                # Abort if stop signals completely vanish while continuing at road speed
                self.transition_to(
                    DockingState.DRIVE,
                    "STOP_APPROACH_CANCELLED_FALSE_ALARM",
                    timestamp=now
                )

        # ----------------------------------------------------------------------
        # STATE: STOPPING
        # ----------------------------------------------------------------------
        elif self.state == DockingState.STOPPING:
            if vehicle_speed_kmh <= self.stop_speed_threshold:
                self._halt_counter += 1
                if self._halt_counter >= 1:  # full halt confirmed
                    self.transition_to(
                        DockingState.STOPPED,
                        f"FULL_HALT_REACHED (speed={vehicle_speed_kmh:.1f} km/h)",
                        timestamp=now
                    )
            else:
                self._halt_counter = 0

        # ----------------------------------------------------------------------
        # STATE: STOPPED
        # ----------------------------------------------------------------------
        elif self.state == DockingState.STOPPED:
            if doors_open or passenger_activity:
                self.transition_to(
                    DockingState.BOARDING,
                    "PASSENGER_BOARDING_INITIATED",
                    timestamp=now
                )

        # ----------------------------------------------------------------------
        # STATE: BOARDING
        # ----------------------------------------------------------------------
        elif self.state == DockingState.BOARDING:
            # Complete boarding once dwell time has elapsed and doors are closed
            if time_in_state >= self.min_dwell_seconds and not doors_open:
                self.transition_to(
                    DockingState.READY_TO_DEPART,
                    f"BOARDING_COMPLETE (dwell={time_in_state:.1f}s, doors closed)",
                    timestamp=now
                )

        # ----------------------------------------------------------------------
        # STATE: READY_TO_DEPART
        # ----------------------------------------------------------------------
        elif self.state == DockingState.READY_TO_DEPART:
            # Resuming speed indicates departure
            if vehicle_speed_kmh >= self.depart_speed_threshold or time_in_state >= 2.0:
                self.transition_to(
                    DockingState.DRIVE,
                    f"DEPARTED_STOP_RESUMING_TRANSIT (speed={vehicle_speed_kmh:.1f} km/h)",
                    timestamp=now
                )

        return {
            "state": self.state.value,
            "previous_state": self.prev_state.value,
            "state_reason": self.current_reason,
            "active_signals": self.active_signals,
            "approach_confidence": self.approach_confidence,
            "time_in_state": round(float(time_in_state), 2),
            "docking_in_progress": bool(self.state != DockingState.DRIVE),
            "vehicle_speed_kmh": round(float(vehicle_speed_kmh), 1),
            "doors_open": bool(doors_open)
        }
