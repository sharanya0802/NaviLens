"""
hazard_monitor.py — Background safety alerts when not in full navigation.

Uses YOLO + simple proximity rules to warn about people, vehicles, and
obstacles suddenly close in the walking path. Fully local.
"""

import time
from typing import Callable, List, Optional

from detection import Detection, PRIORITY_CLASSES
from local_vision import detect_objects


VERY_NEAR = 0.12
NEAR = 0.06
MIN_ALERT_GAP = 3.5


class HazardMonitor:
    """Lightweight always-on hazard watcher for idle / listening mode."""

    def __init__(self, on_alert: Callable[[str, bool], None]):
        """
        Args:
            on_alert: callback(message, urgent)
        """
        self._on_alert = on_alert
        self._last_alert = 0.0
        self._last_message = ""
        self._enabled = True

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def check_frame(self, frame) -> None:
        if not self._enabled:
            return

        now = time.time()
        if now - self._last_alert < MIN_ALERT_GAP:
            return

        h, w = frame.shape[:2]

        try:
            detections = detect_objects(frame)
        except Exception:
            return

        warning = self._build_warning(detections, w)
        if not warning or warning == self._last_message:
            return

        urgent = "stop" in warning.lower() or "caution" in warning.lower()
        self._last_message = warning
        self._last_alert = now
        self._on_alert(warning, urgent)

    @staticmethod
    def _build_warning(detections: List[Detection], frame_w: int) -> Optional[str]:
        center_path = []
        for d in detections:
            if d.label not in PRIORITY_CLASSES:
                continue
            if d.area_ratio >= NEAR:
                zone = HazardMonitor._zone_from_detection(d, frame_w)
                if zone in ("center", "left", "right"):
                    center_path.append((d.area_ratio, d.label, zone))

        if not center_path:
            return None

        center_path.sort(key=lambda x: -x[0])
        area, label, zone = center_path[0]

        if area >= VERY_NEAR:
            return f"Stop! {label} very close {zone}."
        if area >= NEAR:
            return f"Caution. {label} near you on the {zone}."
        return None

    @staticmethod
    def _zone_from_detection(d: Detection, frame_w: int) -> str:
        rel = d.center_x / max(frame_w, 1)
        if rel < 0.35:
            return "left"
        if rel > 0.65:
            return "right"
        return "center"
