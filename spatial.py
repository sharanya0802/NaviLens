"""
spatial.py — Spatial Reasoning Engine
Maps raw detections → directional zone (left/center/right) + proximity (near/far)
→ actionable natural-language navigation instructions.
"""

from typing import List, Optional
from detection import Detection, PRIORITY_CLASSES


# ── Zone thresholds (fraction of frame width) ────────────────────────────────
LEFT_ZONE_MAX   = 0.38   # 0.0 – 0.38  → left
CENTER_ZONE_MIN = 0.38
CENTER_ZONE_MAX = 0.62   # 0.38 – 0.62 → center
RIGHT_ZONE_MIN  = 0.62   # 0.62 – 1.0  → right

# ── Proximity thresholds (fraction of frame area) ────────────────────────────
VERY_NEAR_THRESHOLD = 0.15   # > 15 % of frame → very near / stop
NEAR_THRESHOLD      = 0.07   # > 7  % → near / caution
FAR_THRESHOLD       = 0.02   # > 2  % → far / awareness


class SpatialReasoner:
    def __init__(self):
        self._frame_w: int = 640
        self._frame_h: int = 480

    def set_frame_size(self, w: int, h: int):
        self._frame_w = w
        self._frame_h = h

    # ── Zone classification ───────────────────────────────────────────────────
    def _zone(self, det: Detection) -> str:
        rel_x = det.center_x / self._frame_w
        if rel_x < LEFT_ZONE_MAX:
            return "left"
        elif rel_x > RIGHT_ZONE_MIN:
            return "right"
        return "center"

    def _proximity(self, det: Detection) -> str:
        r = det.area_ratio
        if r >= VERY_NEAR_THRESHOLD:
            return "very near"
        elif r >= NEAR_THRESHOLD:
            return "near"
        elif r >= FAR_THRESHOLD:
            return "far"
        return "distant"

    def _is_urgent(self, det: Detection) -> bool:
        return det.label in PRIORITY_CLASSES and det.area_ratio >= NEAR_THRESHOLD

    # ── Public API ────────────────────────────────────────────────────────────
    def get_urgent_warnings(self, detections: List[Detection]) -> Optional[str]:
        """
        Returns a high-priority speech string if any obstacle is dangerously close,
        or None if the scene is clear.
        """
        urgent = [d for d in detections if self._is_urgent(d)]
        if not urgent:
            return None

        messages = []
        for det in urgent[:2]:   # cap at 2 to keep speech brief
            zone = self._zone(det)
            prox = self._proximity(det)
            if prox == "very near":
                msg = f"Stop! {det.label} directly {zone}."
            else:
                msg = f"Caution. {det.label} {prox} on your {zone}."
            messages.append(msg)

        return " ".join(messages)

    def generate_navigation_instruction(self, detections: List[Detection]) -> str:
        """
        Generates a concise navigation-oriented instruction from the current scene.
        Focuses on the most relevant objects only.
        """
        if not detections:
            return "Path appears clear."

        # Bucket by zone
        zones = {"left": [], "center": [], "right": []}
        for det in detections:
            zones[self._zone(det)].append(det)

        parts = []

        # ── Center path warnings are highest priority ──────────────────────
        center = sorted(zones["center"], key=lambda d: -d.area_ratio)
        if center:
            top = center[0]
            prox = self._proximity(top)
            if prox in ("very near", "near"):
                parts.append(f"{top.label} blocking center path, {prox}.")
            else:
                parts.append(f"{top.label} ahead, {prox}.")

        # ── Side awareness ─────────────────────────────────────────────────
        for side in ("left", "right"):
            items = sorted(zones[side], key=lambda d: -d.area_ratio)
            if items:
                top = items[0]
                prox = self._proximity(top)
                if prox in ("very near", "near"):
                    parts.append(f"{top.label} on your {side}, {prox}.")

        return " ".join(parts) if parts else "Path appears clear."

    def generate_full_description(self, detections: List[Detection]) -> str:
        """
        Generates a fuller scene description for 'describe surroundings' command.
        """
        if not detections:
            return "I cannot detect any objects around you."

        zones = {"left": [], "center": [], "right": []}
        for det in detections:
            zones[self._zone(det)].append(det)

        parts = []
        for zone, items in zones.items():
            if not items:
                continue
            # Describe up to 3 per zone
            items_sorted = sorted(items, key=lambda d: -d.area_ratio)[:3]
            labels = ", ".join(
                f"{d.label} ({self._proximity(d)})" for d in items_sorted
            )
            parts.append(f"On your {zone}: {labels}.")

        count = len(detections)
        header = f"I can see {count} object{'s' if count != 1 else ''}. "
        return header + " ".join(parts)
