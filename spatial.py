"""
spatial.py — Spatial Reasoning Engine
Maps raw detections → directional zone (left/center/right) + proximity (near/far)
→ actionable natural-language navigation instructions.
"""

from typing import List, Optional
from detection import Detection, PRIORITY_CLASSES, OBJECT_CLASSES


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
        Only includes directional info for persons and navigation-critical objects.
        Identifiable objects (products) are skipped here — handled by SmartIdentifier.
        """
        if not detections:
            return "Path appears clear."

        # Filter to only navigation-relevant detections (persons, vehicles, etc.)
        nav_dets = [d for d in detections if d.label not in OBJECT_CLASSES]

        if not nav_dets:
            return ""

        # Bucket by zone
        zones = {"left": [], "center": [], "right": []}
        for det in nav_dets:
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

        return " ".join(parts) if parts else ""

    def generate_full_description(self, detections: List[Detection]) -> str:
        """
        Generates a fuller scene description for 'describe surroundings' command.
        Persons/navigation objects get directional info, identifiable objects just get named.
        """
        if not detections:
            return "I cannot detect any objects around you."

        # Separate navigation objects from identifiable objects
        nav_dets = [d for d in detections if d.label not in OBJECT_CLASSES]
        obj_dets = [d for d in detections if d.label in OBJECT_CLASSES]

        parts = []

        # Navigation objects with spatial info
        if nav_dets:
            zones = {"left": [], "center": [], "right": []}
            for det in nav_dets:
                zones[self._zone(det)].append(det)
            for zone, items in zones.items():
                if not items:
                    continue
                items_sorted = sorted(items, key=lambda d: -d.area_ratio)[:3]
                labels = ", ".join(
                    f"{d.label} ({self._proximity(d)})" for d in items_sorted
                )
                parts.append(f"On your {zone}: {labels}.")

        # Identifiable objects — just list them without direction
        if obj_dets:
            obj_labels = ", ".join(
                d.label for d in sorted(obj_dets, key=lambda d: -d.area_ratio)[:5]
            )
            parts.append(f"I also see: {obj_labels}.")

        count = len(detections)
        header = f"I can see {count} object{'s' if count != 1 else ''}. "
        return header + " ".join(parts)
