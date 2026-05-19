"""
exit_detector.py — Door / Exit Tracker for NaviLens

Tracks door/exit detections across frames, providing:
  - Smoothed bearing (clock-face direction, e.g. "2 o'clock")
  - Estimated distance in steps
  - Confidence score for how long a door has been stably visible
  - Scan directive when no exit is visible
  - Depth-based opening detection as fallback when YOLO misses doors

All local. No APIs.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────────────────────────

# The more frames a door has been confirmed, the higher the confidence
CONFIRM_FRAMES = 3       # minimum frames to confirm a door sighting
HISTORY_LEN    = 10      # smoothing window

# Area ratio → estimated distance (empirical calibration)
# A door filling >25% of the frame is ~1m away; ~3% → ~5m
DISTANCE_TABLE = [
    (0.25, 1.0),   # very close  → ~1 m
    (0.15, 2.0),
    (0.08, 3.0),
    (0.04, 4.5),
    (0.02, 6.0),   # barely visible → ~6 m
]

EXIT_LABELS = {"door"}   # can be extended with "exit sign", "gate", etc.


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExitObservation:
    """One frame's worth of door observation."""
    rel_x: float       # 0.0 (far left) → 1.0 (far right)
    area_ratio: float  # fraction of frame area (proxy for proximity)
    depth_score: float # optional: average normalised depth at door bbox (0..1)


@dataclass
class ExitState:
    """
    Current best estimate of the exit's position.
    Populated only when the tracker has sufficient confidence.
    """
    clock_position: str    # e.g. "12 o'clock", "2 o'clock" …
    bearing_label: str     # e.g. "straight ahead", "to your left"
    distance_m: float      # estimated metres
    distance_steps: int    # estimated footsteps (≈ 0.75 m / step)
    angle_deg: float       # raw angle from centre, negative = left
    confidence: int        # how many consecutive frames we've seen the door
    action: str            # immediate nav instruction, e.g. "Move forward."
    visible: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Tracker
# ─────────────────────────────────────────────────────────────────────────────

class DoorTracker:
    """
    Tracks door detections across frames, smooths position, and provides
    contextual ExitState objects for the guidance layer.
    """

    def __init__(self, frame_w: int = 640, frame_h: int = 480):
        self._frame_w = frame_w
        self._frame_h = frame_h

        # Rolling history of rel_x observations
        self._rel_x_history:    deque = deque(maxlen=HISTORY_LEN)
        self._area_history:     deque = deque(maxlen=HISTORY_LEN)
        self._depth_history:    deque = deque(maxlen=HISTORY_LEN)

        # Consecutive-miss counter (for confidence decay)
        self._miss_frames: int = 0
        self._hit_frames:  int = 0

    def set_frame_size(self, w: int, h: int):
        self._frame_w = w
        self._frame_h = h

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        objects: List[dict],          # list from NavigationEngine._detect_critical_objects
        depth: Optional[np.ndarray],  # normalised depth map (0..1)
        openings: Optional[list] = None,  # from SpatialMap.get_openings()
    ) -> Optional[ExitState]:
        """
        Feed the latest detections.  Returns ExitState if we have a confident
        fix on an exit, else None.

        Fuses YOLO door detections with depth-based openings from SpatialMap.
        `objects` items must have keys: label, zone, area_ratio, bbox
        `openings` items are Opening dataclasses from spatial_map.py
        """
        # Try YOLO door first (high confidence)
        door_obs = self._extract_door(objects, depth)

        # Fallback: use depth-based openings if no YOLO door
        if door_obs is None and openings:
            door_obs = self._extract_opening_as_door(openings)

        if door_obs is not None:
            self._rel_x_history.append(door_obs.rel_x)
            self._area_history.append(door_obs.area_ratio)
            self._depth_history.append(door_obs.depth_score)
            self._miss_frames = 0
            self._hit_frames = min(self._hit_frames + 1, HISTORY_LEN)
        else:
            self._miss_frames += 1
            self._hit_frames = max(self._hit_frames - 1, 0)

            if self._miss_frames > 3:
                # Stale — discard old data
                self._rel_x_history.clear()
                self._area_history.clear()
                self._depth_history.clear()

        if self._hit_frames < CONFIRM_FRAMES or not self._rel_x_history:
            return None

        # Build smoothed state
        rel_x       = float(np.mean(self._rel_x_history))
        area_ratio  = float(np.mean(self._area_history))
        depth_score = float(np.mean(self._depth_history)) if self._depth_history else 0.5

        return self._build_state(rel_x, area_ratio, depth_score)

    def reset(self):
        self._rel_x_history.clear()
        self._area_history.clear()
        self._depth_history.clear()
        self._miss_frames = 0
        self._hit_frames  = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _extract_door(
        self,
        objects: List[dict],
        depth: Optional[np.ndarray],
    ) -> Optional[ExitObservation]:
        """Pick the most prominent door from this frame's detections."""

        doors = [o for o in objects if o.get("label", "") in EXIT_LABELS]
        if not doors:
            return None

        # Take the largest (closest) door
        doors.sort(key=lambda o: -o["area_ratio"])
        best = doors[0]

        x1, y1, x2, y2 = best["bbox"]
        cx = (x1 + x2) / 2
        rel_x = cx / self._frame_w

        # Depth at door centre (optional)
        depth_score = 0.5
        if depth is not None:
            cy_i = min(max((y1 + y2) // 2, 0), depth.shape[0] - 1)
            cx_i = min(max(int(cx), 0), depth.shape[1] - 1)
            depth_score = float(depth[cy_i, cx_i])

        return ExitObservation(
            rel_x=rel_x,
            area_ratio=best["area_ratio"],
            depth_score=depth_score,
        )

    @staticmethod
    def _extract_opening_as_door(openings: list) -> Optional[ExitObservation]:
        """
        Use depth-based openings from SpatialMap as a fallback door signal.
        Returns an ExitObservation if a sufficiently confident opening exists.
        """
        if not openings:
            return None

        # Take the highest-confidence opening
        best = openings[0]   # already sorted by confidence descending

        if best.confidence < 0.3:
            return None   # not confident enough

        # Estimate area ratio from opening width and depth
        # Wider + deeper opening → closer → larger area ratio
        estimated_area = best.width * min(best.depth, 1.0) * 0.15

        return ExitObservation(
            rel_x=best.rel_x,
            area_ratio=estimated_area,
            depth_score=best.depth,
        )

    def _build_state(
        self,
        rel_x: float,
        area_ratio: float,
        depth_score: float,
    ) -> ExitState:
        """Convert smoothed observations into an ExitState."""

        # ── Angle from frame centre ──────────────────────────────────────────
        # rel_x=0.5 → 0°, rel_x=0 → -45°, rel_x=1 → +45°
        angle_deg = (rel_x - 0.5) * 90.0   # degrees, + = right

        # ── Clock face ──────────────────────────────────────────────────────
        clock, bearing = _angle_to_clock(angle_deg)

        # ── Distance ────────────────────────────────────────────────────────
        dist_m = _area_to_distance(area_ratio)
        steps  = max(1, int(round(dist_m / 0.75)))

        # ── Immediate action verb ────────────────────────────────────────────
        action = _build_action(angle_deg, dist_m)

        return ExitState(
            clock_position=clock,
            bearing_label=bearing,
            distance_m=dist_m,
            distance_steps=steps,
            angle_deg=angle_deg,
            confidence=self._hit_frames,
            action=action,
            visible=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _angle_to_clock(angle_deg: float) -> Tuple[str, str]:
    """
    Convert a horizontal bearing (degrees, + = right) to a clock position
    and a natural-language direction label.

      -45° → "10 o'clock" / "to your left"
        0° → "12 o'clock" / "straight ahead"
      +45° → "2 o'clock" / "to your right"
    """
    # Map ±45° range to 10→12→2 on a clock face
    # We model the field-of-view as 9 o'clock (far left) to 3 o'clock (far right)
    # angle_deg is in [-45, +45]; clock goes 10..12..2
    #   normalised = (angle_deg + 45) / 90   → 0..1
    #   clock_hour = 10 + 2 * normalised     → 10..12 (left half)
    #             or use full 9→3 range
    # We'll use 9 o'clock (−45°) → 3 o'clock (+45°)

    norm = (angle_deg + 45.0) / 90.0           # 0..1
    clock_val = 9.0 + norm * 6.0               # 9..15 (15 = 3 on the right)
    clock_hour = int(round(clock_val)) % 12
    if clock_hour == 0:
        clock_hour = 12

    clock_str = f"{clock_hour} o'clock"

    if angle_deg < -20:
        bearing = "to your left"
    elif angle_deg > 20:
        bearing = "to your right"
    else:
        bearing = "straight ahead"

    return clock_str, bearing


def _area_to_distance(area_ratio: float) -> float:
    """Estimate distance (metres) from the door's bounding-box area ratio."""
    for threshold, dist in DISTANCE_TABLE:
        if area_ratio >= threshold:
            return dist
    return 8.0   # very far


def _build_action(angle_deg: float, dist_m: float) -> str:
    """
    Generate a single, immediate navigation command based on bearing and distance.
    """
    if dist_m <= 1.5:
        return "You're right at the door. Reach forward to open it."
    if dist_m <= 3.0:
        if abs(angle_deg) < 15:
            return "Move straight ahead."
        elif angle_deg < 0:
            return "Veer slightly left."
        else:
            return "Veer slightly right."
    # Far away
    if abs(angle_deg) < 15:
        return "Continue straight towards the exit."
    elif angle_deg < -25:
        return "Turn left towards the exit."
    elif angle_deg > 25:
        return "Turn right towards the exit."
    elif angle_deg < 0:
        return "Bear left towards the exit."
    else:
        return "Bear right towards the exit."
