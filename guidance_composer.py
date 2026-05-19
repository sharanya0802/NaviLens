"""
guidance_composer.py — Human-Like Mobility Assistant Speech

Context-aware, temporally-smoothed guidance narrator with a state machine.
Generates natural speech that behaves like a mobility assistant rather than
an object detector.

States: Scanning → Navigating → Approaching → Arrived → Blocked

All local. No APIs.
"""

import time
from enum import Enum, auto
from typing import List, Optional

from path_planner import PlannedPath
from spatial_map import SpatialMap
from exit_detector import ExitState


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HYSTERESIS_ANGLE = 15.0       # degrees — direction must change by this much to re-announce
MIN_ANNOUNCE_GAP = 2.5        # seconds between regular announcements
URGENT_ANNOUNCE_GAP = 1.0     # seconds when obstacles are very close
BLOCKED_DISTANCE = 0.08       # normalised — below this = dead end
APPROACHING_DISTANCE = 0.15   # normalised — below this while heading to opening = approaching
ARRIVED_DISTANCE = 0.06       # normalised — very close to doorway


class NavState(Enum):
    SCANNING = auto()
    NAVIGATING = auto()
    APPROACHING = auto()
    ARRIVED = auto()
    BLOCKED = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Context phrases
# ─────────────────────────────────────────────────────────────────────────────

_CONTEXT_PHRASES = {
    "corridor": "along the corridor",
    "open_room": "through the open space",
    "narrow_passage": "through the narrow passage",
    "near_wall": "away from the wall",
    "along_wall": "along the wall",
    "doorway_approach": "towards the opening",
    "unknown": "forward",
}

_DIRECTION_WORDS = {
    (-45, -20): "Turn left",
    (-20, -8): "Veer left",
    (-8, 8): "Continue straight",
    (8, 20): "Veer right",
    (20, 45): "Turn right",
}


def _angle_to_direction(angle: float) -> str:
    """Convert angle to a direction phrase."""
    for (lo, hi), phrase in _DIRECTION_WORDS.items():
        if lo <= angle < hi:
            return phrase
    if angle < -20:
        return "Turn left"
    return "Turn right"


def _angle_to_side(angle: float) -> str:
    """Convert angle to 'left'/'right'/'ahead'."""
    if angle < -8:
        return "left"
    elif angle > 8:
        return "right"
    return "ahead"


# ─────────────────────────────────────────────────────────────────────────────
# GuidanceComposer
# ─────────────────────────────────────────────────────────────────────────────

class GuidanceComposer:
    """
    Stateful, context-aware guidance narrator.

    Call ``compose()`` every navigation frame. Returns a speech string
    when something meaningful should be announced, or None to stay silent.
    """

    def __init__(self):
        self._state = NavState.SCANNING
        self._last_announce_time = 0.0
        self._last_announced_angle: Optional[float] = None
        self._last_announced_text = ""
        self._first_guidance = True
        self._blocked_count = 0
        self._stable_frames = 0       # how many frames the direction has been stable

    def reset(self):
        """Reset state for a new navigation session."""
        self._state = NavState.SCANNING
        self._last_announce_time = 0.0
        self._last_announced_angle = None
        self._last_announced_text = ""
        self._first_guidance = True
        self._blocked_count = 0
        self._stable_frames = 0

    def compose(
        self,
        path: PlannedPath,
        spatial: SpatialMap,
        objects: List[dict],
        head_obstacle: bool,
        exit_state: Optional[ExitState] = None,
    ) -> Optional[str]:
        """
        Generate guidance speech for the current frame.

        Returns a speech string, or None to stay silent this frame.
        """
        now = time.time()

        # ── Head-level obstacle: highest priority interrupt ───────────────────
        if head_obstacle:
            if now - self._last_announce_time >= URGENT_ANNOUNCE_GAP:
                self._last_announce_time = now
                return "Caution! Low obstacle overhead. Duck slightly."

        # ── State transitions ────────────────────────────────────────────────
        prev_state = self._state
        self._update_state(path, exit_state)

        # ── Throttle announcements ───────────────────────────────────────────
        gap = URGENT_ANNOUNCE_GAP if path.distance < 0.15 else MIN_ANNOUNCE_GAP
        if now - self._last_announce_time < gap:
            # Allow state-change announcements to bypass throttle
            if self._state == prev_state:
                return None

        # ── Check if direction actually changed (hysteresis) ─────────────────
        direction_changed = self._check_direction_changed(path.angle)

        # ── Generate speech based on state ───────────────────────────────────
        speech = self._generate_for_state(
            path, spatial, objects, exit_state, direction_changed
        )

        if speech and speech != self._last_announced_text:
            self._last_announce_time = now
            self._last_announced_angle = path.angle
            self._last_announced_text = speech
            self._first_guidance = False
            return speech

        return None

    # ── State machine ─────────────────────────────────────────────────────────

    def _update_state(self, path: PlannedPath, exit_state: Optional[ExitState]):
        """Transition between navigation states."""
        if path.distance < BLOCKED_DISTANCE and path.width < 0.2:
            self._state = NavState.BLOCKED
            self._blocked_count += 1
            return

        self._blocked_count = 0

        if exit_state and exit_state.visible and exit_state.distance_m <= 1.5:
            self._state = NavState.ARRIVED
            return

        if path.context == "doorway_approach" or (
            exit_state and exit_state.visible and exit_state.distance_m <= 3.0
        ):
            self._state = NavState.APPROACHING
            return

        if path.distance > BLOCKED_DISTANCE:
            self._state = NavState.NAVIGATING
            return

    def _check_direction_changed(self, new_angle: float) -> bool:
        """Return True if the direction has changed enough to re-announce."""
        if self._last_announced_angle is None:
            return True
        delta = abs(new_angle - self._last_announced_angle)
        if delta >= HYSTERESIS_ANGLE:
            self._stable_frames = 0
            return True
        self._stable_frames += 1
        return False

    # ── Speech generation per state ───────────────────────────────────────────

    def _generate_for_state(
        self,
        path: PlannedPath,
        spatial: SpatialMap,
        objects: List[dict],
        exit_state: Optional[ExitState],
        direction_changed: bool,
    ) -> Optional[str]:

        if self._state == NavState.SCANNING:
            return self._speech_scanning(path, spatial)

        if self._state == NavState.BLOCKED:
            return self._speech_blocked(path)

        if self._state == NavState.ARRIVED:
            return self._speech_arrived(exit_state)

        if self._state == NavState.APPROACHING:
            return self._speech_approaching(path, exit_state, objects)

        # NAVIGATING
        return self._speech_navigating(
            path, spatial, objects, direction_changed
        )

    def _speech_scanning(self, path: PlannedPath, spatial: SpatialMap) -> str:
        openings = spatial.get_openings()
        if openings:
            best = openings[0]
            side = best.direction
            if side == "center":
                return "I see a possible exit straight ahead. Walk forward carefully."
            return f"I see a possible exit to your {side}. Turn {side} and walk forward."

        ctx = spatial.get_room_context()
        if ctx == "corridor":
            return "I can see a corridor ahead. Let me guide you through it."
        if ctx == "open_room":
            return "You're in an open space. Scanning for the best path."
        if ctx == "near_wall":
            return "There's a wall close ahead. Let me find an open direction."
        return "Scanning your surroundings. One moment."

    def _speech_blocked(self, path: PlannedPath) -> str:
        if path.alternatives:
            best_alt = path.alternatives[0]
            side = _angle_to_side(best_alt["angle"])
            return f"Dead end ahead. Turn to your {side} where there's open space."
        if self._blocked_count > 3:
            return "Path is blocked in all directions. Please rotate slowly to find an opening."
        return "Dead end ahead. Stop and rotate slowly to find an open path."

    def _speech_arrived(self, exit_state: Optional[ExitState]) -> str:
        if exit_state and exit_state.visible:
            return "You've reached the doorway. Reach forward to open it."
        return "You've reached the opening. Proceed carefully."

    def _speech_approaching(
        self,
        path: PlannedPath,
        exit_state: Optional[ExitState],
        objects: list,
    ) -> str:
        parts = []

        if exit_state and exit_state.visible:
            parts.append(
                f"Door is {exit_state.bearing_label}, "
                f"about {exit_state.distance_steps} steps away."
            )
            parts.append(exit_state.action)
        else:
            direction = _angle_to_direction(path.angle)
            parts.append(f"{direction} towards the opening ahead.")

        # Add obstacle warning if something is in the path
        obs_warning = self._obstacle_warning(objects)
        if obs_warning:
            parts.append(obs_warning)

        return " ".join(parts)

    def _speech_navigating(
        self,
        path: PlannedPath,
        spatial: SpatialMap,
        objects: list,
        direction_changed: bool,
    ) -> Optional[str]:
        # If direction hasn't changed and we've already spoken, stay silent
        if not direction_changed and not self._first_guidance:
            # But still speak if there's a new obstacle
            obs = self._obstacle_warning(objects)
            if obs:
                return obs
            return None

        direction = _angle_to_direction(path.angle)
        ctx_phrase = _CONTEXT_PHRASES.get(path.context, "forward")

        parts = []

        if self._first_guidance:
            # Detailed first announcement
            walls = spatial.get_walls()
            room_ctx = spatial.get_room_context()
            if room_ctx == "corridor":
                parts.append(f"You're in a corridor. {direction} {ctx_phrase}.")
            elif room_ctx == "open_room":
                parts.append(f"Open space around you. {direction} {ctx_phrase}.")
            elif room_ctx == "near_wall":
                parts.append(f"Wall ahead. {direction} {ctx_phrase}.")
            else:
                parts.append(f"{direction} {ctx_phrase}.")
        else:
            # Brief follow-up
            parts.append(f"{direction} {ctx_phrase}.")

        # Obstacle caveat
        obs = self._obstacle_warning(objects)
        if obs:
            parts.append(obs)

        return " ".join(parts)

    # ── Obstacle warning helper ───────────────────────────────────────────────

    @staticmethod
    def _obstacle_warning(objects: list) -> Optional[str]:
        """Build a compact obstacle warning from detected objects."""
        warnings = []
        for obj in objects:
            if obj["area_ratio"] < 0.04:
                continue
            label = obj["label"]
            zone = obj["zone"]
            if label == "door":
                continue
            if label == "stairs":
                warnings.append("Stairs detected, slow down.")
            elif zone == "center":
                warnings.append(f"Watch out, {label} directly ahead.")
            else:
                warnings.append(f"{label} on your {zone}.")
        return " ".join(warnings[:2]) if warnings else None
