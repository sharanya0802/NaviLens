"""
exit_navigation.py — Contextual "Leave the Room" Guidance Layer

Converts ExitState from DoorTracker into rich, natural-language TTS messages
that fully guide a visually impaired person to the exit.

Handles:
  - First sighting announcement ("I can see a door at your 2 o'clock...")
  - Ongoing approach narration with distance updates
  - Scan prompts when no exit is visible ("Please rotate slowly...")
  - Arrival announcement ("You've reached the door...")
  - Obstacle warnings while approaching

All local. No APIs.
"""

import time
from typing import Optional

from exit_detector import ExitState


# ─────────────────────────────────────────────────────────────────────────────
# Timing / suppression constants
# ─────────────────────────────────────────────────────────────────────────────

# How far the door must move (metres) before we announce a distance update
DISTANCE_DELTA_THRESHOLD = 0.8   # metres

# How far (degrees) angle must shift before we re-announce bearing
ANGLE_DELTA_THRESHOLD    = 18.0  # degrees

# Minimum seconds between any two guidance announcements
MIN_ANNOUNCE_GAP = 2.5

# After this many seconds without seeing a door, say "scan" again
SCAN_REPEAT_GAP = 6.0


# ─────────────────────────────────────────────────────────────────────────────
# ExitGuidance
# ─────────────────────────────────────────────────────────────────────────────

class ExitGuidance:
    """
    Stateful guidance narrator for the "exit seeking" mode.

    Usage:
        guidance = ExitGuidance()
        guidance.activate()

        # every nav frame:
        speech = guidance.update(exit_state, obstacle_text)
        if speech:
            tts.speak(speech)

        guidance.deactivate()
    """

    def __init__(self):
        self._active            = False
        self._last_announce_t   = 0.0
        self._last_scan_t       = 0.0

        self._last_dist_m       : Optional[float] = None
        self._last_angle_deg    : Optional[float] = None
        self._door_seen_once    = False   # have we found the door at all yet?
        self._arrived           = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def activate(self):
        """Call when the user initiates "find exit" mode."""
        self._active          = True
        self._last_announce_t = 0.0
        self._last_scan_t     = 0.0
        self._last_dist_m     = None
        self._last_angle_deg  = None
        self._door_seen_once  = False
        self._arrived         = False

    def deactivate(self):
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    # ── Main update ───────────────────────────────────────────────────────────

    def update(
        self,
        exit_state: Optional[ExitState],
        obstacle_text: str = "",
    ) -> Optional[str]:
        """
        Called once per navigation frame.

        Args:
            exit_state:    Current ExitState from DoorTracker, or None.
            obstacle_text: Any obstacle warnings already composed by NavigationEngine.

        Returns:
            A speech string, or None to stay silent this frame.
        """
        if not self._active:
            return None

        now = time.time()

        # ── No door in sight ──────────────────────────────────────────────────
        if exit_state is None:
            return self._handle_no_door(now, obstacle_text)

        # ── Door found ────────────────────────────────────────────────────────

        # Arrival check (very close and roughly centred)
        if exit_state.distance_m <= 1.2 and abs(exit_state.angle_deg) < 20:
            if not self._arrived:
                self._arrived = True
                self._last_announce_t = now
                return "You've reached the door. Reach forward to open it."
            return None  # already announced arrival

        # Reset arrived if they backed away
        self._arrived = False

        # Throttle repeated identical guidance
        if now - self._last_announce_t < MIN_ANNOUNCE_GAP:
            return None

        # Decide if the situation has changed enough to warrant a new announcement
        changed = self._state_changed(exit_state)
        first   = not self._door_seen_once

        if not first and not changed:
            return None

        # Compose the announcement
        speech = self._compose(exit_state, first, obstacle_text)

        self._door_seen_once  = True
        self._last_dist_m     = exit_state.distance_m
        self._last_angle_deg  = exit_state.angle_deg
        self._last_announce_t = now

        return speech

    # ── No-door handler ───────────────────────────────────────────────────────

    def _handle_no_door(self, now: float, obstacle_text: str) -> Optional[str]:
        """Prompt the user to scan if no door has been found yet."""

        if now - self._last_scan_t < SCAN_REPEAT_GAP:
            return None

        self._last_scan_t = now

        if not self._door_seen_once:
            msg = (
                "I cannot see an exit yet. "
                "Please rotate slowly to your right so I can scan the room."
            )
        else:
            msg = (
                "I've lost sight of the door. "
                "Rotate slowly to find it again."
            )

        # Append any obstacle warning
        if obstacle_text:
            msg += f" Caution — {obstacle_text}"

        return msg

    # ── Change detection ──────────────────────────────────────────────────────

    def _state_changed(self, s: ExitState) -> bool:
        """Return True if the exit state has changed enough to re-announce."""

        if self._last_dist_m is None or self._last_angle_deg is None:
            return True

        dist_delta  = abs(s.distance_m - self._last_dist_m)
        angle_delta = abs(s.angle_deg  - self._last_angle_deg)

        return (
            dist_delta  >= DISTANCE_DELTA_THRESHOLD
            or angle_delta >= ANGLE_DELTA_THRESHOLD
        )

    # ── Announcement composer ─────────────────────────────────────────────────

    def _compose(
        self,
        s: ExitState,
        first_sighting: bool,
        obstacle_text: str,
    ) -> str:
        """Build the full natural-language announcement for one frame."""

        parts = []

        # ── Opening ───────────────────────────────────────────────────────────
        if first_sighting:
            parts.append(f"Exit found! The door is {s.bearing_label},")
            parts.append(f"at your {s.clock_position}.")
        else:
            parts.append(f"Door is {s.bearing_label}, at your {s.clock_position}.")

        # ── Distance ─────────────────────────────────────────────────────────
        if s.distance_m <= 1.5:
            parts.append("You are very close.")
        elif s.distance_m <= 3.0:
            parts.append(f"About {s.distance_steps} steps away.")
        else:
            dist_text = (
                f"Roughly {s.distance_steps} steps,"
                f" or about {s.distance_m:.0f} metres away."
            )
            parts.append(dist_text)

        # ── Immediate action ─────────────────────────────────────────────────
        parts.append(s.action)

        # ── Obstacle caveat ──────────────────────────────────────────────────
        if obstacle_text:
            parts.append(f"Watch out — {obstacle_text}")

        return " ".join(parts)
