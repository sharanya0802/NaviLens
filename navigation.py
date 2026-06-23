"""
navigation.py — Local depth-based navigation pipeline.

Combines MiDaS depth estimation + lightweight YOLO object detection
to provide real-time obstacle avoidance and directional guidance.

All processing is LOCAL — no API calls.
"""

import time
import cv2
import numpy as np
from typing import List, Optional, Tuple

from depth_engine import DepthEngine
from spatial import VERY_NEAR_THRESHOLD

# ── Navigation-critical YOLO classes ─────────────────────────────────
NAV_CRITICAL_CLASSES = {"person", "chair", "dining table", "couch",
                        "dog", "cat", "bicycle", "car", "motorbike",
                        "fire hydrant", "stop sign", "bench"}

# Proximity thresholds on normalized depth (0=far, 1=close)
DANGER_CLOSE = 0.75      # depth > this in center = STOP
WARNING_CLOSE = 0.60     # depth > this = caution
FREE_THRESHOLD = 0.50    # depth < this = walkable


class NavigationEngine:
    """
    Real-time local navigation using depth + optional YOLO.

    Splits the frame into LEFT/CENTER/RIGHT zones, analyzes average depth
    in each, detects critical objects, and generates event-based guidance.
    Only announces changes — no TTS spam.
    """

    def __init__(self, model_name: str = "yolov8m.pt", conf: float = 0.30):
        self._model_name = model_name
        self._conf = conf
        self._depth = DepthEngine()
        self._detector = None
        self._detector_loaded = False

        # ── State tracking for event-based announcements ─────────────
        self._last_state: str = ""
        self._last_announce_time: float = 0.0
        self._min_announce_gap: float = 4.0   # seconds between announcements
        self._last_severity: str = ""         # "danger"/"blocked"/"caution"/"clear"
        self._clear_announced: bool = False    # True if we already said "path clear"

        # ── Temporal smoothing (rolling average over N frames) ────────
        self._depth_history: list = []         # list of (left, center, right) tuples
        self._smoothing_window: int = 5        # average over last 5 frames

        # ── Hysteresis (require N consecutive same-state before change) ──
        self._pending_state: str = ""
        self._pending_count: int = 0
        self._hysteresis_threshold: int = 3

        # ── Frame dimensions ─────────────────────────────────────────
        self._frame_w: int = 640
        self._frame_h: int = 480

        # ── Target-guided navigation ──────────────────────────────────
        self._target: Optional[str] = None
        self._target_was_visible: bool = False
        self._target_invisible_frames: int = 0
        self._last_target_zone: Optional[str] = None
        self._last_target_announce_time: float = 0.0
        self.target_reached: bool = False
        self.target_lost: bool = False

    def _ensure_detector(self):
        """Lazy-load YOLOv8n for navigation-critical objects."""
        if self._detector_loaded:
            return
        try:
            from ultralytics import YOLO
            self._detector = YOLO(self._model_name)
            self._detector_loaded = True
            print(f"[Nav] {self._model_name} loaded for navigation objects.")
        except Exception as e:
            print(f"[Nav] YOLO load failed (navigation will use depth only): {e}")
            self._detector_loaded = True

    def set_frame_size(self, w: int, h: int):
        self._frame_w = w
        self._frame_h = h

    # ── Target navigation ────────────────────────────────────────────
    def set_target(self, object_name: str):
        """Set a target object to navigate toward."""
        self._target = object_name.lower()
        self._target_was_visible = False
        self._target_invisible_frames = 0
        self._last_target_zone = None
        self._last_target_announce_time = 0.0
        self.target_reached = False
        self.target_lost = False

    def clear_target(self):
        """Clear the navigation target."""
        self._target = None
        self._target_was_visible = False
        self._target_invisible_frames = 0
        self._last_target_zone = None
        self.target_reached = False
        self.target_lost = False

    # ──────────────────────────────────────────────────────────────────
    #  CORE ANALYSIS
    # ──────────────────────────────────────────────────────────────────
    def analyze(self, frame: np.ndarray) -> Tuple[str, Optional[str], np.ndarray]:
        """
        Analyze a frame for navigation.

        Returns:
            (state_key, announcement, depth_colormap)
            - state_key:    compact state string for change detection
            - announcement: speech text if state changed, else None
            - depth_colormap: visualization of depth map for display
        """
        # 1) Depth estimation
        depth = self._depth.estimate(frame)
        depth_vis = self._depth.depth_to_colormap(depth)

        h, w = depth.shape

        # 2) Split into LEFT / CENTER / RIGHT zones (matching spatial.py: 38/62)
        left_bound = int(w * 0.38)
        right_bound = int(w * 0.62)
        left_depth = depth[:, :left_bound]
        center_depth = depth[:, left_bound:right_bound]
        right_depth = depth[:, right_bound:]

        # Zone depth using 75th percentile (higher = closer / more blocked)
        # Using percentile instead of mean prevents far-background pixels from
        # masking close objects — e.g. a person at depth 0.8 covering 40% of
        # the zone would average to ~0.38 ("clear") with mean, but the 75th
        # percentile correctly reads ~0.7+ ("blocked").
        raw_left = float(np.percentile(left_depth, 75))
        raw_center = float(np.percentile(center_depth, 75))
        raw_right = float(np.percentile(right_depth, 75))

        # ── Temporal smoothing: rolling average over last N frames ────
        self._depth_history.append((raw_left, raw_center, raw_right))
        if len(self._depth_history) > self._smoothing_window:
            self._depth_history.pop(0)

        left_avg = sum(h[0] for h in self._depth_history) / len(self._depth_history)
        center_avg = sum(h[1] for h in self._depth_history) / len(self._depth_history)
        right_avg = sum(h[2] for h in self._depth_history) / len(self._depth_history)

        # Maximum depth in center (closest obstacle — high value = close)
        center_max = float(np.percentile(center_depth, 90))  # 90th percentile = closest stuff

        # 3) Detect critical objects with YOLO (optional)
        critical_objects = self._detect_critical_objects(frame)

        # 4) Build navigation state
        state, announcement = self._evaluate_state(
            left_avg, center_avg, right_avg, center_max, critical_objects
        )

        # ── Target-specific override ─────────────────────────────────
        self.target_reached = False
        self.target_lost = False
        if self._target:
            target_obj = None
            for obj in critical_objects:
                if obj["label"].lower() == self._target:
                    target_obj = obj
                    break

            if target_obj:
                self._target_was_visible = True
                self._target_invisible_frames = 0

                if target_obj["area_ratio"] >= VERY_NEAR_THRESHOLD:
                    self.target_reached = True
                    announcement = f"You have reached the {self._target}"
                elif announcement is None:
                    import time as _time
                    now = _time.time()
                    zone_changed = target_obj["zone"] != self._last_target_zone
                    gap_elapsed = now - self._last_target_announce_time >= self._min_announce_gap
                    if zone_changed or gap_elapsed:
                        self._last_target_zone = target_obj["zone"]
                        self._last_target_announce_time = now
                        announcement = self._build_target_guidance(target_obj["zone"])
            else:
                if self._target_was_visible:
                    self._target_invisible_frames += 1
                    if self._target_invisible_frames >= self._hysteresis_threshold:
                        self.target_lost = True
                        announcement = f"I've lost sight of the {self._target}"

        # 5) Draw zone indicators on depth visualization
        self._draw_zone_info(depth_vis, left_avg, center_avg, right_avg, critical_objects)

        return state, announcement, depth_vis

    def _detect_critical_objects(self, frame: np.ndarray) -> List[dict]:
        """Detect navigation-critical objects using YOLOv8n."""
        self._ensure_detector()
        if self._detector is None:
            return []

        try:
            results = self._detector(frame, conf=self._conf, verbose=False, stream=False)[0]
        except Exception:
            return []

        objects = []
        for box in results.boxes:
            label = self._detector.model.names[int(box.cls)]
            if label not in NAV_CRITICAL_CLASSES and (self._target is None or label.lower() != self._target):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / (self._frame_w * self._frame_h)

            # Determine zone
            rel_x = cx / self._frame_w
            if rel_x < 0.38:
                zone = "left"
            elif rel_x > 0.62:
                zone = "right"
            else:
                zone = "center"

            objects.append({
                "label": label,
                "zone": zone,
                "area_ratio": area_ratio,
                "bbox": (x1, y1, x2, y2),
            })

        # Sort by proximity (larger area = closer)
        objects.sort(key=lambda o: -o["area_ratio"])
        return objects[:4]  # cap at 4

    # ──────────────────────────────────────────────────────────────────
    #  STATE EVALUATION
    # ──────────────────────────────────────────────────────────────────
    def _evaluate_state(
        self,
        left_avg: float,
        center_avg: float,
        right_avg: float,
        center_max: float,
        objects: List[dict],
    ) -> Tuple[str, Optional[str]]:
        """
        Evaluate navigation state and generate announcement if changed.
        Only announces 'clear' once when transitioning from obstacle→clear.
        Returns (state_key, announcement_or_None).
        """
        left_status = self._zone_status(left_avg)
        center_status = self._zone_status(center_avg)
        right_status = self._zone_status(right_avg)
        center_danger = center_max > DANGER_CLOSE

        # Determine current severity
        if center_danger:
            current_severity = "danger"
        elif center_status == "blocked":
            current_severity = "blocked"
        elif center_status == "caution":
            current_severity = "caution"
        else:
            current_severity = "clear"

        # Build state key (depth-only — YOLO labels are too noisy for change detection)
        state_key = f"L:{left_status},C:{center_status},R:{right_status},D:{center_danger}"

        # ── Hysteresis: require N consecutive identical states before changing ──
        now = time.time()
        if state_key == self._last_state:
            # Same as current confirmed state — nothing to do
            self._pending_state = ""
            self._pending_count = 0
            return state_key, None

        if state_key == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = state_key
            self._pending_count = 1

        if self._pending_count < self._hysteresis_threshold:
            # Not enough consecutive frames — don't announce yet
            return self._last_state, None

        # Hysteresis passed — this state is confirmed
        self._pending_state = ""
        self._pending_count = 0

        if now - self._last_announce_time < self._min_announce_gap:
            self._last_state = state_key
            return state_key, None

        # SUPPRESS: if we're clear and we already announced clear → stay silent
        if current_severity == "clear" and self._clear_announced:
            self._last_state = state_key
            return state_key, None

        # State changed — generate announcement
        self._last_state = state_key
        self._last_announce_time = now

        # Track severity transitions
        was_obstructed = self._last_severity in ("danger", "blocked", "caution")
        self._last_severity = current_severity

        # If transitioning TO clear, announce once then set flag
        if current_severity == "clear":
            if was_obstructed:
                self._clear_announced = True
                return state_key, "Path clear ahead."
            else:
                # First frame or already clear → announce once
                self._clear_announced = True
                return state_key, "Path clear ahead."
        else:
            # Obstacle detected — reset clear flag so we re-announce when it clears
            self._clear_announced = False

        announcement = self._build_announcement(
            left_status, center_status, right_status,
            center_danger, left_avg, center_avg, right_avg, objects
        )

        return state_key, announcement

    def _zone_status(self, avg_depth: float) -> str:
        """Higher depth = closer to camera = more blocked."""
        if avg_depth <= FREE_THRESHOLD:
            return "clear"
        elif avg_depth <= WARNING_CLOSE:
            return "caution"
        else:
            return "blocked"

    def _build_announcement(
        self,
        left_s: str, center_s: str, right_s: str,
        center_danger: bool,
        left_avg: float, center_avg: float, right_avg: float,
        objects: List[dict],
    ) -> str:
        parts = []

        # Urgent: something very close in center
        if center_danger:
            parts.append("Stop! Obstacle very close ahead.")
            # Suggest safest direction (lower avg_depth = further away = safer)
            if left_avg < right_avg and left_s != "blocked":
                parts.append("Move left.")
            elif right_s != "blocked":
                parts.append("Move right.")
            elif left_s != "blocked":
                parts.append("Move left.")
            else:
                parts.append("Caution, all paths are crowded.")
            # Mention critical object if any
            center_objs = [o for o in objects if o["zone"] == "center"]
            if center_objs:
                parts.append(f"{center_objs[0]['label']} ahead.")

        elif center_s == "blocked":
            # Center blocked but not danger-close
            parts.append("Path ahead is blocked.")
            if left_avg < right_avg and left_s != "blocked":
                parts.append("Try moving left.")
            elif right_s != "blocked":
                parts.append("Try moving right.")

        elif center_s == "caution":
            parts.append("Obstacle ahead, proceed with caution.")
            if left_s == "clear" and left_avg < center_avg - 0.1:
                parts.append("Left side is clearer.")
            elif right_s == "clear" and right_avg < center_avg - 0.1:
                parts.append("Right side is clearer.")

        # NOTE: "clear" case is handled in _evaluate_state, not here.
        # This method is only called for non-clear states.

        # Mention critical objects on sides (deduplicated)
        seen_side_objects = set()
        for obj in objects:
            if obj["zone"] != "center" and obj["area_ratio"] > 0.03:
                key = (obj["label"], obj["zone"])
                if key not in seen_side_objects:
                    seen_side_objects.add(key)
                    parts.append(f"{obj['label']} on your {obj['zone']}.")

        return " ".join(parts)

    def _build_target_guidance(self, zone: str) -> str:
        """Generate directional guidance toward the target object."""
        if zone == "left":
            return f"The {self._target} is on your left"
        elif zone == "right":
            return f"The {self._target} is on your right"
        return f"The {self._target} is straight ahead"

    # ──────────────────────────────────────────────────────────────────
    #  VISUALIZATION
    # ──────────────────────────────────────────────────────────────────
    def _draw_zone_info(
        self, vis: np.ndarray,
        left_avg: float, center_avg: float, right_avg: float,
        objects: List[dict],
    ):
        """Draw zone status and objects on the depth visualization."""
        h, w = vis.shape[:2]
        left_bound = int(w * 0.38)
        right_bound = int(w * 0.62)

        # Zone labels with status
        for i, (label, avg, x_start) in enumerate([
            ("LEFT", left_avg, 10),
            ("CENTER", center_avg, left_bound + 10),
            ("RIGHT", right_avg, right_bound + 10),
        ]):
            status = self._zone_status(avg)
            color = (0, 255, 0) if status == "clear" else \
                    (0, 200, 255) if status == "caution" else (0, 0, 255)
            cv2.putText(vis, f"{label}: {status}", (x_start, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(vis, f"d={avg:.2f}", (x_start, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Zone divider lines (matching 38/62 split)
        cv2.line(vis, (left_bound, 0), (left_bound, h), (255, 255, 255), 1)
        cv2.line(vis, (right_bound, 0), (right_bound, h), (255, 255, 255), 1)

        # Draw detected critical objects
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis, obj["label"], (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    def reset_state(self):
        """Reset navigation state (called when entering nav mode)."""
        self._last_state = ""
        self._last_announce_time = 0.0
        self._last_severity = ""
        self._clear_announced = False
        self._depth_history.clear()
        self._pending_state = ""
        self._pending_count = 0
        self.clear_target()
