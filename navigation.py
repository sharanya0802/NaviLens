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

# ── Navigation-critical YOLO classes ─────────────────────────────────
NAV_CRITICAL_CLASSES = {"person", "chair", "dining table", "couch",
                        "dog", "cat", "bicycle", "car", "motorbike",
                        "fire hydrant", "stop sign", "bench"}

# Proximity thresholds on normalized depth (0=close, 1=far)
DANGER_CLOSE = 0.25      # depth < this in center = STOP
WARNING_CLOSE = 0.40     # depth < this = caution
FREE_THRESHOLD = 0.50    # depth > this = walkable


class NavigationEngine:
    """
    Real-time local navigation using depth + optional YOLO.

    Splits the frame into LEFT/CENTER/RIGHT zones, analyzes average depth
    in each, detects critical objects, and generates event-based guidance.
    Only announces changes — no TTS spam.
    """

    def __init__(self):
        self._depth = DepthEngine()
        self._detector = None       # lazy-loaded YOLOv8n
        self._detector_loaded = False

        # ── State tracking for event-based announcements ─────────────
        self._last_state: str = ""
        self._last_announce_time: float = 0.0
        self._min_announce_gap: float = 2.5   # seconds between announcements
        self._last_severity: str = ""         # "danger"/"blocked"/"caution"/"clear"
        self._clear_announced: bool = False    # True if we already said "path clear"

        # ── Frame dimensions ─────────────────────────────────────────
        self._frame_w: int = 640
        self._frame_h: int = 480

    def _ensure_detector(self):
        """Lazy-load YOLOv8n for navigation-critical objects."""
        if self._detector_loaded:
            return
        try:
            from ultralytics import YOLO
            self._detector = YOLO("yolov8n.pt")
            self._detector_loaded = True
            print("[Nav] YOLOv8n loaded for navigation objects.")
        except Exception as e:
            print(f"[Nav] YOLO load failed (navigation will use depth only): {e}")
            self._detector_loaded = True  # don't retry

    def set_frame_size(self, w: int, h: int):
        self._frame_w = w
        self._frame_h = h

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

        # 2) Split into LEFT / CENTER / RIGHT zones
        third = w // 3
        left_depth = depth[:, :third]
        center_depth = depth[:, third:2*third]
        right_depth = depth[:, 2*third:]

        # Average depth per zone (higher = more free space)
        left_avg = float(np.mean(left_depth))
        center_avg = float(np.mean(center_depth))
        right_avg = float(np.mean(right_depth))

        # Minimum depth in center (closest obstacle)
        center_min = float(np.percentile(center_depth, 10))  # 10th percentile = close stuff

        # 3) Detect critical objects with YOLO (optional)
        critical_objects = self._detect_critical_objects(frame)

        # 4) Build navigation state
        state, announcement = self._evaluate_state(
            left_avg, center_avg, right_avg, center_min, critical_objects
        )

        # 5) Draw zone indicators on depth visualization
        self._draw_zone_info(depth_vis, left_avg, center_avg, right_avg, critical_objects)

        return state, announcement, depth_vis

    def _detect_critical_objects(self, frame: np.ndarray) -> List[dict]:
        """Detect navigation-critical objects using YOLOv8n."""
        self._ensure_detector()
        if self._detector is None:
            return []

        try:
            results = self._detector(frame, conf=0.40, verbose=False, stream=False)[0]
        except Exception:
            return []

        objects = []
        for box in results.boxes:
            label = self._detector.model.names[int(box.cls)]
            if label not in NAV_CRITICAL_CLASSES:
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
        center_min: float,
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
        center_danger = center_min < DANGER_CLOSE

        # Determine current severity
        if center_danger:
            current_severity = "danger"
        elif center_status == "blocked":
            current_severity = "blocked"
        elif center_status == "caution":
            current_severity = "caution"
        else:
            current_severity = "clear"

        # Build state key
        obj_key = "|".join(f"{o['label']}@{o['zone']}" for o in objects[:2])
        state_key = f"L:{left_status},C:{center_status},R:{right_status},D:{center_danger},{obj_key}"

        # Skip if same state
        now = time.time()
        if state_key == self._last_state:
            return state_key, None
        if now - self._last_announce_time < self._min_announce_gap:
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
        if avg_depth >= FREE_THRESHOLD:
            return "clear"
        elif avg_depth >= WARNING_CLOSE:
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
            # Suggest safest direction
            if left_avg > right_avg and left_s != "blocked":
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
            if left_avg > right_avg and left_s != "blocked":
                parts.append("Try moving left.")
            elif right_s != "blocked":
                parts.append("Try moving right.")

        elif center_s == "caution":
            parts.append("Obstacle ahead, proceed with caution.")
            if left_s == "clear" and left_avg > center_avg + 0.1:
                parts.append("Left side is clearer.")
            elif right_s == "clear" and right_avg > center_avg + 0.1:
                parts.append("Right side is clearer.")

        # NOTE: "clear" case is handled in _evaluate_state, not here.
        # This method is only called for non-clear states.

        # Mention critical objects on sides
        for obj in objects:
            if obj["zone"] != "center" and obj["area_ratio"] > 0.03:
                parts.append(f"{obj['label']} on your {obj['zone']}.")

        return " ".join(parts)

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
        third = w // 3

        # Zone labels with status
        for i, (label, avg, x_start) in enumerate([
            ("LEFT", left_avg, 10),
            ("CENTER", center_avg, third + 10),
            ("RIGHT", right_avg, 2 * third + 10),
        ]):
            status = self._zone_status(avg)
            color = (0, 255, 0) if status == "clear" else \
                    (0, 200, 255) if status == "caution" else (0, 0, 255)
            cv2.putText(vis, f"{label}: {status}", (x_start, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(vis, f"d={avg:.2f}", (x_start, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Zone divider lines
        cv2.line(vis, (third, 0), (third, h), (255, 255, 255), 1)
        cv2.line(vis, (2 * third, 0), (2 * third, h), (255, 255, 255), 1)

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
