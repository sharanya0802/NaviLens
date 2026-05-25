"""
navigation.py — Spatial navigation pipeline for blind mobility assistance.

Pipeline (all local, no cloud APIs):
  1. MiDaS depth → SpatialMap (walls, openings, room context)
  2. YOLO → semantic obstacles
  3. PathPlanner → radial free-space path with momentum
  4. GuidanceComposer / ExitGuidance → natural-language turn-by-turn speech
  5. DoorTracker → clock-face exit guidance when leaving a room

Visualization shows planned path, openings, and live guidance — not a raw heatmap.
"""

import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import YOLO_MODEL
from depth_engine import DepthEngine
from exit_detector import DoorTracker
from exit_navigation import ExitGuidance
from guidance_composer import GuidanceComposer
from path_planner import PathPlanner, PlannedPath
from spatial_map import SpatialMap

# ───────────────────────────────────────────────────────────────
# Navigation-critical objects (YOLO)
# ───────────────────────────────────────────────────────────────
NAV_CRITICAL_CLASSES = {
    "person", "chair", "dining table", "couch", "dog", "cat",
    "bicycle", "car", "motorbike", "fire hydrant", "stop sign",
    "bench", "door", "stairs",
}

WARNING_CLOSE = 0.45


class NavigationEngine:
    """
    Full spatial navigation engine: depth + spatial map + path planning
    + mobility-assistant speech. Supports standard walk guidance and
    dedicated exit-seeking mode.
    """

    def __init__(self):
        self._depth = DepthEngine()
        self._spatial = SpatialMap()
        self._planner = PathPlanner()
        self._composer = GuidanceComposer()
        self._door_tracker = DoorTracker()
        self._exit_guidance = ExitGuidance()

        self._detector = None
        self._detector_loaded = False

        self._frame_w = 640
        self._frame_h = 480
        self._exit_mode = False

        self._last_state = ""
        self._last_guidance = ""
        self._guidance_history = deque(maxlen=3)

    # ───────────────────────────────────────────────────────────
    # YOLO
    # ───────────────────────────────────────────────────────────
    def _ensure_detector(self):
        if self._detector_loaded:
            return
        try:
            from ultralytics import YOLO
            self._detector = YOLO(YOLO_MODEL)
            print(f"[Nav] YOLO loaded: {YOLO_MODEL}")
        except Exception as e:
            print(f"[Nav] YOLO failed: {e}")
            self._detector = None
        self._detector_loaded = True

    def set_frame_size(self, w: int, h: int):
        self._frame_w = w
        self._frame_h = h
        self._door_tracker.set_frame_size(w, h)

    # ───────────────────────────────────────────────────────────
    # EXIT MODE
    # ───────────────────────────────────────────────────────────
    def start_exit_mode(self):
        self._exit_mode = True
        self._door_tracker.reset()
        self._exit_guidance.activate()
        self._composer.reset()
        print("[Nav] Exit-seeking mode ON")

    def stop_exit_mode(self):
        self._exit_mode = False
        self._exit_guidance.deactivate()
        self._door_tracker.reset()
        print("[Nav] Exit-seeking mode OFF")

    # ───────────────────────────────────────────────────────────
    # MAIN ANALYSIS
    # ───────────────────────────────────────────────────────────
    def analyze(
        self,
        frame: np.ndarray,
    ) -> Tuple[str, Optional[str], np.ndarray]:
        """
        Run one navigation frame.

        Returns:
            state_key: compact state fingerprint (dedup / logging)
            announcement: speech string or None if silent this frame
            nav_vis: BGR visualization for the guidance window
        """
        depth = self._depth.estimate(frame)

        # Room-level spatial model
        self._spatial.update(depth)
        openings = self._spatial.get_openings()

        objects = self._detect_critical_objects(frame)
        head_obstacle = self._detect_head_obstacles(depth)
        obstacle_text = self._build_obstacle_text(objects, head_obstacle)

        # Free-space path planning
        path = self._planner.plan(depth, self._spatial, objects)

        # Exit tracking (YOLO door + depth openings fallback)
        exit_state = None
        if self._exit_mode:
            exit_state = self._door_tracker.update(objects, depth, openings)

        # Compose spoken guidance
        announcement = self._compose_speech(
            path, objects, head_obstacle, exit_state, obstacle_text
        )

        if announcement:
            self._last_guidance = announcement
            self._guidance_history.append(announcement)

        state_key = self._build_state_key(path, exit_state, announcement)

        nav_vis = self._render_guidance_view(
            frame=frame,
            depth=depth,
            path=path,
            objects=objects,
            exit_state=exit_state,
            guidance=announcement or self._last_guidance,
        )

        return state_key, announcement, nav_vis

    def _compose_speech(
        self,
        path: PlannedPath,
        objects: list,
        head_obstacle: bool,
        exit_state,
        obstacle_text: str,
    ) -> Optional[str]:
        """Exit mode prefers ExitGuidance; both modes fall back to GuidanceComposer."""
        if self._exit_mode:
            speech = self._exit_guidance.update(exit_state, obstacle_text)
            if speech:
                return speech

        return self._composer.compose(
            path=path,
            spatial=self._spatial,
            objects=objects,
            head_obstacle=head_obstacle,
            exit_state=exit_state,
        )

    @staticmethod
    def _build_state_key(path, exit_state, announcement) -> str:
        parts = [
            f"a{path.angle:.0f}",
            f"d{path.distance:.2f}",
            f"c{path.context}",
        ]
        if exit_state:
            parts.append(f"ex{exit_state.clock_position}")
            parts.append(f"m{exit_state.distance_m:.1f}")
        if announcement:
            parts.append(announcement[:40])
        return "|".join(parts)

    # ───────────────────────────────────────────────────────────
    # OBSTACLE TEXT
    # ───────────────────────────────────────────────────────────
    def _build_obstacle_text(self, objects: list, low_clearance: bool) -> str:
        parts = []
        if low_clearance:
            parts.append("low obstacle overhead")
        for obj in objects:
            if obj["area_ratio"] < 0.04:
                continue
            label = obj["label"]
            zone = obj["zone"]
            if label == "door":
                continue
            if label == "stairs":
                parts.append("stairs ahead")
            elif zone == "center":
                parts.append(f"{label} directly ahead")
            else:
                parts.append(f"{label} on your {zone}")
        return ", ".join(parts)

    def _detect_head_obstacles(self, depth: np.ndarray) -> bool:
        h, w = depth.shape
        head_region = depth[: int(h * 0.25), :]
        close_ratio = np.mean(head_region < WARNING_CLOSE)
        return close_ratio > 0.18

    # ───────────────────────────────────────────────────────────
    # YOLO OBJECTS
    # ───────────────────────────────────────────────────────────
    def _detect_critical_objects(self, frame: np.ndarray) -> List[dict]:
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
            area_ratio = (x2 - x1) * (y2 - y1) / (self._frame_w * self._frame_h)
            rel_x = cx / self._frame_w

            if rel_x < 0.2:
                zone = "hard left"
            elif rel_x < 0.4:
                zone = "left"
            elif rel_x > 0.8:
                zone = "hard right"
            elif rel_x > 0.6:
                zone = "right"
            else:
                zone = "center"

            objects.append({
                "label": label,
                "zone": zone,
                "area_ratio": area_ratio,
                "bbox": (x1, y1, x2, y2),
            })

        objects.sort(key=lambda o: -o["area_ratio"])
        return objects[:5]

    # ───────────────────────────────────────────────────────────
    # VISUALIZATION — mobility guidance HUD (not raw depth heatmap)
    # ───────────────────────────────────────────────────────────
    def _render_guidance_view(
        self,
        frame: np.ndarray,
        depth: np.ndarray,
        path: PlannedPath,
        objects: list,
        exit_state,
        guidance: str,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        vis = frame.copy()

        # Depth tint on lower floor region only (subtle, not dominant)
        floor_y = int(h * 0.50)
        depth_tint = self._depth.depth_to_colormap(depth)
        floor_overlay = vis[floor_y:, :].astype(np.float32)
        tint_region = depth_tint[floor_y:, :].astype(np.float32)
        vis[floor_y:, :] = cv2.addWeighted(
            floor_overlay, 0.55, tint_region, 0.45, 0
        ).astype(np.uint8)

        origin = (w // 2, h - 10)
        self._draw_path_fan(vis, path, origin)
        self._draw_direction_arrow(vis, path.angle, origin)
        self._draw_openings(vis, depth)
        self._draw_walls(vis)
        self._draw_objects(vis, objects, exit_state)
        self._draw_hud(vis, path, guidance)

        return vis

    def _draw_path_fan(self, vis: np.ndarray, path: PlannedPath, origin: Tuple[int, int]):
        """Radial rays showing evaluated paths; best path highlighted."""
        h, w = vis.shape[:2]
        ox, oy = origin
        span = 70

        for alt in path.alternatives:
            angle = alt["angle"]
            dist = alt["distance"]
            length = int(dist * min(h, w) * 0.35)
            end = self._ray_endpoint(ox, oy, angle, length, w, h)
            cv2.line(vis, origin, end, (80, 80, 80), 1, cv2.LINE_AA)

        best_len = int(path.distance * min(h, w) * 0.45)
        best_end = self._ray_endpoint(ox, oy, path.angle, best_len, w, h)
        color = (0, 255, 80) if path.distance > 0.12 else (0, 140, 255)
        cv2.line(vis, origin, best_end, color, 3, cv2.LINE_AA)
        cv2.circle(vis, best_end, 8, color, -1, cv2.LINE_AA)

    @staticmethod
    def _ray_endpoint(ox, oy, angle_deg, length, w, h):
        rad = np.deg2rad(-90 + angle_deg)
        ex = int(np.clip(ox + np.cos(rad) * length, 0, w - 1))
        ey = int(np.clip(oy + np.sin(rad) * length, 0, h - 1))
        return (ex, ey)

    def _draw_direction_arrow(self, vis: np.ndarray, angle: float, origin: Tuple[int, int]):
        """Large compass-style turn indicator at top-centre."""
        h, w = vis.shape[:2]
        cx, cy = w // 2, 55
        radius = 38

        cv2.circle(vis, (cx, cy), radius, (40, 40, 40), -1)
        cv2.circle(vis, (cx, cy), radius, (200, 200, 200), 2)

        # Needle: 0° = up (straight)
        rad = np.deg2rad(angle)
        tip_x = int(cx + radius * 0.75 * np.sin(rad))
        tip_y = int(cy - radius * 0.75 * np.cos(rad))
        cv2.arrowedLine(vis, (cx, cy), (tip_x, tip_y), (0, 255, 120), 3, tipLength=0.35)

        label = "STRAIGHT"
        if angle < -20:
            label = "TURN LEFT"
        elif angle < -8:
            label = "VEER LEFT"
        elif angle > 20:
            label = "TURN RIGHT"
        elif angle > 8:
            label = "VEER RIGHT"

        cv2.putText(
            vis, label, (cx - 55, cy + radius + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2,
        )

    def _draw_openings(self, vis: np.ndarray, depth: np.ndarray):
        h, w = vis.shape[:2]
        for opening in self._spatial.get_openings():
            x1 = int((opening.rel_x - opening.width / 2) * w)
            x2 = int((opening.rel_x + opening.width / 2) * w)
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = int(h * 0.20), int(h * 0.58)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 255), 2)
            cv2.putText(
                vis, "OPENING", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 255), 1,
            )

    def _draw_walls(self, vis: np.ndarray):
        h, w = vis.shape[:2]
        walls = self._spatial.get_walls()
        bar_h = int(h * 0.35)
        if walls.left:
            cv2.rectangle(vis, (0, int(h * 0.25)), (18, int(h * 0.25) + bar_h), (0, 0, 220), -1)
            cv2.putText(vis, "WALL", (2, int(h * 0.25) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if walls.right:
            cv2.rectangle(vis, (w - 18, int(h * 0.25)), (w, int(h * 0.25) + bar_h), (0, 0, 220), -1)
            cv2.putText(vis, "WALL", (w - 42, int(h * 0.25) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _draw_objects(self, vis: np.ndarray, objects: list, exit_state):
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            is_door = obj["label"] == "door"
            color = (255, 0, 255) if is_door else (0, 200, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            tag = obj["label"]
            if is_door and self._exit_mode:
                tag = "EXIT / DOOR"
            cv2.putText(vis, tag, (x1, max(y1 - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if exit_state and self._exit_mode:
            h, w = vis.shape[:2]
            rel_x = (exit_state.angle_deg / 90.0) + 0.5
            door_x = int(np.clip(rel_x * w, 0, w - 1))
            cv2.line(vis, (door_x, 0), (door_x, h), (255, 0, 255), 2)
            info = f"{exit_state.clock_position}  ~{exit_state.distance_steps} steps"
            cv2.putText(
                vis, info, (max(door_x - 80, 5), h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
            )

    def _draw_hud(self, vis: np.ndarray, path: PlannedPath, guidance: str):
        h, w = vis.shape[:2]
        mode = "EXIT GUIDANCE" if self._exit_mode else "NAVIGATION"
        mode_color = (255, 0, 255) if self._exit_mode else (0, 220, 255)
        room = self._spatial.get_room_context().replace("_", " ").upper()

        cv2.rectangle(vis, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(
            vis, f"{mode}  |  {room}  |  path {path.angle:+.0f}°",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2,
        )

        # Guidance banner at bottom
        banner_h = 72
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 0), -1)
        vis[:] = cv2.addWeighted(overlay, 0.72, vis, 0.28, 0)

        if not guidance:
            guidance = "Listening to surroundings..."
        self._draw_wrapped_text(vis, guidance, (10, h - banner_h + 22), w - 20, 0.52, (255, 255, 255))

    @staticmethod
    def _draw_wrapped_text(img, text, origin, max_width, scale, color):
        x, y = origin
        words = text.split()
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            (tw, _), _ = cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            if tw > max_width and line:
                cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
                line = word
                y += int(22 * scale / 0.5)
            else:
                line = test
        if line:
            cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    # ───────────────────────────────────────────────────────────
    # RESET
    # ───────────────────────────────────────────────────────────
    def reset_state(self):
        self._last_state = ""
        self._last_guidance = ""
        self._guidance_history.clear()
        self._spatial.reset()
        self._planner.reset()
        self._composer.reset()
        self._door_tracker.reset()
        self._exit_guidance.deactivate()
        self._exit_mode = False
