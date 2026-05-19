"""
path_planner.py — Free-Space Path Extraction

Radial ray-casting from the bottom-center of the depth frame.
Evaluates multiple candidate paths scored by distance, width,
center-bias, momentum, and doorway boost.

All local. No APIs.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from spatial_map import SpatialMap


NUM_RAYS = 11
RAY_SPAN_DEG = 70.0
OBSTACLE_DEPTH = 0.40
RAY_STEP_PX = 3

W_DISTANCE = 1.0
W_WIDTH = 0.6
W_CENTER = 0.3
W_MOMENTUM = 0.4
W_OPENING = 0.5
MOMENTUM_DECAY = 0.85

BLOCKING_LABELS = {
    "person", "chair", "dining table", "couch", "dog", "cat",
    "bicycle", "car", "motorbike", "bench",
}


@dataclass
class PlannedPath:
    """The recommended path for this frame."""
    angle: float
    distance: float
    width: float
    confidence: float
    context: str
    obstacle_map: np.ndarray
    alternatives: List[dict] = field(default_factory=list)


class PathPlanner:
    """Radial ray-casting path planner."""

    def __init__(self):
        self._prev_heading: float = 0.0

    def reset(self) -> None:
        """Clear heading momentum for a new navigation session."""
        self._prev_heading = 0.0

    def plan(
        self,
        depth: np.ndarray,
        spatial_map: SpatialMap,
        objects: list,
        current_heading: float = 0.0,
    ) -> PlannedPath:
        h, w = depth.shape

        # Build object mask
        obj_mask = np.zeros((h, w), dtype=np.float32)
        for obj in objects:
            if obj["label"] not in BLOCKING_LABELS:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mid_y = (y1 + y2) // 2
            obj_mask[mid_y:y2, x1:x2] = 1.0

        traversable = ((depth > OBSTACLE_DEPTH) & (obj_mask < 0.5)).astype(np.float32)

        origin_x = w // 2
        origin_y = h - 1
        angles = np.linspace(-RAY_SPAN_DEG / 2, RAY_SPAN_DEG / 2, NUM_RAYS)

        ray_results = []
        for angle_deg in angles:
            dist = self._cast_ray(traversable, origin_x, origin_y, angle_deg, h, w)
            ray_results.append({"angle": float(angle_deg), "distance": dist})

        # Compute width for each ray
        for i, ray in enumerate(ray_results):
            width = 1
            threshold = ray["distance"] * 0.5
            for j in range(i - 1, -1, -1):
                if ray_results[j]["distance"] >= threshold:
                    width += 1
                else:
                    break
            for j in range(i + 1, len(ray_results)):
                if ray_results[j]["distance"] >= threshold:
                    width += 1
                else:
                    break
            ray["width"] = width / NUM_RAYS

        openings = spatial_map.get_openings()

        # Score each ray
        best_score = -1.0
        best_ray = ray_results[NUM_RAYS // 2]

        for ray in ray_results:
            s_dist = ray["distance"]
            s_width = ray["width"]
            s_center = float(np.exp(-0.5 * (ray["angle"] / 25.0) ** 2))
            heading_diff = abs(ray["angle"] - self._prev_heading)
            s_momentum = float(np.exp(-0.5 * (heading_diff / 20.0) ** 2))

            s_opening = 0.0
            for opening in openings:
                opening_angle = (opening.rel_x - 0.5) * RAY_SPAN_DEG * 2
                if abs(ray["angle"] - opening_angle) < 15:
                    s_opening = max(s_opening, opening.confidence * opening.depth)

            score = (
                W_DISTANCE * s_dist + W_WIDTH * s_width
                + W_CENTER * s_center + W_MOMENTUM * s_momentum
                + W_OPENING * s_opening
            )
            ray["score"] = score
            if score > best_score:
                best_score = score
                best_ray = ray

        self._prev_heading = (
            MOMENTUM_DECAY * self._prev_heading
            + (1 - MOMENTUM_DECAY) * best_ray["angle"]
        )

        # Obstacle proximity map
        floor_region = depth[int(h * 0.55):, :]
        obstacle_map = (
            np.min(floor_region, axis=0) if floor_region.size > 0
            else np.ones(w, dtype=np.float32)
        )

        # Alternatives
        sorted_rays = sorted(ray_results, key=lambda r: -r.get("score", 0))
        alternatives = [
            {"angle": r["angle"], "distance": r["distance"]}
            for r in sorted_rays[1:4]
        ]

        # Context
        room_ctx = spatial_map.get_room_context()
        if openings and any(
            abs(best_ray["angle"] - (o.rel_x - 0.5) * RAY_SPAN_DEG * 2) < 20
            for o in openings
        ):
            context = "doorway_approach"
        else:
            context = room_ctx

        confidence = min(1.0, best_ray["distance"] * 0.5 + best_ray["width"] * 0.5)

        return PlannedPath(
            angle=best_ray["angle"],
            distance=best_ray["distance"],
            width=best_ray["width"],
            confidence=confidence,
            context=context,
            obstacle_map=obstacle_map,
            alternatives=alternatives,
        )

    @staticmethod
    def _cast_ray(traversable, origin_x, origin_y, angle_deg, h, w):
        angle_rad = np.deg2rad(-90 + angle_deg)
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        max_steps = int(np.sqrt(h * h + w * w) / RAY_STEP_PX)
        steps_clear = 0
        for step in range(1, max_steps + 1):
            px = int(origin_x + dx * step * RAY_STEP_PX)
            py = int(origin_y + dy * step * RAY_STEP_PX)
            if px < 0 or px >= w or py < 0 or py >= h:
                break
            if traversable[py, px] < 0.5:
                break
            steps_clear = step
        return steps_clear / max(max_steps, 1)
