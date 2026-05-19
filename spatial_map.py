"""
spatial_map.py — Room-Level Spatial Model

Maintains a persistent, temporally-smoothed spatial understanding of the
environment by dividing each depth frame into a grid and tracking:
  - Wall positions (left / right / front)
  - Openings / gaps in walls (potential doorways or corridors)
  - Room context classification (corridor, open room, narrow passage, near wall)
  - Floor depth profile for path planning

All local. No APIs.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

GRID_COLS = 5       # horizontal divisions
GRID_ROWS = 3       # vertical: top (ceiling), mid (walls/openings), bottom (floor)

# Exponential moving average weight for temporal smoothing
EMA_ALPHA = 0.35

# Wall detection: a region counts as "wall" if its average depth is below this
WALL_DEPTH_THRESHOLD = 0.35

# Opening detection: depth jump required between adjacent mid-row cells
OPENING_DEPTH_JUMP = 0.20

# Minimum opening width (fraction of frame width)
MIN_OPENING_WIDTH = 0.08

# Room context thresholds
CORRIDOR_ASPECT = 0.40   # if both side walls present and narrow
NEAR_WALL_DEPTH = 0.30   # front wall depth that triggers "near wall"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Opening:
    """A detected gap/opening in the walls (potential doorway or corridor)."""
    rel_x: float          # normalised centre x (0..1)
    width: float          # normalised width (0..1)
    depth: float          # average depth at the opening (higher = further = deeper opening)
    direction: str        # "left", "center", "right"
    confidence: float     # 0..1, based on depth contrast and persistence


@dataclass
class WallInfo:
    """Wall presence information."""
    left: bool = False
    right: bool = False
    front_depth: float = 1.0   # 0 = wall right in front, 1 = no front wall


# ─────────────────────────────────────────────────────────────────────────────
# SpatialMap
# ─────────────────────────────────────────────────────────────────────────────

class SpatialMap:
    """
    Persistent room-level spatial model.

    Call ``update(depth)`` every frame. Query walls, openings, room context
    at any time — results are temporally smoothed.
    """

    def __init__(self):
        # Smoothed grid: GRID_ROWS × GRID_COLS of average depth values
        self._grid: Optional[np.ndarray] = None          # (ROWS, COLS)
        self._grid_var: Optional[np.ndarray] = None       # variance per cell
        self._frame_count: int = 0

        # Cached results (updated each frame)
        self._walls = WallInfo()
        self._openings: List[Opening] = []
        self._floor_profile: Optional[np.ndarray] = None  # 1-D, one value per column pixel
        self._room_context: str = "unknown"
        self._safest_angle: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, depth: np.ndarray) -> None:
        """
        Feed a new depth frame (H×W, float32, 0..1, higher = further).
        Updates the internal spatial model with temporal smoothing.
        """
        h, w = depth.shape

        # Build raw grid for this frame
        raw_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        raw_var = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)

        row_edges = [0, int(h * 0.25), int(h * 0.55), h]
        col_edges = [int(w * c / GRID_COLS) for c in range(GRID_COLS + 1)]

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                cell = depth[row_edges[r]:row_edges[r + 1],
                             col_edges[c]:col_edges[c + 1]]
                if cell.size > 0:
                    raw_grid[r, c] = float(np.mean(cell))
                    raw_var[r, c] = float(np.var(cell))

        # Temporal smoothing (EMA)
        if self._grid is None:
            self._grid = raw_grid.copy()
            self._grid_var = raw_var.copy()
        else:
            self._grid = EMA_ALPHA * raw_grid + (1 - EMA_ALPHA) * self._grid
            self._grid_var = EMA_ALPHA * raw_var + (1 - EMA_ALPHA) * self._grid_var

        self._frame_count += 1

        # Derive spatial features
        self._detect_walls()
        self._detect_openings(depth)
        self._compute_floor_profile(depth)
        self._classify_room_context()
        self._compute_safest_direction(depth)

    def get_walls(self) -> WallInfo:
        """Return current wall detection results."""
        return self._walls

    def get_openings(self) -> List[Opening]:
        """Return detected openings (potential doorways/corridors)."""
        return self._openings

    def get_floor_profile(self) -> Optional[np.ndarray]:
        """Return 1-D floor depth profile across the frame width."""
        return self._floor_profile

    def get_room_context(self) -> str:
        """Return room context: 'corridor', 'open_room', 'narrow_passage', 'near_wall', 'unknown'."""
        return self._room_context

    def get_safest_direction(self) -> float:
        """Return the safest steering angle (degrees, −=left, +=right)."""
        return self._safest_angle

    def get_grid(self) -> Optional[np.ndarray]:
        """Return the smoothed depth grid (ROWS × COLS) for debugging."""
        return self._grid

    def reset(self) -> None:
        """Clear temporal state for a new navigation session."""
        self._grid = None
        self._grid_var = None
        self._frame_count = 0
        self._walls = WallInfo()
        self._openings = []
        self._floor_profile = None
        self._room_context = "unknown"
        self._safest_angle = 0.0

    # ── Wall detection ────────────────────────────────────────────────────────

    def _detect_walls(self) -> None:
        if self._grid is None:
            return

        mid_row = self._grid[1, :]    # mid row = wall/opening region

        # Left wall: leftmost column of mid row
        self._walls.left = bool(mid_row[0] < WALL_DEPTH_THRESHOLD)

        # Right wall: rightmost column of mid row
        self._walls.right = bool(mid_row[-1] < WALL_DEPTH_THRESHOLD)

        # Front wall: centre column(s) of mid row
        center_idx = GRID_COLS // 2
        self._walls.front_depth = float(mid_row[center_idx])

    # ── Opening / gap detection ───────────────────────────────────────────────

    def _detect_openings(self, depth: np.ndarray) -> None:
        """Scan mid-row for depth jumps that indicate doorways or corridor openings."""
        if self._grid is None:
            self._openings = []
            return

        h, w = depth.shape
        mid_row = self._grid[1, :]  # smoothed mid-row depths
        openings: List[Opening] = []

        for c in range(GRID_COLS):
            cell_depth = mid_row[c]

            # Check if this column is significantly deeper than its neighbours
            left_depth = mid_row[c - 1] if c > 0 else 0.0
            right_depth = mid_row[c + 1] if c < GRID_COLS - 1 else 0.0

            # An opening: this cell is deeper than at least one neighbour by OPENING_DEPTH_JUMP
            left_contrast = cell_depth - left_depth
            right_contrast = cell_depth - right_depth

            if left_contrast > OPENING_DEPTH_JUMP or right_contrast > OPENING_DEPTH_JUMP:
                # This cell looks like an opening
                rel_x = (c + 0.5) / GRID_COLS
                width_est = 1.0 / GRID_COLS  # approximate

                # Refine width by scanning the actual depth column
                col_start = int(w * c / GRID_COLS)
                col_end = int(w * (c + 1) / GRID_COLS)
                mid_strip = depth[int(h * 0.25):int(h * 0.55), col_start:col_end]

                # Check if the opening extends beyond this grid cell
                if c > 0:
                    prev_start = int(w * (c - 1) / GRID_COLS)
                    prev_strip = depth[int(h * 0.25):int(h * 0.55), prev_start:col_start]
                    if prev_strip.size > 0 and np.mean(prev_strip) > WALL_DEPTH_THRESHOLD:
                        width_est += 1.0 / GRID_COLS
                        rel_x -= 0.5 / GRID_COLS

                if c < GRID_COLS - 1:
                    next_end = int(w * (c + 2) / GRID_COLS)
                    next_strip = depth[int(h * 0.25):int(h * 0.55), col_end:next_end]
                    if next_strip.size > 0 and np.mean(next_strip) > WALL_DEPTH_THRESHOLD:
                        width_est += 1.0 / GRID_COLS

                # Direction label
                if rel_x < 0.35:
                    direction = "left"
                elif rel_x > 0.65:
                    direction = "right"
                else:
                    direction = "center"

                contrast = max(left_contrast, right_contrast)
                confidence = min(1.0, contrast / 0.4)

                if width_est >= MIN_OPENING_WIDTH:
                    openings.append(Opening(
                        rel_x=rel_x,
                        width=width_est,
                        depth=float(cell_depth),
                        direction=direction,
                        confidence=confidence,
                    ))

        # Sort by confidence descending
        openings.sort(key=lambda o: -o.confidence)
        self._openings = openings[:3]   # keep top 3

    # ── Floor profile ─────────────────────────────────────────────────────────

    def _compute_floor_profile(self, depth: np.ndarray) -> None:
        """Compute a 1-D depth profile across the floor region."""
        h, w = depth.shape
        floor = depth[int(h * 0.60):, :]   # bottom 40%

        if floor.size == 0:
            self._floor_profile = np.ones(w, dtype=np.float32)
            return

        # Column-wise mean depth in the floor region
        profile = np.mean(floor, axis=0).astype(np.float32)

        # Light smoothing
        if len(profile) > 15:
            kernel = np.ones(15) / 15
            profile = np.convolve(profile, kernel, mode="same")

        self._floor_profile = profile

    # ── Room context classification ───────────────────────────────────────────

    def _classify_room_context(self) -> None:
        if self._grid is None:
            self._room_context = "unknown"
            return

        walls = self._walls

        # Near wall: front wall is very close
        if walls.front_depth < NEAR_WALL_DEPTH:
            self._room_context = "near_wall"
            return

        # Corridor: both side walls, moderate front depth
        if walls.left and walls.right:
            mid_center = self._grid[1, GRID_COLS // 2]
            if mid_center > 0.5:
                self._room_context = "corridor"
            else:
                self._room_context = "narrow_passage"
            return

        # One wall: passage along a wall
        if walls.left or walls.right:
            self._room_context = "along_wall"
            return

        # No walls detected: open room
        self._room_context = "open_room"

    # ── Safest direction ──────────────────────────────────────────────────────

    def _compute_safest_direction(self, depth: np.ndarray) -> None:
        """Compute the safest heading angle based on floor profile."""
        if self._floor_profile is None:
            self._safest_angle = 0.0
            return

        profile = self._floor_profile
        w = len(profile)
        if w == 0:
            self._safest_angle = 0.0
            return

        # Find the column with maximum depth (most open space)
        # Apply a gentle Gaussian centered at the middle for center-bias
        center = w // 2
        x = np.arange(w, dtype=np.float32)
        center_weight = np.exp(-0.5 * ((x - center) / (w * 0.4)) ** 2)

        weighted = profile * center_weight
        best_x = int(np.argmax(weighted))

        # Convert to angle (−45° to +45°)
        offset = best_x - center
        self._safest_angle = float(offset / center * 45.0)
