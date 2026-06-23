"""
depth_engine.py — MiDaS monocular depth estimation.

Generates relative depth maps from single camera frames.
Higher values = closer to camera, lower values = further away.
Used by the navigation pipeline for obstacle detection and free-space analysis.
"""

import threading
import cv2
import numpy as np
import torch


class DepthEngine:
    """
    MiDaS small depth estimator. Lazy-loads on first use (~40 MB download).
    Thread-safe, runs on CPU by default.
    """

    def __init__(self):
        self._model = None
        self._transform = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.Lock()
        self._initialized = False

    def _ensure_loaded(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            print("[Depth] Loading MiDaS small model...")
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self._model.to(self._device).eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self._transform = midas_transforms.small_transform
            self._initialized = True
            print(f"[Depth] MiDaS ready (device={self._device}).")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from a BGR frame.

        Args:
            frame: BGR image (H, W, 3) uint8.

        Returns:
            Depth map (H, W) float32, normalized 0-1.
            Higher values = closer to camera.
        """
        self._ensure_loaded()

        # MiDaS expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            # Interpolate to original frame size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # MiDaS outputs inverse depth (closer = higher value).
        # Normalize to 0-1 range, keeping higher = closer.
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    def depth_to_colormap(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth map to a color visualization (for display)."""
        depth_uint8 = (depth * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
