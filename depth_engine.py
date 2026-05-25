"""
depth_engine.py — Monocular depth estimation (MiDaS / DPT).

Profiles (see config.py):
  small  — MiDaS_small, fastest
  hybrid — DPT_Hybrid, recommended balance
  large  — DPT_Large, best quality
"""

import threading

import cv2
import numpy as np
import torch

from config import DEPTH_MODELS, DEPTH_PROFILE


class DepthEngine:
    """Lazy-loaded depth estimator. Thread-safe."""

    def __init__(self, profile: str = None):
        self._profile = profile or DEPTH_PROFILE
        if self._profile not in DEPTH_MODELS:
            print(f"[Depth] Unknown profile '{self._profile}', using hybrid")
            self._profile = "hybrid"

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

            model_name, transform_name = DEPTH_MODELS[self._profile]
            print(f"[Depth] Loading {model_name} ({self._profile})...")

            self._model = torch.hub.load(
                "intel-isl/MiDaS", model_name, trust_repo=True
            )
            self._model.to(self._device).eval()

            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self._transform = getattr(midas_transforms, transform_name)
            self._initialized = True
            print(f"[Depth] Ready — {model_name} on {self._device}")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        self._ensure_loaded()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)

        depth = 1.0 - depth
        return depth.astype(np.float32)

    def depth_to_colormap(self, depth: np.ndarray) -> np.ndarray:
        depth_uint8 = (depth * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_MAGMA)
