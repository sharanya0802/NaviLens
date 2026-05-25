"""
local_vision.py — Local vision helpers (no cloud APIs).

Used for:
  - Scene / surroundings description (YOLO + spatial reasoning)
  - Cropping the object the user is pointing at (for CLIP / OCR / optional Gemini)
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import YOLO_MODEL
from detection import Detection, OBJECT_CLASSES, ObjectDetector
from spatial import SpatialReasoner

_detector: Optional[ObjectDetector] = None
_spatial = SpatialReasoner()


def _get_detector() -> ObjectDetector:
    global _detector
    if _detector is None:
        self_conf = 0.38 if "n.pt" in YOLO_MODEL else 0.42
        _detector = ObjectDetector(model_name=YOLO_MODEL, conf=self_conf)
        print(f"[Vision] YOLO loaded: {YOLO_MODEL}")
    return _detector


def set_frame_size(w: int, h: int) -> None:
    _spatial.set_frame_size(w, h)
    det = _get_detector()
    det.set_frame_area(w, h)


def detect_objects(frame: np.ndarray) -> List[Detection]:
    return _get_detector().detect(frame)


def describe_scene(frame: np.ndarray) -> str:
    """Full local scene description — no API."""
    detections = detect_objects(frame)
    return _spatial.generate_full_description(detections)


def describe_ahead(frame: np.ndarray) -> str:
    """Short path / obstacle summary — no API."""
    detections = detect_objects(frame)
    urgent = _spatial.get_urgent_warnings(detections)
    if urgent:
        return urgent
    nav_text = _spatial.generate_navigation_instruction(detections)
    return nav_text or "Path appears clear ahead."


def get_product_crop(
    frame: np.ndarray,
    detections: Optional[List[Detection]] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], List[Detection]]:
    """
    Pick the best crop for product identification.

    Prefers identifiable COCO objects near frame centre; otherwise uses a
    central region (typical "holding item in front of camera" pose).
    """
    h, w = frame.shape[:2]
    if detections is None:
        detections = detect_objects(frame)

    cx_frame, cy_frame = w // 2, h // 2

    candidates = [
        d for d in detections
        if d.label in OBJECT_CLASSES or d.area_ratio >= 0.02
    ]
    if not candidates:
        candidates = detections

    if candidates:
        def score(d: Detection) -> float:
            dist = abs(d.center_x - cx_frame) / max(w, 1)
            return d.area_ratio * 2.0 - dist

        best = max(candidates, key=score)
        x1, y1, x2, y2 = best.bbox
        pad = 12
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2), detections

    # Fallback: central 55% of frame
    mx, my = int(w * 0.225), int(h * 0.225)
    x2, y2 = w - mx, h - my
    crop = frame[my:y2, mx:x2]
    return crop, (mx, my, x2, y2), detections
