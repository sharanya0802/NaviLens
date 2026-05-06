"""
detection.py — YOLOv10 Object Detector
Uses Ultralytics YOLOv10 (NMS-free). Auto-downloads model weights on first run.
"""

from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics not installed. Run: pip install ultralytics"
    )


@dataclass
class Detection:
    """Single detected object."""
    label: str          # e.g. "person", "chair"
    confidence: float   # 0.0 – 1.0
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)
    center_x: int       # bbox center X
    center_y: int       # bbox center Y
    area_ratio: float   # fraction of frame area (proxy for proximity)

    @property
    def is_person(self) -> bool:
        """True if this detection is a person."""
        return self.label == "person"


# Objects that are navigation-critical get priority
PRIORITY_CLASSES = {
    "person", "bicycle", "car", "motorbike", "bus", "truck",
    "traffic light", "stop sign", "stairs", "door", "chair",
    "couch", "dining table", "dog", "cat", "fire hydrant"
}

# COCO classes considered as identifiable objects (OCR + smart ID)
OBJECT_CLASSES = {
    "bottle", "cup", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "spoon", "fork", "knife", "wine glass", "cell phone",
    "laptop", "mouse", "remote", "keyboard", "tv", "microwave", "oven",
    "toaster", "sink", "refrigerator", "handbag", "suitcase", "umbrella",
    "backpack", "tie", "sports ball", "frisbee", "skis", "snowboard",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket",
}


class ObjectDetector:
    """
    Wraps YOLOv10 for NaviLens.
    
    Model options (auto-downloaded):
        yolov10n.pt — nano   (fastest,  ~2.3M params) ← recommended for RPi
        yolov10s.pt — small  (~7.2M  params)
        yolov10m.pt — medium (~15.4M params)
        yolov10b.pt — balanced (~19.1M params)
        yolov10l.pt — large  (~24.4M params)
        yolov10x.pt — xlarge (~29.5M params, most accurate)
    """

    def __init__(self, model_name: str = "yolov10s.pt", conf: float = 0.45):
        self.model = YOLO(model_name)
        self.conf = conf
        self._frame_area: int = 1  # updated once frame size is known

        # Colour palette for drawing (BGR)
        self._colors = {
            "priority": (0, 60, 255),    # red-ish for urgent
            "normal":   (0, 200, 80),    # green for normal
        }

    def set_frame_area(self, w: int, h: int):
        self._frame_area = w * h

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame.
        Returns a list of Detection objects, sorted by proximity (area_ratio desc).
        """
        if self._frame_area == 1:
            h, w = frame.shape[:2]
            self._frame_area = w * h

        results = self.model(
            frame,
            conf=self.conf,
            verbose=False,        # suppress per-frame console spam
            stream=False,
        )[0]

        detections: List[Detection] = []

        for box in results.boxes:
            label = self.model.names[int(box.cls)]

            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            area_ratio = area / self._frame_area

            detections.append(Detection(
                label=label,
                confidence=conf,
                bbox=(x1, y1, x2, y2),
                center_x=cx,
                center_y=cy,
                area_ratio=area_ratio,
            ))

        # Sort: priority classes first, then by area (larger = closer)
        detections.sort(
            key=lambda d: (d.label not in PRIORITY_CLASSES, -d.area_ratio)
        )
        return detections

    def draw(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels onto the frame."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = (
                self._colors["priority"]
                if det.label in PRIORITY_CLASSES
                else self._colors["normal"]
            )
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label_text = f"{det.label} {det.confidence:.0%}"
            cv2.putText(
                out, label_text,
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )
        return out
