"""
ocr_engine.py — EasyOCR-based text extraction for product labels and objects.

Extracts visible text (brand names, prices, descriptions) from cropped
object regions detected by YOLOv8.
"""

import threading
import numpy as np

try:
    import easyocr
except ImportError:
    raise ImportError(
        "EasyOCR not installed. Run: pip install easyocr"
    )


class OCREngine:
    """
    Thread-safe OCR engine using EasyOCR.

    Lazily initialises the reader on first use (model download ~100 MB first time).
    Designed to run on cropped bounding-box regions, not full frames.
    """

    def __init__(self, languages: list = None):
        self._languages = languages or ["en"]
        self._reader = None
        self._lock = threading.Lock()
        self._initialised = False

    def _ensure_reader(self):
        """Lazy-init: only load model when first needed."""
        if self._initialised:
            return
        with self._lock:
            if self._initialised:
                return
            print("[OCR] Loading EasyOCR model (first run may download ~100 MB)...")
            self._reader = easyocr.Reader(
                self._languages,
                gpu=True,       # auto-falls back to CPU if no GPU
                verbose=False,
            )
            self._initialised = True
            print("[OCR] EasyOCR ready.")

    def extract_text(self, frame: np.ndarray, bbox: tuple) -> str:
        """
        Extract text from a cropped region of the frame.

        Args:
            frame: Full BGR frame (numpy array).
            bbox:  (x1, y1, x2, y2) bounding box in pixels.

        Returns:
            Cleaned, concatenated text found in the region.
            Empty string if nothing readable is found.
        """
        self._ensure_reader()

        x1, y1, x2, y2 = bbox

        # Add a small padding to capture text near edges
        h, w = frame.shape[:2]
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame[y1:y2, x1:x2]

        # Skip tiny crops that won't have readable text
        if crop.shape[0] < 20 or crop.shape[1] < 20:
            return ""

        try:
            results = self._reader.readtext(
                crop,
                detail=1,           # returns (bbox, text, confidence)
                paragraph=False,
                min_size=10,
                text_threshold=0.5,
                low_text=0.3,
            )
        except Exception as e:
            print(f"[OCR] Error during text extraction: {e}")
            return ""

        if not results:
            return ""

        # Filter by confidence and concatenate
        texts = []
        for (_, text, conf) in results:
            if conf >= 0.3 and len(text.strip()) >= 2:
                texts.append(text.strip())

        return " ".join(texts)
