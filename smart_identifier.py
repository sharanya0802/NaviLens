"""
smart_identifier.py — Product identification (CLIP + OCR + optional Gemini).

Used only for product/label voice commands ("what is this", "read the label").
Navigation and scene description do not use this module.

  - identify_local()     → CLIP + OCR (offline fallback)
  - identify_with_gemini() → Gemini Vision for brand, product type, MRP
"""

import re
import threading
import time
import hashlib
import cv2
import numpy as np
from typing import Optional, Tuple

from clip_classifier import CLIPClassifier


class SmartIdentifier:
    """
    Combines CLIP visual classification + OCR text into smart descriptions.
    Gemini is reserved for on-demand requests only.
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._client = None
        self._gemini_available = False
        self._lock = threading.Lock()

        # Cache: image_hash → (description, price, timestamp)
        self._cache: dict = {}
        self._cache_ttl = 20.0

        # CLIP classifier (local, no API)
        self._clip = CLIPClassifier()

        # Gemini (on-demand only)
        if self._api_key:
            self._init_gemini()

    def _init_gemini(self):
        try:
            from google import genai  # pyrefly: ignore [missing-import]
            self._client = genai.Client(api_key=self._api_key)
            self._gemini_available = True
            print("[SmartID] Gemini available for on-demand identification.")
        except ImportError:
            print("[SmartID] google-genai not installed. On-demand ID disabled.")
        except Exception as e:
            print(f"[SmartID] Gemini init error: {e}")

    def _image_hash(self, image: np.ndarray) -> str:
        small = cv2.resize(image, (32, 32))
        return hashlib.md5(small.tobytes()).hexdigest()

    # ──────────────────────────────────────────────────────────────────
    #  LOCAL IDENTIFICATION  (CLIP + OCR, no API)
    # ──────────────────────────────────────────────────────────────────
    def identify_local(
        self, ocr_text: str, image_crop: Optional[np.ndarray] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Identify an object using CLIP + OCR locally. No Gemini call.
        Used for automatic announcements.
        """
        ocr_text = ocr_text.strip() if ocr_text else ""

        # Check cache
        cache_key = None
        if image_crop is not None and image_crop.size > 0:
            cache_key = self._image_hash(image_crop)
            with self._lock:
                if cache_key in self._cache:
                    desc, price, ts = self._cache[cache_key]
                    if time.time() - ts < self._cache_ttl:
                        return desc, price

        # Extract price from OCR text
        price = self._extract_price(ocr_text)

        # Get CLIP classification
        clip_label = ""
        if image_crop is not None and image_crop.size > 0:
            results = self._clip.classify(image_crop, top_k=1)
            if results:
                clip_label = results[0][0]   # e.g. "face serum"

        # Build description from OCR text + CLIP
        desc = self._build_local_description(ocr_text, clip_label, price)

        if cache_key:
            with self._lock:
                self._cache[cache_key] = (desc, price, time.time())

        return desc, price

    # ──────────────────────────────────────────────────────────────────
    #  ON-DEMAND IDENTIFICATION  (Gemini Vision)
    # ──────────────────────────────────────────────────────────────────
    def identify_with_gemini(
        self, ocr_text: str, image_crop: Optional[np.ndarray] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Identify using Gemini Vision. Only called on explicit voice command.
        Falls back to local identification if Gemini unavailable.
        """
        if not self._gemini_available or image_crop is None or image_crop.size == 0:
            return self.identify_local(ocr_text, image_crop)

        ocr_text = ocr_text.strip() if ocr_text else ""

        from google.genai import types
        success, jpeg_bytes = cv2.imencode(
            ".jpg", image_crop, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        if not success:
            return self.identify_local(ocr_text, image_crop)

        image_part = types.Part.from_bytes(
            data=jpeg_bytes.tobytes(), mime_type="image/jpeg"
        )

        ocr_ctx = f"\nOCR text on object: '{ocr_text}'" if ocr_text else ""
        prompt = (
            f"Identify this object for a visually impaired person.\n"
            f"Trust what you SEE in the image.{ocr_ctx}\n\n"
            f"1. Give the actual object name (2-6 words), include brand if visible.\n"
            f"2. Extract price if visible (look for Rs., MRP, ₹, $).\n"
            f"Respond EXACTLY as:\nNAME: <name>\nPRICE: <price or NONE>\n"
        )

        try:
            response = self._client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[image_part, prompt],
            )
            name, price = self._parse_gemini_response(response.text)
            if name:
                return name, price
        except Exception as e:
            print(f"[SmartID] Gemini error: {e}")

        return self.identify_local(ocr_text, image_crop)

    # ──────────────────────────────────────────────────────────────────
    #  PRICE EXTRACTION  (robust local regex)
    # ──────────────────────────────────────────────────────────────────
    def _extract_price(self, text: str) -> Optional[str]:
        """Extract price from OCR text using robust pattern matching."""
        if not text:
            return None

        patterns = [
            # ₹ symbol variants
            r'[₹]\s*(\d+[\.,]?\d*)',
            # Rs / Rs. / RS
            r'(?:Rs\.?|RS\.?)\s*:?\s*(\d+[\.,]?\d*)',
            # MRP
            r'(?:MRP|mrp|M\.R\.P)\s*:?\s*[₹]?\s*(\d+[\.,]?\d*)',
            # INR
            r'(?:INR|inr)\s*:?\s*(\d+[\.,]?\d*)',
            # Price: / Cost:
            r'(?:Price|PRICE|Cost|COST)\s*:?\s*[₹]?\s*(\d+[\.,]?\d*)',
            # Dollar / Euro / Pound
            r'[\$€£]\s*(\d+[\.,]?\d*)',
            # Number followed by currency
            r'(\d+[\.,]?\d*)\s*[₹\$€£]',
            # Standalone "Rs" followed by number anywhere
            r'[Rr][Ss]\.?\s*(\d{2,})',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                amount = match.group(1) if match.lastindex else match.group()
                # Clean up: "450.00" → "450", "3,999" → "3999"
                amount = amount.replace(",", "")
                if amount.endswith(".00"):
                    amount = amount[:-3]
                return f"Rs. {amount}"

        return None

    # ──────────────────────────────────────────────────────────────────
    #  LOCAL DESCRIPTION BUILDER
    # ──────────────────────────────────────────────────────────────────
    def _build_local_description(
        self, ocr_text: str, clip_label: str, price: Optional[str]
    ) -> str:
        """Combine OCR text + CLIP label into a human-friendly description."""
        brand = self._extract_brand(ocr_text, price)

        if brand and clip_label:
            return f"{brand} {clip_label}"
        elif brand:
            return brand
        elif clip_label:
            return clip_label
        else:
            return "object"

    def _extract_brand(self, ocr_text: str, price: Optional[str]) -> str:
        """
        Extract brand/product name from OCR text.
        Removes price-related words, numbers, and noise.
        """
        if not ocr_text:
            return ""

        # Remove price substring if found
        cleaned = ocr_text
        if price:
            # Remove any price-like patterns from the text
            cleaned = re.sub(
                r'(?:Rs\.?|RS\.?|MRP|mrp|INR|inr|Price|PRICE|Cost|COST)\s*:?\s*[₹]?\s*\d+[\.,]?\d*',
                '', cleaned
            )
            cleaned = re.sub(r'[₹\$€£]\s*\d+[\.,]?\d*', '', cleaned)

        # Clean OCR noise
        cleaned = re.sub(r'[{}\[\]|\\/<>]', '', cleaned)  # remove bracket noise
        cleaned = re.sub(r'\b\d{1,2}\b', '', cleaned)     # remove stray 1-2 digit numbers
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Fix common OCR errors
        cleaned = re.sub(r'&\d', '&', cleaned)   # "&5" → "&"
        cleaned = re.sub(r'\{(\d)', r'(\1', cleaned)

        # Take first meaningful words (up to 6)
        words = [w for w in cleaned.split() if len(w) >= 2]
        if words:
            return " ".join(words[:6]).title()

        return ""

    def _parse_gemini_response(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        name = None
        price = None
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("NAME:"):
                name = line[5:].strip()
            elif line.upper().startswith("PRICE:"):
                p = line[6:].strip()
                if p.upper() != "NONE" and p:
                    price = p
        return name, price
