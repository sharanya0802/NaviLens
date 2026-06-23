"""
gemini_vision.py — Gemini Vision API for precise scene description.

Sends a camera frame to Google's Gemini model for detailed, AI-powered
scene description. Used as a complement to local YOLO detection when
the user wants a more precise and natural description.

Requires: pip install google-generativeai python-dotenv
"""

import os
import threading
import cv2
import numpy as np
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


VISION_PROMPT = (
    "Tou are seeing what I am seeing."
    "You are a navigation assistant for a visually impaired user wearing a camera. "
    "Describe the scene in 2-3 complete sentences. "
    "Mention: what objects are present, where they are (left, center, right), "
    "how close they appear, and whether the path ahead is clear or blocked. "
    "Example: 'A wooden table is directly ahead, about 2 meters away. "
    "A person is standing to your left. The path to the right appears clear.' "
    "Always finish your sentences. Be concise and helpful."
)


class GeminiVision:
    """Sends camera frames to Gemini for precise AI scene description."""

    def __init__(self):
        self._model = None
        self._available = False
        self._model_name = ""
        self._lock = threading.Lock()
        self._init()

    def _init(self):
        if not _GENAI_AVAILABLE:
            print("[Gemini] google-generativeai not installed. Run: pip install google-generativeai python-dotenv")
            return

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            print("[Gemini] No GEMINI_API_KEY found in .env file.")
            return

        try:
            genai.configure(api_key=api_key)

            # Safety settings — disable filters for this assistive use case.
            # Describing people, obstacles, and surroundings should never be blocked.
            self._safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            # Try models in order of preference (quota availability varies)
            for model_name in ["gemini-2.5-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]:
                try:
                    self._model = genai.GenerativeModel(model_name)
                    self._available = True
                    self._model_name = model_name
                    print(f"[Gemini] Gemini Vision ready ({model_name}).")
                    return
                except Exception:
                    continue
            print("[Gemini] No available Gemini model found.")
        except Exception as e:
            print(f"[Gemini] Failed to initialize: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def describe_frame(self, frame: np.ndarray) -> Optional[str]:
        """
        Send a camera frame to Gemini and get a scene description.

        Args:
            frame: BGR image (H, W, 3) uint8 from OpenCV.

        Returns:
            Description string, or None on failure.
        """
        if not self._available:
            return None

        with self._lock:
            try:
                # Encode frame as JPEG (smaller than PNG, fast to encode)
                _, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )
                image_bytes = buffer.tobytes()

                image_part = {
                    "mime_type": "image/jpeg",
                    "data": image_bytes,
                }

                response = self._model.generate_content(
                    [VISION_PROMPT, image_part],
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=10000,
                        temperature=0.4,
                    ),
                    safety_settings=self._safety_settings,
                )

                # Check for blocked/truncated responses
                if not response.candidates:
                    print("[Gemini] Response blocked (no candidates).")
                    return None

                candidate = response.candidates[0]
                finish = candidate.finish_reason

                # finish_reason: 1=STOP (normal), 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION
                if finish and finish != 1 and finish != "STOP":
                    print(f"[Gemini] Response may be truncated (finish_reason={finish}).")

                text = response.text.strip()
                return text if text else None

            except Exception as e:
                print(f"[Gemini] API error: {e}")
                return None

