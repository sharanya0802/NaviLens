"""
config.py — Central model and runtime settings for NaviLens.

Use environment variables or CLI flags in main.py to override.
"""

import os

# ── Depth (MiDaS / DPT) ─────────────────────────────────────────────────────
# small  = fastest (~40 MB), good for Raspberry Pi
# hybrid = better accuracy (~400 MB), recommended on laptop
# large  = best quality, slowest
DEPTH_PROFILE = os.environ.get("NAVILENS_DEPTH", "hybrid")

DEPTH_MODELS = {
    "small": ("MiDaS_small", "small_transform"),
    "hybrid": ("DPT_Hybrid", "dpt_transform"),
    "large": ("DPT_Large", "dpt_transform"),
}

# ── Object detection (Ultralytics YOLO) ─────────────────────────────────────
# yolov8n = fastest | yolov8s = better for doors/products | yolov8m = heavy
YOLO_MODEL = os.environ.get("NAVILENS_YOLO", "yolov8s.pt")

# ── Speech-to-text (faster-whisper) ─────────────────────────────────────────
WHISPER_MODEL = os.environ.get("NAVILENS_WHISPER", "small")

# ── TTS ─────────────────────────────────────────────────────────────────────
# auto = macOS `say` on Darwin, else pyttsx3; also: say, pyttsx3, edge
TTS_ENGINE = os.environ.get("NAVILENS_TTS", "auto")

# ── Navigation timing ───────────────────────────────────────────────────────
NAV_FRAME_INTERVAL = float(os.environ.get("NAVILENS_NAV_INTERVAL", "0.55"))
MIN_SPEECH_GAP = float(os.environ.get("NAVILENS_SPEECH_GAP", "2.8"))

# ── Hazard watch (idle obstacle alerts) ───────────────────────────────────────
HAZARD_WATCH_ENABLED = os.environ.get("NAVILENS_HAZARD_WATCH", "1") != "0"
HAZARD_WATCH_INTERVAL = float(os.environ.get("NAVILENS_HAZARD_INTERVAL", "1.2"))
