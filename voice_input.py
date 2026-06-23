"""
voice_input.py — Whisper-based always-on voice listener with intent routing.

Routes user speech to either:
  1. Describe (YOLO + spatial description)
  2. Navigation (depth-based guidance)

Prevents TTS echo feedback loop by:
  - Skipping audio while TTS is speaking or has just finished (cooldown gate)
  - Filtering transcribed text against known TTS output patterns
  - Raising energy threshold to reduce sensitivity to speaker output
"""

import threading
import time
import numpy as np
from typing import Callable, Optional

# ── Keyword sets for routing ─────────────────────────────────────────
# Includes common Whisper misheard variants for robustness
DESCRIBE_PHRASES = [
    "what's ahead", "what is ahead", "what's in front", "what is in front",
    "describe surroundings", "describe", "what do you see", "what can you see",
    "what's around me", "what is around me", "what's there", "what is there",
    "tell me what you see", "scene description",
    # Common Whisper misheard variants
    "it's a head", "its a head", "what said", "what's a head", "whats a head",
    "what ahead", "was ahead", "what's at head",
]

GEMINI_PHRASES = [
    "describe precisely", "precise description", "detailed description",
    "what exactly is ahead", "what exactly do you see",
    "gemini describe", "gemini", "use ai", "ai describe",
    "look carefully", "tell me exactly", "tell me precisely",
]

NAVIGATE_TO_PHRASES = [
    "take me to", "navigate to", "guide me to", "go to", "lead me to",
]

NAV_PHRASES = [
    "navigate me", "guide me", "help me walk", "start navigation",
    "where is the exit", "where is the door",
    "can i walk", "is the path", "any obstacle", "which way",
    "where should i go", "where do i go", "how do i get",
    # Common Whisper misheard variants
    "go to me", "got me", "go me", "never get me", "navigating",
    "guy me", "guard me", "guiding me",
]

NAV_KEYWORDS = {
    "navigate", "navigation", "move", "obstacle", "path", "walk",
    "avoid", "direction", "exit", "stairs", "door",
    "guide", "forward", "backward", "straight",
    "start navigation", "help me walk", "take me", "lead me",
}

STOP_NAV_PHRASES = [
    "stop navigation", "stop navigating", "end navigation",
    "cancel navigation", "stop guiding",
]

STOP_WORDS = {"stop", "quit", "exit", "shut down", "shutdown", "turn off"}

# ── TTS echo patterns (every phrase the system can speak) ────────────
TTS_ECHO_PATTERNS = [
    # navilens.py
    "navilens ready", "navilens shutting down", "navilens goodbye",
    "camera not ready", "couldn't describe",
    "navigation is active", "navigation mode activated",
    "say stop navigation to exit",
    "navigation stopped", "stopping navilens", "stay safe",
    # navigation.py
    "stop obstacle", "stop obstacle very close",
    "move left", "move right",
    "person ahead", "all paths are crowded",
    "path ahead is blocked", "try moving",
    "obstacle ahead proceed with caution",
    "left side is clearer", "right side is clearer",
    "on your left", "on your right",
    "path clear ahead", "path clear",
    # spatial.py
    "i can see", "on your center",
    "cannot detect any objects", "i also see",
    # common fragments that get echoed
    "navigation mode", "navigation stopped",
    "caution", "blocking center",
    "very near", "distant",
]

# Cooldown: discard any audio captured within this many seconds after TTS finishes
TTS_ECHO_COOLDOWN = 2.0


def _is_tts_echo(text: str) -> bool:
    """Check if transcribed text looks like TTS output being re-heard."""
    t = text.lower().strip()
    if len(t) < 5:
        return False
    for pattern in TTS_ECHO_PATTERNS:
        if pattern in t:
            return True
    # 3+ spatial/nav keywords together = likely echo
    keywords = {"left", "right", "center", "ahead", "person", "couch",
                "object", "very near", "near", "far", "distant",
                "navigation", "obstacle", "clear", "stop", "move",
                "blocking", "path", "guide", "activated", "deactivated"}
    words = set(t.split())
    matches = words & keywords
    return len(matches) >= 3


def classify_intent(text: str) -> str:
    """
    Classify user speech into one of:
      'stop'         — shutdown the app
      'stop_nav'     — stop navigation mode only
      'gemini'       — precise AI description (Gemini Vision)
      'describe'     — one-shot scene description
      'navigate_to'  — navigate to a specific object
      'navigate'     — activate/continue local navigation
      'unknown'      — unrecognized (ignored)
    """
    t = text.lower().strip().rstrip(".!?")

    if t in STOP_WORDS:
        return "stop"

    for phrase in STOP_NAV_PHRASES:
        if phrase in t:
            return "stop_nav"

    for phrase in GEMINI_PHRASES:
        if phrase in t:
            return "gemini"

    for phrase in DESCRIBE_PHRASES:
        if phrase in t:
            return "describe"

    for phrase in NAVIGATE_TO_PHRASES:
        if phrase in t:
            return "navigate_to"

    for phrase in NAV_PHRASES:
        if phrase in t:
            return "navigate"

    words = set(t.split())
    nav_matches = words & NAV_KEYWORDS
    if nav_matches and len(nav_matches) >= 2:
        return "navigate"

    # Don't default to describe — ignore unrecognized speech
    return "unknown"


class WhisperListener:
    """
    Always-on voice listener with describe/navigate routing.
    Suppresses TTS echo by coordinating with the TTS engine.
    """

    def __init__(
        self,
        describe_callback: Callable[[], None],
        nav_callback: Callable[[str], None],
        stop_callback: Callable[[], None],
        stop_nav_callback: Callable[[], None],
        navigate_to_callback: Optional[Callable[[str], None]] = None,
        gemini_callback: Optional[Callable[[], None]] = None,
        tts_is_speaking: Optional[Callable[[], bool]] = None,
        tts_last_spoke_ago: Optional[Callable[[], float]] = None,
    ):
        self._describe_cb = describe_callback
        self._nav_cb = nav_callback
        self._stop_cb = stop_callback
        self._stop_nav_cb = stop_nav_callback
        self._navigate_to_cb = navigate_to_callback or nav_callback  # fallback to nav
        self._gemini_cb = gemini_callback or describe_callback  # fallback to describe
        self._tts_is_speaking = tts_is_speaking or (lambda: False)
        self._tts_last_spoke_ago = tts_last_spoke_ago or (lambda: float("inf"))
        self._thread = None
        self._stop_event = threading.Event()
        self._whisper = None
        self._available = self._check_deps()

    def _check_deps(self) -> bool:
        try:
            import speech_recognition
            import pyaudio
            from faster_whisper import WhisperModel
            return True
        except ImportError as e:
            print(f"[Voice] Missing dependency: {e}")
            print("[Voice] Fix: pip install SpeechRecognition pyaudio faster-whisper")
            return False

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _load_whisper(self):
        if self._whisper is not None:
            return
        from faster_whisper import WhisperModel
                # Force CPU execution
        _device = "cpu"
        _compute = "int8"
        print(f"[Voice] Loading Whisper 'small' model on {_device}...")
        self._whisper = WhisperModel(
            "small",
            device=_device,
            compute_type=_compute,
        )
        print(f"[Voice] Whisper ready ({_device}, {_compute}).")

    def start(self):
        if not self._available:
            return
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[Voice] Listener started — speak naturally, I'm always listening.")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _listen_loop(self):
        import speech_recognition as sr

        self._load_whisper()

        r = sr.Recognizer()
        r.energy_threshold = 500
        r.dynamic_energy_threshold = True
        r.pause_threshold = 1.0
        r.phrase_threshold = 0.3
        mic = sr.Microphone(sample_rate=16000)

        with mic as source:
            print("[Voice] Calibrating microphone (2s)...")
            r.adjust_for_ambient_noise(source, duration=2)
            print("[Voice] Ready — ask what's ahead or say navigate me!")

            while not self._stop_event.is_set():
                try:
                    # Pre-listen gate: skip if TTS is speaking or just finished
                    if self._tts_is_speaking():
                        time.sleep(0.2)
                        continue
                    if self._tts_last_spoke_ago() < TTS_ECHO_COOLDOWN:
                        time.sleep(0.2)
                        continue

                    audio = r.listen(source, timeout=8, phrase_time_limit=10)

                    # Post-listen gate: discard if TTS started/finished during capture
                    if self._tts_is_speaking():
                        continue
                    if self._tts_last_spoke_ago() < TTS_ECHO_COOLDOWN:
                        continue

                    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                    segments, info = self._whisper.transcribe(
                        samples,
                        language="en",
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=600),
                    )
                    text = " ".join(s.text for s in segments).strip()

                    if not text or len(text) < 2:
                        continue

                    # Text-based TTS echo filter
                    if _is_tts_echo(text):
                        print(f"[Voice] Suppressed TTS echo: '{text}'")
                        continue

                    print(f"[Voice] Heard: '{text}'")

                    intent = classify_intent(text)
                    print(f"[Voice] Intent -> {intent}")

                    if intent == "stop":
                        self._stop_cb()
                    elif intent == "stop_nav":
                        self._stop_nav_cb()
                    elif intent == "navigate_to":
                        self._navigate_to_cb(text)
                    elif intent == "navigate":
                        self._nav_cb(text)
                    elif intent == "gemini":
                        self._gemini_cb()
                    elif intent == "describe":
                        self._describe_cb()
                    else:
                        # Unknown intent — ignore silently
                        print(f"[Voice] Ignored unrecognized speech: '{text}'")

                except Exception as e:
                    err = str(e)
                    if "timeout" in err.lower() or "WaitTimeoutError" in err:
                        continue
                    print(f"[Voice] Error: {e}")
