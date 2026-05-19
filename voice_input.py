"""
voice_input.py — Whisper-based always-on voice listener with intent routing.

Routes user speech to one of:
  1. Gemini Q&A pipeline (semantic/visual questions)
  2. Local navigation pipeline (spatial/movement commands)
  3. Exit-seeking pipeline ("find the exit" / "help me leave")

Default → Gemini, unless navigation/exit keywords are explicitly present.
"""

import threading
import numpy as np
from typing import Callable

# ── Keyword sets for routing ─────────────────────────────────────────
NAV_KEYWORDS = {
    "navigate", "navigation", "move", "obstacle", "path", "walk",
    "avoid", "direction", "exit", "stairs", "door", "left", "right",
    "ahead", "guide", "forward", "backward", "straight",
    "start navigation", "help me walk", "take me", "lead me",
}

NAV_PHRASES = [
    "navigate me", "guide me", "help me walk", "start navigation",
    "take me to", "lead me to",
    "can i walk", "is the path", "any obstacle", "which way",
    "where should i go", "where do i go", "how do i get",
]

EXIT_PHRASES = [
    "find the exit", "find exit", "find the door", "find a door",
    "help me leave", "help me get out", "guide me out",
    "take me outside", "where is the exit", "where is the door",
    "where's the exit", "where's the door", "leave the room",
    "get out of here", "exit the room", "how do i leave",
    "i want to leave", "i need to leave", "way out",
]

STOP_NAV_PHRASES = [
    "stop navigation", "stop navigating", "end navigation",
    "cancel navigation", "stop guiding",
]

STOP_WORDS = {"stop", "quit", "exit", "shut down", "shutdown", "turn off"}


def classify_intent(text: str) -> str:
    """
    Classify user speech into one of:
      'stop'       — shutdown the app
      'stop_nav'   — stop navigation mode only
      'find_exit'  — activate exit-seeking mode
      'navigate'   — activate/continue local navigation
      'question'   — send to Gemini (default)
    """
    t = text.lower().strip().rstrip(".!?")

    # Check for app stop
    if t in STOP_WORDS:
        return "stop"

    # Check for stop-navigation phrases
    for phrase in STOP_NAV_PHRASES:
        if phrase in t:
            return "stop_nav"

    # Check for exit-seeking phrases (before general nav)
    for phrase in EXIT_PHRASES:
        if phrase in t:
            return "find_exit"

    # Check for navigation phrases (multi-word, higher priority)
    for phrase in NAV_PHRASES:
        if phrase in t:
            return "navigate"

    # Check for navigation keywords (single words)
    words = set(t.split())
    nav_matches = words & NAV_KEYWORDS
    if nav_matches and len(nav_matches) >= 1:
        # Extra check: if "where" appears with semantic words, it's a question
        semantic_words = {"what", "who", "identify", "describe", "read",
                          "product", "price", "text", "color", "brand", "label"}
        if words & semantic_words:
            return "question"  # semantic takes priority
        return "navigate"

    # Default: Gemini Q&A
    return "question"


class WhisperListener:
    """
    Always-on voice listener with intelligent routing.
    """

    def __init__(
        self,
        question_callback: Callable[[str], None],
        nav_callback: Callable[[str], None],
        exit_callback: Callable[[str], None],
        stop_callback: Callable[[], None],
        stop_nav_callback: Callable[[], None],
        whisper_model: str = "base",
    ):
        self._question_cb = question_callback
        self._nav_cb = nav_callback
        self._exit_cb = exit_callback
        self._stop_cb = stop_callback
        self._stop_nav_cb = stop_nav_callback
        self._whisper_model_name = whisper_model
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

    def _load_whisper(self):
        if self._whisper is not None:
            return
        from faster_whisper import WhisperModel
        print(f"[Voice] Loading Whisper '{self._whisper_model_name}' model...")
        self._whisper = WhisperModel(
            self._whisper_model_name,
            device="cpu",
            compute_type="int8",
        )
        print("[Voice] Whisper ready.")

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
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 1.2
        r.phrase_threshold = 0.3
        mic = sr.Microphone(sample_rate=16000)

        with mic as source:
            print("[Voice] Calibrating microphone (2s)...")
            r.adjust_for_ambient_noise(source, duration=2)
            print("[Voice] Ready — ask questions or say 'navigate me'!")

            while not self._stop_event.is_set():
                try:
                    audio = r.listen(source, timeout=8, phrase_time_limit=15)

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

                    print(f"[Voice] Heard: '{text}'")

                    # Route based on intent
                    intent = classify_intent(text)
                    print(f"[Voice] Intent → {intent}")

                    if intent == "stop":
                        self._stop_cb()
                    elif intent == "stop_nav":
                        self._stop_nav_cb()
                    elif intent == "find_exit":
                        self._exit_cb(text)
                    elif intent == "navigate":
                        self._nav_cb(text)
                    else:
                        self._question_cb(text)

                except Exception as e:
                    err = str(e)
                    if "timeout" in err.lower() or "WaitTimeoutError" in err:
                        continue
                    print(f"[Voice] Error: {e}")