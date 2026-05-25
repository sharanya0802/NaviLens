"""
voice_input.py — Whisper-based voice listener with intent routing.

Routes to:
  - Local navigation (depth + path planner)
  - Local scene description (YOLO + spatial)
  - Product identification (Gemini Vision — products / labels only)
"""

import threading
import numpy as np
from typing import Callable

NAV_KEYWORDS = {
    "navigate", "navigation", "move", "obstacle", "path", "walk",
    "avoid", "direction", "stairs", "door", "left", "right",
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

PRODUCT_PHRASES = [
    "what is this", "what's this", "what am i holding", "what am i holding",
    "identify this", "identify the product", "what product",
    "read this label", "read the label", "read the price",
    "what brand", "how much", "what does this say",
    "what is on this", "what's on this", "tell me about this product",
]

SCENE_PHRASES = [
    "what's ahead", "whats ahead", "what is ahead",
    "describe surroundings", "describe the scene", "what do you see",
    "what can you see", "what is around me", "what's around me",
    "describe the room", "what is in front of me", "what is in front",
    "is the path clear", "anything in my way",
]

STOP_NAV_PHRASES = [
    "stop navigation", "stop navigating", "end navigation",
    "cancel navigation", "stop guiding",
]

STOP_WORDS = {"stop", "quit", "shut down", "shutdown", "turn off"}

PRODUCT_WORDS = {
    "product", "brand", "price", "label", "package", "bottle",
    "holding", "item", "buy", "mrp",
}

SCENE_WORDS = {
    "ahead", "around", "surroundings", "scene", "room", "front",
    "see", "detect", "obstacle", "clear", "path",
}


def classify_intent(text: str) -> str:
    """
    Returns one of:
      stop, stop_nav, find_exit, navigate,
      identify_product, describe_scene
    """
    t = text.lower().strip().rstrip(".!?")

    if t in STOP_WORDS:
        return "stop"

    for phrase in STOP_NAV_PHRASES:
        if phrase in t:
            return "stop_nav"

    for phrase in EXIT_PHRASES:
        if phrase in t:
            return "find_exit"

    for phrase in NAV_PHRASES:
        if phrase in t:
            return "navigate"

    for phrase in PRODUCT_PHRASES:
        if phrase in t:
            return "identify_product"

    for phrase in SCENE_PHRASES:
        if phrase in t:
            return "describe_scene"

    words = set(t.split())

    nav_matches = words & NAV_KEYWORDS
    if nav_matches:
        if words & (PRODUCT_WORDS | {"what", "read", "identify"}):
            return "identify_product"
        if words & SCENE_WORDS:
            return "describe_scene"
        return "navigate"

    if words & PRODUCT_WORDS or (
        "what" in words and any(w in t for w in ("holding", "this", "label", "product"))
    ):
        return "identify_product"

    if words & SCENE_WORDS or t.startswith("describe"):
        return "describe_scene"

    if "what" in words or "read" in words:
        return "identify_product"

    return "describe_scene"


class WhisperListener:
    def __init__(
        self,
        nav_callback: Callable[[str], None],
        exit_callback: Callable[[str], None],
        scene_callback: Callable[[str], None],
        product_callback: Callable[[str], None],
        stop_callback: Callable[[], None],
        stop_nav_callback: Callable[[], None],
        whisper_model: str = "base",
    ):
        self._nav_cb = nav_callback
        self._exit_cb = exit_callback
        self._scene_cb = scene_callback
        self._product_cb = product_callback
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
        print("[Voice] Listener started.")

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
            print("[Voice] Ready — navigate, describe scene, or identify product.")

            while not self._stop_event.is_set():
                try:
                    audio = r.listen(source, timeout=8, phrase_time_limit=15)

                    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

                    segments, _info = self._whisper.transcribe(
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
                    elif intent == "identify_product":
                        self._product_cb(text)
                    else:
                        self._scene_cb(text)

                except Exception as e:
                    err = str(e)
                    if "timeout" in err.lower() or "WaitTimeoutError" in err:
                        continue
                    print(f"[Voice] Error: {e}")
