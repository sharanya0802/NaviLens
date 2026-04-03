"""
tts_engine.py — Priority-Based Text-to-Speech Engine

Two modes:
  pyttsx3 — offline, works on RPi without internet, lower latency
  gtts    — Google TTS, better voice quality, requires internet

Priority queue: urgent messages preempt descriptive ones.
"""

import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Literal


@dataclass(order=True)
class _SpeechItem:
    priority: int           # 0 = urgent, 1 = normal
    timestamp: float = field(compare=False)
    text: str = field(compare=False)


class TTSEngine:
    """
    Thread-safe, priority-queued TTS engine.
    
    Usage:
        tts = TTSEngine(engine="pyttsx3")
        tts.speak("Obstacle ahead!", priority=True)     # preempts queue
        tts.speak("You are walking on a footpath.", priority=False)
        tts.shutdown()
    """

    def __init__(self, engine: Literal["pyttsx3", "gtts"] = "pyttsx3"):
        self._engine_type = engine
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._current_lock = threading.Lock()
        self._speaking = False

        self._init_engine()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ── Engine initialisation ─────────────────────────────────────────────────
    def _init_engine(self):
        if self._engine_type == "pyttsx3":
            try:
                import pyttsx3
                self._tts = pyttsx3.init()
                self._tts.setProperty("rate", 160)      # slightly slower for clarity
                self._tts.setProperty("volume", 1.0)
                # Prefer a female voice if available (often clearer)
                voices = self._tts.getProperty("voices")
                for v in voices:
                    if "female" in v.name.lower() or "zira" in v.name.lower():
                        self._tts.setProperty("voice", v.id)
                        break
            except ImportError:
                print("[TTS] pyttsx3 not found. Falling back to print mode.")
                self._engine_type = "print"
                self._tts = None

        elif self._engine_type == "gtts":
            try:
                from gtts import gTTS
                import pygame
                pygame.mixer.init()
                self._gtts_cls = gTTS
                self._pygame = pygame
            except ImportError:
                print("[TTS] gTTS or pygame not found. Falling back to print mode.")
                self._engine_type = "print"
                self._tts = None

    # ── Public API ────────────────────────────────────────────────────────────
    def speak(self, text: str, priority: bool = False):
        """
        Enqueue a speech item.
          priority=True  → obstacle warnings (queue priority 0)
          priority=False → descriptive info   (queue priority 1)
        """
        if not text or not text.strip():
            return
        item = _SpeechItem(
            priority=0 if priority else 1,
            timestamp=time.time(),
            text=text.strip(),
        )
        self._queue.put(item)

    def shutdown(self):
        self._stop_event.set()
        self._worker.join(timeout=3)

    # ── Worker thread ─────────────────────────────────────────────────────────
    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            # Drop stale low-priority items (older than 5 s)
            if item.priority == 1 and (time.time() - item.timestamp) > 5.0:
                self._queue.task_done()
                continue

            self._say(item.text)
            self._queue.task_done()

    def _say(self, text: str):
        print(f"[TTS] ▶ {text}")

        if self._engine_type == "pyttsx3":
            try:
                self._tts.say(text)
                self._tts.runAndWait()
            except Exception as e:
                print(f"[TTS] pyttsx3 error: {e}")

        elif self._engine_type == "gtts":
            try:
                import io
                import os
                tts_obj = self._gtts_cls(text=text, lang="en", slow=False)
                mp3_fp = io.BytesIO()
                tts_obj.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                self._pygame.mixer.music.load(mp3_fp)
                self._pygame.mixer.music.play()
                while self._pygame.mixer.music.get_busy():
                    time.sleep(0.05)
            except Exception as e:
                print(f"[TTS] gTTS error: {e}")

        else:
            # Print-only fallback (no audio library available)
            pass
