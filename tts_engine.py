"""
tts_engine.py — Reliable text-to-speech for NaviLens.

macOS: uses the built-in `say` command (most reliable from background threads).
Linux/Windows: pyttsx3, with optional edge-tts for higher quality.

Fixes common pyttsx3 issues: silent failures, cut-off speech, queue flooding.
"""

import platform
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(order=True)
class _SpeechItem:
    priority: int
    timestamp: float = field(compare=False)
    text: str = field(compare=False)
    interrupt: bool = field(compare=False)


def _split_sentences(text: str, max_len: int = 220) -> list:
    """Split long guidance into speakable chunks."""
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_len:
        return [text] if text else []

    parts = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current = ""
    for part in parts:
        if len(current) + len(part) + 1 <= max_len:
            current = f"{current} {part}".strip()
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks or [text[:max_len]]


class TTSEngine:
    """
    Thread-safe TTS with one utterance at a time and optional interrupt.

    Engines:
      auto   — `say` on macOS, else pyttsx3
      say    — macOS / Linux `say` or `espeak`
      pyttsx3 — offline cross-platform
      edge   — Microsoft Edge neural voice (needs internet + edge-tts)
    """

    def __init__(
        self,
        engine: Literal["auto", "say", "pyttsx3", "edge", "gtts"] = "auto",
        rate: int = 175,
        min_gap: float = 0.4,
    ):
        self._requested = engine
        self._engine_type = self._resolve_engine(engine)
        self._rate = rate
        self._min_gap = min_gap
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._speak_lock = threading.Lock()
        self._last_spoke_at = 0.0
        self._current_proc: Optional[subprocess.Popen] = None
        self._pyttsx3 = None
        self._pygame = None
        self._gtts_cls = None
        self._edge_voice = "en-US-AriaNeural"

        self._init_backend()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="tts-worker")
        self._worker.start()
        print(f"[TTS] Backend: {self._engine_type}")

    def _resolve_engine(self, engine: str) -> str:
        if engine != "auto":
            return engine
        if platform.system() == "Darwin":
            return "say"
        return "pyttsx3"

    def _init_backend(self):
        if self._engine_type == "pyttsx3":
            try:
                import pyttsx3
                driver = "nsss" if platform.system() == "Darwin" else None
                self._pyttsx3 = pyttsx3.init(driverName=driver) if driver else pyttsx3.init()
                self._pyttsx3.setProperty("rate", self._rate)
                self._pyttsx3.setProperty("volume", 1.0)
            except Exception as e:
                print(f"[TTS] pyttsx3 failed ({e}), trying say/espeak")
                self._engine_type = "say"

        elif self._engine_type == "edge":
            try:
                import edge_tts  # noqa: F401
            except ImportError:
                print("[TTS] edge-tts not installed. pip install edge-tts")
                self._engine_type = "pyttsx3"
                self._init_backend()
                return

        elif self._engine_type == "gtts":
            try:
                from gtts import gTTS
                import pygame
                pygame.mixer.init(frequency=24000)
                self._gtts_cls = gTTS
                self._pygame = pygame
            except ImportError:
                print("[TTS] gTTS/pygame missing, falling back")
                self._engine_type = "say" if platform.system() == "Darwin" else "pyttsx3"
                self._init_backend()

    def speak(self, text: str, priority: bool = False, interrupt: bool = False):
        if not text or not str(text).strip():
            return
        item = _SpeechItem(
            priority=0 if priority else 1,
            timestamp=time.time(),
            text=str(text).strip(),
            interrupt=interrupt or priority,
        )
        if item.interrupt:
            self._flush_queue(keep_priority_zero=True)
        self._queue.put(item)

    def speak_blocking(self, text: str) -> bool:
        """Speak immediately (startup test). Returns True if audio likely played."""
        return self._utter(text)

    def test_audio(self) -> bool:
        print("[TTS] Running audio test...")
        ok = self.speak_blocking("NaviLens audio test.")
        if ok:
            print("[TTS] Audio test OK.")
        else:
            print("[TTS] Audio test FAILED — check volume and output device.")
        return ok

    def shutdown(self):
        self._stop_event.set()
        self._kill_current()
        self._worker.join(timeout=5)

    def _flush_queue(self, keep_priority_zero: bool = False):
        drained = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if keep_priority_zero:
            for item in drained:
                if item.priority == 0:
                    self._queue.put(item)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            age = time.time() - item.timestamp
            if item.priority == 1 and age > 8.0:
                self._queue.task_done()
                continue

            gap = self._min_gap if item.priority == 1 else 0.15
            wait = gap - (time.time() - self._last_spoke_at)
            if wait > 0:
                time.sleep(wait)

            if item.interrupt:
                self._kill_current()

            for chunk in _split_sentences(item.text):
                if self._stop_event.is_set():
                    break
                self._utter(chunk)

            self._last_spoke_at = time.time()
            self._queue.task_done()

    def _utter(self, text: str) -> bool:
        print(f"[TTS] ▶ {text}")
        with self._speak_lock:
            try:
                if self._engine_type == "say":
                    return self._say_native(text)
                if self._engine_type == "pyttsx3":
                    return self._say_pyttsx3(text)
                if self._engine_type == "edge":
                    return self._say_edge(text)
                if self._engine_type == "gtts":
                    return self._say_gtts(text)
            except Exception as e:
                print(f"[TTS] Error ({self._engine_type}): {e}")
        return False

    def _say_native(self, text: str) -> bool:
        system = platform.system()
        if system == "Darwin":
            cmd = ["say", "-r", str(self._rate), text]
        else:
            if self._has_espeak():
                cmd = ["espeak", "-s", str(self._rate), text]
            else:
                return self._say_pyttsx3(text)
        self._current_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._current_proc.wait(timeout=max(60, len(text) * 0.15))
        self._current_proc = None
        return True

    @staticmethod
    def _has_espeak() -> bool:
        from shutil import which
        return which("espeak") is not None

    def _say_pyttsx3(self, text: str) -> bool:
        import pyttsx3
        # Fresh engine per utterance avoids macOS thread deadlock / silent drop
        driver = "nsss" if platform.system() == "Darwin" else None
        engine = pyttsx3.init(driverName=driver) if driver else pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        try:
            engine.stop()
        except Exception:
            pass
        return True

    def _say_edge(self, text: str) -> bool:
        import asyncio
        import tempfile
        import os

        async def _run():
            import edge_tts
            communicate = edge_tts.Communicate(text, self._edge_voice)
            path = tempfile.mktemp(suffix=".mp3")
            await communicate.save(path)
            return path

        path = asyncio.run(_run())
        try:
            if platform.system() == "Darwin":
                self._current_proc = subprocess.Popen(["afplay", path])
                self._current_proc.wait()
            else:
                import pygame
                pygame.mixer.init()
                pygame.mixer.music.load(path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
        self._current_proc = None
        return True

    def _say_gtts(self, text: str) -> bool:
        import io
        tts_obj = self._gtts_cls(text=text, lang="en", slow=False)
        mp3_fp = io.BytesIO()
        tts_obj.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        self._pygame.mixer.music.load(mp3_fp)
        self._pygame.mixer.music.play()
        while self._pygame.mixer.music.get_busy():
            time.sleep(0.05)
        return True

    def _kill_current(self):
        proc = self._current_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=0.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._current_proc = None
        if self._pygame:
            try:
                self._pygame.mixer.music.stop()
            except Exception:
                pass
