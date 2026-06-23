"""
tts_engine.py — Text-to-Speech Engine (Windows System.Speech)

Uses PowerShell's built-in System.Speech synthesis — no extra dependencies.
Priority queue: urgent messages preempt descriptive ones.
"""

import subprocess
import threading
import queue
import time
from dataclasses import dataclass, field


@dataclass(order=True)
class _SpeechItem:
    priority: int
    timestamp: float = field(compare=False)
    text: str = field(compare=False)


class TTSEngine:
    """
    Thread-safe, priority-queued TTS engine using Windows System.Speech.

    Usage:
        tts = TTSEngine()
        tts.speak("Obstacle ahead!", priority=True)
        tts.speak("Path appears clear.", priority=False)
        tts.shutdown()
    """

    def __init__(self):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._stop_event = threading.Event()
        self._available = self._check_available()
        self._speaking = False
        self._last_spoke_time = 0.0
        self._speak_lock = threading.Lock()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    @property
    def is_speaking(self) -> bool:
        """True while TTS is actively outputting speech."""
        return self._speaking

    @property
    def last_spoke_ago(self) -> float:
        """Seconds since last speech finished. Inf if never spoken."""
        if self._last_spoke_time == 0.0:
            return float("inf")
        return time.time() - self._last_spoke_time

    def _check_available(self) -> bool:
        try:
            result = subprocess.run(
                [
                    "powershell", "-ExecutionPolicy", "Bypass", "-Command",
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    "Write-Host $($s.GetInstalledVoices().Count)"
                ],
                capture_output=True, text=True, timeout=10,
            )
            available = result.returncode == 0 and result.stdout.strip() not in ("", "0")
            if available:
                print(f"[TTS] Windows System.Speech ready ({result.stdout.strip()} voice(s)).")
            else:
                print(f"[TTS] Windows System.Speech not available: {result.stderr.strip()}")
            return available
        except Exception as e:
            print(f"[TTS] TTS check failed: {e}")
            return False

    def speak(self, text: str, priority: bool = False):
        if not text or not text.strip():
            return
        item = _SpeechItem(
            priority=0 if priority else 1,
            timestamp=time.time(),
            text=text.strip(),
        )
        self._queue.put(item)

    def flush(self):
        """Clear all pending messages from the queue (e.g. on navigation stop)."""
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            print(f"[TTS] Flushed {cleared} stale message(s).")

    def shutdown(self):
        self._stop_event.set()
        self._worker.join(timeout=3)

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item.priority == 1 and (time.time() - item.timestamp) > 5.0:
                self._queue.task_done()
                continue

            self._say(item.text)
            self._queue.task_done()

    def _say(self, text: str):
        print(f"[TTS] {text}")

        if not self._available:
            return

        with self._speak_lock:
            self._speaking = True

        escaped = text.replace("'", "''")
        ps_command = (
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{escaped}')"
        )
        try:
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_command],
                capture_output=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            print("[TTS] PowerShell speech timed out.")
        except Exception as e:
            print(f"[TTS] PowerShell speech error: {e}")
        finally:
            with self._speak_lock:
                self._speaking = False
                self._last_spoke_time = time.time()
