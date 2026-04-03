"""
voice_input.py — adds 'nearest_exit' and 'emergency' intents
"""
import threading
from typing import Callable, Optional

INTENT_MAP = {
    "whats_ahead":           ["what's ahead","whats ahead","ahead","what do you see"],
    "describe_surroundings": ["describe surroundings","look around","what's around","surroundings"],
    "nearest_exit":          ["nearest exit","find exit","where is the exit","where's the exit","exit","how far is the exit"],
    "emergency":             ["emergency","fire","evacuate","evacuation","fire alarm"],
    "stop":                  ["stop","quit","shutdown","turn off"],
    "help":                  ["help","commands","what can you do"],
}

def _classify_intent(text: str) -> Optional[str]:
    t = text.lower().strip()
    for intent, phrases in INTENT_MAP.items():
        for phrase in phrases:
            if phrase in t:
                return intent
    return None

class VoiceCommandListener:
    def __init__(self, command_callback: Callable[[str], None]):
        self._callback = command_callback
        self._thread = None
        self._stop_event = threading.Event()
        self._available = self._check_deps()

    def _check_deps(self) -> bool:
        try:
            import speech_recognition, pyaudio
            return True
        except ImportError:
            print("[Voice] SpeechRecognition or PyAudio missing. Voice disabled.")
            print("[Voice] Fix: pip install SpeechRecognition pyaudio")
            return False

    def start(self):
        if not self._available: return
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[Voice] Listener started.")

    def stop(self):
        self._stop_event.set()
        if self._thread: self._thread.join(timeout=2)

    def _listen_loop(self):
        import speech_recognition as sr
        r = sr.Recognizer()
        r.energy_threshold = 300
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.7
        mic = sr.Microphone(sample_rate=16000)
        with mic as source:
            print("[Voice] Calibrating (2s)...")
            r.adjust_for_ambient_noise(source, duration=2)
            print("[Voice] Ready.")
            while not self._stop_event.is_set():
                try:
                    audio = r.listen(source, timeout=5, phrase_time_limit=5)
                    text = r.recognize_google(audio)
                    print(f"[Voice] Heard: '{text}'")
                    intent = _classify_intent(text)
                    if intent:
                        print(f"[Voice] Intent → {intent}")
                        self._callback(intent)
                except sr.WaitTimeoutError: pass
                except sr.UnknownValueError: pass
                except sr.RequestError as e: print(f"[Voice] STT error: {e}")
                except Exception as e: print(f"[Voice] Error: {e}")