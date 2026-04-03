"""
navilens.py — Core Orchestrator
Ties together: YOLOv8 detection → Spatial Reasoning → TTS Output → Voice Input
"""

import threading
import time
import cv2
from detection import ObjectDetector
from spatial import SpatialReasoner
from tts_engine import TTSEngine
from voice_input import VoiceCommandListener


class NaviLens:
    def __init__(
        self,
        source="0",
        model_name="yolov8n.pt",
        conf_threshold=0.45,
        show_window=True,
        tts_engine="pyttsx3",
        enable_voice_input=True,
    ):
        self.source = int(source) if source.isdigit() else source
        self.show_window = show_window
        self.enable_voice_input = enable_voice_input
        self.running = False

        # ── Subsystems ──────────────────────────────────────────────
        print("[NaviLens] Loading YOLOv8 model...")
        self.detector = ObjectDetector(model_name=model_name, conf=conf_threshold)

        print("[NaviLens] Initialising spatial reasoner...")
        self.reasoner = SpatialReasoner()

        print("[NaviLens] Initialising TTS engine...")
        self.tts = TTSEngine(engine=tts_engine)

        if self.enable_voice_input:
            print("[NaviLens] Starting voice command listener...")
            self.voice = VoiceCommandListener(command_callback=self._on_voice_command)
        else:
            self.voice = None

        # ── State ────────────────────────────────────────────────────
        self._last_scene_description: str = ""
        self._last_auto_announce: float = 0.0
        self._auto_announce_interval: float = 4.0   # seconds between proactive announcements
        self._frame_lock = threading.Lock()
        self._latest_detections = []

    # ── Voice command callback (runs in voice thread) ────────────────
    def _on_voice_command(self, intent: str):
        if intent == "whats_ahead":
            self.tts.speak(self._latest_ahead_summary(), priority=True)
        elif intent == "describe_surroundings":
            self.tts.speak(self._latest_full_description(), priority=True)
        elif intent == "stop":
            self.tts.speak("Stopping navigation.", priority=True)
            self.running = False
        elif intent == "help":
            self.tts.speak(
                "Available commands: what's ahead, describe surroundings, stop.", priority=True
            )

    # ── Summaries derived from latest detections ─────────────────────
    def _latest_ahead_summary(self) -> str:
        with self._frame_lock:
            dets = list(self._latest_detections)
        if not dets:
            return "Nothing detected ahead."
        return self.reasoner.generate_navigation_instruction(dets)

    def _latest_full_description(self) -> str:
        with self._frame_lock:
            dets = list(self._latest_detections)
        if not dets:
            return "Surroundings appear clear."
        return self.reasoner.generate_full_description(dets)

    # ── Main loop ────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[NaviLens] ERROR: Cannot open source '{self.source}'")
            return

        # Get frame dimensions for spatial reasoning calibration
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.reasoner.set_frame_size(frame_w, frame_h)

        if self.voice:
            self.voice.start()

        self.running = True
        self.tts.speak("NaviLens activated. I will guide you.", priority=True)

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[NaviLens] Frame read failed. Exiting.")
                    break

                # ── Detection ────────────────────────────────────────
                detections = self.detector.detect(frame)

                with self._frame_lock:
                    self._latest_detections = detections

                # ── Draw bounding boxes ──────────────────────────────
                annotated = self.detector.draw(frame, detections)

                # ── Proactive navigation announcements ───────────────
                now = time.time()
                if now - self._last_auto_announce >= self._auto_announce_interval:
                    urgent = self.reasoner.get_urgent_warnings(detections)
                    if urgent:
                        self.tts.speak(urgent, priority=True)
                    else:
                        nav = self.reasoner.generate_navigation_instruction(detections)
                        if nav:
                            self.tts.speak(nav, priority=False)
                    self._last_auto_announce = now

                # ── Overlay info on frame ────────────────────────────
                if self.show_window:
                    status_text = f"Objects: {len(detections)} | Press Q to quit"
                    cv2.putText(
                        annotated, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                    cv2.imshow("NaviLens — Live Feed", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            self.running = False
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            if self.voice:
                self.voice.stop()
            self.tts.speak("NaviLens shutting down. Stay safe.", priority=True)
            time.sleep(2)
            self.tts.shutdown()
            print("[NaviLens] Goodbye.")
