"""
navilens.py — Core Orchestrator

Two modes:
  1. DESCRIBE  — One-shot: YOLO detection → spatial description → TTS
  2. NAVIGATE  — Continuous: MiDaS depth + YOLO → event-based TTS guidance

All processing is LOCAL — no API calls.
"""

import threading
import time
from typing import List, Optional
import cv2
from tts_engine import TTSEngine
from voice_input import WhisperListener
from navigation import NavigationEngine
from detection import Detection, ObjectDetector
from spatial import SpatialReasoner
from gemini_vision import GeminiVision


class NaviLens:
    def __init__(
        self,
        source="0",
        show_window=True,
        model="yolov8m.pt",
    ):
        self.source = int(source) if source.isdigit() else source
        self.show_window = show_window
        self.running = False

        # ── TTS ──────────────────────────────────────────────────────
        print("[NaviLens] Initialising TTS engine...")
        self.tts = TTSEngine()

        # ── Object detector (for one-shot describe — uses large model) ───
        print(f"[NaviLens] Loading detection model: {model}...")
        self._detector = ObjectDetector(model_name=model, conf=0.40)

        # ── Spatial reasoner ─────────────────────────────────────────
        self._spatial = SpatialReasoner()

        # ── Navigation engine (uses lighter model for real-time speed) ────
        print("[NaviLens] Initialising navigation engine...")
        nav_model = model.replace("26l", "26s").replace("26x", "26m") if "yolo26" in model else model
        self._nav = NavigationEngine(model_name=nav_model, conf=0.35)

        # ── Gemini Vision (precise AI scene description) ───────────────
        print("[NaviLens] Initialising Gemini Vision...")
        self._gemini = GeminiVision()

        # ── Whisper listener (describe + nav + gemini routing) ──────────
        print("[NaviLens] Initialising Whisper voice listener...")
        self.voice = WhisperListener(
            describe_callback=self._on_describe,
            nav_callback=self._on_nav_command,
            stop_callback=self._on_stop,
            stop_nav_callback=self._on_stop_nav,
            navigate_to_callback=self._navigate_to_object,
            gemini_callback=self._on_gemini_describe,
            tts_is_speaking=lambda: self.tts.is_speaking,
            tts_last_spoke_ago=lambda: self.tts.last_spoke_ago,
        )

        # ── Frame state ──────────────────────────────────────────────
        self._frame_lock = threading.Lock()
        self._latest_frame = None

        # ── Navigation mode state ────────────────────────────────────
        self._nav_active = False
        self._nav_lock = threading.Lock()
        self._nav_thread = None
        self._nav_interval = 0.5
        self._latest_depth_vis = None
        self._depth_vis_lock = threading.Lock()

        # ── Command cooldown (prevents echo re-trigger) ───────────────
        self._last_cmd_time = 0.0
        self._cmd_cooldown = 2.0

        # ── Last describe detections (for target navigation) ─────────
        self._last_detections: List[Detection] = []

    # ══════════════════════════════════════════════════════════════════
    #  DESCRIBE MODE  (one-shot YOLO + spatial → TTS)
    # ══════════════════════════════════════════════════════════════════
    def _on_describe(self):
        """Called when Whisper routes a describe command (e.g. 'what's ahead?')."""
        now = time.time()
        if now - self._last_cmd_time < self._cmd_cooldown:
            return
        self._last_cmd_time = now
        # Stop navigation if active
        with self._nav_lock:
            if self._nav_active:
                self._nav_active = False
                print("[Nav] Navigation auto-stopped (describe command)")
                with self._depth_vis_lock:
                    self._latest_depth_vis = None

        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is None:
            self.tts.speak("Camera not ready yet.", priority=True)
            return

        threading.Thread(
            target=self._describe_scene,
            args=(frame,),
            daemon=True,
        ).start()

    def _describe_scene(self, frame):
        """Run YOLO detection + spatial reasoning, speak the description."""
        try:
            detections = self._detector.detect(frame, augment=True)
            self._last_detections = detections
            description = self._spatial.generate_full_description(detections)
            print(f"[Describe] {description}")
            self.tts.speak(description, priority=True)
        except Exception as e:
            print(f"[Describe] Error: {e}")
            self.tts.speak("Sorry, I couldn't describe the scene.", priority=True)

    # ── Target navigation ─────────────────────────────────────────────
    @staticmethod
    def _extract_target_name(text: str) -> Optional[str]:
        """Extract object name from voice command like 'take me to the chair'."""
        triggers = [
            "take me to", "navigate to", "guide me to", "go to", "lead me to",
        ]
        t = text.lower().strip()
        for phrase in triggers:
            if phrase in t:
                name = t.replace(phrase, "", 1).strip().rstrip(".,!?")
                for article in ["the ", "a ", "an "]:
                    if name.startswith(article):
                        name = name[len(article):].strip()
                return name if name else None
        return None

    def _find_target_in_frame(self, target: str) -> bool:
        """Run detection on current frame; return True if target is found."""
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is None:
            return False
        try:
            detections = self._detector.detect(frame, augment=False)
            t = target.lower().strip()
            for det in detections:
                label = det.label.lower().strip()
                if label == t or label + "s" == t or label == t + "s":
                    return True
        except Exception as e:
            print(f"[Nav] Detection error: {e}")
        return False

    def _navigate_to_object(self, text: str):
        """Called when Whisper routes 'take me to [object]'."""
        now = time.time()
        if now - self._last_cmd_time < self._cmd_cooldown:
            return
        self._last_cmd_time = now

        with self._nav_lock:
            if self._nav_active:
                self._nav_active = False
                print("[Nav] Navigation stopped for target navigation")

        target = self._extract_target_name(text)
        if not target:
            self.tts.speak("What object should I navigate you to?", priority=True)
            return

        if not self._find_target_in_frame(target):
            self.tts.speak(
                f"I don't see a {target} nearby.",
                priority=True,
            )
            return

        self._start_navigation()
        self._nav.set_target(target)
        self.tts.speak(f"Navigating to {target}.", priority=True)

    # ════════════════════════════════════════════════════════════════
    #  GEMINI VISION  (precise AI scene description)
    # ════════════════════════════════════════════════════════════════
    def _on_gemini_describe(self):
        """Called when user says 'describe precisely' or presses G."""
        now = time.time()
        if now - self._last_cmd_time < self._cmd_cooldown:
            return
        self._last_cmd_time = now

        if not self._gemini.available:
            self.tts.speak("Gemini Vision is not available. Check your API key.", priority=True)
            return

        # Stop navigation if active
        with self._nav_lock:
            if self._nav_active:
                self._nav_active = False
                print("[Nav] Navigation auto-stopped (gemini command)")
                with self._depth_vis_lock:
                    self._latest_depth_vis = None

        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is None:
            self.tts.speak("Camera not ready yet.", priority=True)
            return

        self.tts.speak("Analyzing scene with AI, one moment.", priority=True)
        threading.Thread(
            target=self._gemini_describe_scene,
            args=(frame,),
            daemon=True,
        ).start()

    def _gemini_describe_scene(self, frame):
        """Send frame to Gemini API and speak the response."""
        try:
            description = self._gemini.describe_frame(frame)
            if description:
                print(f"[Gemini] {description}")
                self.tts.speak(description, priority=True)
            else:
                self.tts.speak("Sorry, I couldn't get a description from Gemini.", priority=True)
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            self.tts.speak("Gemini Vision encountered an error.", priority=True)

    # ══════════════════════════════════════════════════════════════════
    #  LOCAL NAVIGATION  (depth + YOLO, fully local)
    # ══════════════════════════════════════════════════════════════════
    def _on_nav_command(self, text: str):
        """Called when Whisper routes a navigation command."""
        now = time.time()
        if now - self._last_cmd_time < self._cmd_cooldown:
            return
        self._last_cmd_time = now
        with self._nav_lock:
            if self._nav_active:
                self.tts.speak("Navigation is active. Say stop navigation to exit.", priority=True)
                return
        self._start_navigation()

    def _start_navigation(self):
        with self._nav_lock:
            if self._nav_active:
                return
            self._nav_active = True

        self._nav.reset_state()
        print("[Nav] Navigation mode ACTIVATED")
        self.tts.speak("Navigation mode activated. I'll guide you.", priority=True)

        self._nav_thread = threading.Thread(
            target=self._nav_loop,
            daemon=True,
        )
        self._nav_thread.start()

    def _stop_navigation(self):
        with self._nav_lock:
            if not self._nav_active:
                return
            self._nav_active = False

        print("[Nav] Navigation mode DEACTIVATED")
        self.tts.flush()
        with self._depth_vis_lock:
            self._latest_depth_vis = None

    def _on_stop_nav(self):
        self._stop_navigation()

    def _nav_loop(self):
        """Continuous navigation analysis loop (background thread)."""
        while True:
            with self._nav_lock:
                if not self._nav_active:
                    break

            with self._frame_lock:
                frame = self._latest_frame
                if frame is not None:
                    frame = frame.copy()

            if frame is None:
                time.sleep(0.1)
                continue

            try:
                state_key, announcement, depth_vis = self._nav.analyze(frame)
                with self._depth_vis_lock:
                    self._latest_depth_vis = depth_vis
                if announcement:
                    print(f"[Nav] {announcement}")
                    self.tts.speak(announcement, priority=True)

                # Auto-stop on target reached or lost
                if self._nav.target_reached or self._nav.target_lost:
                    reason = "reached" if self._nav.target_reached else "lost"
                    print(f"[Nav] Target {reason} — stopping navigation")
                    self._nav.clear_target()
                    self._stop_navigation()
            except Exception as e:
                print(f"[Nav] Analysis error: {e}")

            time.sleep(self._nav_interval)

    # ══════════════════════════════════════════════════════════════════
    #  APP LIFECYCLE
    # ══════════════════════════════════════════════════════════════════
    def _on_stop(self):
        self._stop_navigation()
        self.running = False

    # ══════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ══════════════════════════════════════════════════════════════════
    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[NaviLens] ERROR: Cannot open source '{self.source}'")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._nav.set_frame_size(frame_w, frame_h)
        self._spatial.set_frame_size(frame_w, frame_h)
        self._detector.set_frame_area(frame_w, frame_h)

        self.voice.start()
        self.running = True
        self.tts.speak(
            "NaviLens ready. Say what's ahead to describe the scene, or navigate me for guidance.",
            priority=True,
        )

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("[NaviLens] Frame read failed. Exiting.")
                    break

                with self._frame_lock:
                    self._latest_frame = frame

                if self.show_window:
                    display = frame.copy()

                    with self._nav_lock:
                        nav_on = self._nav_active

                    if nav_on:
                        status = "NAVIGATING | Say 'stop navigation' to exit | Q to quit"
                        color = (0, 200, 255)
                    else:
                        gemini_hint = " | G = Gemini AI" if self._gemini.available else ""
                        status = f"Listening... | Say 'what's ahead' or 'navigate me'{gemini_hint} | Q to quit"
                        color = (0, 255, 0)

                    cv2.putText(
                        display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )
                    cv2.imshow("NaviLens", display)

                    if nav_on:
                        with self._depth_vis_lock:
                            dv = self._latest_depth_vis
                        if dv is not None:
                            cv2.imshow("NaviLens - Depth", dv)
                    else:
                        try:
                            cv2.destroyWindow("NaviLens - Depth")
                        except cv2.error:
                            pass

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("g"):
                        # G key = Gemini AI precise description
                        threading.Thread(
                            target=self._on_gemini_describe,
                            daemon=True,
                        ).start()

        finally:
            self._stop_navigation()
            self.running = False
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            self.voice.stop()
            self.tts.speak("NaviLens shutting down. Stay safe.", priority=True)
            time.sleep(2)
            self.tts.shutdown()
            print("[NaviLens] Goodbye.")
