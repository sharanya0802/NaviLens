"""
navilens.py — Core Orchestrator

  LOCAL: Navigation (DPT/MiDaS + YOLO + path planner), scene description, hazard alerts
  API:   Gemini for product/label identification only
"""

import threading
import time

import cv2

from config import (
    HAZARD_WATCH_ENABLED,
    HAZARD_WATCH_INTERVAL,
    MIN_SPEECH_GAP,
    NAV_FRAME_INTERVAL,
    TTS_ENGINE,
    WHISPER_MODEL,
)
from hazard_monitor import HazardMonitor
from local_vision import (
    describe_ahead,
    describe_scene,
    detect_objects,
    get_product_crop,
    set_frame_size as set_vision_frame_size,
)
from ocr_engine import OCREngine
from smart_identifier import SmartIdentifier
from tts_engine import TTSEngine
from voice_input import WhisperListener
from navigation import NavigationEngine


def _is_urgent_speech(text: str) -> bool:
    t = text.lower()
    return any(
        k in t
        for k in (
            "stop!", "stop.", "caution", "duck", "dead end",
            "very close", "blocking", "overhead", "watch out",
            "you've reached the door",
        )
    )


class NaviLens:
    def __init__(
        self,
        source="0",
        show_window=True,
        tts_engine=None,
        gemini_api_key=None,
        whisper_model=None,
    ):
        self.source = int(source) if source.isdigit() else source
        self.show_window = show_window
        self.running = False

        print("[NaviLens] Initialising TTS engine...")
        self.tts = TTSEngine(engine=tts_engine or TTS_ENGINE, min_gap=MIN_SPEECH_GAP)

        print("[NaviLens] Initialising local navigation engine...")
        self._nav = NavigationEngine()

        print("[NaviLens] Initialising local vision...")
        self._ocr = OCREngine()
        self._smart_id = SmartIdentifier(api_key=gemini_api_key)
        self._gemini_products = self._smart_id._gemini_available

        self._hazard = HazardMonitor(on_alert=self._on_hazard_alert)
        if not HAZARD_WATCH_ENABLED:
            self._hazard.set_enabled(False)

        self.voice = WhisperListener(
            nav_callback=self._on_nav_command,
            exit_callback=self._on_exit_command,
            scene_callback=self._on_scene_query,
            product_callback=self._on_identify_product,
            help_callback=self._on_help,
            repeat_callback=self._on_repeat,
            pause_nav_callback=self._on_pause_nav,
            resume_nav_callback=self._on_resume_nav,
            stop_callback=self._on_stop,
            stop_nav_callback=self._on_stop_nav,
            whisper_model=whisper_model or WHISPER_MODEL,
        )

        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._processing = False
        self._processing_lock = threading.Lock()

        self._nav_active = False
        self._nav_paused = False
        self._nav_lock = threading.Lock()
        self._nav_thread = None
        self._nav_interval = NAV_FRAME_INTERVAL
        self._latest_depth_vis = None
        self._depth_vis_lock = threading.Lock()
        self._exit_mode = False

        self._last_spoken = ""
        self._last_hazard_check = 0.0

    def _speak(self, text: str, urgent: bool = False):
        """Enqueue speech; urgent interrupts current utterance."""
        if not text:
            return
        self._last_spoken = text
        self.tts.speak(text, priority=urgent, interrupt=urgent)

    def _on_hazard_alert(self, message: str, urgent: bool):
        with self._nav_lock:
            if self._nav_active or self._processing:
                return
        print(f"[Hazard] {message}")
        self._speak(message, urgent=urgent)

    # ── Help / repeat / pause ─────────────────────────────────────────────
    def _on_help(self):
        self._speak(
            "NaviLens commands. "
            "Navigate me or guide me for walking directions. "
            "Find the exit to leave a room. "
            "What is this or read the label for products. "
            "Describe surroundings or what's ahead for the scene. "
            "Say repeat to hear the last message. "
            "Pause navigation or stop navigation. "
            "Say stop to quit.",
            urgent=False,
        )

    def _on_repeat(self):
        if self._last_spoken:
            self._speak(self._last_spoken, urgent=False)
        else:
            self._speak("Nothing to repeat yet.", urgent=False)

    def _on_pause_nav(self):
        with self._nav_lock:
            if not self._nav_active:
                self._speak("Navigation is not running.", urgent=False)
                return
            self._nav_paused = True
        self._speak("Navigation paused. Say resume navigation to continue.", urgent=False)

    def _on_resume_nav(self):
        with self._nav_lock:
            if not self._nav_active:
                self._speak("Say navigate me to start.", urgent=False)
                return
            self._nav_paused = False
        self._speak("Resuming navigation.", urgent=False)

    # ── Scene (local) ─────────────────────────────────────────────────────
    def _on_scene_query(self, question: str):
        self._stop_nav_for_vision()
        self._run_vision_task(self._answer_scene_local, question)

    def _answer_scene_local(self, question: str, frame):
        q = question.lower()
        if any(k in q for k in ("ahead", "front", "path", "obstacle", "clear")):
            answer = describe_ahead(frame)
        else:
            answer = describe_scene(frame)
        print(f"[Local] Scene: '{answer}'")
        self._speak(answer, urgent=_is_urgent_speech(answer))

    # ── Product (Gemini or CLIP) ──────────────────────────────────────────
    def _on_identify_product(self, question: str):
        self._stop_nav_for_vision()
        self._run_vision_task(self._answer_product, question)

    def _answer_product(self, question: str, frame):
        detections = detect_objects(frame)
        crop, bbox, _ = get_product_crop(frame, detections)

        ocr_text = ""
        try:
            ocr_text = self._ocr.extract_text(frame, bbox)
        except Exception as e:
            print(f"[OCR] Skipped: {e}")

        if self._gemini_products:
            name, price = self._smart_id.identify_with_gemini(ocr_text, crop)
            source = "Gemini"
        else:
            name, price = self._smart_id.identify_local(ocr_text, crop)
            source = "local CLIP"

        if price:
            answer = f"This looks like {name}. Price on label: {price}."
        elif name and name != "object":
            answer = f"This looks like {name}."
        else:
            answer = (
                "I couldn't identify the product clearly. "
                "Hold it closer to the camera and try again."
            )

        print(f"[Product/{source}] '{answer}'")
        self._speak(answer, urgent=False)

    def _run_vision_task(self, task_fn, question: str):
        with self._processing_lock:
            if self._processing:
                self._speak("One moment, still processing.", urgent=False)
                return
            self._processing = True

        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is None:
            self._speak("Camera not ready yet.", urgent=False)
            with self._processing_lock:
                self._processing = False
            return

        def worker():
            try:
                task_fn(question, frame)
            except Exception as e:
                print(f"[Vision] Error: {e}")
                self._speak("Sorry, I couldn't process that. Try again.", urgent=False)
            finally:
                with self._processing_lock:
                    self._processing = False

        threading.Thread(target=worker, daemon=True).start()

    def _stop_nav_for_vision(self):
        with self._nav_lock:
            if self._nav_active:
                self._nav_active = False
                self._nav_paused = False
                with self._depth_vis_lock:
                    self._latest_depth_vis = None

    # ── Navigation (local) ────────────────────────────────────────────────
    def _on_nav_command(self, text: str):
        with self._nav_lock:
            if self._nav_active:
                self._speak(
                    "Navigation is active. Say stop navigation to exit.",
                    urgent=False,
                )
                return
        self._start_navigation(exit_mode=False)

    def _on_exit_command(self, text: str):
        with self._nav_lock:
            nav_on = self._nav_active

        if nav_on and self._exit_mode:
            self._speak(
                "Already looking for the exit.",
                urgent=False,
            )
            return

        if nav_on and not self._exit_mode:
            self._exit_mode = True
            self._nav.start_exit_mode()
            self._speak("Switching to exit mode.", urgent=False)
            return

        self._start_navigation(exit_mode=True)

    def _start_navigation(self, exit_mode: bool = False):
        with self._nav_lock:
            if self._nav_active:
                return
            self._nav_active = True
            self._nav_paused = False

        self._exit_mode = exit_mode
        self._nav.reset_state()
        self._hazard.set_enabled(False)

        if exit_mode:
            self._nav.start_exit_mode()
            self._speak("Exit mode activated. I'll guide you to the nearest door.", urgent=False)
        else:
            self._speak(
                "Navigation activated. Fully local guidance on this device.",
                urgent=False,
            )

        self._nav_thread = threading.Thread(target=self._nav_loop, daemon=True)
        self._nav_thread.start()

    def _stop_navigation(self):
        with self._nav_lock:
            if not self._nav_active:
                return
            self._nav_active = False
            self._nav_paused = False

        self._exit_mode = False
        self._hazard.set_enabled(HAZARD_WATCH_ENABLED)
        self._speak("Navigation stopped.", urgent=False)
        with self._depth_vis_lock:
            self._latest_depth_vis = None

    def _on_stop_nav(self):
        self._stop_navigation()

    def _nav_loop(self):
        while True:
            with self._nav_lock:
                if not self._nav_active:
                    break
                paused = self._nav_paused

            if paused:
                time.sleep(0.2)
                continue

            with self._frame_lock:
                frame = self._latest_frame
                if frame is not None:
                    frame = frame.copy()

            if frame is None:
                time.sleep(0.1)
                continue

            try:
                state_key, announcement, nav_vis = self._nav.analyze(frame)
                with self._depth_vis_lock:
                    self._latest_depth_vis = nav_vis
                if announcement:
                    print(f"[Nav] {announcement}")
                    urgent = _is_urgent_speech(announcement)
                    self._speak(announcement, urgent=urgent)
            except Exception as e:
                print(f"[Nav] Analysis error: {e}")

            time.sleep(self._nav_interval)

    def _on_stop(self):
        self._stop_navigation()
        self._speak("Stopping NaviLens.", urgent=False)
        self.running = False

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"[NaviLens] ERROR: Cannot open source '{self.source}'")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._nav.set_frame_size(frame_w, frame_h)
        set_vision_frame_size(frame_w, frame_h)

        self.voice.start()
        self.running = True

        # Audio test — must hear this or fix TTS/volume first
        time.sleep(0.5)
        if not self.tts.test_audio():
            print(
                "[NaviLens] TIP: On Mac, TTS uses the `say` command. "
                "Check System Settings → Sound → Output volume."
            )

        gemini_note = (
            "Gemini is on for product labels."
            if self._gemini_products
            else "Set GEMINI_API_KEY for product brands and prices."
        )
        self._speak(
            f"NaviLens ready. {gemini_note} "
            "Say navigate me, find the exit, what is this, or help.",
            urgent=False,
        )

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break

                with self._frame_lock:
                    self._latest_frame = frame

                # Idle hazard watch
                now = time.time()
                with self._nav_lock:
                    idle = not self._nav_active
                with self._processing_lock:
                    idle = idle and not self._processing

                if idle and HAZARD_WATCH_ENABLED and now - self._last_hazard_check > HAZARD_WATCH_INTERVAL:
                    self._last_hazard_check = now
                    self._hazard.check_frame(frame)

                if self.show_window:
                    display = frame.copy()
                    with self._nav_lock:
                        nav_on = self._nav_active
                        paused = self._nav_paused

                    if nav_on:
                        if paused:
                            status = "NAV PAUSED | resume | Q quit"
                            color = (0, 165, 255)
                        elif self._exit_mode:
                            status = "EXIT (local) | stop nav | Q quit"
                            color = (255, 0, 255)
                        else:
                            status = "NAV (local) | stop nav | Q quit"
                            color = (0, 200, 255)
                    elif self._processing:
                        status = "Vision... | Q quit"
                        color = (255, 200, 0)
                    else:
                        status = "Listening + hazard watch | Q quit"
                        color = (0, 255, 0)

                    cv2.putText(
                        display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )
                    cv2.imshow("NaviLens", display)

                    if nav_on:
                        with self._depth_vis_lock:
                            guide_vis = self._latest_depth_vis
                        if guide_vis is not None:
                            cv2.imshow("NaviLens — Guidance", guide_vis)
                    else:
                        try:
                            cv2.destroyWindow("NaviLens — Guidance")
                        except cv2.error:
                            pass

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            self._stop_navigation()
            self.running = False
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()
            self.voice.stop()
            self._speak("NaviLens shutting down. Stay safe.", urgent=False)
            time.sleep(3)
            self.tts.shutdown()
            print("[NaviLens] Goodbye.")
