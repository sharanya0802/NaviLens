"""
navilens.py — Core Orchestrator (Dual Pipeline)

  1. LOCAL NAVIGATION  — MiDaS depth + spatial map + path planner → TTS
  2. LOCAL SCENE       — YOLO + spatial reasoning → TTS (what's ahead, describe)
  3. PRODUCT ID (API)  — YOLO crop + OCR + Gemini Vision → TTS (labels, brands, prices)

Navigation and scene description never call Gemini.
Gemini is only used when the user asks to identify a product / read a label.
"""

import threading
import time

import cv2

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


class NaviLens:
    def __init__(
        self,
        source="0",
        show_window=True,
        tts_engine="pyttsx3",
        gemini_api_key=None,
        whisper_model="base",
    ):
        self.source = int(source) if source.isdigit() else source
        self.show_window = show_window
        self.running = False

        print("[NaviLens] Initialising TTS engine...")
        self.tts = TTSEngine(engine=tts_engine)

        print("[NaviLens] Initialising local navigation engine...")
        self._nav = NavigationEngine()

        print("[NaviLens] Initialising local vision (scene + product crop)...")
        self._ocr = OCREngine()
        self._smart_id = SmartIdentifier(api_key=gemini_api_key)
        self._gemini_products = self._smart_id._gemini_available

        print("[NaviLens] Initialising Whisper voice listener...")
        self.voice = WhisperListener(
            nav_callback=self._on_nav_command,
            exit_callback=self._on_exit_command,
            scene_callback=self._on_scene_query,
            product_callback=self._on_identify_product,
            stop_callback=self._on_stop,
            stop_nav_callback=self._on_stop_nav,
            whisper_model=whisper_model,
        )

        self._frame_lock = threading.Lock()
        self._latest_frame = None

        self._processing = False
        self._processing_lock = threading.Lock()

        self._nav_active = False
        self._nav_lock = threading.Lock()
        self._nav_thread = None
        self._nav_interval = 0.5
        self._latest_depth_vis = None
        self._depth_vis_lock = threading.Lock()
        self._exit_mode = False

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 1: LOCAL SCENE (YOLO + spatial — no API)
    # ══════════════════════════════════════════════════════════════════
    def _on_scene_query(self, question: str):
        """Describe surroundings or what's ahead using local models only."""
        self._stop_nav_for_vision()
        self._run_vision_task(self._answer_scene_local, question)

    def _answer_scene_local(self, question: str, frame):
        q = question.lower()
        if any(k in q for k in ("ahead", "front", "path", "obstacle", "clear")):
            answer = describe_ahead(frame)
        else:
            answer = describe_scene(frame)
        print(f"[Local] Scene: '{answer}'")
        self.tts.speak(answer, priority=True)

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 2: PRODUCT ID (Gemini Vision — products / labels only)
    # ══════════════════════════════════════════════════════════════════
    def _on_identify_product(self, question: str):
        """Identify product, brand, price via Gemini; CLIP+OCR fallback if offline."""
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
        self.tts.speak(answer, priority=True)

    def _run_vision_task(self, task_fn, question: str):
        with self._processing_lock:
            if self._processing:
                self.tts.speak("One moment, still processing.", priority=True)
                return
            self._processing = True

        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is None:
            self.tts.speak("Camera not ready yet.", priority=True)
            with self._processing_lock:
                self._processing = False
            return

        def worker():
            try:
                task_fn(question, frame)
            except Exception as e:
                print(f"[Vision] Error: {e}")
                self.tts.speak("Sorry, I couldn't process that. Try again.", priority=True)
            finally:
                with self._processing_lock:
                    self._processing = False

        threading.Thread(target=worker, daemon=True).start()

    def _stop_nav_for_vision(self):
        with self._nav_lock:
            if self._nav_active:
                self._nav_active = False
                print("[Nav] Stopped for vision query")
                with self._depth_vis_lock:
                    self._latest_depth_vis = None

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 3: LOCAL NAVIGATION (depth + path planner — no API)
    # ══════════════════════════════════════════════════════════════════
    def _on_nav_command(self, text: str):
        with self._nav_lock:
            if self._nav_active:
                self.tts.speak(
                    "Navigation is active. Say stop navigation to exit.",
                    priority=True,
                )
                return
        self._start_navigation(exit_mode=False)

    def _on_exit_command(self, text: str):
        with self._nav_lock:
            nav_on = self._nav_active

        if nav_on and self._exit_mode:
            self.tts.speak(
                "Already looking for the exit. I'll let you know when I find it.",
                priority=True,
            )
            return

        if nav_on and not self._exit_mode:
            self._exit_mode = True
            self._nav.start_exit_mode()
            self.tts.speak(
                "Switching to exit mode. I'll guide you to the nearest door.",
                priority=True,
            )
            return

        self._start_navigation(exit_mode=True)

    def _start_navigation(self, exit_mode: bool = False):
        with self._nav_lock:
            if self._nav_active:
                return
            self._nav_active = True

        self._exit_mode = exit_mode
        self._nav.reset_state()

        if exit_mode:
            self._nav.start_exit_mode()
            print("[Nav] Exit-seeking mode ACTIVATED")
            self.tts.speak(
                "Exit mode activated. I'll guide you to the nearest door.",
                priority=True,
            )
        else:
            print("[Nav] Navigation mode ACTIVATED")
            self.tts.speak(
                "Navigation mode activated. "
                "Turn-by-turn guidance is fully local on this device.",
                priority=True,
            )

        self._nav_thread = threading.Thread(target=self._nav_loop, daemon=True)
        self._nav_thread.start()

    def _stop_navigation(self):
        with self._nav_lock:
            if not self._nav_active:
                return
            self._nav_active = False

        self._exit_mode = False
        print("[Nav] Navigation DEACTIVATED")
        self.tts.speak("Navigation stopped.", priority=True)
        with self._depth_vis_lock:
            self._latest_depth_vis = None

    def _on_stop_nav(self):
        self._stop_navigation()

    def _nav_loop(self):
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
                state_key, announcement, nav_vis = self._nav.analyze(frame)
                with self._depth_vis_lock:
                    self._latest_depth_vis = nav_vis
                if announcement:
                    print(f"[Nav] {announcement}")
                    self.tts.speak(announcement, priority=True)
            except Exception as e:
                print(f"[Nav] Analysis error: {e}")

            time.sleep(self._nav_interval)

    def _on_stop(self):
        self._stop_navigation()
        self.tts.speak("Stopping NaviLens.", priority=True)
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

        gemini_status = (
            "Product identification uses Gemini."
            if self._gemini_products
            else "Product ID uses local CLIP only; set GEMINI_API_KEY for brands and prices."
        )
        self.tts.speak(
            "NaviLens ready. Navigation and scene description run locally on this device. "
            f"{gemini_status} "
            "Say navigate me to walk, find the exit to leave a room, "
            "what is this for a product, or describe surroundings.",
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
                        if self._exit_mode:
                            status = "EXIT (local) | stop navigation | Q quit"
                            color = (255, 0, 255)
                        else:
                            status = "NAV (local) | stop navigation | Q quit"
                            color = (0, 200, 255)
                    elif self._processing:
                        status = "Vision... | Q quit"
                        color = (255, 200, 0)
                    else:
                        status = "Listening | Q quit"
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
            self.tts.speak("NaviLens shutting down. Stay safe.", priority=True)
            time.sleep(2)
            self.tts.shutdown()
            print("[NaviLens] Goodbye.")
