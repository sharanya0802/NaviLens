"""
navilens.py — Core Orchestrator (Dual Pipeline)

Two co-existing pipelines:
  1. GEMINI Q&A   — Semantic/visual questions → frame + question → Gemini → TTS
  2. LOCAL NAV    — Navigation commands → MiDaS depth + YOLO → event-based TTS

Routing is keyword-based (see voice_input.py).
Gemini pipeline is NEVER used for navigation decisions.
Navigation pipeline is FULLY LOCAL.
"""

import threading
import time
# pyrefly: ignore [missing-import]
import cv2
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

        # ── TTS ──────────────────────────────────────────────────────
        print("[NaviLens] Initialising TTS engine...")
        self.tts = TTSEngine(engine=tts_engine)

        # ── Gemini (unchanged) ───────────────────────────────────────
        self._gemini_client = None
        self._gemini_available = False
        if gemini_api_key:
            self._init_gemini(gemini_api_key)

        # ── Navigation engine (local) ────────────────────────────────
        print("[NaviLens] Initialising navigation engine...")
        self._nav = NavigationEngine()

        # ── Whisper listener (with routing) ──────────────────────────
        print("[NaviLens] Initialising Whisper voice listener...")
        self.voice = WhisperListener(
            question_callback=self._on_question,
            nav_callback=self._on_nav_command,
            exit_callback=self._on_exit_command,
            stop_callback=self._on_stop,
            stop_nav_callback=self._on_stop_nav,
            whisper_model=whisper_model,
        )

        # ── Frame state ──────────────────────────────────────────────
        self._frame_lock = threading.Lock()
        self._latest_frame = None

        # ── Gemini processing guard ──────────────────────────────────
        self._processing = False
        self._processing_lock = threading.Lock()

        # ── Navigation mode state ────────────────────────────────────
        self._nav_active = False
        self._nav_lock = threading.Lock()
        self._nav_thread = None
        self._nav_interval = 0.5    # seconds between nav analysis frames
        self._latest_depth_vis = None
        self._depth_vis_lock = threading.Lock()

    def _init_gemini(self, api_key: str):
        try:
            from google import genai
            self._gemini_client = genai.Client(api_key=api_key)
            self._gemini_available = True
            print("[NaviLens] Gemini 2.0 Flash connected.")
        except ImportError:
            print("[NaviLens] google-genai not installed. Run: pip install google-genai")
        except Exception as e:
            print(f"[NaviLens] Gemini error: {e}")

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 1: GEMINI VISUAL Q&A  (unchanged from original)
    # ══════════════════════════════════════════════════════════════════
    def _on_question(self, question: str):
        """Called when Whisper routes a semantic question to Gemini.
        Auto-stops navigation mode — Gemini takes priority."""
        # Stop navigation if active (user switched to Q&A mode)
        with self._nav_lock:
            if self._nav_active:
                self._nav_active = False
                print("[Nav] ═══ Navigation auto-stopped (Gemini question) ═══")
                with self._depth_vis_lock:
                    self._latest_depth_vis = None

        with self._processing_lock:
            if self._processing:
                print("[NaviLens] Already processing a question, skipping.")
                return
            self._processing = True

        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None

        if frame is None:
            self.tts.speak("Camera not ready yet.", priority=True)
            with self._processing_lock:
                self._processing = False
            return

        if not self._gemini_available:
            self.tts.speak("Gemini is not configured. Please provide an API key.", priority=True)
            with self._processing_lock:
                self._processing = False
            return

        threading.Thread(
            target=self._ask_gemini,
            args=(question, frame),
            daemon=True,
        ).start()

    def _ask_gemini(self, question: str, frame):
        """Send frame + question to Gemini, speak the answer."""
        try:
            from google.genai import types

            success, jpeg_bytes = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )
            if not success:
                self.tts.speak("Failed to capture image.", priority=True)
                return

            image_part = types.Part.from_bytes(
                data=jpeg_bytes.tobytes(),
                mime_type="image/jpeg",
            )

            prompt = (
                f"You are a visual assistant for a visually impaired person. "
                f"They are asking: \"{question}\"\n\n"
                f"Look at the image from their camera and answer their question. "
                f"Be concise (1-2 sentences max), accurate, and helpful. "
                f"If they ask about an object, identify it specifically "
                f"(brand name, product type, color, etc.). "
                f"If you see a price (Rs., MRP, ₹), mention it. "
                f"Speak naturally as if talking to them."
            )

            print(f"[Gemini] Asking: '{question}'")
            response = self._gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[image_part, prompt],
            )

            answer = response.text.strip()
            print(f"[Gemini] Answer: '{answer}'")
            self.tts.speak(answer, priority=True)

        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                self.tts.speak("Please wait a moment and try again.", priority=True)
                print("[Gemini] Rate limited.")
            else:
                print(f"[Gemini] Error: {e}")
                self.tts.speak("Sorry, I couldn't process that. Try again.", priority=True)
        finally:
            with self._processing_lock:
                self._processing = False

    # ══════════════════════════════════════════════════════════════════
    #  PIPELINE 2: LOCAL NAVIGATION  (depth + YOLO, fully local)
    # ══════════════════════════════════════════════════════════════════
    def _on_nav_command(self, text: str):
        """Called when Whisper routes a general navigation command."""
        with self._nav_lock:
            if self._nav_active:
                # Already navigating — treat as status request
                self.tts.speak("Navigation is active. Say stop navigation to exit.", priority=True)
                return

        self._start_navigation(exit_mode=False)

    def _on_exit_command(self, text: str):
        """Called when the user says 'find the exit' / 'help me leave the room'."""
        with self._nav_lock:
            nav_on = self._nav_active

        if nav_on and self._exit_mode:
            self.tts.speak(
                "Already looking for the exit. I'll let you know when I find it.",
                priority=True,
            )
            return

        if nav_on and not self._exit_mode:
            # Switch from standard nav to exit mode without restarting the loop
            self._exit_mode = True
            self._nav.start_exit_mode()
            self.tts.speak(
                "Switching to exit mode. I'll guide you to the nearest door.",
                priority=True,
            )
            return

        # Start fresh in exit mode
        self._start_navigation(exit_mode=True)

    def _start_navigation(self, exit_mode: bool = False):
        """Activate continuous local navigation mode (standard or exit-seeking)."""
        with self._nav_lock:
            if self._nav_active:
                return
            self._nav_active = True

        self._exit_mode = exit_mode
        self._nav.reset_state()

        if exit_mode:
            self._nav.start_exit_mode()
            print("[Nav] ═══ Exit-seeking mode ACTIVATED ═══")
            self.tts.speak(
                "Exit mode activated. "
                "I'll find the nearest door and guide you out. "
                "Please look around slowly if I ask you to scan.",
                priority=True,
            )
        else:
            print("[Nav] ═══ Navigation mode ACTIVATED ═══")
            self.tts.speak(
                "Navigation mode activated. "
                "I'll give you turn-by-turn directions based on free space ahead.",
                priority=True,
            )

        self._nav_thread = threading.Thread(
            target=self._nav_loop,
            daemon=True,
        )
        self._nav_thread.start()

    def _stop_navigation(self):
        """Deactivate navigation mode (both standard and exit)."""
        with self._nav_lock:
            if not self._nav_active:
                return
            self._nav_active = False

        self._exit_mode = False
        print("[Nav] ═══ Navigation mode DEACTIVATED ═══")
        self.tts.speak("Navigation stopped.", priority=True)

        # Clear depth visualization
        with self._depth_vis_lock:
            self._latest_depth_vis = None

    def _on_stop_nav(self):
        """Called when user says 'stop navigation'."""
        self._stop_navigation()

    def _nav_loop(self):
        """Continuous navigation analysis loop (runs in background thread)."""
        while True:
            with self._nav_lock:
                if not self._nav_active:
                    break

            # Get latest frame
            with self._frame_lock:
                frame = self._latest_frame
                if frame is not None:
                    frame = frame.copy()

            if frame is None:
                time.sleep(0.1)
                continue

            # Analyze with depth + YOLO
            try:
                state_key, announcement, depth_vis = self._nav.analyze(frame)

                # Store depth visualization for display
                with self._depth_vis_lock:
                    self._latest_depth_vis = depth_vis

                # Speak only if state changed
                if announcement:
                    print(f"[Nav] {announcement}")
                    self.tts.speak(announcement, priority=True)

            except Exception as e:
                print(f"[Nav] Analysis error: {e}")

            # Throttle to ~2 fps
            time.sleep(self._nav_interval)

    # ══════════════════════════════════════════════════════════════════
    #  APP LIFECYCLE
    # ══════════════════════════════════════════════════════════════════
    def _on_stop(self):
        """Called when user says stop/quit/exit."""
        self._stop_navigation()
        self.tts.speak("Stopping NaviLens.", priority=True)
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

        self.voice.start()
        self.running = True
        self.tts.speak(
            "NaviLens ready. "
            "Ask me a question, say navigate me to start navigation, "
            "or say find the exit to guide you out of the room.",
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

                    # Status bar
                    with self._nav_lock:
                        nav_on = self._nav_active

                    if nav_on:
                        if self._exit_mode:
                            status = "EXIT MODE | Say 'stop navigation' to cancel | Q to quit"
                            color  = (255, 0, 255)
                        else:
                            status = "NAVIGATING | Say 'stop navigation' to exit | Q to quit"
                            color  = (0, 200, 255)
                    elif self._processing:
                        status = "Thinking... | Q to quit"
                        color = (255, 200, 0)
                    else:
                        status = "Listening... | Q to quit"
                        color = (0, 255, 0)

                    cv2.putText(
                        display, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                    )
                    cv2.imshow("NaviLens", display)

                    # Guidance HUD (path + directions) when navigating
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
