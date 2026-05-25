"""
NaviLens - Voice-Activated Visual Assistant + Navigation
Main entry point. Run with: python main.py
Optional: --gemini-api-key for product/label identification only.
"""

import argparse
import os
from navilens import NaviLens


def parse_args():
    parser = argparse.ArgumentParser(
        description="NaviLens: Local navigation + Gemini product ID"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0 for webcam (default: 0)",
    )
    parser.add_argument(
        "--gemini-api-key", type=str, default=None,
        help="Gemini API key for product/label ID only (or GEMINI_API_KEY env)",
    )
    parser.add_argument(
        "--whisper-model", type=str, default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--tts-engine", type=str, default="pyttsx3",
        choices=["pyttsx3", "gtts"],
        help="TTS engine (default: pyttsx3)",
    )
    parser.add_argument(
        "--no-show", action="store_true", default=False,
        help="Hide camera window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")

    print("=" * 60)
    print("  NaviLens — Local Navigation + Product Vision")
    print("  RV College of Engineering | Team UH38")
    print("=" * 60)
    print(f"  Source   : {args.source}")
    print(f"  Whisper  : {args.whisper_model}")
    print(f"  TTS      : {args.tts_engine}")
    print(f"  Gemini   : {'ON (products only)' if gemini_key else 'OFF (CLIP fallback)'}")
    print("=" * 60)
    print("  LOCAL (no API):")
    print("    Navigation — 'Navigate me' / 'Find the exit'")
    print("    Scene      — 'What's ahead?' / 'Describe surroundings'")
    print()
    print("  GEMINI (products only):")
    print("    'What is this?' / 'Read this label' / 'What brand is this?'")
    print()
    print("  Say 'stop navigation' or 'stop' to quit.")
    print("=" * 60 + "\n")

    if not gemini_key:
        print(
            "Note: No Gemini key — product ID uses local CLIP+OCR only.\n"
            "      Set GEMINI_API_KEY for brand names and prices on packages.\n"
        )

    app = NaviLens(
        source=args.source,
        show_window=not args.no_show,
        tts_engine=args.tts_engine,
        gemini_api_key=gemini_key,
        whisper_model=args.whisper_model,
    )
    app.run()
