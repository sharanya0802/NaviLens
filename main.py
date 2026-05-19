"""
NaviLens - Voice-Activated Visual Assistant + Navigation
Main entry point. Run with: python main.py --gemini-api-key YOUR_KEY
"""

import argparse
import os
from navilens import NaviLens


def parse_args():
    parser = argparse.ArgumentParser(
        description="NaviLens: Visual Q&A + Depth-Based Navigation for the Visually Impaired"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0 for webcam (default: 0)",
    )
    parser.add_argument(
        "--gemini-api-key", type=str, default=None,
        help="Gemini 2.5 Flash API key (or set GEMINI_API_KEY env var)",
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
    print("  NaviLens — Visual Assistant + Navigation")
    print("  RV College of Engineering | Team UH38")
    print("=" * 60)
    print(f"  Source   : {args.source}")
    print(f"  Whisper  : {args.whisper_model}")
    print(f"  TTS      : {args.tts_engine}")
    print(f"  Gemini   : {'Connected' if gemini_key else 'NOT SET'}")
    print("=" * 60)
    print("  QUESTION MODE (→ Gemini):")
    print("    'What am I holding?'")
    print("    'Read this label'")
    print("    'What color is this?'")
    print()
    print("  NAVIGATION MODE (→ Local depth + path planning):")
    print("    'Navigate me forward' / 'Guide me'")
    print("    'Find the exit' / 'Help me leave the room'")
    print("    'Stop navigation'")
    print()
    print("  Say 'stop' to quit.")
    print("=" * 60 + "\n")

    if not gemini_key:
        print("WARNING: No Gemini API key. Q&A mode disabled. Navigation still works.")

    app = NaviLens(
        source=args.source,
        show_window=not args.no_show,
        tts_engine=args.tts_engine,
        gemini_api_key=gemini_key,
        whisper_model=args.whisper_model,
    )
    app.run()
