"""
NaviLens - Voice Activated Visual Recognition and Audio Navigation
Main entry point. Run with: python main.py
"""

import argparse
import sys
from navilens import NaviLens


def parse_args():
    parser = argparse.ArgumentParser(description="NaviLens: AI Navigation Assistant for the Visually Impaired")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: 0 for webcam, path for video file, or RTSP URL (default: 0)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="YOLOv8 model size (n=nano fastest, x=xlarge most accurate)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Detection confidence threshold (default: 0.45)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Show live detection window (disable on Pi with no display)"
    )
    parser.add_argument(
        "--tts-engine",
        type=str,
        default="pyttsx3",
        choices=["pyttsx3", "gtts"],
        help="TTS engine to use (pyttsx3 = offline, gtts = online)"
    )
    parser.add_argument(
        "--no-voice-input",
        action="store_true",
        default=False,
        help="Disable voice command input (useful for testing)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 55)
    print("  NaviLens — AI Navigation Assistant")
    print("  RV College of Engineering | Team UH38")
    print("=" * 55)
    print(f"  Model   : {args.model}")
    print(f"  Source  : {args.source}")
    print(f"  Conf    : {args.conf}")
    print(f"  TTS     : {args.tts_engine}")
    print("=" * 55)
    print("  Voice commands: 'what's ahead' | 'describe surroundings'")
    print("                  'stop' | 'help'")
    print("  Press Q in video window to quit.")
    print("=" * 55 + "\n")

    app = NaviLens(
        source=args.source,
        model_name=args.model,
        conf_threshold=args.conf,
        show_window=args.show,
        tts_engine=args.tts_engine,
        enable_voice_input=not args.no_voice_input,
    )
    app.run()
