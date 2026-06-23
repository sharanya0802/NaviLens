"""
NaviLens - Voice-Activated Visual Assistant + Navigation
Main entry point. Run with: python main.py
"""

import argparse
from navilens import NaviLens


def parse_args():
    parser = argparse.ArgumentParser(
        description="NaviLens: Voice-guided navigation and scene description for the visually impaired"
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: 0 for webcam (default: 0)",
    )
    parser.add_argument(
        "--model", type=str, default="yolo26l.pt",
        help="YOLO model (yolov8n/s/m/l/x.pt or yolo26n/s/m/l/x.pt)",
    )
    parser.add_argument(
        "--no-show", action="store_true", default=False,
        help="Hide camera window",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("  NaviLens — Voice-Guided Navigation & Scene Description")
    print("  RV College of Engineering | Team UH38")
    print("=" * 60)
    print(f"  Source   : {args.source}")
    print(f"  Model    : {args.model}")
    print("=" * 60)
    print("  VOICE COMMANDS:")
    print("    'What's ahead?'   — Describe objects in front of you")
    print("    'Describe'        — Full scene description")
    print("    'Navigate me'     — Start depth-based navigation guidance")
    print("    'Stop navigation' — Stop navigation mode")
    print("    'Stop'            — Quit NaviLens")
    print("=" * 60 + "\n")

    app = NaviLens(
        source=args.source,
        show_window=not args.no_show,
        model=args.model,
    )
    app.run()
