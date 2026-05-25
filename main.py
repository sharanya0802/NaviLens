"""
NaviLens - Voice-Activated Visual Assistant + Navigation
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="NaviLens: Local navigation + Gemini product ID"
    )
    parser.add_argument("--source", type=str, default="0", help="Webcam index or video path")
    parser.add_argument(
        "--gemini-api-key", type=str, default=None,
        help="Gemini key for products/labels only (or GEMINI_API_KEY)",
    )
    parser.add_argument(
        "--whisper-model", type=str, default=None,
        choices=["tiny", "base", "small", "medium"],
        help="Whisper STT size (default: small)",
    )
    parser.add_argument(
        "--depth", type=str, default=None,
        choices=["small", "hybrid", "large"],
        help="Depth: small=fast, hybrid=recommended, large=best",
    )
    parser.add_argument(
        "--yolo-model", type=str, default=None,
        help="YOLO weights, e.g. yolov8s.pt",
    )
    parser.add_argument(
        "--tts-engine", type=str, default=None,
        choices=["auto", "say", "pyttsx3", "edge", "gtts"],
        help="TTS: auto uses macOS `say` on Mac (most reliable)",
    )
    parser.add_argument("--no-show", action="store_true", help="Hide camera windows")
    parser.add_argument("--no-hazard-watch", action="store_true", help="Disable idle alerts")
    return parser.parse_args()


def _apply_env(args):
    if args.depth:
        os.environ["NAVILENS_DEPTH"] = args.depth
    if args.yolo_model:
        os.environ["NAVILENS_YOLO"] = args.yolo_model
    if args.whisper_model:
        os.environ["NAVILENS_WHISPER"] = args.whisper_model
    if args.tts_engine:
        os.environ["NAVILENS_TTS"] = args.tts_engine
    if args.no_hazard_watch:
        os.environ["NAVILENS_HAZARD_WATCH"] = "0"


if __name__ == "__main__":
    args = parse_args()
    _apply_env(args)

    # Import after env so config.py picks up overrides
    import importlib
    import config as cfg
    importlib.reload(cfg)

    from navilens import NaviLens

    gemini_key = args.gemini_api_key or os.environ.get("GEMINI_API_KEY")

    print("=" * 60)
    print("  NaviLens")
    print("=" * 60)
    print(f"  Depth    : {cfg.DEPTH_PROFILE}")
    print(f"  YOLO     : {cfg.YOLO_MODEL}")
    print(f"  Whisper  : {cfg.WHISPER_MODEL}")
    print(f"  TTS      : {cfg.TTS_ENGINE}")
    print(f"  Gemini   : {'ON (products)' if gemini_key else 'OFF'}")
    print(f"  Hazard   : {'ON' if cfg.HAZARD_WATCH_ENABLED else 'OFF'}")
    print("=" * 60)
    print("  TTS tip: On Mac, default is `say` — you should hear an audio test at startup.")
    print("=" * 60)

    app = NaviLens(
        source=args.source,
        show_window=not args.no_show,
        tts_engine=args.tts_engine,
        gemini_api_key=gemini_key,
        whisper_model=args.whisper_model,
    )
    app.run()
