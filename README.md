# NaviLens
### Voice-Activated Visual Navigation & Scene Description for the Blind

---

## Overview
NaviLens is a wearable AI assistant that provides real-time audio navigation guidance and scene descriptions to visually impaired users via a spectacle-mounted camera, microphone, and bone-conduction speaker.

**All processing is LOCAL — no API calls, no internet required.**

---

## File Structure
```
navilens/
├── main.py           ← Entry point (run this)
├── navilens.py       ← Core orchestrator
├── detection.py      ← YOLOv8 object detection (80 COCO classes)
├── spatial.py        ← Spatial reasoning engine (zones + proximity)
├── depth_engine.py   ← MiDaS monocular depth estimation
├── navigation.py     ← Depth + YOLO navigation pipeline
├── tts_engine.py     ← Windows System.Speech TTS (built-in)
├── voice_input.py    ← Whisper voice command listener
└── requirements.txt
```

---

## Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** TTS uses Windows built-in System.Speech. Only works on Windows 10/11.

> **Linux / Raspberry Pi extra step** (for PyAudio):
> ```bash
> sudo apt install portaudio19-dev python3-pyaudio
> ```

> **Linux / Raspberry Pi TTS:** The TTS engine currently requires Windows. On Linux, install `pyttsx3` and `espeak`:
> ```bash
> pip install pyttsx3
> sudo apt install espeak
> ```
> Then modify `tts_engine.py` to use pyttsx3 instead of PowerShell.

---

## Running

### Basic (webcam, YOLOv8-nano)
```bash
python main.py
```

### Use a video file (no mic needed)
```bash
python main.py --source path/to/video.mp4 --no-show
```

### All options
```bash
python main.py --help
```

---

## Voice Commands

| You say | What it does |
|---------|-------------|
| "What's ahead?" / "Describe surroundings" | One-shot YOLO detection + spatial description |
| "Navigate me" / "Guide me" | Starts continuous depth-based navigation guidance |
| "Stop navigation" | Stops navigation mode |
| "Stop" / "Quit" | Shuts down NaviLens |

---

## How It Works

### Describe Mode
Voice command -> Whisper STT -> YOLOv8 object detection -> spatial reasoning (zone + proximity) -> TTS

### Navigation Mode
Voice command -> Whisper STT -> continuous loop:
  - MiDaS depth estimation (3 zones: left/center/right)
  - YOLOv8 critical object detection (persons, obstacles)
  - Event-based guidance announcements (only on state change)

### Spatial Zone System
```
 LEFT (0-38%) | CENTER (38-62%) | RIGHT (62-100%)
```

Proximity (by bounding box area):
- > 15% of frame -> VERY NEAR
- > 7%  of frame -> NEAR
- > 2%  of frame -> FAR
- <= 2% of frame -> DISTANT

---

## Hardware
Currently designed for Windows with:
- Webcam or USB camera
- USB microphone
- Speakers or headphones (for TTS output)

For Raspberry Pi deployment:
- TTS backend needs to be changed to pyttsx3 (see Setup notes)
- Use `yolov8n.pt` for best performance (~8-12 FPS on Pi 4)
