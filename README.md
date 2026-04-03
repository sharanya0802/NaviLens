# NaviLens 🦯
### Voice Activated Visual Recognition and Audio Navigation for the Blind


---

## Project Overview
NaviLens is a wearable AI assistant that provides real-time audio navigation guidance to visually
impaired users via a spectacle-mounted camera, microphone, and bone-conduction speaker.

---

## File Structure
```
navilens/
├── main.py           ← Entry point (run this)
├── navilens.py       ← Core orchestrator
├── detection.py      ← YOLOv8 object detection
├── spatial.py        ← Spatial reasoning engine (zones + proximity)
├── tts_engine.py     ← Priority-based TTS (pyttsx3 / gTTS)
├── voice_input.py    ← Voice command listener (Google STT)
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

> **Linux / Raspberry Pi extra step** (for PyAudio):
> ```bash
> sudo apt install portaudio19-dev python3-pyaudio
> ```

---

## Running

### Basic (webcam, YOLOv8-nano, pyttsx3 TTS)
```bash
python main.py
```

### Use a larger, more accurate model
```bash
python main.py --model yolov8s.pt
```

### Use gTTS (better voice, needs internet)
```bash
python main.py --tts-engine gtts
```

### Test with a video file (no mic needed)
```bash
python main.py --source path/to/video.mp4 --no-voice-input
```

### All options
```bash
python main.py --help
```

---

## Voice Commands (say these aloud)
| What you say              | What it does                        |
|---------------------------|-------------------------------------|
| "What's ahead?"           | Describes objects in your path      |
| "Describe surroundings"   | Full scene description (all zones)  |
| "Help"                    | Lists available commands            |
| "Stop"                    | Shuts down NaviLens                 |

---

## Spatial Zone System
```
┌──────────────────────────────┐
│  LEFT  │    CENTER   │ RIGHT │
│ 0–38%  │   38–62%   │ 62–100│  (frame width %)
└──────────────────────────────┘

Proximity (by bounding box area):
  > 15% of frame → VERY NEAR (Stop!)
  > 7%  of frame → NEAR (Caution)
  > 2%  of frame → FAR (Awareness)
  ≤ 2%  of frame → DISTANT
```

---

## Hardware Integration (Phase 5)
When ready to deploy on Raspberry Pi:
1. Change `--source 0` to the Pi camera index (usually `0`)
2. For Pi Camera Module v3: use `libcamera` or `picamera2` and pipe frames into OpenCV
3. PyAudio will use the USB mic automatically if it's the only audio device
4. Bone-conduction speaker connects via 3.5mm or Bluetooth (pyttsx3 uses system audio)
5. Use `--model yolov8n.pt` (nano) for best performance on Pi 4

---

## Model Performance Reference
| Model       | Size   | Speed (RPi 4) | mAP50-95 |
|-------------|--------|---------------|----------|
| yolov8n.pt  | 6 MB   | ~8–12 FPS     | 37.3     |
| yolov8s.pt  | 22 MB  | ~4–6 FPS      | 44.9     |
| yolov8m.pt  | 52 MB  | ~2–3 FPS      | 50.2     |

> Weights are auto-downloaded on first run from Ultralytics servers.

---

## Future Additions
- [ ] Google Gemini 2.0 Flash integration for richer scene descriptions
- [ ] FastAPI dashboard for remote monitoring
- [ ] Whisper STT (offline alternative to Google STT)
- [ ] Edge-optimized model via TensorRT / ONNX export for faster Pi inference
