# NaviLens

**Voice-activated visual assistant and spatial navigation for blind and low-vision users.**

NaviLens runs on a wearable camera + microphone and speaks concise, actionable guidance — not object-detection labels. It combines monocular depth (MiDaS), room-level spatial mapping, radial path planning, and natural-language TTS.

---

## How it works

```
Camera frame
    │
    ├─► MiDaS depth ──► SpatialMap (walls, openings, room type)
    │
    ├─► YOLOv8 ───────► Obstacle semantics (chair, person, stairs…)
    │
    └─► PathPlanner ──► Best free-space heading (−45° … +45°)
              │
              ├─► GuidanceComposer ──► "Turn left along the corridor."
              │
              └─► ExitGuidance + DoorTracker (exit mode)
                        └── "Door at your 2 o'clock, about 5 steps."
```

**Two pipelines (by voice intent):**

| Mode | Backend | Example commands |
|------|---------|------------------|
| **Q&A** | Gemini 2.0 Flash | "What am I holding?", "Read this label" |
| **Navigation** | 100% local (depth + YOLO) | "Navigate forward", "Guide me" |
| **Exit** | Local + door/opening fusion | "Find the exit", "Help me leave the room" |

Navigation never calls the cloud. Gemini is only used for semantic questions.

---

## File structure

```
NaviLens/
├── main.py              ← Entry point
├── navilens.py          ← Orchestrator (camera, voice routing, TTS)
├── navigation.py        ← Full spatial nav pipeline + guidance HUD
├── depth_engine.py      ← MiDaS monocular depth
├── spatial_map.py       ← Walls, openings, room context (temporal)
├── path_planner.py      ← Radial ray-cast path scoring
├── guidance_composer.py ← Turn-by-turn speech state machine
├── exit_detector.py     ← Door / opening tracker (clock-face bearing)
├── exit_navigation.py   ← "Leave the room" narrator
├── voice_input.py       ← Whisper + intent routing
├── tts_engine.py        ← Priority speech queue
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / Raspberry Pi** (PyAudio):

```bash
sudo apt install portaudio19-dev python3-pyaudio
```

Set Gemini key for Q&A (optional for navigation):

```bash
export GEMINI_API_KEY=your_key_here
```

---

## Running

```bash
# Webcam + local navigation (navigation works without Gemini)
python main.py

# With Gemini Q&A
python main.py --gemini-api-key YOUR_KEY

# Hide preview windows (audio-only)
python main.py --no-show
```

---

## Voice commands

| Say | Action |
|-----|--------|
| "Navigate me forward" / "Guide me" | Start turn-by-turn navigation |
| "Find the exit" / "Help me leave the room" | Exit mode — door + depth openings |
| "Stop navigation" | End navigation only |
| "What is this?" / "Read the label" | Gemini visual Q&A |
| "Stop" | Quit application |

---

## What you hear (examples)

**Corridor:** "You're in a corridor. Continue straight along the corridor."

**Obstacle:** "Veer left through the open space. Watch out, chair on your right."

**Blocked:** "Dead end ahead. Turn to your left where there's open space."

**Exit:** "Exit found! The door is straight ahead, at your 12 o'clock. About 4 steps away. Continue straight towards the exit."

---

## Guidance window (for testers / caregivers)

When navigation is active, a second window **NaviLens — Guidance** shows:

- Planned path rays (green = best route)
- Compass needle (STRAIGHT / VEER LEFT / TURN RIGHT)
- Wall bars, detected openings, YOLO obstacles
- Live guidance text banner

This is a debug/verification view; blind users rely on **audio only**.

---

## Hardware notes (Raspberry Pi / glasses)

- Use `yolov8n.pt` on Pi 4 for ~8–12 FPS depth+detect
- Camera index: `python main.py --source 0`
- Bone-conduction speaker via system audio (pyttsx3) or `--tts-engine gtts`

---

## ML stack

| Component | Model | Role |
|-----------|-------|------|
| Depth | MiDaS small | Free space, walls, openings |
| Detection | YOLOv8n | Semantic obstacles, doors |
| Speech | Whisper base | Voice commands (local) |
| Q&A | Gemini 2.0 Flash | Visual questions only |

---

## Team

RV College of Engineering — Team UH38
