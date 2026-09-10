# Hand Gesture Drawing Application

A Python app that lets you draw and interact with 2D shapes using hand gestures via webcam. Built as a way to test out real-time camera tracking and see if I could make drawing feel physical.

## Features

- **Hand tracking**: MediaPipe tracks hand landmarks in real time.
- **Drawing mode**:
  - Left hand pinch sets Point 1, right hand pinch sets Point 2.
  - Thumbs up finalizes the line.
  - Lines auto-straighten to horizontal/vertical axes.
  - Points magnetically snap to existing line endpoints for easy shape closure.
- **Shape detection**: Automatically detects when 4 connected lines form a rectangle or square.
- **Rotation mode**: Triggers automatically after shape detection.
  - Left hand thumb/index controls rotation.
  - Thumbs up finalizes the shape and drops it into the physics world.
- **Physics**: Uses `pymunk` for 2D physics — shapes fall under gravity and collide with each other and the screen boundaries.

## Requirements

- Python 3.x
- `opencv-python`, `mediapipe`, `pymunk`, `numpy`

## Installation

```bash
git clone https://github.com/JeevikaS-19/Building_blocks.git
pip install opencv-python mediapipe pymunk numpy
```

## Usage

```bash
python hand_tracker.py
```

1. Pinch with left and right hands to define a line segment, thumbs up to save it.
2. Draw 4 connected lines to form a box — the app detects it and enters rotation mode.
3. Rotate with your left hand.
4. Thumbs up to drop the shape into the physics simulation.

## Controls

- `u` — undo last line
- `c` — clear all lines and shapes
- `q` — quit application
