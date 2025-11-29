# Hand Gesture Drawing Application

A Python-based application that allows you to draw and interact with 2D shapes using hand gestures detected via webcam.

## Features

*   **Hand Tracking**: Uses MediaPipe to track hand landmarks in real-time.
*   **Drawing Mode**:
    *   **Left Hand Pinch**: Set Point 1.
    *   **Right Hand Pinch**: Set Point 2.
    *   **Thumbs Up**: Finalize the line.
    *   **Auto-Straightening**: Lines automatically snap to horizontal or vertical axes.
    *   **Magnetic Snapping**: Points snap to existing line endpoints for easy shape closure.
*   **Shape Detection**: Automatically detects when 4 connected lines form a rectangle or square.
*   **Rotation Mode**:
    *   Triggered automatically after shape detection.
    *   **Left Hand Thumb/Index**: Control the rotation of the shape.
    *   **Thumbs Up**: Finalize the shape and convert it into a physics object.
*   **Physics Integration**:
    *   Uses `pymunk` for 2D physics.
    *   Shapes fall with gravity and collide with each other and the screen boundaries.

## Requirements

*   Python 3.x
*   `opencv-python`
*   `mediapipe`
*   `pymunk`
*   `numpy`

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/JeevikaS-19/Building_blocks.git
    ```
2.  Install dependencies:
    ```bash
    pip install opencv-python mediapipe pymunk numpy
    ```

## Usage

1.  Run the application:
    ```bash
    python hand_tracker.py
    ```
2.  **Draw**: Use your left and right hands to pinch and define line segments. Give a thumbs up to save each line.
3.  **Create Shape**: Draw 4 connected lines to form a box. The app will detect it and enter Rotation Mode.
4.  **Rotate**: Use your left hand to rotate the shape.
5.  **Release**: Give a thumbs up to drop the shape into the physics world.

## Controls

*   **'u'**: Undo last line
*   **'c'**: Clear all lines and shapes
*   **'q'**: Quit application
