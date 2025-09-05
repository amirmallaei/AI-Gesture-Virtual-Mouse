# AI-Powered Virtual Mouse with Hand Gesture Control

Tired of your mouse and keyboard? This project transforms your webcam into a next-generation human-computer interface, allowing you to control your computer with intuitive hand gestures.

This isn't just a simple detector; it's a complete, real-time pipeline that integrates state-of-the-art **YOLOv8** for robust detection with **Google's MediaPipe** for detailed pose estimation, resulting in a smooth, responsive, and powerful control experience.

---

## ✨ Core Features

*   **Full Virtual Mouse:** Control your cursor, scroll pages, and click, all without touching a physical mouse.
*   **Advanced System Control:** Go beyond the mouse! Use intuitive gestures to take screenshots, close applications, and even pinch-to-zoom.
*   **High-Performance AI Pipeline:** Leverages a two-stage detection/estimation pipeline, ensuring both high accuracy and real-time performance.
*   **Smooth & Responsive:** Implements motion smoothing to prevent jittery cursor movement, making the control feel natural and predictable.
*   **Clean, Object-Oriented Code:** Written with a professional, class-based structure that is easy to read, maintain, and extend.

---

## 🙋‍♂️ The Gesture Guide

Control your computer with these intuitive hand gestures. The system intelligently switches between single-hand and two-hand modes.

| Gesture | Fingers / Hands | Action |
| :--- | :--- | :--- |
| **Mouse Mode** | 👆 Index Finger Up | Moves the mouse cursor. The on-screen circle tracks your fingertip. |
| **Scroll Mode** | ✌️ Index & Middle Up | Move your hand up or down to scroll web pages and documents. |
| **Click** | 🤏 Pinch Thumb & Index | Performs a left mouse click. |
| **Zoom** | ✋🤚 Two Hands | Move your hands apart to zoom in (`Ctrl` + `+`) and closer to zoom out (`Ctrl` + `-`). |
| **Screenshot** | 🖐️ Open Palm | Takes a full-screen screenshot and saves it as `screenshot.png`. |
| **Close App** | ✊ Fist | Closes the currently active window (`Alt` + `F4`). **Use with care!** |

---

## ⚙️ How It Works: The Tech Stack

This project is built on a modern computer vision pipeline:

1.  **Person Detection (YOLOv8):** A high-speed YOLOv8 model first locates any person in the camera's view, creating a region of interest.
2.  **Hand Pose Estimation (MediaPipe):** Within that region, Google's MediaPipe Hands is used to find up to two hands and calculate the precise 3D coordinates of 21 keypoints for each.
3.  **Gesture Recognition Engine:** A custom logic module analyzes the landmark data in real-time to recognize specific, robust gestures (e.g., "is the index finger up AND are the others down?").
4.  **Action Controller (PyAutoGUI):** Once a gesture is recognized, the PyAutoGUI library translates it into a system command, like moving the mouse or pressing a key.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- An active internet connection (for the first run to download the YOLO model).
- A webcam.

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/AI-Gesture-Virtual-Mouse.git
    cd AI-Gesture-Virtual-Mouse
    ```

2.  **Install all required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

### ▶️ How to Run
Execute the main script from your terminal.

```bash
python myfile.py
```
A window will pop up showing your camera feed. Hold up your hand and start controlling your computer! **Press 'q' to quit.**

---

## 🔧 Tuning & Configuration

All key parameters are located at the top of the `GestureController` class in `virtual_mouse.py`, making them easy to adjust:

*   **`SMOOTHENING`:** Increase this value for smoother but slightly slower cursor movement.
*   **`CLICK_COOLDOWN`, `SCREENSHOT_COOLDOWN`:** Increase these to prevent accidental double-actions.
*   **Gesture Rules:** The `_recognize_gesture` method contains the clear, readable logic for each gesture, which you can easily modify or extend.

## 📬 Contact
Have questions, ideas, or want to collaborate? Feel free to reach out!

**Amir Mallaei** - [amirmallaei@gmail.com](mailto:amirmallaei@gmail.com)
