# Real-Time Hand-Gesture Image Filter App 🖐️✨

An interactive Computer Vision application built using Python, OpenCV, and the `cvzone` tracking engine (powered by MediaPipe backend infrastructure). The application detects complex 21-joint hand skeletal architecture in real-time, allowing users to dynamically blend distinct visual filters directly onto live camera feeds through physical hand gestures or keyboard shortcuts.

---

## 🚀 Features

* **Custom 21-Joint Skeletal Tracking**: Draws complete structural bone paths and high-visibility tracking joints live on the camera feed.
* **Dynamic Dual-Mode Bounding Boxes**:
  * **Finger Box Mode (Single Hand)**: Isolates and bounds image filters inside the spatial coordinate box between the user's thumb and index finger.
  * **Dual Hand Box Mode**: Stretches a coordinate-mapped responsive filter mask across both hands simultaneously.
* **5 Classic Visual Filters**:
  * `GRAY`: Monochromatic grayscale conversion.
  * `THERMAL`: Simulated heat map tracking using OpenCV's `COLORMAP_JET`.
  * `INVERT`: Complete color channel bitwise negation.
  * `SKETCH`: Edge-detected pencil contour drawing using Gaussian blurring.
  * `VINTAGE`: Sepia matrix transformation.
* **Classic UI HUD**: Features dedicated top and bottom banner bars styled with crisp geometric typography (`FONT_HERSHEY_DUPLEX`) for optimal readability without overlapping screen text.

---

## 🛠️ Tech Stack & Dependencies

* **Language**: Python 3.x
* **Core Libraries**:
  * `opencv-python` (Frame processing and matrix math calculations)
  * `cvzone` & `mediapipe` (Advanced neural network tracking models)
  * `numpy` (High-speed multi-dimensional array operations)

---

## 📦 Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/btth-7/hand-gesture-filter-app.git](https://github.com/btth-7/hand-gesture-filter-app.git)
   cd hand-gesture-filter-app
   pip install -r requirements.txt
   python hand-ditaction.py

  ---
  
## 🎮 How to Control the Application

| Action | Control Method | Details |
| :--- | :--- | :--- |
| **Cycle Filters** | **Pinch Gesture** | Bring your **Thumb** and **Index finger** together firmly on screen. |
| **Cycle Filters (Manual)** | **`N` Key** | Press **`N`** on your keyboard to instantly switch to the next filter. |
| **Finger Box Filter** | **One Hand** | Hold up **1 hand** to apply the filter strictly inside the box between your thumb and index finger. |
| **Dual-Hand Box Filter** | **Two Hands** | Hold up **both hands** to stretch the filter inside the quadrilateral box spanned between them. |
| **Quit Application** | **`Q` Key** | Click on the camera window and press **`Q`** to safely exit and release your webcam. |
