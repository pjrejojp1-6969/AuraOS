# AuraOS — Eye-Controlled Hands-Free Web Operating System

**Author:** Jayan Rejo  
**Project:** AuraOS (Assistive Ocular Web Operating System)  
**Target Environment:** Windows 10 / 11, Google Chrome  

## 1. Executive Summary

**AuraOS** is an eye-controlled web operating system designed to give individuals with severe motor disabilities, ALS, quadriplegia, or spinal cord injuries complete independence when browsing the web and communicating online.

Using a standard webcam, AuraOS tracks pupil micro-movements, eye-aspect ratios, and ocular gestures in real time. Users can seamlessly navigate web pages, search Google, stream YouTube videos, compose messages using an intelligent on-screen virtual keyboard, and monitor their live vital signs—all without needing hands, voice commands, or physical input devices.

## 2. Key Features & Innovations

* **100% Hands-Free Web Navigation:** Browse websites like YouTube, Google, Wikipedia, and BBC News using only natural eye movements and winks.
* **Effortless Eye-Joystick Control:** Look Up, Down, Left, or Right to move one step at a time, then look back at the center dot to reset—preventing eye strain and accidental clicks.
* **9-Grid Progressive Zoom & Click:** Divides any live website into a 9-box grid so users can zoom in with a Left Wink and click the exact button or link with a Right Wink.
* **Smart Auto-Opening Virtual Keyboard:** Automatically opens an on-screen keyboard with smart word suggestions whenever a text or search box is clicked, then types and submits the text back into the website.
* **Audio-Guided Eyes-Closed Shortcuts:** Close both eyes and listen for simple audio beeps to go Back (5 seconds), Recalibrate (11 seconds), or return Home to the Dashboard (20 seconds).
* **3-Blink Wake & Sleep Mode:** Prevents accidental actions while resting or watching a video—simply blink 3 times slowly to wake the system up or put it to sleep.
* **Contactless Camera Heart-Rate & SOS Monitor:** Measures live heart rate (BPM) directly through the webcam and sounds an SOS alert if an abnormal heart rate is detected.
* **Safe, Locked Full-Screen Experience:** Runs in a dedicated full-screen window with no accidental exit buttons, ensuring users never get stuck on the Windows desktop (only a caregiver can exit using the physical `Esc` key).

## 3. System Architecture & Tech Stack

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                           AuraOS Engine                                │
 │                                                                        │
 │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
 │  │  Webcam Stream   │───>│  MediaPipe Face  │───>│ Ocular Signal    │  │
 │  │  (OpenCV CV2)    │    │   Landmarker     │    │ Filtering (EAR)  │  │
 │  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Core Processing & Automation                      │
 │                                                                        │
 │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
 │  │ rPPG Heart Rate  │    │  Chrome CDP      │    │ Dual WebSockets  │  │
 │  │ (SciPy Bandpass) │    │  (Port 9222)     │    │ (5006 & 8765)    │  │
 │  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
 └────────────────────────────────────┬───────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        Single-Window UI Layer                          │
 │                                                                        │
 │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
 │  │  Dashboard UI    │    │ Virtual Keyboard │    │  9-Grid Zoom     │  │
 │  │ (dashboard.html) │    │  (keyboard.html) │    │     Overlay      │  │
 │  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
 └────────────────────────────────────────────────────────────────────────┘
```

* **Core Language:** Python 3.9+
* **Computer Vision:** OpenCV (`opencv-python`), MediaPipe Face Landmarker (`face_landmarker.task`)
* **Signal Processing & ML:** NumPy, SciPy (Signal Filtering), Scikit-Learn, Joblib
* **OS & Browser Automation:** Win32 API (`ctypes`), PyAutoGUI, Chrome DevTools Protocol (`CDP`), Asynchronous WebSockets (`websockets`)
* **User Interface:** HTML5, CSS3, JavaScript (ES6+), Canvas 2D Stipple Engine, Web Speech API

## 4. Gesture & Control Cheat Sheet

| Gesture / Input | Timing / Condition | Sound Effect | Action / Function |
| :--- | :--- | :--- | :--- |
| **Gaze Shift** | Look Up / Down / Left / Right | *Silent* | Highlight adjacent portal card / grid cell |
| **Gaze Return** | Look at Center Red Dot | *Silent* | Re-arm 1-step joystick action |
| **Left Wink** | 0.48s – 1.60s (Left eye only) | `Zoom Chime` | **Select Card** / **Zoom Grid Level** |
| **Right Wink** | 0.48s – 1.60s (Right eye only) | `Click Sound` | **Execute Click** / **Speak Text** |
| **3 Slow Blinks** | 3 blinks within 5.5 seconds | `Wake Chime` | **Toggle System Wake / Sleep** |
| **Hold Eyes Shut** | 5.0s – 10.9s | `Single Beep` | **Contextual Back** / **Undo Zoom** |
| **Hold Eyes Shut** | 11.0s – 19.9s | `Double Beep` | **Start 9-Point Recalibration** |
| **Hold Eyes Shut** | 20.0s+ | `Triple Chime` | **Return Home to Dashboard** |
| **Physical ESC Key** | Press `Esc` key on keyboard | *None* | **Caregiver Emergency OS Exit** |

## 5. Project Directory Structure

```
AuraOS_Project/
├── master_engine.py            # Unified Python Master Engine (Vision, rPPG, Gestures, CDP, WS)
├── dashboard.html              # Title Page & 3D Web Portal Dashboard UI
├── keyboard.html               # Smart Virtual Keyboard & QWERTY Predictive Engine
├── requirements.txt            # Project Dependencies
├── face_landmarker.task        # MediaPipe Face Landmark Model
├── gaze_grid_calibration.npz   # Unified Calibration Profile Data
├── ocular_ml_model.pkl         # Shadow ML Classification Model
└── README.md                   # System Documentation
```

## 6. How the AI & Vision Pipeline Works

* **Eye Openness Tracking (EAR):** AuraOS uses MediaPipe to track key points around both eyes, measuring how open or closed each eye is on every camera frame.
* **Smart Wink & Blink Filtering:** By comparing the left and right eyes together, the system easily tells the difference between an intentional wink, a normal fast blink, or an involuntary facial movement like a yawn or sneeze.
* **Camera Heart-Rate Sensing (rPPG):** The webcam reads tiny, invisible color changes on the user's forehead caused by blood flow, filtering that signal to calculate live Heart Rate (BPM) without any physical sensors.
* **Machine Learning Gesture Classifier:** A trained Random Forest model (`ocular_ml_model.pkl`) runs alongside the vision engine, classifying eye states (`Neutral`, `Left Wink`, `Right Wink`, `Long Blink`) and logging real-time predictions to the console.

## 7. Challenges Faced & How We Solved Them

1. **Accidental Clicks from Natural Blinks & Sneezes**
   * *Problem:* Standard eye trackers often mistake natural blinks, flinches, or facial twitches for deliberate mouse clicks.
   * *Solution:* We built a strict asymmetry and timing filter. Natural blinks are ignored because both eyes close together, while deliberate winks only trigger when one eye closes independently for at least 0.48 seconds.

2. **Eye Fatigue & Cursor Jitter**
   * *Problem:* Controlling a continuous mouse cursor with gaze causes severe eye fatigue and jittery movement.
   * *Solution:* We implemented a 4-way spring-centered "Eye Joystick" (1 look = 1 step) and a 9-Grid progressive zoom overlay. This lets users make precise selections on websites without needing to strain their eyes.

3. **Calibration Offset Across Different Screens & Views**
   * *Problem:* Moving between the Dashboard, Virtual Keyboard, and external web pages caused slight screen offsets, throwing off gaze tracking.
   * *Solution:* We unified all system components around a single absolute center coordinate (`HUD_HEIGHT_PX = 0`), ensuring one 9-point calibration profile (`gaze_grid_calibration.npz`) works perfectly across every screen mode.

4. **Overlay Disappearing Behind Browser Windows**
   * *Problem:* When loading external websites in Chrome, Windows would occasionally render the web page over the 9-Grid zoom overlay.
   * *Solution:* Using Win32 `ctypes` API calls (`WS_EX_TOPMOST` and `WS_EX_TRANSPARENT`), we locked the overlay as a transparent, click-through window that always remains on top of Chrome while letting mouse clicks pass directly to the browser.

5. **Risk of Motor-Disabled Users Getting Stuck on Desktop**
   * *Problem:* Accidental window closes or pop-up dismissals could drop a motor-disabled user onto the Windows desktop with no way to recover.
   * *Solution:* AuraOS locks Chrome into a single fullscreen Kiosk mode with no on-screen close buttons. All page navigation happens in-place via Chrome DevTools Protocol (`CDP`), and only a caregiver can exit using the physical `Esc` key.

## 8. Prerequisites & Installation

### Hardware Requirements
* Any standard PC running **Windows 10 or Windows 11**.
* Standard USB or built-in webcam (720p resolution or higher recommended).
* Screen resolution of 1920 x 1080 (DPI Aware).

### Software Setup
1. **Ensure Google Chrome is installed** on your system.
2. Clone or download this project folder.
3. Open a terminal in the project directory and install all required dependencies:

```bash
pip install -r requirements.txt
```

## 9. How to Run & Calibrate

1. Run the unified master engine from your terminal:
   ```bash
   python master_engine.py
   ```
2. **Kiosk Launch:** Google Chrome will automatically open in locked Kiosk mode displaying the AuraOS Dashboard.
3. **First-Time Calibration:**
   * Look directly at the bright red center dot inside each ring during the 9-point calibration routine.
   * As each point completes, a confirmation sound will play.
4. **Wake the System:**
   * Blink slowly **3 times** to wake AuraOS from surveillance mode.
   * Look at any web portal card and **Wink Left** to launch it.
5. **Testing Shortcuts:**
   * Press **`C`** on your keyboard anytime to start a manual recalibration.
   * Press **`Esc`** on your keyboard to perform a caregiver system shutdown.

## 10. Author & Acknowledgments

* **Developer:** Jayan Rejo  
* **Platform:** AuraOS 
* **Built With:** Python, MediaPipe, OpenCV, WebSockets, Chrome DevTools Protocol, and Web Speech API.