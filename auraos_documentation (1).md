# AuraOS — Eye-Controlled Hands-Free Web Operating System

**Author:** Jayan Rejo  
**Project:** AuraOS (Assistive Ocular Web Operating System)  
**Target Environment:** Windows 10 / 11, Google Chrome  

## 1. Executive Summary

**AuraOS** is an eye-controlled web operating system designed to give individuals with severe motor disabilities, ALS, quadriplegia, or spinal cord injuries complete independence when browsing the web and communicating online.

Using a standard webcam, AuraOS tracks pupil micro-movements, eye-aspect ratios, and ocular gestures in real time. Users can launch any website on the internet from the customizable 3D Dashboard, zoom into any part of a webpage, click any link or button, search Google, stream YouTube videos, compose messages using an intelligent on-screen virtual keyboard, and monitor their live vital signs—all without needing hands, voice commands, or physical input devices.

## 🎥 Project Demo Video

> 📺 [**Click Here to Watch the Full AuraOS Demo Video**](demo.mp4)
>
> *(Tip for GitHub: If uploading to GitHub, edit this README on GitHub.com and drag-and-drop your `.mp4` file directly here to generate an automatic embedded player!)*

## 2. Key Features & Innovations

* **100% Hands-Free Custom Web Portal:** Launch built-in favourites (YouTube, Google, Wikipedia, BBC News) or use the **Add Website / App** and **Remove App Mode** cards on the 3D Dashboard carousel to add and visit *any* website on the internet completely hands-free.
* **Effortless Eye-Joystick Control:** Look Up, Down, Left, or Right to move one step at a time, then look back at the center dot to reset—preventing eye strain and accidental clicks.
* **9-Grid Progressive Zoom & Click on Any Webpage:** Once inside any website, AuraOS overlays a transparent 9-box grid—allowing users to progressively zoom into any area of the page with a **Left Wink** for pinpoint accuracy and click any button, link, video, or input field with a **Right Wink**.
* **Smart Auto-Opening Virtual Keyboard:** Automatically opens an on-screen keyboard with smart word suggestions whenever a text or search box is clicked on any website, then types and submits the text directly back into the webpage.
* **Audio-Guided Eyes-Closed Shortcuts:** Close both eyes and listen for simple audio beeps to go Back (5 seconds), Recalibrate (11 seconds), or return Home to the Dashboard (20 seconds).
* **3-Blink Wake & Sleep Mode:** Prevents accidental actions while resting or watching a video—simply blink 3 times slowly to wake the system up or put it to sleep.
* **Contactless Camera Heart-Rate & SOS Monitor:** Measures live heart rate (BPM) directly through the webcam and sounds an SOS alert if an abnormal heart rate is detected.
* **Safe, Locked Full-Screen Experience:** Runs in a dedicated full-screen window with no accidental exit buttons, ensuring users never get stuck on the Windows desktop (only a caregiver can exit using the physical `Esc` key).

## 3. System Architecture & Tech Stack

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Offline ML Training Pipeline                      │
 │                                                                        │
 │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
 │  │   Data Collector │───>│  Ocular Dataset  │───>│   Train Model    │  │
 │  │(data_collector.py│    │(ocular_dataset)  │    │ (train_model.py) │  │
 │  └──────────────────┘    └──────────────────┘    └────────┬─────────┘  │
 └───────────────────────────────────────────────────────────┼────────────┘
                                                             │ Exports
                                                             ▼ (.pkl)
 ┌────────────────────────────────────────────────────────────────────────┐
 │                           AuraOS Engine                                │
 │                                                                        │
 │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
 │  │  Webcam Stream   │───>│  MediaPipe Face  │───>│ EAR + ML Model   │  │
 │  │  (OpenCV CV2)    │    │   Landmarker     │    │(ocular_ml_model) │  │
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
* **Signal Processing & ML:** NumPy, SciPy (Signal Filtering), Scikit-Learn (`RandomForestClassifier`), Joblib
* **OS & Browser Automation:** Win32 API (`ctypes`), PyAutoGUI, Chrome DevTools Protocol (`CDP`), Asynchronous WebSockets (`websockets`)
* **User Interface:** HTML5, CSS3, JavaScript (ES6+), Canvas 2D Stipple Engine, Windows SAPI5 / Web Speech API

## 4. Gesture & Control Cheat Sheet

| Gesture / Input | Timing / Condition | Sound Effect | Action / Function |
| :--- | :--- | :--- | :--- |
| **Gaze Shift** | Look Up / Down / Left / Right | *Silent* | Highlight adjacent portal card / grid cell / key |
| **Gaze Return** | Look at Center Dot | *Silent* | Re-arm 1-step joystick action |
| **Left Wink** | Left eye closed, Right eye open | `Zoom Chime` | **Open Website Card** / **Zoom Into Grid Cell** / **Select Key** |
| **Right Wink** | Right eye closed, Left eye open | `Click Sound` | **Click Anything in Cell** / **Speak & Inject Text** |
| **3 Slow Blinks** | 3 blinks within 2.4 seconds | `Wake Chime` | **Toggle System Wake / Sleep** |
| **Hold Eyes Shut** | 5.0s – 10.9s | `Single Beep` | **Contextual Back** / **Undo Zoom** |
| **Hold Eyes Shut** | 11.0s – 19.9s | `Double Beep` | **Start 9-Point Recalibration** |
| **Hold Eyes Shut** | 20.0s+ | `Triple Chime` | **Return Home to Dashboard** |
| **Physical ESC Key** | Press `Esc` key on keyboard | *None* | **Caregiver Emergency OS Exit** |

## 5. Project Directory Structure

```text
AuraOS_Project/
├── master_engine.py            # Unified Python Master Engine (Vision, rPPG, Gestures, CDP, WS)
├── data_collector.py           # Live Ocular Feature Recorder (EAR_L, EAR_R, Diff, Avg, Iris)
├── train_model.py              # Scikit-Learn Random Forest Training & Evaluation Script
├── dashboard.html              # Title Page & 3D Web Portal Dashboard UI
├── keyboard.html               # Smart Virtual Keyboard & QWERTY Predictive Engine
├── requirements.txt            # Project Dependencies
├── face_landmarker.task        # MediaPipe Face Landmark Model
├── ocular_dataset.csv          # Recorded Ocular Gesture Training Dataset
├── ocular_ml_model.pkl         # Trained Random Forest Ocular Classification Model
├── gaze_grid_calibration.npz   # Unified 9-Point Calibration Profile Data
├── demo.mp4                    # Project Walkthrough & Demo Video
└── README.md                   # System Documentation
```

## 6. How the AI & Vision Pipeline Works

* **Data Collection (`data_collector.py`):** Captures real-time MediaPipe facial landmarks from the webcam, extracts ocular features (`ear_left`, `ear_right`, `ear_diff`, `ear_avg`, and `iris_x`), and logs labeled samples (`0: Neutral`, `1: Left Wink`, `2: Right Wink`, `3: Blink / Closed`) into `ocular_dataset.csv`.
* **Model Training (`train_model.py`):** Trains a Scikit-Learn `RandomForestClassifier` on `ocular_dataset.csv`, evaluates classification accuracy, and serializes the trained weights to `ocular_ml_model.pkl` using `joblib`.
* **Eye Openness Tracking (EAR):** AuraOS uses MediaPipe to track key points around both eyes, measuring how open or closed each eye is on every camera frame.
* **Smart Wink & Blink Filtering:** By combining the trained Random Forest classifier (`ocular_ml_model.pkl`) with bilateral asymmetry thresholds, the system reliably distinguishes between an intentional wink, a normal fast blink, or an involuntary facial movement.
* **Camera Heart-Rate Sensing (rPPG):** The webcam reads tiny, invisible color changes on the user's forehead caused by blood flow, filtering that signal via a 4th-order Butterworth bandpass filter to calculate live Heart Rate (BPM) without any physical sensors.

## 7. Challenges Faced & How We Solved Them

1. **Accidental Clicks from Natural Blinks & Sneezes**
   * *Problem:* Standard eye trackers often mistake natural blinks, flinches, or facial twitches for deliberate mouse clicks.
   * *Solution:* We built a strict asymmetry and timing filter backed by our trained ML model. Natural blinks are ignored because both eyes close symmetrically, while deliberate winks only trigger when one eye closes independently.

2. **Eye Fatigue & Cursor Jitter**
   * *Problem:* Controlling a continuous mouse cursor with gaze causes severe eye fatigue and jittery movement.
   * *Solution:* We implemented a 4-way spring-centered "Eye Joystick" (1 look = 1 step) and a 9-Grid progressive zoom overlay. This lets users make precise selections on websites without needing to strain their eyes.

3. **Calibration Offset Across Different Screens & Views**
   * *Problem:* Moving between the Dashboard, Virtual Keyboard, and external web pages caused slight screen offsets, throwing off gaze tracking.
   * *Solution:* We unified all system components around a single absolute center coordinate (`HUD_HEIGHT_PX = 0`), ensuring one 9-point calibration profile (`gaze_grid_calibration.npz`) works perfectly across every screen mode.

4. **Overlay Disappearing Behind Browser Windows**
   * *Problem:* When loading external websites in Chrome, Windows would occasionally render the web page over the 9-Grid zoom overlay.
   * *Solution:* Using Win32 `ctypes` API calls (`WS_EX_TOPMOST`, `WS_EX_TRANSPARENT`, and `WS_EX_NOACTIVATE`), we locked the overlay as a transparent, click-through window that always remains on top of Chrome without stealing keyboard focus.

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

### Optional: Collect Custom Data & Retrain ML Model
1. To record custom ocular gesture samples into `ocular_dataset.csv`:
   ```bash
   python data_collector.py
   ```
2. To train the Random Forest classifier and generate an updated `ocular_ml_model.pkl`:
   ```bash
   python train_model.py
   ```

### Launch AuraOS
1. Run the unified master engine from your terminal:
   ```bash
   python master_engine.py
   ```
2. **Kiosk Launch:** Google Chrome will automatically open in locked Kiosk mode displaying the AuraOS Title Page.
3. **Wake the System:**
   * Blink slowly **3 times** to wake AuraOS from surveillance mode.
4. **9-Point Blink Calibration:**
   * Look directly at the bright red center dot inside each active ring and **blink once** (or Left Wink / press `Space`) to capture each point.
   * As each point completes, a confirmation sound will play.
5. **Navigate the Dashboard & Browse Any Website:**
   * Look **Left** or **Right** to rotate the 3D Dashboard carousel (including the **Add Website / App** card to add any custom website URL) and **Left Wink** to launch the selected site.
   * On any live website, look at a grid cell, **Left Wink** to progressively zoom in for pinpoint accuracy, and **Right Wink** to click any button, link, video, or search box on the page.
6. **Testing Shortcuts:**
   * Press **`C`** on your keyboard anytime to start a manual recalibration.
   * Press **`R`** while looking at the center dot to recenter head-posture bias.
   * Press **`Esc`** on your keyboard to perform a caregiver system shutdown.

## 10. Author & Acknowledgments

* **Developer:** Jayan Rejo  
* **Project:** AuraOS (Assistive Ocular Web Operating System)  
* **Target Platform:** Windows 10 / 11 (Google Chrome Kiosk)  
* **Built With:** Python, MediaPipe, OpenCV, Scikit-Learn, WebSockets, Chrome DevTools Protocol, and Windows SAPI5 Speech API.