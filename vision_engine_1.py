import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request

# 1. Automatically download the new Face Landmarker AI model if it doesn't exist
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe Face Landmarker AI model (~9MB)...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# 2. Initialize the Modern MediaPipe Tasks API
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 3. Setup CLAHE for lighting correction (handles normal ceiling lights)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Pulse Tracking Variables
pulse_buffer = []
time_buffer = []

def get_eye_metrics(landmarks, frame_w, frame_h):
    """Calculates Eye Aspect Ratio (EAR) and pupil X/Y ratios."""
    def distance(p1_idx, p2_idx):
        p1 = landmarks[p1_idx]
        p2 = landmarks[p2_idx]
        return np.hypot((p1.x - p2.x) * frame_w, (p1.y - p2.y) * frame_h)

    # Left Eye: 33 (inner), 133 (outer), 159 (top), 145 (bottom), 468 (iris)
    ear_left = (distance(159, 145) + distance(158, 153)) / (2.0 * distance(33, 133) + 1e-6)
    
    # Right Eye: 362 (inner), 263 (outer), 386 (top), 374 (bottom), 473 (iris)
    ear_right = (distance(386, 374) + distance(385, 380)) / (2.0 * distance(362, 263) + 1e-6)
    
    # Iris X-Ratio (Horizontal) for Left Eye
    iris_x = distance(33, 468) / (distance(33, 133) + 1e-6)
    
    return ear_left, ear_right, iris_x

# 4. Start Webcam
cap = cv2.VideoCapture(0)
print("Starting AuraOS Vision Engine... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1) # Mirror image
    h, w, _ = frame.shape
    
    # Convert to grayscale for CLAHE lighting fix, then back to RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhanced_gray = clahe.apply(gray)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to MediaPipe Image format and run detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    timestamp_ms = int(time.time() * 1000)
    results = detector.detect_for_video(mp_image, timestamp_ms)
    
    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            
            # --- TASK A: EYE TRACKING ---
            ear_l, ear_r, iris_x = get_eye_metrics(face_landmarks, w, h)
            avg_ear = (ear_l + ear_r) / 2.0
            
            # Draw Iris markers for visual feedback
            for idx in [468, 473]: # Left and Right Iris centers
                lx = int(face_landmarks[idx].x * w)
                ly = int(face_landmarks[idx].y * h)
                cv2.circle(frame, (lx, ly), 4, (0, 255, 255), -1)
                
            cv2.putText(frame, f"Eye Ratio (Iris X): {iris_x:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink EAR: {avg_ear:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if avg_ear < 0.20:
                cv2.putText(frame, "BLINK DETECTED", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- TASK B: HEART RATE (PULSE) VIA FOREHEAD ---
            # Forehead landmarks: 151 (mid)
            fh_x = int(face_landmarks[151].x * w)
            fh_y = int(face_landmarks[151].y * h)
            
            # Extract a 20x20 pixel box from the forehead
            box_size = 10
            if fh_y - box_size > 0 and fh_y + box_size < h and fh_x - box_size > 0 and fh_x + box_size < w:
                roi = frame[fh_y-box_size:fh_y+box_size, fh_x-box_size:fh_x+box_size]
                
                # Split channels and normalize Green against Red+Blue (cancels ceiling light flicker)
                b, g, r = cv2.split(roi)
                bg_mean = np.mean(g)
                rb_avg = (np.mean(r) + np.mean(b)) / 2.0 + 1e-6
                normalized_pulse = bg_mean / rb_avg
                
                pulse_buffer.append(normalized_pulse)
                time_buffer.append(time.time())
                
                # Keep only the last 5 seconds of data (approx 150 frames at 30fps)
                if len(pulse_buffer) > 150:
                    pulse_buffer.pop(0)
                    time_buffer.pop(0)
                    
                    # Simple frequency analysis (FFT) to get BPM
                    fps = 1.0 / (np.mean(np.diff(time_buffer)))
                    signal = np.array(pulse_buffer)
                    signal = signal - np.mean(signal) # Detrend
                    
                    fft_data = np.abs(np.fft.rfft(signal))
                    freqs = np.fft.rfftfreq(len(signal), d=1.0/fps)
                    
                    # Filter for human heart rate (0.8 Hz to 3.0 Hz -> 48 to 180 BPM)
                    valid_idx = np.where((freqs >= 0.8) & (freqs <= 3.0))
                    if len(valid_idx[0]) > 0:
                        peak_freq = freqs[valid_idx][np.argmax(fft_data[valid_idx])]
                        bpm = peak_freq * 60.0
                        cv2.putText(frame, f"BPM: {bpm:.1f}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                cv2.rectangle(frame, (fh_x-box_size, fh_y-box_size), (fh_x+box_size, fh_y+box_size), (255, 0, 0), 2)

    cv2.imshow("AuraOS - Raw Engineering View", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()