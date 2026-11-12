import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import csv
import urllib.request

# 1. Download Model if Missing
model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading MediaPipe Face Landmarker AI model...")
    urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", model_path)

# 2. Initialize MediaPipe
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 3. Create the CSV File Header
csv_filename = "ocular_dataset.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ear_left", "ear_right", "iris_x", "label"])
    print(f"Created new dataset file: {csv_filename}")

def get_eye_metrics(landmarks, frame_w, frame_h):
    def distance(p1_idx, p2_idx):
        return np.hypot((landmarks[p1_idx].x - landmarks[p2_idx].x) * frame_w, (landmarks[p1_idx].y - landmarks[p2_idx].y) * frame_h)
    
    ear_left = (distance(159, 145) + distance(158, 153)) / (2.0 * distance(33, 133) + 1e-6)
    ear_right = (distance(386, 374) + distance(385, 380)) / (2.0 * distance(362, 263) + 1e-6)
    iris_x = distance(33, 468) / (distance(33, 133) + 1e-6)
    return ear_left, ear_right, iris_x

# 4. Start Webcam Loop
cap = cv2.VideoCapture(0)
print("\n--- DATA COLLECTION MODE ACTIVE ---")
print("Hold '0' for Neutral (Reading normally)")
print("Hold '1' for Left Wink")
print("Hold '2' for Right Wink")
print("Hold '3' for Intentional Long Blink")
print("Press 'q' to Quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    results = detector.detect_for_video(mp_image, int(time.time() * 1000))
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            ear_left, ear_right, iris_x = get_eye_metrics(face_landmarks, w, h)
            
            cv2.putText(frame, f"L-EAR: {ear_left:.2f} | R-EAR: {ear_right:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Save data if a designated key is held down
            label = -1
            if key == ord('0'): label = 0
            elif key == ord('1'): label = 1
            elif key == ord('2'): label = 2
            elif key == ord('3'): label = 3
            
            if label != -1:
                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([ear_left, ear_right, iris_x, label])
                cv2.putText(frame, f"SAVING CLASS: {label}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow("AuraOS Data Collector", frame)

cap.release()
cv2.destroyAllWindows()