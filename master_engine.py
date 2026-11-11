import asyncio
from collections import deque
import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
import winsound

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import scipy.signal
import websockets

# Disable PyAutoGUI corner fail-safe so eye-tracking near corners never crashes the OS
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01

# ==============================================================================
# 1. GLOBAL CONFIGURATION & UNIFIED GEOMETRY (HUD_HEIGHT_PX = 0)
# ==============================================================================
HUD_HEIGHT_PX = 0  # Absolute zero offset: Dashboard, Keyboard, and ZoomGrid share true center
CALIBRATION_FILE = "gaze_grid_calibration.npz"
ML_MODEL_FILE = "ocular_ml_model.pkl"
FACE_MODEL_PATH = "face_landmarker.task"
CDP_PORT = 9222
WS_PORT_MAIN = 8765       # Port for Keyboard & Unified state
WS_PORT_DASHBOARD = 5006  # Port for Dashboard so dashboard.html works without any URL changes

# Screen dimensions
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)
CENTER_X = SCREEN_W // 2
CENTER_Y = (SCREEN_H - HUD_HEIGHT_PX) // 2

# Gesture Timing Constants (Seconds)
EAR_THRESHOLD = 0.21          # Eye closure threshold
WINK_ASYMMETRY_MIN = 0.055    # Minimum difference between left & right EAR for a valid wink
WINK_MIN_DUR = 0.48           # Filters out natural fast blinks/flinches
WINK_MAX_DUR = 1.60           # Max duration for a single wink
WAKE_BLINK_MIN = 0.35         # Deliberate blink minimum for 3-blink wake
WAKE_BLINK_MAX = 1.25         # Deliberate blink maximum for 3-blink wake
WAKE_WINDOW_SEC = 5.5         # Time window to complete 3 slow blinks

# Eyes-Shut Duration Checkpoints (Seconds)
BACK_MIN_SEC = 5.0            # 5–10s: Contextual Back / Undo Zoom
BACK_MAX_SEC = 10.9
CALIB_MIN_SEC = 11.0          # 11–19s: Hands-Free Recalibration
CALIB_MAX_SEC = 19.9
HOME_MIN_SEC = 20.0           # 20–30s+: Global Home (Return to Dashboard)

# ==============================================================================
# 2. LIVE CAMERA rPPG HEART RATE & SOS CONSTANTS (EXTRACTED VERBATIM)
# ==============================================================================
FOREHEAD_LANDMARKS = (10, 67, 69, 104, 108, 109, 151, 297, 299, 333, 337, 338)
RPPG_BUFFER_MAX = 300
RPPG_MIN_SAMPLES = 90
RPPG_BANDPASS_LOW_HZ = 0.75
RPPG_BANDPASS_HIGH_HZ = 2.50
RPPG_UPDATE_INTERVAL_S = 0.50
BPM_ROLLING_WINDOW_S = 15.0
BPM_SOS_LOW = 50.0
BPM_SOS_HIGH = 120.0
SOS_SUSTAINED_DURATION_S = 15.0
SOS_BEEP_INTERVAL_S = 1.50


def trigger_sos_beep_async():
    def _beep():
        try:
            winsound.Beep(2500, 400)
        except Exception:
            sys.stdout.write("\a")
            sys.stdout.flush()

    threading.Thread(target=_beep, daemon=True).start()


def extract_forehead_green(frame, landmarks, fw, fh):
    xs, ys = [], []
    for idx in FOREHEAD_LANDMARKS:
        if idx < len(landmarks):
            xs.append(int(landmarks[idx].x * fw))
            ys.append(int(landmarks[idx].y * fh))
    if not xs or not ys:
        return None, None
    x_min, x_max = max(0, min(xs)), min(fw, max(xs))
    y_min, y_max = max(0, min(ys)), min(fh, max(ys))
    if x_max <= x_min + 2 or y_max <= y_min + 2:
        return None, None
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None, None
    return float(np.mean(roi[:, :, 1])), (x_min, y_min, x_max, y_max)


def compute_filtered_rppg_bpm(green_buf, time_buf):
    n = len(green_buf)
    if n < RPPG_MIN_SAMPLES or len(time_buf) != n:
        return None
    duration = float(time_buf[-1] - time_buf[0])
    if duration <= 1.5:
        return None
    fs = (n - 1) / duration
    nyq = 0.5 * fs
    if nyq <= RPPG_BANDPASS_HIGH_HZ + 0.1:
        return None
    try:
        t_raw = np.asarray(time_buf, dtype=np.float64)
        y_raw = np.asarray(green_buf, dtype=np.float64)
        t_uni = np.linspace(t_raw[0], t_raw[-1], n)
        sig = np.interp(t_uni, t_raw, y_raw)
        detrended = scipy.signal.detrend(sig, type="linear")
        b, a = scipy.signal.butter(
            4,
            [RPPG_BANDPASS_LOW_HZ / nyq, RPPG_BANDPASS_HIGH_HZ / nyq],
            btype="bandpass",
        )
        filtered = scipy.signal.filtfilt(b, a, detrended)
        windowed = filtered * np.hanning(n)
        fft_mag = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mask = (freqs >= RPPG_BANDPASS_LOW_HZ) & (freqs <= RPPG_BANDPASS_HIGH_HZ)
        if not np.any(mask):
            return None
        peak_freq = float(freqs[mask][int(np.argmax(fft_mag[mask]))])
        return peak_freq * 60.0
    except Exception:
        return None


# ==============================================================================
# 3. AUDIO CHECKPOINT ENGINE (NON-BLOCKING)
# ==============================================================================
def play_sound_async(pattern):
    """Plays non-blocking system audio cues while the user's eyes are closed."""
    def _worker():
        try:
            if pattern == "BEEP_5S_BACK":
                winsound.Beep(750, 260)
            elif pattern == "BEEP_11S_CALIB":
                winsound.Beep(980, 160)
                time.sleep(0.08)
                winsound.Beep(980, 160)
            elif pattern == "CHIME_20S_HOME":
                winsound.Beep(660, 140)
                winsound.Beep(880, 140)
                winsound.Beep(1100, 260)
            elif pattern == "CLICK":
                winsound.Beep(1200, 60)
            elif pattern == "ZOOM":
                winsound.Beep(900, 70)
            elif pattern == "WAKE":
                winsound.Beep(700, 100)
                winsound.Beep(1050, 150)
        except Exception:
            pass
    threading.Thread(target=_worker, daemon=True).start()


# ==============================================================================
# 4. WINDOWS I-BEAM CURSOR DETECTOR & TRANSPARENT OVERLAY HELPERS
# ==============================================================================
class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),
        ("ptScreenPos", wintypes.POINT),
    ]

IDC_IBEAM = 32513
HCURSOR_IBEAM = user32.LoadCursorW(0, IDC_IBEAM)

def is_ibeam_cursor_active():
    """Detects if the Windows mouse cursor has turned into a text-input I-Beam."""
    try:
        ci = CURSORINFO()
        ci.cbSize = ctypes.sizeof(CURSORINFO)
        if user32.GetCursorInfo(ctypes.byref(ci)):
            return ci.hCursor == HCURSOR_IBEAM
    except Exception:
        pass
    return False

def make_window_transparent_clickthrough(window_name):
    """Makes an OpenCV window borderless, topmost, transparent, and click-through."""
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return
    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x00080000
    WS_EX_TRANSPARENT = 0x00000020
    WS_EX_TOPMOST = 0x00000008
    WS_EX_TOOLWINDOW = 0x00000080
    LWA_COLORKEY = 0x00000001

    ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    user32.SetWindowLongW(
        hwnd,
        GWL_EXSTYLE,
        ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
    )
    user32.SetLayeredWindowAttributes(hwnd, 0x000000, 255, LWA_COLORKEY)
    user32.SetWindowPos(hwnd, -1, 0, 0, SCREEN_W, SCREEN_H, 0x0040)

def copy_to_clipboard(text):
    """Copies text to Windows clipboard using standard library subprocess."""
    p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, close_fds=True)
    p.communicate(input=text.encode("utf-16-le"))


# ==============================================================================
# 5. SINGLE-WINDOW CHROME KIOSK CONTROLLER
# ==============================================================================
class KioskBrowserManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.dashboard_url = "file:///" + os.path.abspath(os.path.join(base_dir, "dashboard.html")).replace("\\", "/")
        self.keyboard_url = "file:///" + os.path.abspath(os.path.join(base_dir, "keyboard.html")).replace("\\", "/")
        self.chrome_proc = None
        self.last_external_url = "https://www.youtube.com"

    def find_chrome_exe(self):
        candidates = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return "chrome"

    def launch_kiosk(self):
        chrome_exe = self.find_chrome_exe()
        profile_dir = os.path.join(self.base_dir, "aura_kiosk_profile")
        cmd = [
            chrome_exe,
            "--kiosk",
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={profile_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-infobars",
            "--disable-session-crashed-bubble",
            "--autoplay-policy=no-user-gesture-required",
            self.dashboard_url,
        ]
        try:
            self.chrome_proc = subprocess.Popen(cmd)
            print("[Kiosk] Launched single-window Chrome Kiosk.")
        except Exception as e:
            print(f"[Kiosk] Fallback browser launch: {e}")
            webbrowser.open(self.dashboard_url)

    def navigate_single_window(self, target_url):
        """Swaps the active URL inside the single locked Chrome Kiosk window via CDP."""
        def _cdp_nav():
            try:
                req = urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=1.5)
                tabs = json.loads(req.read().decode("utf-8"))
                ws_url = None
                for t in tabs:
                    if t.get("type") == "page" and "webSocketDebuggerUrl" in t:
                        ws_url = t["webSocketDebuggerUrl"]
                        break
                if ws_url:
                    async def _send_nav():
                        async with websockets.connect(ws_url) as ws:
                            payload = {
                                "id": 1,
                                "method": "Page.navigate",
                                "params": {"url": target_url},
                            }
                            await ws.send(json.dumps(payload))
                            await ws.recv()
                    asyncio.run(_send_nav())
            except Exception as e:
                print(f"[Kiosk] CDP navigation notice: {e}")
        threading.Thread(target=_cdp_nav, daemon=True).start()

    def shutdown(self):
        try:
            if self.chrome_proc and self.chrome_proc.poll() is None:
                self.chrome_proc.terminate()
            else:
                pyautogui.hotkey("alt", "f4")
        except Exception:
            pass


# ==============================================================================
# 6. UNIFIED GAZE MATH & 9-POINT RED-DOT CALIBRATION
# ==============================================================================
class UnifiedGazeCalibration:
    def __init__(self):
        self.center_x = 0.50
        self.center_y = 0.50
        self.span_x = 0.085
        self.span_y = 0.065
        self.grid_points_gaze = None
        self.is_calibrated = False
        self.load()

    def load(self):
        if os.path.exists(CALIBRATION_FILE):
            try:
                data = np.load(CALIBRATION_FILE)
                if "center_x" in data:
                    self.center_x = float(data["center_x"])
                    self.center_y = float(data["center_y"])
                    self.span_x = max(float(data["span_x"]), 0.025)
                    self.span_y = max(float(data["span_y"]), 0.020)
                    if "grid_points_gaze" in data:
                        self.grid_points_gaze = data["grid_points_gaze"]
                    self.is_calibrated = True
                    print(f"[Calibration] Loaded unified {CALIBRATION_FILE} (HUD_HEIGHT_PX = 0).")
            except Exception as e:
                print(f"[Calibration] Could not load {CALIBRATION_FILE}: {e}")

    def save(self, points_gaze):
        pts = np.array(points_gaze, dtype=np.float32)
        self.center_x = float(pts[4, 0])
        self.center_y = float(pts[4, 1])
        min_x, max_x = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        min_y, max_y = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        self.span_x = max((max_x - min_x) * 0.45, 0.025)
        self.span_y = max((max_y - min_y) * 0.45, 0.020)
        self.grid_points_gaze = pts
        self.is_calibrated = True

        np.savez(
            CALIBRATION_FILE,
            center_x=self.center_x,
            center_y=self.center_y,
            span_x=self.span_x,
            span_y=self.span_y,
            grid_points_gaze=pts,
            hud_height_px=HUD_HEIGHT_PX,
        )
        print(f"[Calibration] Saved unified {CALIBRATION_FILE} successfully.")

    def evaluate_4way_dpad(self, gx, gy):
        """
        Single universal joystick mechanism shared by Dashboard, Keyboard, and ZoomGrid.
        Returns: (dpad_direction, grid_cell_1_to_9, norm_dx, norm_dy)
        """
        dx = (gx - self.center_x) / (self.span_x + 1e-6)
        dy = (gy - self.center_y) / (self.span_y + 1e-6)

        deadzone = 0.38
        if abs(dx) < deadzone and abs(dy) < deadzone:
            direction = "CENTER"
        elif abs(dx) >= abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"

        col = 1
        if dx < -deadzone:
            col = 0
        elif dx > deadzone:
            col = 2

        row = 1
        if dy < -deadzone:
            row = 0
        elif dy > deadzone:
            row = 2

        cell_index = row * 3 + col + 1  # 1..9 (5 is exact center)
        return direction, cell_index, float(np.clip(dx, -2.0, 2.0)), float(np.clip(dy, -2.0, 2.0))


# ==============================================================================
# 7. MASTER ENGINE CORE
# ==============================================================================
class AuraOSMasterEngine:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.kiosk = KioskBrowserManager(self.base_dir)
        self.calibration = UnifiedGazeCalibration()
        self.ws_clients = set()
        self.ws_loop = None

        # Operating State (Starts in SLEEPING mode so 3-Blink Particle Intro runs first)
        self.running = True
        self.awake = False
        self.mode = "DASHBOARD"  # "DASHBOARD", "KEYBOARD", "EXTERNAL"
        self.calibration_requested = False

        # Smoothed Gaze & Spring-Return Joystick State
        self.smooth_gx = self.calibration.center_x
        self.smooth_gy = self.calibration.center_y
        self.current_direction = "CENTER"
        self.current_cell = 5
        self.dpad_armed = True
        self.held_dir = "CENTER"
        self.held_dir_since = 0.0

        # ZoomGrid State (for External Websites)
        self.zoom_bounds = [0, 0, SCREEN_W, SCREEN_H]
        self.zoom_history = []
        self.overlay_win_name = "AuraOS_ZoomGrid_Overlay"
        self.overlay_initialized = False

        # Gesture Timers & Counters
        self.left_closed_start = None
        self.right_closed_start = None
        self.both_closed_start = None
        self.beep_5s_played = False
        self.beep_11s_played = False
        self.beep_20s_played = False
        self.slow_blink_timestamps = []

        # Live Camera rPPG Heart Rate & SOS State
        self.rppg_green_buf = deque(maxlen=RPPG_BUFFER_MAX)
        self.rppg_time_buf = deque(maxlen=RPPG_BUFFER_MAX)
        self.bpm_rolling_buf = deque()
        self.last_rppg_calc_t = 0.0
        self.rolling_bpm = 0.0
        self.abnormal_bpm_since = None
        self.sos_active = False
        self.last_sos_beep_t = 0.0

        # Shadow ML Model (Loaded purely for evaluator console verification)
        self.shadow_ml_model = None
        self.last_ml_label = "Neutral"
        self.last_ml_log_time = 0.0
        self._load_shadow_ml_model()

        # MediaPipe Face Landmarker Initialization
        self.detector = self._init_mediapipe()

    def _load_shadow_ml_model(self):
        model_path = os.path.join(self.base_dir, ML_MODEL_FILE)
        if os.path.exists(model_path):
            try:
                self.shadow_ml_model = joblib.load(model_path)
                print(f"[ML Shadow Model] Loaded '{ML_MODEL_FILE}' for evaluator logging.")
            except Exception as e:
                print(f"[ML Shadow Model] Could not load '{ML_MODEL_FILE}': {e}")
        else:
            print(f"[ML Shadow Model] '{ML_MODEL_FILE}' not found yet. Running OpenCV core.")

    def _init_mediapipe(self):
        model_full_path = os.path.join(self.base_dir, FACE_MODEL_PATH)
        if not os.path.exists(model_full_path):
            print("[MediaPipe] Downloading face_landmarker.task...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                model_full_path,
            )
        base_options = python.BaseOptions(model_asset_path=model_full_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        return vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def extract_eye_metrics(landmarks, frame_w, frame_h):
        def dist(p1, p2):
            return np.hypot(
                (landmarks[p1].x - landmarks[p2].x) * frame_w,
                (landmarks[p1].y - landmarks[p2].y) * frame_h,
            )

        ear_left = (dist(159, 145) + dist(158, 153)) / (2.0 * dist(33, 133) + 1e-6)
        ear_right = (dist(386, 374) + dist(385, 380)) / (2.0 * dist(362, 263) + 1e-6)
        collector_iris_x = dist(33, 468) / (dist(33, 133) + 1e-6)

        l_ratio_x = (landmarks[468].x - landmarks[33].x) / (landmarks[133].x - landmarks[33].x + 1e-6)
        r_ratio_x = (landmarks[473].x - landmarks[362].x) / (landmarks[263].x - landmarks[362].x + 1e-6)
        gaze_x = 0.5 * (l_ratio_x + r_ratio_x) * 0.75 + landmarks[1].x * 0.25

        l_ratio_y = (landmarks[468].y - landmarks[159].y) / (landmarks[145].y - landmarks[159].y + 1e-6)
        r_ratio_y = (landmarks[473].y - landmarks[386].y) / (landmarks[374].y - landmarks[386].y + 1e-6)
        gaze_y = 0.5 * (l_ratio_y + r_ratio_y) * 0.45 + landmarks[1].y * 0.55

        return ear_left, ear_right, collector_iris_x, gaze_x, gaze_y

    # --------------------------------------------------------------------------
    # 9-POINT RED-DOT CALIBRATION ROUTINE
    # --------------------------------------------------------------------------
    def run_9point_calibration(self, cap, timestamp_offset_ms):
        cal_win = "AuraOS_Calibration"
        cv2.namedWindow(cal_win, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(cal_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(cal_win, cv2.WND_PROP_TOPMOST, 1)

        margin_x = int(SCREEN_W * 0.14)
        margin_y = int(SCREEN_H * 0.14)
        xs = [margin_x, CENTER_X, SCREEN_W - margin_x]
        ys = [margin_y, CENTER_Y, SCREEN_H - margin_y]
        targets = [(x, y) for y in ys for x in xs]

        collected_points = []
        play_sound_async("WAKE")

        for idx, (tx, ty) in enumerate(targets):
            samples = []
            start_pt = time.time()
            while time.time() - start_pt < 2.1 and self.running:
                if user32.GetAsyncKeyState(0x1B) & 0x8000:
                    self.running = False
                    break

                ok, frame = cap.read()
                if not ok:
                    continue
                frame = cv2.flip(frame, 1)
                fh, fw, _ = frame.shape
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = self.detector.detect_for_video(mp_img, int(time.time() * 1000) + timestamp_offset_ms)

                if res.face_landmarks:
                    _, _, _, gx, gy = self.extract_eye_metrics(res.face_landmarks[0], fw, fh)
                    if time.time() - start_pt > 0.65:
                        samples.append([gx, gy])

                canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
                progress = min(1.0, (time.time() - start_pt) / 2.1)

                cv2.putText(
                    canvas,
                    f"AURAOS UNIFIED CALIBRATION ({idx + 1}/9) - FOCUS ON THE RED CENTER DOT",
                    (CENTER_X - 420, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (200, 230, 255),
                    2,
                )

                outer_radius = int(34 - 12 * progress)
                cv2.circle(canvas, (tx, ty), outer_radius, (0, 220, 120), 2, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 10, (25, 25, 25), -1, cv2.LINE_AA)
                cv2.circle(canvas, (tx, ty), 10, (255, 255, 255), 1, cv2.LINE_AA)
                # Solid 3-pixel bright red dot (0, 0, 255) dead in the center
                cv2.circle(canvas, (tx, ty), 3, (0, 0, 255), -1, cv2.LINE_AA)

                cv2.imshow(cal_win, canvas)
                cv2.waitKey(1)

            if samples:
                collected_points.append(np.median(samples, axis=0))
            else:
                collected_points.append([0.5, 0.5])
            play_sound_async("CLICK")

        cv2.destroyWindow(cal_win)
        if len(collected_points) == 9:
            self.calibration.save(collected_points)
            self.smooth_gx = self.calibration.center_x
            self.smooth_gy = self.calibration.center_y
            self.dpad_armed = True
            self.broadcast_ws({"type": "calib_state", "active": False, "completed": True})

    # --------------------------------------------------------------------------
    # TRANSPARENT 9-GRID ZOOM OVERLAY (EXTERNAL WEB BROWSING)
    # --------------------------------------------------------------------------
    def update_zoomgrid_overlay(self, eyes_closed_sec=0.0):
        if self.mode != "EXTERNAL":
            if self.overlay_initialized:
                cv2.destroyWindow(self.overlay_win_name)
                self.overlay_initialized = False
            return

        if not self.overlay_initialized:
            cv2.namedWindow(self.overlay_win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.overlay_win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(self.overlay_win_name, np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8))
            cv2.waitKey(1)
            make_window_transparent_clickthrough(self.overlay_win_name)
            self.overlay_initialized = True

        canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        x0, y0, x1, y1 = self.zoom_bounds
        w = x1 - x0
        h = y1 - y0

        for r in range(3):
            for c in range(3):
                cell_num = r * 3 + c + 1
                cx0 = x0 + int(c * w / 3)
                cy0 = y0 + int(r * h / 3)
                cx1 = x0 + int((c + 1) * w / 3)
                cy1 = y0 + int((r + 1) * h / 3)

                is_selected = (cell_num == self.current_cell)
                color = (0, 255, 180) if is_selected else (0, 140, 255)
                thickness = 3 if is_selected else 1
                cv2.rectangle(canvas, (cx0, cy0), (cx1, cy1), color, thickness)
                cv2.putText(
                    canvas,
                    str(cell_num),
                    (cx0 + 14, cy0 + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        # Manual Virtual Keyboard Fallback Badge in Top-Right of Cell 3
        kb_x0, kb_y0 = SCREEN_W - 210, 16
        kb_x1, kb_y1 = SCREEN_W - 16, 68
        kb_highlight = (self.current_cell == 3 and len(self.zoom_history) == 0)
        cv2.rectangle(canvas, (kb_x0, kb_y0), (kb_x1, kb_y1), (30, 30, 30), -1)
        cv2.rectangle(
            canvas,
            (kb_x0, kb_y0),
            (kb_x1, kb_y1),
            (0, 255, 255) if kb_highlight else (180, 180, 180),
            2,
        )
        cv2.putText(
            canvas,
            "[KB] KEYBOARD",
            (kb_x0 + 14, kb_y0 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255) if kb_highlight else (240, 240, 240),
            2,
        )

        # Absolute Center Reset Ring (HUD_HEIGHT_PX = 0)
        ring_color = (0, 255, 120) if self.current_direction == "CENTER" else (0, 215, 255)
        cv2.circle(canvas, (CENTER_X, CENTER_Y), 16, ring_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (CENTER_X, CENTER_Y), 3, (0, 0, 255), -1, cv2.LINE_AA)

        cv2.imshow(self.overlay_win_name, canvas)
        cv2.waitKey(1)

    # --------------------------------------------------------------------------
    # GESTURE EXECUTION (LEFT WINK, RIGHT WINK, 5S BACK, 11S CALIB, 20S HOME)
    # --------------------------------------------------------------------------
    def handle_left_wink(self):
        if not self.awake:
            return
        play_sound_async("ZOOM")

        if self.mode == "EXTERNAL":
            x0, y0, x1, y1 = self.zoom_bounds
            w, h = x1 - x0, y1 - y0
            if w > 160 and h > 120:
                self.zoom_history.append(list(self.zoom_bounds))
                r = (self.current_cell - 1) // 3
                c = (self.current_cell - 1) % 3
                nx0 = x0 + int(c * w / 3)
                ny0 = y0 + int(r * h / 3)
                nx1 = x0 + int((c + 1) * w / 3)
                ny1 = y0 + int((r + 1) * h / 3)
                self.zoom_bounds = [nx0, ny0, nx1, ny1]
        else:
            # Broadcast both formats so Dashboard and Keyboard both respond immediately
            self.broadcast_ws({"type": "action", "action": "LEFT_WINK_SELECT"})
            self.broadcast_ws({"type": "gesture", "action": "LEFT_WINK"})

    def handle_right_wink(self):
        if not self.awake:
            return
        play_sound_async("CLICK")

        if self.mode == "EXTERNAL":
            x0, y0, x1, y1 = self.zoom_bounds
            w, h = x1 - x0, y1 - y0
            r = (self.current_cell - 1) // 3
            c = (self.current_cell - 1) % 3
            target_x = x0 + int((c + 0.5) * w / 3)
            target_y = y0 + int((r + 0.5) * h / 3)

            if target_x >= SCREEN_W - 220 and target_y <= 90:
                self.switch_to_keyboard()
                return

            pyautogui.click(target_x, target_y)
            self.zoom_bounds = [0, 0, SCREEN_W, SCREEN_H]
            self.zoom_history.clear()

            def _check_ibeam():
                time.sleep(0.38)
                if self.mode == "EXTERNAL" and is_ibeam_cursor_active():
                    print("[Smart Typing] Windows I-Beam cursor detected! Switching to Virtual Keyboard...")
                    self.switch_to_keyboard()
            threading.Thread(target=_check_ibeam, daemon=True).start()
        elif self.mode == "DASHBOARD":
            self.broadcast_ws({"type": "action", "action": "LEFT_WINK_SELECT"})
            self.broadcast_ws({"type": "gesture", "action": "RIGHT_WINK"})
        else:
            self.broadcast_ws({"type": "action", "action": "RIGHT_WINK_SPEAK"})
            self.broadcast_ws({"type": "gesture", "action": "RIGHT_WINK"})

    def handle_5s_back(self):
        play_sound_async("CLICK")
        if self.mode == "EXTERNAL":
            if self.zoom_history:
                self.zoom_bounds = self.zoom_history.pop()
                print("[Gesture 5-10s] Undid ZoomGrid step.")
            else:
                pyautogui.hotkey("alt", "left")
                print("[Gesture 5-10s] Triggered Browser Previous Page.")
        else:
            self.broadcast_ws({"type": "move", "dir": "LEFT"})
            self.broadcast_ws({"type": "gesture", "action": "BACK"})

    def handle_11s_calibrate(self):
        print("[Gesture 11-19s] Triggering Hands-Free 9-Point Recalibration...")
        self.calibration_requested = True

    def handle_20s_home(self):
        print("[Gesture 20s+] Returning Home to AuraOS Dashboard...")
        self.switch_to_dashboard()

    # --------------------------------------------------------------------------
    # MODE & SCREEN ROUTING
    # --------------------------------------------------------------------------
    def switch_to_dashboard(self):
        self.mode = "DASHBOARD"
        self.zoom_bounds = [0, 0, SCREEN_W, SCREEN_H]
        self.zoom_history.clear()
        self.kiosk.navigate_single_window(self.kiosk.dashboard_url)

    def switch_to_keyboard(self):
        self.mode = "KEYBOARD"
        self.zoom_bounds = [0, 0, SCREEN_W, SCREEN_H]
        self.zoom_history.clear()
        self.kiosk.navigate_single_window(self.kiosk.keyboard_url)

    def switch_to_external(self, url):
        self.kiosk.last_external_url = url
        self.mode = "EXTERNAL"
        self.zoom_bounds = [0, 0, SCREEN_W, SCREEN_H]
        self.zoom_history.clear()
        self.kiosk.navigate_single_window(url)

    def inject_typed_text_and_return(self, text):
        copy_to_clipboard(text)
        target_url = self.kiosk.last_external_url or "https://www.google.com"
        self.switch_to_external(target_url)

        def _paste_worker():
            time.sleep(2.2)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.15)
            pyautogui.press("enter")
        threading.Thread(target=_paste_worker, daemon=True).start()

    # --------------------------------------------------------------------------
    # DUAL-PORT WEBSOCKET SERVER BRIDGE (ws://localhost:5006 & ws://localhost:8765)
    # --------------------------------------------------------------------------
    def broadcast_ws(self, payload):
        if not self.ws_loop or not self.ws_clients:
            return
        msg = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(self._broadcast_coroutine(msg), self.ws_loop)

    async def _broadcast_coroutine(self, msg):
        for ws in list(self.ws_clients):
            try:
                await ws.send(msg)
            except Exception:
                self.ws_clients.discard(ws)

    async def _ws_handler(self, websocket, *args):
        self.ws_clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                cmd = data.get("command") or data.get("type")

                if cmd == "open_url":
                    url = data.get("url", "https://www.youtube.com")
                    self.switch_to_external(url)
                elif cmd == "open_keyboard":
                    self.switch_to_keyboard()
                elif cmd == "open_dashboard":
                    self.switch_to_dashboard()
                elif cmd in ("calibrate", "start_calib"):
                    self.awake = True
                    self.calibration_requested = True
                elif cmd == "force_wake":
                    self.awake = True
                    play_sound_async("WAKE")
                elif cmd in ("inject_text", "speak_text"):
                    text = data.get("text", "")
                    self.inject_typed_text_and_return(text)
        except Exception:
            pass
        finally:
            self.ws_clients.discard(websocket)

    def start_ws_server(self):
        def _thread():
            self.ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.ws_loop)

            async def _serve():
                async with websockets.serve(self._ws_handler, "localhost", WS_PORT_MAIN), \
                           websockets.serve(self._ws_handler, "localhost", WS_PORT_DASHBOARD):
                    await asyncio.Future()

            self.ws_loop.run_until_complete(_serve())

        threading.Thread(target=_thread, daemon=True).start()

    # --------------------------------------------------------------------------
    # MAIN VISION, rPPG & GESTURE LOOP
    # --------------------------------------------------------------------------
    def run(self):
        self.start_ws_server()
        self.kiosk.launch_kiosk()

        cap = cv2.VideoCapture(0)
        if not self.calibration.is_calibrated:
            self.calibration_requested = True

        print("\n=== AURAOS MASTER ENGINE RUNNING ===")
        print(f"WebSocket Ports Active: ws://localhost:{WS_PORT_DASHBOARD} (Dashboard) & ws://localhost:{WS_PORT_MAIN} (Keyboard)")
        print("Caregiver Override: Press physical 'ESC' key at any time to exit AuraOS safely.")

        while self.running and cap.isOpened():
            # Caregiver-Only Global Physical ESC Key Listener
            if user32.GetAsyncKeyState(0x1B) & 0x8000:
                print("\n[Caregiver Override] Physical ESC key detected. Shutting down AuraOS safely...")
                self.running = False
                break

            # Optional 'C' key shortcut to trigger calibration anytime
            if user32.GetAsyncKeyState(0x43) & 0x8000:
                self.awake = True
                self.calibration_requested = True

            if self.calibration_requested:
                self.calibration_requested = False
                self.run_9point_calibration(cap, 10000)
                continue

            ok, frame = cap.read()
            if not ok:
                continue

            frame = cv2.flip(frame, 1)
            fh, fw, _ = frame.shape
            now = time.time()

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = self.detector.detect_for_video(mp_img, int(now * 1000))

            eyes_closed_sec = 0.0

            if results.face_landmarks:
                lms = results.face_landmarks[0]
                ear_l, ear_r, collector_iris_x, gx, gy = self.extract_eye_metrics(lms, fw, fh)

                # --------------------------------------------------------------
                # 1. LIVE CAMERA rPPG HEART RATE (BPM/RPM) + SOS ALERT
                # --------------------------------------------------------------
                green_val, _ = extract_forehead_green(frame, lms, fw, fh)
                if green_val is not None:
                    self.rppg_green_buf.append(green_val)
                    self.rppg_time_buf.append(now)

                if (
                    len(self.rppg_green_buf) >= RPPG_MIN_SAMPLES
                    and (now - self.last_rppg_calc_t) >= RPPG_UPDATE_INTERVAL_S
                ):
                    self.last_rppg_calc_t = now
                    est_bpm = compute_filtered_rppg_bpm(self.rppg_green_buf, self.rppg_time_buf)
                    if est_bpm is not None:
                        self.bpm_rolling_buf.append((now, est_bpm))

                while (
                    self.bpm_rolling_buf
                    and (now - self.bpm_rolling_buf[0][0]) > BPM_ROLLING_WINDOW_S
                ):
                    self.bpm_rolling_buf.popleft()

                if self.bpm_rolling_buf:
                    self.rolling_bpm = float(np.mean([b[1] for b in self.bpm_rolling_buf]))
                    if self.rolling_bpm > BPM_SOS_HIGH or self.rolling_bpm < BPM_SOS_LOW:
                        if self.abnormal_bpm_since is None:
                            self.abnormal_bpm_since = now
                        elif (now - self.abnormal_bpm_since) >= SOS_SUSTAINED_DURATION_S:
                            self.sos_active = True
                            if (now - self.last_sos_beep_t) >= SOS_BEEP_INTERVAL_S:
                                self.last_sos_beep_t = now
                                trigger_sos_beep_async()
                    else:
                        self.abnormal_bpm_since = None
                        self.sos_active = False

                # --------------------------------------------------------------
                # 2. SHADOW ML MODEL EXECUTION (FOR EVALUATOR CONSOLE LOGS ONLY)
                # --------------------------------------------------------------
                if self.shadow_ml_model is not None:
                    try:
                        ml_pred = int(self.shadow_ml_model.predict([[ear_l, ear_r, collector_iris_x]])[0])
                        ml_map = {0: "Neutral", 1: "Left Wink", 2: "Right Wink", 3: "Long Blink"}
                        self.last_ml_label = ml_map.get(ml_pred, "Neutral")
                        if now - self.last_ml_log_time > 1.5 and ml_pred != 0:
                            print(f"[ML Shadow Model Active] Evaluator Log: {self.last_ml_label}")
                            self.last_ml_log_time = now
                    except Exception:
                        pass

                # --------------------------------------------------------------
                # 3. OPENCV SPEED, SYMMETRY & PATTERN FILTERS (ACTUAL OS NAVIGATION)
                # --------------------------------------------------------------
                left_closed = ear_l < EAR_THRESHOLD
                right_closed = ear_r < EAR_THRESHOLD
                both_closed = left_closed and right_closed

                # Update Gaze & 4-Way Spring-Return D-Pad when both eyes are open
                if not left_closed and not right_closed:
                    self.smooth_gx = 0.80 * self.smooth_gx + 0.20 * gx
                    self.smooth_gy = 0.80 * self.smooth_gy + 0.20 * gy
                    direction, cell_idx, _, _ = self.calibration.evaluate_4way_dpad(
                        self.smooth_gx, self.smooth_gy
                    )
                    self.current_direction = direction
                    self.current_cell = cell_idx

                    # 1-Step Spring-Return Logic (Look at Center Dot to re-arm)
                    if not self.dpad_armed:
                        if direction == "CENTER":
                            self.dpad_armed = True
                            self.held_dir = "CENTER"
                    else:
                        if direction != "CENTER":
                            if direction != self.held_dir:
                                self.held_dir = direction
                                self.held_dir_since = now
                            elif (now - self.held_dir_since) >= 0.12:
                                if self.awake:
                                    self.broadcast_ws({"type": "move", "dir": direction, "repeat": False})
                                self.dpad_armed = False
                                self.held_dir = "CENTER"
                        else:
                            self.held_dir = "CENTER"

                # Both Eyes Closed Timeline (3-Blink Wake, 5s Back, 11s Calib, 20s Home)
                if both_closed:
                    self.left_closed_start = None
                    self.right_closed_start = None

                    if self.both_closed_start is None:
                        self.both_closed_start = now
                        self.beep_5s_played = False
                        self.beep_11s_played = False
                        self.beep_20s_played = False

                    eyes_closed_sec = now - self.both_closed_start

                    if eyes_closed_sec >= BACK_MIN_SEC and not self.beep_5s_played:
                        self.beep_5s_played = True
                        play_sound_async("BEEP_5S_BACK")
                    if eyes_closed_sec >= CALIB_MIN_SEC and not self.beep_11s_played:
                        self.beep_11s_played = True
                        play_sound_async("BEEP_11S_CALIB")
                    if eyes_closed_sec >= HOME_MIN_SEC and not self.beep_20s_played:
                        self.beep_20s_played = True
                        play_sound_async("CHIME_20S_HOME")
                else:
                    if self.both_closed_start is not None:
                        closed_dur = now - self.both_closed_start
                        self.both_closed_start = None

                        if closed_dur >= HOME_MIN_SEC:
                            self.handle_20s_home()
                        elif CALIB_MIN_SEC <= closed_dur <= CALIB_MAX_SEC:
                            self.handle_11s_calibrate()
                        elif BACK_MIN_SEC <= closed_dur <= BACK_MAX_SEC:
                            self.handle_5s_back()
                        elif WAKE_BLINK_MIN <= closed_dur <= WAKE_BLINK_MAX:
                            self.slow_blink_timestamps = [
                                t for t in self.slow_blink_timestamps if now - t <= WAKE_WINDOW_SEC
                            ]
                            self.slow_blink_timestamps.append(now)
                            if len(self.slow_blink_timestamps) >= 3:
                                self.slow_blink_timestamps.clear()
                                self.awake = not self.awake
                                play_sound_async("WAKE")
                                self.broadcast_ws({"type": "gesture", "action": "WAKE_TOGGLE", "awake": self.awake})

                    # Strict Asymmetry Filter for Left Wink vs Right Wink
                    asymmetry = abs(ear_l - ear_r)
                    if left_closed and not right_closed and asymmetry >= WINK_ASYMMETRY_MIN:
                        if self.left_closed_start is None:
                            self.left_closed_start = now
                    else:
                        if self.left_closed_start is not None:
                            dur = now - self.left_closed_start
                            self.left_closed_start = None
                            if WINK_MIN_DUR <= dur <= WINK_MAX_DUR:
                                self.handle_left_wink()

                    if right_closed and not left_closed and asymmetry >= WINK_ASYMMETRY_MIN:
                        if self.right_closed_start is None:
                            self.right_closed_start = now
                    else:
                        if self.right_closed_start is not None:
                            dur = now - self.right_closed_start
                            self.right_closed_start = None
                            if WINK_MIN_DUR <= dur <= WINK_MAX_DUR:
                                self.handle_right_wink()

                bpm_val = round(self.rolling_bpm, 1) if self.rolling_bpm else 0.0
                blink_prog = len(self.slow_blink_timestamps)

                # Broadcast telemetry for dashboard.html (matches exact telemetry schema)
                self.broadcast_ws({
                    "type": "telemetry",
                    "awake": self.awake,
                    "calibrated": self.calibration.is_calibrated,
                    "blink_progress": blink_prog,
                    "armed": self.dpad_armed,
                    "dir": self.current_direction,
                    "in_center": (self.current_direction == "CENTER"),
                    "bpm": bpm_val,
                    "rpm": bpm_val,
                    "sos": self.sos_active,
                    "status": "PORTAL ARMED" if self.awake else "SURVEILLANCE SLEEP",
                })

                # Broadcast state for keyboard.html
                self.broadcast_ws({
                    "type": "state",
                    "awake": self.awake,
                    "armed": self.dpad_armed,
                    "blink_count": blink_prog,
                    "direction": self.current_direction,
                    "cell": self.current_cell,
                    "ear_l": round(float(ear_l), 3),
                    "ear_r": round(float(ear_r), 3),
                    "eyes_closed_sec": round(float(eyes_closed_sec), 1),
                    "bpm": bpm_val,
                    "sos": self.sos_active,
                    "ml_class": self.last_ml_label,
                })

            self.update_zoomgrid_overlay(eyes_closed_sec)
            time.sleep(0.015)

        cap.release()
        cv2.destroyAllWindows()
        self.kiosk.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    engine = AuraOSMasterEngine()
    engine.run()