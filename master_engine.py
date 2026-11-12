"""
AuraOS Unified Master Engine (master_engine.py)
- Unified Zero-Offset Center Geometry (HUD_HEIGHT_PX = 0, BORDER_MARGIN_PX = 28)
- Wait-for-Blink 9-Point Calibration (CALIB_SETTLE_FRAMES = 6, CALIB_SAMPLES_NEEDED = 22, Monotonic mp_ts += 33)
- Verbatim Zero-Flicker Spring-Return 4-Way Joystick (extract_all_features + evaluate_4way_dpad)
- Smart Cell-Box Input Snap + Multi-Input Form Memory + Direct DOM Value & CDP Enter Injection
- WS_EX_NOACTIVATE Overlay Flag (9-Grid never steals OS focus from Chrome)
- Symmetric Eyes-Shut Timeline with 0.28s Grace Buffer (5s Back, 11s Calibrate, 20s Home)
- Live Camera Forehead rPPG Heart Rate (BPM/RPM) + Sustained Abnormal BPM SOS Alert
"""

import asyncio
from collections import deque
import ctypes
from ctypes import wintypes
import json
import math
import os
import pathlib
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import scipy.signal
import websockets

try:
    import winsound
except ImportError:
    winsound = None

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ==============================================================================
# 1. WIN32 CONSTANTS, HARDWARE MOUSE CLICK, I-BEAM DETECTOR & OVERLAY HELPERS
# ==============================================================================
user32 = ctypes.windll.user32

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
LWA_COLORKEY = 0x0001
SW_RESTORE = 9
SW_SHOWNOACTIVATE = 4

VK_SPACE = 0x20
VK_ESCAPE = 0x1B
VK_C = 0x43
VK_R = 0x52
VK_BACK = 0x08


def is_vk_down(vk_code):
    try:
        return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0
    except Exception:
        return False


def raw_win32_click(x, y, button="left"):
    user32.SetCursorPos(int(x), int(y))
    time.sleep(0.015)
    if button == "left":
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.025)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    elif button == "right":
        user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        time.sleep(0.025)
        user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)


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
    try:
        ci = CURSORINFO()
        ci.cbSize = ctypes.sizeof(CURSORINFO)
        if user32.GetCursorInfo(ctypes.byref(ci)):
            return ci.hCursor == HCURSOR_IBEAM
    except Exception:
        pass
    return False


def setup_transparent_overlay(window_name):
    """Configures the OpenCV window to be transparent, click-through, and non-activating."""
    hwnd = user32.FindWindowW(None, window_name)
    if hwnd:
        try:
            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                ex_style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            )
            user32.SetLayeredWindowAttributes(hwnd, 0, 0, LWA_COLORKEY)
            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
        except Exception:
            pass
    return hwnd


def force_window_topmost(window_name, hwnd):
    if not hwnd or not user32.IsWindow(hwnd):
        hwnd = setup_transparent_overlay(window_name)
    if hwnd:
        try:
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, SW_RESTORE)
            else:
                user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)

            ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            req_flags = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            if (ex_style & req_flags) != req_flags:
                user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style | req_flags)
                user32.SetLayeredWindowAttributes(hwnd, 0, 0, LWA_COLORKEY)

            user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
        except Exception:
            pass
    return hwnd


def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash_length=12):
    x0, y0 = pt1
    x1, y1 = pt2
    for x in range(x0, x1, dash_length * 2):
        cv2.line(img, (x, y0), (min(x + dash_length, x1), y0), color, thickness)
        cv2.line(img, (x, y1), (min(x + dash_length, x1), y1), color, thickness)
    for y in range(y0, y1, dash_length * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash_length, y1)), color, thickness)
        cv2.line(img, (x1, y), (x1, min(y + dash_length, y1)), color, thickness)


def copy_to_clipboard(text):
    try:
        p = subprocess.Popen(["clip"], stdin=subprocess.PIPE, close_fds=True)
        p.communicate(input=str(text).encode("utf-16-le"))
    except Exception:
        pass


# ==============================================================================
# 2. AUDIO CHECKPOINT ENGINE & PERSISTENT WINDOWS SAPI5 VOICE WORKER
# ==============================================================================
def play_sound_async(pattern):
    def _worker():
        if winsound is None:
            return
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
            elif pattern == "CALIB_PT":
                winsound.Beep(1800, 60)
            elif pattern == "WAKE":
                winsound.Beep(1400, 110)
            elif pattern == "SLEEP":
                winsound.Beep(900, 90)
            elif pattern == "ZOOM":
                winsound.Beep(1300, 60)
            elif pattern == "CLICK":
                winsound.Beep(2000, 80)
            elif pattern == "SOS":
                winsound.Beep(2500, 400)
        except Exception:
            pass

    threading.Thread(target=_worker, daemon=True).start()


tts_queue = queue.Queue()


def start_persistent_tts_worker():
    def _worker():
        if winsound is not None:
            try:
                proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-NoLogo", "-Command", "-"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                proc.stdin.write("$voice = New-Object -ComObject Sapi.SpVoice\n")
                proc.stdin.write("$voice.Rate = 1\n")
                proc.stdin.write("$voice.Volume = 100\n")
                proc.stdin.flush()

                while True:
                    text = tts_queue.get()
                    if text is None:
                        break
                    clean = str(text).replace("'", "").replace('"', "").replace("\n", " ").strip()
                    if not clean:
                        clean = "Please type something first"
                    try:
                        winsound.Beep(1200, 90)
                    except Exception:
                        pass
                    proc.stdin.write(f"[void]$voice.Speak('{clean}', 3)\n")
                    proc.stdin.flush()
            except Exception as e:
                print(f"[TTS Warning] Persistent voice fallback error: {e}")

    threading.Thread(target=_worker, daemon=True).start()


start_persistent_tts_worker()


def speak_out_loud(text):
    while not tts_queue.empty():
        try:
            tts_queue.get_nowait()
        except queue.Empty:
            break
    tts_queue.put(text)


# ==============================================================================
# 3. EXACT CONSTANTS FROM VISION_ENGINE_1.PY (WITH HUD_HEIGHT_PX = 0)
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")
ML_MODEL_PATH = os.path.join(SCRIPT_DIR, "ocular_ml_model.pkl")
CALIB_LOG_PATH = os.path.join(SCRIPT_DIR, "gaze_grid_calibration.npz")

CDP_PORT = 9222
WS_PORT_DASHBOARD = 5006
WS_PORT_KEYBOARD = 8765
WS_PORT_KEYBOARD_LEGACY = 5005

HUD_HEIGHT_PX = 0
BORDER_MARGIN_PX = 28
CALIB_SETTLE_FRAMES = 6
CALIB_SAMPLES_NEEDED = 22
CALIB_MAX_FRAMES = 120

DPAD_FIRE_RATIO = 0.42
DPAD_RESET_RATIO = 0.22
AXIS_DOMINANCE_RATIO = 1.30
UP_GAIN_BOOST = 1.15
DOWN_GAIN_BOOST = 1.28
DPAD_HOLD_CONFIRM_S = 0.12
DPAD_REARM_SETTLE_S = 0.18
MAX_SACCADE_VEL = 0.018
IRIS_SMOOTH_WINDOW = 5

SLOW_BLINK_MIN_DUR_S = 0.13
SLOW_BLINK_MAX_DUR_S = 0.90
SLOW_BLINK_MAX_GAP_S = 2.40
CLUTCH_BLINKS_REQUIRED = 3

ACTION_COOLDOWN_S = 1.05
MAX_ZOOM_DEPTH = 6

EYES_SHUT_EAR_THRESH = 0.178
EYES_SHUT_SYMMETRY_MAX = 0.045
EYES_SHUT_GRACE_SEC = 0.28
BACK_MIN_SEC = 5.0
BACK_MAX_SEC = 10.9
CALIB_MIN_SEC = 11.0
CALIB_MAX_SEC = 19.9
HOME_MIN_SEC = 20.0

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


# ==============================================================================
# 4. SINGLE-WINDOW CHROME KIOSK CONTROLLER & SMART DOM INPUT ENGINE
# ==============================================================================
class KioskBrowserManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.dashboard_url = pathlib.Path(os.path.join(base_dir, "dashboard.html")).as_uri()
        self.keyboard_url = pathlib.Path(os.path.join(base_dir, "keyboard.html")).as_uri()
        self.chrome_proc = None
        self.last_external_url = "https://www.youtube.com"
        self.last_clicked_coords = None
        self.last_clicked_norm = None
        self.last_input_index = 0
        self.page_form_values = {}  # {url_key: {input_index: typed_value}} for multi-input forms

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
            "--remote-allow-origins=*",
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
            print(f"[Kiosk] Auto-launched single-window Chrome Kiosk: {self.dashboard_url}")
        except Exception as e:
            print(f"[Kiosk] Fallback browser launch: {e}")
            webbrowser.open(self.dashboard_url)

    def _get_visible_page_ws_url(self):
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json", timeout=1.2)
            tabs = json.loads(req.read().decode("utf-8"))
            for t in tabs:
                url = t.get("url", "")
                if (
                    t.get("type") == "page"
                    and "webSocketDebuggerUrl" in t
                    and not url.startswith(("devtools://", "chrome-extension://"))
                ):
                    return t["webSocketDebuggerUrl"]
            for t in tabs:
                if t.get("type") == "page" and "webSocketDebuggerUrl" in t:
                    return t["webSocketDebuggerUrl"]
        except Exception:
            pass
        return None

    def navigate_single_window(self, target_url):
        def _cdp_nav():
            ws_url = self._get_visible_page_ws_url()
            if ws_url:
                try:
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
                    return
                except Exception as e:
                    print(f"[Kiosk] CDP navigation notice: {e}")
        threading.Thread(target=_cdp_nav, daemon=True).start()

    def find_input_inside_cell_or_focused(self, norm_x0, norm_y0, norm_x1, norm_y1):
        """
        Checks if the clicked 9-Grid cell rectangle contains a visible <input>, <textarea>,
        or [contenteditable]. If found, returns (nx, ny, input_index, current_page_url).
        """
        ws_url = self._get_visible_page_ws_url()
        if not ws_url:
            return None

        js_expr = f"""
        (() => {{
            function isEditable(el) {{
                if (!el) return false;
                const tag = (el.tagName || '').toUpperCase();
                if (tag === 'INPUT') {{
                    const t = (el.type || 'text').toLowerCase();
                    const nonText = ['button', 'checkbox', 'color', 'file', 'hidden', 'image', 'radio', 'range', 'reset', 'submit'];
                    return !nonText.includes(t) && !el.disabled && !el.readOnly;
                }}
                if (tag === 'TEXTAREA') return !el.disabled && !el.readOnly;
                if (el.isContentEditable === true) return true;
                const role = (el.getAttribute && (el.getAttribute('role') || '')).toLowerCase();
                return role === 'textbox' || role === 'searchbox' || role === 'combobox';
            }}

            function collectVisibleInputs(root, out) {{
                if (!root || !root.querySelectorAll) return;
                const nodes = root.querySelectorAll('input, textarea, [contenteditable="true"], [role="searchbox"], [role="textbox"]');
                for (const n of nodes) {{
                    if (!isEditable(n)) continue;
                    const r = n.getBoundingClientRect();
                    if (r.width < 24 || r.height < 12) continue;
                    const s = window.getComputedStyle(n);
                    if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) continue;
                    if (!out.includes(n)) out.push(n);
                }}
                const all = root.querySelectorAll('*');
                for (const el of all) {{
                    if (el.shadowRoot) collectVisibleInputs(el.shadowRoot, out);
                }}
            }}

            const vw = window.innerWidth || 1;
            const vh = window.innerHeight || 1;
            const x0 = {norm_x0} * vw;
            const y0 = {norm_y0} * vh;
            const x1 = {norm_x1} * vw;
            const y1 = {norm_y1} * vh;
            const cellCx = 0.5 * (x0 + x1);
            const cellCy = 0.5 * (y0 + y1);

            const candidates = [];
            collectVisibleInputs(document, candidates);

            let bestIdx = -1;
            let bestDist = 1e9;
            let bestCenter = null;

            for (let i = 0; i < candidates.length; i++) {{
                const el = candidates[i];
                const r = el.getBoundingClientRect();
                const intersects = (r.right >= x0 && r.left <= x1 && r.bottom >= y0 && r.top <= y1);
                if (intersects) {{
                    const cx = r.left + r.width * 0.5;
                    const cy = r.top + r.height * 0.5;
                    const d = Math.hypot(cx - cellCx, cy - cellCy);
                    if (d < bestDist) {{
                        bestDist = d;
                        bestIdx = i;
                        bestCenter = {{ nx: cx / vw, ny: cy / vh }};
                    }}
                }}
            }}

            if (bestIdx >= 0) {{
                const el = candidates[bestIdx];
                el.focus();
                if (typeof el.click === 'function') el.click();
                return {{
                    found: true,
                    nx: bestCenter.nx,
                    ny: bestCenter.ny,
                    idx: bestIdx,
                    href: window.location.href
                }};
            }}

            return {{ found: false }};
        }})()
        """
        try:
            async def _eval():
                async with websockets.connect(ws_url) as ws:
                    payload = {
                        "id": 2,
                        "method": "Runtime.evaluate",
                        "params": {"expression": js_expr, "returnByValue": True},
                    }
                    await ws.send(json.dumps(payload))
                    raw_resp = await asyncio.wait_for(ws.recv(), timeout=0.9)
                    resp = json.loads(raw_resp)
                    val = resp.get("result", {}).get("result", {}).get("value")
                    if isinstance(val, dict) and val.get("found"):
                        return (
                            float(val.get("nx", 0.5)),
                            float(val.get("ny", 0.5)),
                            int(val.get("idx", 0)),
                            str(val.get("href", "")),
                        )
                    return None
            return asyncio.run(_eval())
        except Exception:
            return None

    def navigate_and_inject_text_via_cdp(self, target_url, text_to_type, saved_norm=None, target_idx=0):
        """
        Navigates Chrome back to target_url, waits until keyboard.html has unloaded,
        restores any previously typed multi-input form fields for this URL, sets the
        new text into target_idx (or closest input to saved_norm), and submits if it is
        a search bar or the final input field in the form.
        """
        ws_url = self._get_visible_page_ws_url()
        if not ws_url:
            return False

        url_key = target_url.split("#")[0]
        if url_key not in self.page_form_values:
            self.page_form_values[url_key] = {}
        self.page_form_values[url_key][int(target_idx)] = text_to_type

        norm_x = saved_norm[0] if saved_norm else -1.0
        norm_y = saved_norm[1] if saved_norm else -1.0
        saved_map_json = json.dumps(self.page_form_values[url_key])
        safe_text_json = json.dumps(text_to_type)

        inject_and_submit_js = f"""
        (() => {{
            const textToType = {safe_text_json};
            const targetIdx = {int(target_idx)};
            const savedMap = {saved_map_json};

            function isEditable(el) {{
                if (!el) return false;
                const tag = (el.tagName || '').toUpperCase();
                if (tag === 'INPUT') {{
                    const t = (el.type || 'text').toLowerCase();
                    const nonText = ['button', 'checkbox', 'color', 'file', 'hidden', 'image', 'radio', 'range', 'reset', 'submit'];
                    return !nonText.includes(t) && !el.disabled && !el.readOnly;
                }}
                if (tag === 'TEXTAREA') return !el.disabled && !el.readOnly;
                if (el.isContentEditable === true) return true;
                const role = (el.getAttribute && (el.getAttribute('role') || '')).toLowerCase();
                return role === 'textbox' || role === 'searchbox' || role === 'combobox';
            }}

            function collectVisibleInputs(root, out) {{
                if (!root || !root.querySelectorAll) return;
                const nodes = root.querySelectorAll('input, textarea, [contenteditable="true"], [role="searchbox"], [role="textbox"]');
                for (const n of nodes) {{
                    if (!isEditable(n)) continue;
                    const r = n.getBoundingClientRect();
                    if (r.width < 24 || r.height < 12) continue;
                    const s = window.getComputedStyle(n);
                    if (s.display === 'none' || s.visibility === 'hidden' || parseFloat(s.opacity) === 0) continue;
                    if (!out.includes(n)) out.push(n);
                }}
                const all = root.querySelectorAll('*');
                for (const el of all) {{
                    if (el.shadowRoot) collectVisibleInputs(el.shadowRoot, out);
                }}
            }}

            function setElementValue(el, val) {{
                el.focus();
                if (typeof el.click === 'function') el.click();
                const tag = (el.tagName || '').toUpperCase();
                if (tag === 'INPUT') {{
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
                    if (setter) setter.call(el, val);
                    else el.value = val;
                }} else if (tag === 'TEXTAREA') {{
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
                    if (setter) setter.call(el, val);
                    else el.value = val;
                }} else if (el.isContentEditable) {{
                    el.textContent = val;
                }}
                el.dispatchEvent(new InputEvent('input', {{ bubbles: true, cancelable: true, inputType: 'insertText', data: val }}));
                el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            }}

            const visibleInputs = [];
            collectVisibleInputs(document, visibleInputs);
            if (visibleInputs.length === 0) return {{ ok: false, reason: 'no_inputs' }};

            // 1. Restore all previously filled fields on this page (for multi-input login/signup forms)
            for (const [k, v] of Object.entries(savedMap)) {{
                const idx = parseInt(k, 10);
                if (!isNaN(idx) && idx >= 0 && idx < visibleInputs.length && idx !== targetIdx) {{
                    setElementValue(visibleInputs[idx], v);
                }}
            }}

            // 2. Pick the target input by index or closest distance to saved_norm
            let target = null;
            if (targetIdx >= 0 && targetIdx < visibleInputs.length) {{
                target = visibleInputs[targetIdx];
            }} else if ({norm_x} >= 0 && {norm_y} >= 0) {{
                const vw = window.innerWidth || 1;
                const vh = window.innerHeight || 1;
                const tx = {norm_x} * vw;
                const ty = {norm_y} * vh;
                let bestDist = 1e9;
                for (const el of visibleInputs) {{
                    const r = el.getBoundingClientRect();
                    const d = Math.hypot((r.left + r.width * 0.5) - tx, (r.top + r.height * 0.5) - ty);
                    if (d < bestDist) {{
                        bestDist = d;
                        target = el;
                    }}
                }}
            }}
            if (!target) target = visibleInputs[0];

            // 3. Inject value into target
            setElementValue(target, textToType);

            // 4. Determine whether to auto-submit (search boxes or single-input or last field in form)
            const typeAttr = (target.type || '').toLowerCase();
            const nameAttr = (target.name || '').toLowerCase();
            const idAttr = (target.id || '').toLowerCase();
            const roleAttr = (target.getAttribute && (target.getAttribute('role') || '')).toLowerCase();
            const isSearchBox = (
                typeAttr === 'search' ||
                nameAttr === 'q' ||
                nameAttr === 'search' ||
                nameAttr === 'search_query' ||
                idAttr.includes('search') ||
                roleAttr === 'searchbox' ||
                roleAttr === 'combobox'
            );
            const isLastInput = (visibleInputs.indexOf(target) === visibleInputs.length - 1);

            if (isSearchBox || visibleInputs.length === 1 || isLastInput) {{
                const enterOpts = {{ key: 'Enter', code: 'Enter', keyCode: 13, which: 13, bubbles: true, cancelable: true }};
                target.dispatchEvent(new KeyboardEvent('keydown', enterOpts));
                target.dispatchEvent(new KeyboardEvent('keypress', enterOpts));
                target.dispatchEvent(new KeyboardEvent('keyup', enterOpts));

                const host = (window.location.hostname || '').toLowerCase();
                if (host.includes('youtube.com')) {{
                    const ytBtn = document.querySelector('button#search-icon-legacy, ytd-searchbox button.ytSearchboxComponentSearchButton');
                    if (ytBtn) {{
                        ytBtn.click();
                        return {{ ok: true, submitted: true, method: 'yt_btn' }};
                    }}
                }}

                const form = target.form || (target.closest && target.closest('form'));
                if (form) {{
                    setTimeout(() => {{
                        try {{
                            if (typeof form.requestSubmit === 'function') form.requestSubmit();
                            else form.submit();
                        }} catch (e) {{
                            try {{ form.submit(); }} catch (e2) {{}}
                        }}
                    }}, 80);
                    return {{ ok: true, submitted: true, method: 'form_submit' }};
                }}
                return {{ ok: true, submitted: true, method: 'enter_key' }};
            }}

            return {{ ok: true, submitted: false, method: 'field_filled_only' }};
        }})()
        """

        async def _run_sequence():
            async with websockets.connect(ws_url) as ws:
                msg_id = 10

                async def cdp_call(method, params=None, timeout_s=4.0):
                    nonlocal msg_id
                    msg_id += 1
                    cur_id = msg_id
                    await ws.send(json.dumps({"id": cur_id, "method": method, "params": params or {}}))
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                        data = json.loads(raw)
                        if data.get("id") == cur_id:
                            return data

                await cdp_call("Page.BringToFront", {})
                await cdp_call("Page.navigate", {"url": target_url}, timeout_s=6.0)

                for _ in range(45):
                    await asyncio.sleep(0.15)
                    try:
                        res = await cdp_call(
                            "Runtime.evaluate",
                            {
                                "expression": """
                                (() => {
                                    const href = (window.location.href || '').toLowerCase();
                                    if (href.includes('keyboard.html') || href.startsWith('about:')) return false;
                                    if (document.readyState !== 'complete' && document.readyState !== 'interactive') return false;
                                    return document.querySelectorAll('input, textarea, [contenteditable="true"]').length > 0;
                                })()
                                """,
                                "returnByValue": True,
                            },
                            timeout_s=1.5,
                        )
                        if res.get("result", {}).get("result", {}).get("value") is True:
                            break
                    except Exception:
                        pass

                await asyncio.sleep(0.35)

                for _ in range(5):
                    try:
                        res = await cdp_call(
                            "Runtime.evaluate",
                            {"expression": inject_and_submit_js, "returnByValue": True},
                            timeout_s=2.0,
                        )
                        val = res.get("result", {}).get("result", {}).get("value")
                        if isinstance(val, dict) and val.get("ok"):
                            print(f"[CDP Inject Success] Injected '{text_to_type}' ({val.get('method')})")
                            if val.get("submitted"):
                                await asyncio.sleep(0.1)
                                await cdp_call(
                                    "Input.dispatchKeyEvent",
                                    {
                                        "type": "rawKeyDown",
                                        "windowsVirtualKeyCode": 13,
                                        "nativeVirtualKeyCode": 13,
                                        "key": "Enter",
                                        "code": "Enter",
                                        "text": "\r",
                                        "unmodifiedText": "\r",
                                    },
                                )
                                await cdp_call(
                                    "Input.dispatchKeyEvent",
                                    {
                                        "type": "keyUp",
                                        "windowsVirtualKeyCode": 13,
                                        "nativeVirtualKeyCode": 13,
                                        "key": "Enter",
                                        "code": "Enter",
                                    },
                                )
                            return True
                    except Exception:
                        pass
                    await asyncio.sleep(0.25)

                return False

        try:
            return asyncio.run(_run_sequence())
        except Exception as e:
            print(f"[CDP Inject Notice] Error: {e}")
            return False

    def shutdown(self):
        try:
            if self.chrome_proc and self.chrome_proc.poll() is None:
                self.chrome_proc.terminate()
        except Exception:
            pass


# ==============================================================================
# 5. VERBATIM VISION_ENGINE_1.PY FUNCTIONS (CALIBRATION, DPAD, WINK, rPPG)
# ==============================================================================
def ensure_landmarker_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading MediaPipe Face Landmarker AI model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)


def make_border_targets(screen_w, screen_h):
    m = BORDER_MARGIN_PX
    top_y = float(HUD_HEIGHT_PX + m)
    bot_y = float(screen_h - m)
    mid_y = float(HUD_HEIGHT_PX + (screen_h - HUD_HEIGHT_PX) // 2)
    xs = [float(m), float(screen_w // 2), float(screen_w - m)]
    ys = [top_y, mid_y, bot_y]
    points = [(xs[c], ys[r]) for r in range(3) for c in range(3)]
    return points, xs, ys


def load_saved_gaze_grid():
    if os.path.exists(CALIB_LOG_PATH):
        try:
            data = np.load(CALIB_LOG_PATH)
            if "fx" in data and "fy" in data:
                fx = np.asarray(data["fx"], dtype=np.float64)
                fy = np.asarray(data["fy"], dtype=np.float64)
                if fx.shape == (3, 3) and fy.shape == (3, 3):
                    print(f"[AuraOS] Loaded saved 9-point calibration from {CALIB_LOG_PATH}")
                    return fx, fy, True
        except Exception:
            pass
    return np.zeros((3, 3), dtype=np.float64), np.zeros((3, 3), dtype=np.float64), False


def extract_all_features(landmarks, fw, fh):
    def pt(i):
        return landmarks[i].x, landmarks[i].y

    def pdist(i, j):
        x1, y1 = pt(i)
        x2, y2 = pt(j)
        return math.hypot((x1 - x2) * fw, (y1 - y2) * fh)

    ear_left = (pdist(159, 145) + pdist(158, 153)) / (2.0 * pdist(33, 133) + 1e-6)
    ear_right = (pdist(386, 374) + pdist(385, 380)) / (2.0 * pdist(362, 263) + 1e-6)
    ear_diff = ear_left - ear_right
    ear_avg = 0.5 * (ear_left + ear_right)
    compat_iris_x = (landmarks[468].x - landmarks[33].x) / (
        (landmarks[133].x - landmarks[33].x) + 1e-6
    )

    p33 = np.array(pt(33), dtype=np.float64)
    p263 = np.array(pt(263), dtype=np.float64)
    axis = p33 - p263
    if axis[0] < 0.0:
        axis = -axis
    ax = axis / (np.linalg.norm(axis) + 1e-6)
    perp = np.array([-ax[1], ax[0]], dtype=np.float64)
    if perp[1] < 0.0:
        perp = -perp

    def iris_offset(outer_i, inner_i, iris_i):
        outer = np.array(pt(outer_i), dtype=np.float64)
        inner = np.array(pt(inner_i), dtype=np.float64)
        iris = np.array(pt(iris_i), dtype=np.float64)
        center = 0.5 * (outer + inner)
        width = float(np.linalg.norm(inner - outer)) + 1e-6
        rel = iris - center
        gx = float(np.dot(rel, ax) / width)
        gy = float(np.dot(rel, perp) / width)
        return gx, gy

    rx, ry = iris_offset(33, 133, 468)
    lx, ly = iris_offset(263, 362, 473)

    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear_diff": ear_diff,
        "ear_avg": ear_avg,
        "compat_iris_x": compat_iris_x,
        "iris_x": 0.5 * (rx + lx),
        "iris_y": 0.5 * (ry + ly),
    }


def evaluate_4way_dpad(fx_grid, fy_grid, live_fx, live_fy, bias_x=0.0, bias_y=0.0):
    fx = live_fx - bias_x
    fy = live_fy - bias_y

    left_mid = float(np.median(fx_grid[:, 0]))
    right_mid = float(np.median(fx_grid[:, 2]))
    h_total = right_mid - left_mid
    col_t = (
        float(np.clip((fx - left_mid) / h_total, 0.0, 1.0) * 2.0)
        if abs(h_total) > 1e-6
        else 1.0
    )
    c_idx = int(min(1, math.floor(col_t)))
    c_frac = col_t - c_idx

    up_anchor = float((1.0 - c_frac) * fy_grid[0, c_idx] + c_frac * fy_grid[0, c_idx + 1])
    cy = float((1.0 - c_frac) * fy_grid[1, c_idx] + c_frac * fy_grid[1, c_idx + 1])
    down_anchor = float((1.0 - c_frac) * fy_grid[2, c_idx] + c_frac * fy_grid[2, c_idx + 1])

    v_total = down_anchor - up_anchor
    row_t = (
        float(np.clip((fy - up_anchor) / v_total, 0.0, 1.0) * 2.0)
        if abs(v_total) > 1e-6
        else 1.0
    )
    r_idx = int(min(1, math.floor(row_t)))
    r_frac = row_t - r_idx

    left_anchor = float((1.0 - r_frac) * fx_grid[r_idx, 0] + r_frac * fx_grid[r_idx + 1, 0])
    cx = float((1.0 - r_frac) * fx_grid[r_idx, 1] + r_frac * fx_grid[r_idx + 1, 1])
    right_anchor = float((1.0 - r_frac) * fx_grid[r_idx, 2] + r_frac * fx_grid[r_idx + 1, 2])

    span_left, span_right = left_anchor - cx, right_anchor - cx
    span_up, span_down = up_anchor - cy, down_anchor - cy

    dx, dy = fx - cx, fy - cy

    raw_left = (dx / span_left) if abs(span_left) > 1e-5 else 0.0
    raw_right = (dx / span_right) if abs(span_right) > 1e-5 else 0.0
    raw_up = ((dy / span_up) * UP_GAIN_BOOST) if abs(span_up) > 1e-5 else 0.0
    raw_down = ((dy / span_down) * DOWN_GAIN_BOOST) if abs(span_down) > 1e-5 else 0.0

    scores = {
        "LEFT": max(0.0, float(raw_left)),
        "RIGHT": max(0.0, float(raw_right)),
        "UP": max(0.0, float(raw_up)),
        "DOWN": max(0.0, float(raw_down)),
    }

    max_h, max_v = max(scores["LEFT"], scores["RIGHT"]), max(scores["UP"], scores["DOWN"])
    best_score = max(max_h, max_v)

    in_center_reset_zone = best_score <= DPAD_RESET_RATIO
    direction = "CENTER"

    if best_score >= DPAD_FIRE_RATIO:
        if max_h >= max_v * AXIS_DOMINANCE_RATIO:
            direction = "LEFT" if scores["LEFT"] >= scores["RIGHT"] else "RIGHT"
        elif max_v >= max_h * AXIS_DOMINANCE_RATIO:
            direction = "UP" if scores["UP"] >= scores["DOWN"] else "DOWN"

    return direction, scores, in_center_reset_zone


def predict_ocular_action(ml_model, feats):
    el, er, ediff, eavg = (
        feats["ear_left"],
        feats["ear_right"],
        feats["ear_diff"],
        feats["ear_avg"],
    )

    if el < 0.175 and er < 0.175 and abs(ediff) < 0.045:
        return 3

    if ml_model is not None:
        try:
            n_features = getattr(ml_model, "n_features_in_", 4)
            X = (
                [[el, er, ediff, eavg]]
                if n_features == 4
                else [[el, er, feats["compat_iris_x"]]]
            )
            pred = int(ml_model.predict(X)[0])
            if pred == 1 and ediff <= -0.052:
                return 1
            if pred == 2 and ediff >= 0.038:
                return 2
            if pred == 3 and eavg < 0.180 and abs(ediff) < 0.045:
                return 3
        except Exception:
            pass

    # Left Wink (1)
    if el < 0.192 and er > 0.208 and ediff <= -0.052:
        return 1

    # Right Wink (2) - Relaxed left-eye squint tolerance so right-wink click is effortless
    if er < 0.210 and el > 0.172 and ediff >= 0.038:
        return 2

    return 0


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


class SlowBlinkClutch:
    def __init__(self):
        self.blink_active = False
        self.blink_start_t = 0.0
        self.slow_blink_timestamps = deque(maxlen=CLUTCH_BLINKS_REQUIRED)
        self.single_blink_ready = False
        self.single_blink_time = 0.0

    def update(self, now, is_blinking):
        clutch_toggled = False
        single_action_blink = False
        immediate_blink_released = False

        if (
            self.slow_blink_timestamps
            and (now - self.slow_blink_timestamps[-1]) > SLOW_BLINK_MAX_GAP_S
        ):
            if len(self.slow_blink_timestamps) == 1 and self.single_blink_ready:
                single_action_blink = True
                self.single_blink_ready = False
            self.slow_blink_timestamps.clear()

        if is_blinking:
            if not self.blink_active:
                self.blink_active = True
                self.blink_start_t = now
        else:
            if self.blink_active:
                dur = now - self.blink_start_t
                self.blink_active = False
                if SLOW_BLINK_MIN_DUR_S <= dur <= SLOW_BLINK_MAX_DUR_S:
                    immediate_blink_released = True
                    self.slow_blink_timestamps.append(now)
                    self.single_blink_ready = len(self.slow_blink_timestamps) == 1
                    self.single_blink_time = now
                    if len(self.slow_blink_timestamps) >= CLUTCH_BLINKS_REQUIRED:
                        clutch_toggled = True
                        self.slow_blink_timestamps.clear()
                        self.single_blink_ready = False

        if (
            self.single_blink_ready
            and len(self.slow_blink_timestamps) == 1
            and not self.blink_active
            and (now - self.single_blink_time) > 0.65
        ):
            single_action_blink = True
            self.single_blink_ready = False

        return (
            clutch_toggled,
            single_action_blink,
            len(self.slow_blink_timestamps),
            immediate_blink_released,
        )


# ==============================================================================
# 6. MULTI-PORT WEBSOCKET SERVER (5006 FOR DASHBOARD, 8765 & 5005 FOR KEYBOARD)
# ==============================================================================
ws_queue = queue.Queue(maxsize=60)
client_cmd_queue = queue.Queue(maxsize=25)


def start_websocket_server():
    connected_clients = set()

    async def ws_handler(websocket, *args):
        connected_clients.add(websocket)
        try:
            async for raw_msg in websocket:
                try:
                    data = json.loads(raw_msg)
                    msg_type = data.get("type") or data.get("command")
                    if msg_type in (
                        "open_url",
                        "open_keyboard",
                        "open_dashboard",
                        "start_calib",
                        "calibrate",
                        "cancel_calib",
                        "force_wake",
                        "speak_text",
                        "inject_text",
                    ):
                        if not client_cmd_queue.full():
                            client_cmd_queue.put(data)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            connected_clients.discard(websocket)

    async def broadcast_loop():
        while True:
            await asyncio.sleep(0.015)
            while not ws_queue.empty():
                try:
                    msg = ws_queue.get_nowait()
                except queue.Empty:
                    break
                if msg and connected_clients:
                    msg_str = json.dumps(msg)
                    for ws in list(connected_clients):
                        try:
                            await ws.send(msg_str)
                        except Exception:
                            connected_clients.discard(ws)

    async def run_server():
        async with websockets.serve(ws_handler, "localhost", WS_PORT_DASHBOARD), \
                   websockets.serve(ws_handler, "localhost", WS_PORT_KEYBOARD), \
                   websockets.serve(ws_handler, "localhost", WS_PORT_KEYBOARD_LEGACY):
            await broadcast_loop()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_server())


def emit_ws_event(payload):
    if ws_queue.full():
        try:
            ws_queue.get_nowait()
        except queue.Empty:
            pass
    ws_queue.put(payload)


# ==============================================================================
# 7. MAIN UNIFIED MASTER ENGINE LOOP
# ==============================================================================
def main():
    ensure_landmarker_model()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    ml_model = joblib.load(ML_MODEL_PATH) if os.path.exists(ML_MODEL_PATH) else None
    if ml_model is not None:
        print(f"[ML Shadow Model] Loaded '{ML_MODEL_PATH}' for evaluator logging.")

    threading.Thread(target=start_websocket_server, daemon=True).start()

    screen_w, screen_h = pyautogui.size()
    screen_w, screen_h = int(screen_w), int(screen_h)
    calib_points, xs, ys = make_border_targets(screen_w, screen_h)

    kiosk = KioskBrowserManager(SCRIPT_DIR)

    fx_grid, fy_grid, has_saved_calib = load_saved_gaze_grid()
    bias_x, bias_y = 0.0, 0.0

    mode = "DASHBOARD"
    awake_state = False
    is_calibrated = has_saved_calib
    has_completed_calib_once = has_saved_calib

    current_calib_pt = 0
    calib_recording = False
    calib_burst = []
    calib_frames_seen = 0
    calib_msg = "Look at the red center dot & BLINK (or Left Wink / SPACE) to capture"
    space_armed = True
    space_lockout_until = 0.0
    c_was_down = False
    r_was_down = False
    back_was_down = False

    iris_x_buf = deque(maxlen=IRIS_SMOOTH_WINDOW)
    iris_y_buf = deque(maxlen=IRIS_SMOOTH_WINDOW)
    prev_smooth_fx, prev_smooth_fy = None, None
    smooth_fx, smooth_fy, iris_vel = 0.0, 0.0, 0.0

    clutch = SlowBlinkClutch()
    dpad_armed, rearmed_since_t, held_dir, held_dir_since = True, 0.0, "CENTER", 0.0
    last_action_t = 0.0
    last_telemetry_t = 0.0
    last_ml_log_t = 0.0

    eyes_shut_start_t = None
    eyes_open_grace_start_t = None
    beep_5s_played = False
    beep_11s_played = False
    beep_20s_played = False

    rppg_green_buf = deque(maxlen=RPPG_BUFFER_MAX)
    rppg_time_buf = deque(maxlen=RPPG_BUFFER_MAX)
    bpm_rolling_buf = deque()
    last_rppg_calc_t = 0.0
    rolling_bpm = None
    abnormal_bpm_since = None
    sos_active = False
    last_sos_beep_t = 0.0

    overlay_win = "AuraOS - 9-Grid Zoom Overlay"
    overlay_visible = False
    hwnd_overlay = None
    grid_row, grid_col = 1, 1
    zoom_level = 1
    zoomed_bg = None
    visual_bounds = [0, HUD_HEIGHT_PX, screen_w, screen_h]
    logical_region = [0, HUD_HEIGHT_PX, screen_w, screen_h]
    zoom_stack = []
    cursor_hold_until = 0.0
    inject_in_progress = False

    mp_ts = 0

    def ensure_overlay_window():
        nonlocal overlay_visible, hwnd_overlay
        if not overlay_visible:
            cv2.namedWindow(overlay_win, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(overlay_win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            try:
                cv2.setWindowProperty(overlay_win, cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
            cv2.imshow(overlay_win, np.zeros((screen_h, screen_w, 3), dtype=np.uint8))
            cv2.waitKey(20)
            hwnd_overlay = setup_transparent_overlay(overlay_win)
            overlay_visible = True

    def reset_grid_to_center():
        nonlocal grid_row, grid_col, zoom_level, logical_region, dpad_armed, held_dir, zoomed_bg
        grid_row, grid_col = 1, 1
        zoom_level = 1
        dpad_armed = True
        held_dir = "CENTER"
        zoomed_bg = None
        logical_region = [0, HUD_HEIGHT_PX, screen_w, screen_h]
        zoom_stack.clear()

    def begin_9_point_calibration(lockout_delay=0.45):
        nonlocal is_calibrated, current_calib_pt, calib_recording, calib_burst, calib_frames_seen, calib_msg
        nonlocal dpad_armed, held_dir, space_lockout_until
        is_calibrated = False
        current_calib_pt = 0
        calib_recording = False
        calib_burst = []
        calib_frames_seen = 0
        space_lockout_until = time.monotonic() + lockout_delay
        calib_msg = "Look at the red center dot & BLINK (or Left Wink / SPACE)"
        dpad_armed = True
        held_dir = "CENTER"
        reset_grid_to_center()
        iris_x_buf.clear()
        iris_y_buf.clear()
        clutch.slow_blink_timestamps.clear()
        clutch.single_blink_ready = False
        if mode == "EXTERNAL":
            ensure_overlay_window()
        print("[AuraOS] Started Hands-Free 9-Point Blink Calibration (1/9)")

    def switch_mode(new_mode, target_url=None, skip_nav=False):
        nonlocal mode, overlay_visible, hwnd_overlay, dpad_armed, held_dir
        mode = new_mode
        dpad_armed = True
        held_dir = "CENTER"
        reset_grid_to_center()

        if new_mode == "DASHBOARD":
            if overlay_visible:
                try:
                    cv2.destroyWindow(overlay_win)
                    cv2.waitKey(1)
                except Exception:
                    pass
                overlay_visible = False
                hwnd_overlay = None
            kiosk.navigate_single_window(kiosk.dashboard_url)
            print("[AuraOS] Switched to Dashboard")

        elif new_mode == "KEYBOARD":
            if overlay_visible:
                try:
                    cv2.destroyWindow(overlay_win)
                    cv2.waitKey(1)
                except Exception:
                    pass
                overlay_visible = False
                hwnd_overlay = None
            kiosk.navigate_single_window(kiosk.keyboard_url)
            print("[AuraOS] Switched to Virtual Keyboard")

        elif new_mode == "EXTERNAL":
            clean_url = (target_url or kiosk.last_external_url or "https://www.youtube.com").strip()
            if not clean_url.startswith(("http://", "https://", "file://")):
                clean_url = "https://" + clean_url
            kiosk.last_external_url = clean_url
            if not skip_nav:
                kiosk.navigate_single_window(clean_url)

            if not inject_in_progress:
                ensure_overlay_window()
            print(f"[AuraOS] Active External Website: {clean_url}")

    def inject_text_and_return_to_site(text):
        nonlocal cursor_hold_until, inject_in_progress, overlay_visible, hwnd_overlay
        clean = (text or "").strip()
        speak_out_loud(clean if clean else "Please type something first")
        if not clean:
            return

        copy_to_clipboard(clean)
        saved_coords = kiosk.last_clicked_coords
        saved_norm = kiosk.last_clicked_norm
        target_idx = kiosk.last_input_index
        target_url = (kiosk.last_external_url or "https://www.google.com").strip()

        inject_in_progress = True
        if overlay_visible:
            try:
                cv2.destroyWindow(overlay_win)
                cv2.waitKey(1)
            except Exception:
                pass
            overlay_visible = False
            hwnd_overlay = None

        cursor_hold_until = time.monotonic() + 6.0

        def _inject_worker():
            nonlocal cursor_hold_until, inject_in_progress
            print(f"[Keyboard Inject] Navigating to {target_url} and injecting: '{clean}'")
            ok_cdp = kiosk.navigate_and_inject_text_via_cdp(
                target_url, clean, saved_norm=saved_norm, target_idx=target_idx
            )
            if not ok_cdp:
                kiosk.navigate_single_window(target_url)
                time.sleep(2.4)
                if saved_coords is not None:
                    raw_win32_click(saved_coords[0], saved_coords[1], button="left")
                else:
                    raw_win32_click(screen_w // 2, int(screen_h * 0.43), button="left")
                time.sleep(0.25)
                pyautogui.hotkey("ctrl", "v")
                time.sleep(0.18)
                pyautogui.press("enter")

            time.sleep(0.35)
            inject_in_progress = False
            cursor_hold_until = time.monotonic() + 0.5

        switch_mode("EXTERNAL", target_url, skip_nav=True)
        threading.Thread(target=_inject_worker, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return 1

    kiosk.launch_kiosk()

    print("=" * 78)
    print(" AuraOS Unified Master Engine Running (HUD_HEIGHT_PX = 0 True Center)")
    print(" - 3 Slow Blinks      : Wake / Sleep Toggle")
    print(" - 9-Point Calibration: Waits for 1 Blink / Left Wink / SPACE on Red Center Dot")
    print(" - Eye Joystick       : Look UP/DOWN/LEFT/RIGHT, return to Center Dot to re-arm")
    print(" - ZoomGrid Top Bar   : Look UP from Top Row to highlight [VIRTUAL KEYBOARD]")
    print(" - Left Wink          : Open Card / Select Key / Progressive Zoom In")
    print(" - Right Wink         : Smart Cell Click (+ Auto Search-Box Keyboard) / Inject")
    print(" - 5-10s Eyes Shut    : Single Beep -> Contextual Back / Undo Zoom / Close Modal")
    print(" - 11-19s Eyes Shut   : Double Beep -> Hands-Free 9-Point Recalibration")
    print(" - 20s+ Eyes Shut     : Rising Chime -> Global Home (Return to Dashboard)")
    print(" - Physical ESC Key   : Caregiver-Only Safe Shutdown")
    print("=" * 78)

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            now = time.monotonic()
            mp_ts += 33

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect_for_video(mp_image, mp_ts)

            key = cv2.waitKey(1) & 0xFF
            space_down = (key == ord(" ")) or is_vk_down(VK_SPACE)
            c_down = (key == ord("c")) or is_vk_down(VK_C)
            r_down = (key == ord("r")) or is_vk_down(VK_R)
            back_down = (key == 8) or is_vk_down(VK_BACK)
            quit_down = (key == 27) or is_vk_down(VK_ESCAPE)

            if quit_down:
                print("\n[Caregiver Override] Physical ESC key pressed. Shutting down AuraOS safely...")
                break
            if not space_down:
                space_armed = True

            while not client_cmd_queue.empty():
                cmd_data = client_cmd_queue.get_nowait()
                cmd_type = cmd_data.get("type") or cmd_data.get("command")

                if cmd_type == "open_url":
                    url = cmd_data.get("url", "https://www.youtube.com")
                    kiosk.last_clicked_coords = None
                    kiosk.last_clicked_norm = None
                    kiosk.last_input_index = 0
                    switch_mode("EXTERNAL", url)
                elif cmd_type == "open_keyboard":
                    switch_mode("KEYBOARD")
                elif cmd_type == "open_dashboard":
                    switch_mode("DASHBOARD")
                elif cmd_type in ("start_calib", "calibrate"):
                    awake_state = True
                    begin_9_point_calibration(lockout_delay=0.45)
                elif cmd_type == "cancel_calib":
                    is_calibrated = True
                    calib_recording = False
                    emit_ws_event({"type": "calib_state", "active": False})
                elif cmd_type == "force_wake":
                    awake_state = True
                    play_sound_async("WAKE")
                    if has_completed_calib_once:
                        is_calibrated = True
                    else:
                        begin_9_point_calibration(lockout_delay=1.45)
                elif cmd_type == "speak_text":
                    speak_out_loud(cmd_data.get("text", ""))
                elif cmd_type == "inject_text":
                    inject_text_and_return_to_site(cmd_data.get("text", ""))

            if c_down and not c_was_down:
                awake_state = True
                begin_9_point_calibration(lockout_delay=0.45)
            c_was_down = c_down

            if back_down and not back_was_down and mode == "EXTERNAL" and awake_state and is_calibrated:
                reset_grid_to_center()
            back_was_down = back_down

            feats = None
            action_class = 0
            is_blinking = False
            single_blink_fired = False
            blink_progress = 0
            immediate_blink_released = False
            direction = "CENTER"
            scores = {"UP": 0.0, "DOWN": 0.0, "LEFT": 0.0, "RIGHT": 0.0}
            in_center_reset_zone = False
            eyes_closed_sec = (now - eyes_shut_start_t) if eyes_shut_start_t is not None else 0.0

            if results and results.face_landmarks:
                lms = results.face_landmarks[0]
                feats = extract_all_features(lms, fw, fh)

                # --- 1. LIVE CAMERA FOREHEAD rPPG HEART RATE & SOS ALERT ---
                green_val, _ = extract_forehead_green(frame, lms, fw, fh)
                if green_val is not None:
                    rppg_green_buf.append(green_val)
                    rppg_time_buf.append(now)

                if (
                    len(rppg_green_buf) >= RPPG_MIN_SAMPLES
                    and (now - last_rppg_calc_t) >= RPPG_UPDATE_INTERVAL_S
                ):
                    last_rppg_calc_t = now
                    est_bpm = compute_filtered_rppg_bpm(rppg_green_buf, rppg_time_buf)
                    if est_bpm is not None:
                        bpm_rolling_buf.append((now, est_bpm))

                while (
                    bpm_rolling_buf
                    and (now - bpm_rolling_buf[0][0]) > BPM_ROLLING_WINDOW_S
                ):
                    bpm_rolling_buf.popleft()

                if bpm_rolling_buf:
                    rolling_bpm = float(np.mean([b[1] for b in bpm_rolling_buf]))
                    if rolling_bpm > BPM_SOS_HIGH or rolling_bpm < BPM_SOS_LOW:
                        if abnormal_bpm_since is None:
                            abnormal_bpm_since = now
                        elif (now - abnormal_bpm_since) >= SOS_SUSTAINED_DURATION_S:
                            sos_active = True
                            if (now - last_sos_beep_t) >= SOS_BEEP_INTERVAL_S:
                                last_sos_beep_t = now
                                play_sound_async("SOS")
                    else:
                        abnormal_bpm_since, sos_active = None, False

                # --- 2. IRIS SMOOTHING & SACCADE VELOCITY ---
                iris_x_buf.append(feats["iris_x"])
                iris_y_buf.append(feats["iris_y"])
                smooth_fx, smooth_fy = (
                    float(np.median(iris_x_buf)),
                    float(np.median(iris_y_buf)),
                )
                iris_vel = (
                    math.hypot(smooth_fx - prev_smooth_fx, smooth_fy - prev_smooth_fy)
                    if prev_smooth_fx is not None
                    else 0.0
                )
                prev_smooth_fx, prev_smooth_fy = smooth_fx, smooth_fy

                # --- 3. OCULAR ACTION & SHADOW ML EVALUATOR LOG ---
                action_class = predict_ocular_action(ml_model, feats)
                if action_class in (1, 2) and (now - last_ml_log_t) > 1.5:
                    lbl = "Left Wink" if action_class == 1 else "Right Wink"
                    print(f"[Ocular Classifier Active] Detected: {lbl}")
                    last_ml_log_t = now

                is_blinking = (
                    feats["ear_left"] < 0.175
                    and feats["ear_right"] < 0.175
                    and abs(feats["ear_diff"]) < EYES_SHUT_SYMMETRY_MAX
                ) or (action_class == 3 and feats["ear_avg"] < 0.180)

                # --- 4. SYMMETRIC SUSTAINED EYES-SHUT TIMELINE (WITH 0.28s GRACE BUFFER) ---
                both_eyes_shut_now = is_blinking or (
                    feats["ear_left"] < EYES_SHUT_EAR_THRESH
                    and feats["ear_right"] < EYES_SHUT_EAR_THRESH
                    and abs(feats["ear_diff"]) < EYES_SHUT_SYMMETRY_MAX
                )

                if both_eyes_shut_now:
                    eyes_open_grace_start_t = None
                    if eyes_shut_start_t is None:
                        eyes_shut_start_t = now
                        beep_5s_played = False
                        beep_11s_played = False
                        beep_20s_played = False

                    eyes_closed_sec = now - eyes_shut_start_t

                    if eyes_closed_sec >= BACK_MIN_SEC and not beep_5s_played:
                        beep_5s_played = True
                        play_sound_async("BEEP_5S_BACK")
                        print("[Eyes-Shut Checkpoint] 5s reached: Contextual Back armed.")
                    if eyes_closed_sec >= CALIB_MIN_SEC and not beep_11s_played:
                        beep_11s_played = True
                        play_sound_async("BEEP_11S_CALIB")
                        print("[Eyes-Shut Checkpoint] 11s reached: 9-Point Calibration armed.")
                    if eyes_closed_sec >= HOME_MIN_SEC and not beep_20s_played:
                        beep_20s_played = True
                        play_sound_async("CHIME_20S_HOME")
                        print("[Eyes-Shut Checkpoint] 20s reached: Global Home armed.")
                else:
                    if eyes_shut_start_t is not None:
                        if eyes_open_grace_start_t is None:
                            eyes_open_grace_start_t = now
                        elif (now - eyes_open_grace_start_t) >= EYES_SHUT_GRACE_SEC:
                            closed_dur = eyes_open_grace_start_t - eyes_shut_start_t
                            eyes_shut_start_t = None
                            eyes_open_grace_start_t = None
                            eyes_closed_sec = 0.0

                            if closed_dur >= HOME_MIN_SEC:
                                print(f"[Eyes-Shut Action] Held {closed_dur:.1f}s -> Returning Home to Dashboard!")
                                awake_state = True
                                switch_mode("DASHBOARD")
                                last_action_t = now
                            elif CALIB_MIN_SEC <= closed_dur <= CALIB_MAX_SEC:
                                print(f"[Eyes-Shut Action] Held {closed_dur:.1f}s -> Starting 9-Point Calibration!")
                                awake_state = True
                                begin_9_point_calibration(lockout_delay=0.45)
                                last_action_t = now
                            elif BACK_MIN_SEC <= closed_dur <= BACK_MAX_SEC:
                                print(f"[Eyes-Shut Action] Held {closed_dur:.1f}s -> Contextual Back!")
                                play_sound_async("CLICK")
                                if mode == "EXTERNAL":
                                    if zoom_stack:
                                        logical_region, zoomed_bg, zoom_level = zoom_stack.pop()
                                        grid_row, grid_col = 1, 1
                                        dpad_armed, held_dir = True, "CENTER"
                                    else:
                                        pyautogui.hotkey("alt", "left")
                                else:
                                    emit_ws_event({"type": "gesture", "action": "BACK"})
                                last_action_t = now

                # --- 5. 3-SLOW-BLINK WAKE / SLEEP CLUTCH ---
                if not calib_recording:
                    (
                        clutch_toggled,
                        single_blink_fired,
                        blink_progress,
                        immediate_blink_released,
                    ) = clutch.update(now, is_blinking)

                    if clutch_toggled and (not awake_state or is_calibrated):
                        if awake_state:
                            awake_state = False
                            dpad_armed, held_dir = True, "CENTER"
                            reset_grid_to_center()
                            play_sound_async("SLEEP")
                            print("[AuraOS] Entered Surveillance Sleep State")
                        else:
                            awake_state = True
                            play_sound_async("WAKE")
                            if has_completed_calib_once:
                                is_calibrated = True
                                dpad_armed, held_dir = True, "CENTER"
                                print("[AuraOS] Woke up (Using saved session calibration)")
                            else:
                                begin_9_point_calibration(lockout_delay=1.45)

            # ==================================================================
            # STATE 1: SURVEILLANCE SLEEP STATE (WAITING FOR 3 SLOW BLINKS)
            # ==================================================================
            if not awake_state:
                state_str = "SURVEILLANCE: Perform 3 Slow Blinks to Wake"

            # ==================================================================
            # STATE 2: HANDS-FREE 9-POINT BLINK CALIBRATION (WAIT-FOR-BLINK)
            # ==================================================================
            elif not is_calibrated:
                row, col = divmod(current_calib_pt, 3)
                state_str = f"WAKING: Calibrating Point ({current_calib_pt + 1}/9)"

                if (
                    (
                        immediate_blink_released
                        or action_class == 1
                        or (space_down and space_armed)
                    )
                    and now >= space_lockout_until
                    and not calib_recording
                    and feats is not None
                ):
                    space_armed = False
                    calib_recording = True
                    calib_burst = []
                    calib_frames_seen = 0
                    calib_msg = ""
                    space_lockout_until = now + 0.25

                if calib_recording:
                    calib_frames_seen += 1
                    if feats is not None and calib_frames_seen > CALIB_SETTLE_FRAMES:
                        if min(feats["ear_left"], feats["ear_right"]) > (
                            0.125 if row == 2 else 0.155
                        ):
                            calib_burst.append(feats)

                    if (
                        len(calib_burst) >= CALIB_SAMPLES_NEEDED
                        or calib_frames_seen >= CALIB_MAX_FRAMES
                    ):
                        calib_recording = False
                        if len(calib_burst) < 8:
                            calib_msg = "Eyes closed! Look at red center dot & BLINK to retry."
                        else:
                            fx_grid[row, col] = float(
                                np.median([b["iris_x"] for b in calib_burst])
                            )
                            fy_grid[row, col] = float(
                                np.median([b["iris_y"] for b in calib_burst])
                            )
                            current_calib_pt += 1
                            space_lockout_until = now + 0.35
                            play_sound_async("CALIB_PT")

                            if current_calib_pt >= 9:
                                is_calibrated = True
                                has_completed_calib_once = True
                                bias_x, bias_y = 0.0, 0.0
                                dpad_armed, held_dir = True, "CENTER"
                                reset_grid_to_center()
                                last_action_t = now + 0.80
                                clutch.slow_blink_timestamps.clear()
                                clutch.single_blink_ready = False
                                np.savez(
                                    CALIB_LOG_PATH,
                                    fx=fx_grid,
                                    fy=fy_grid,
                                    xs=np.asarray(xs),
                                    ys=np.asarray(ys),
                                )
                                print(f"[AuraOS] 9-Point Calibration saved to {CALIB_LOG_PATH}")
                                emit_ws_event(
                                    {
                                        "type": "calib_state",
                                        "active": False,
                                        "completed": True,
                                    }
                                )

                if not is_calibrated and (now - last_telemetry_t) >= 0.04:
                    last_telemetry_t = now
                    emit_ws_event(
                        {
                            "type": "calib_state",
                            "active": True,
                            "pt": current_calib_pt,
                            "recording": calib_recording,
                            "blink_primed": clutch.blink_active,
                            "samples": len(calib_burst),
                            "needed": CALIB_SAMPLES_NEEDED,
                            "msg": calib_msg,
                        }
                    )

            # ==================================================================
            # STATE 3: ARMED NAVIGATION (SPRING-RETURN JOYSTICK + WINKS)
            # ==================================================================
            else:
                state_str = "ACTIVE: Eye-Tracking Portal Armed (3 Blinks to Sleep)"

                if r_down and not r_was_down and feats is not None:
                    bias_x = smooth_fx - float(fx_grid[1, 1])
                    bias_y = smooth_fy - float(fy_grid[1, 1])
                    dpad_armed, held_dir = True, "CENTER"
                    emit_ws_event({"type": "recentered"})
                    print("[AuraOS] Recentered Eye Bias to Center Dot")
                r_was_down = r_down

                eyes_open_for_gaze = (
                    not is_blinking
                    and feats is not None
                    and feats["ear_avg"] > 0.135
                    and abs(feats["ear_diff"]) < 0.075
                )
                in_center_reset_zone = False
                if eyes_open_for_gaze:
                    direction, scores, in_center_reset_zone = evaluate_4way_dpad(
                        fx_grid, fy_grid, smooth_fx, smooth_fy, bias_x, bias_y
                    )

                if not dpad_armed:
                    held_dir = "CENTER"
                    if in_center_reset_zone and iris_vel < MAX_SACCADE_VEL:
                        dpad_armed, rearmed_since_t = True, now
                else:
                    if (
                        direction != "CENTER"
                        and iris_vel < MAX_SACCADE_VEL
                        and (now - rearmed_since_t) >= DPAD_REARM_SETTLE_S
                    ):
                        if direction != held_dir:
                            held_dir, held_dir_since = direction, now
                        elif (now - held_dir_since) >= DPAD_HOLD_CONFIRM_S:
                            if mode == "EXTERNAL":
                                if direction == "UP":
                                    min_r = -1 if zoom_level == 1 else 0
                                    grid_row = max(min_r, grid_row - 1)
                                elif direction == "DOWN":
                                    grid_row = min(2, grid_row + 1)
                                elif direction == "LEFT" and grid_row >= 0:
                                    grid_col = max(0, grid_col - 1)
                                elif direction == "RIGHT" and grid_row >= 0:
                                    grid_col = min(2, grid_col + 1)
                            else:
                                emit_ws_event(
                                    {"type": "move", "dir": direction, "repeat": False}
                                )
                            dpad_armed, held_dir = False, "CENTER"
                    else:
                        held_dir = "CENTER"

                # --- WINK ACTIONS PER ACTIVE SCREEN ---
                if (
                    eyes_closed_sec < 0.40
                    and (now - last_action_t) >= ACTION_COOLDOWN_S
                    and feats is not None
                    and not inject_in_progress
                ):
                    if mode == "DASHBOARD":
                        if action_class == 1:
                            print("[Dashboard] Left Wink -> SELECT / OPEN CARD")
                            emit_ws_event({"type": "action", "action": "LEFT_WINK_SELECT"})
                            last_action_t = now

                    elif mode == "KEYBOARD":
                        if action_class == 1:
                            print("[Keyboard] Left Wink -> SELECT KEY")
                            emit_ws_event({"type": "action", "action": "LEFT_WINK_SELECT"})
                            last_action_t = now
                        elif action_class == 2:
                            print("[Keyboard] Right Wink -> SPEAK & INJECT")
                            emit_ws_event({"type": "action", "action": "RIGHT_WINK_SPEAK"})
                            last_action_t = now

                    elif mode == "EXTERNAL":
                        if grid_row == -1:
                            if action_class in (1, 2):
                                print("[9-Grid Top Bar] Opening Virtual Keyboard!")
                                play_sound_async("CLICK")
                                switch_mode("KEYBOARD")
                                last_action_t = now
                        else:
                            lx0, ly0, lx1, ly1 = logical_region
                            lcw, lch = (lx1 - lx0) / 3.0, (ly1 - ly0) / 3.0
                            true_cell_x0 = int(lx0 + grid_col * lcw)
                            true_cell_y0 = int(ly0 + grid_row * lch)
                            true_cell_x1 = int(lx0 + (grid_col + 1) * lcw)
                            true_cell_y1 = int(ly0 + (grid_row + 1) * lch)
                            target_cx = (true_cell_x0 + true_cell_x1) // 2
                            target_cy = (true_cell_y0 + true_cell_y1) // 2

                            if action_class == 1:
                                cell_w = true_cell_x1 - true_cell_x0
                                cell_h = true_cell_y1 - true_cell_y0
                                if zoom_level < MAX_ZOOM_DEPTH and cell_w >= 18 and cell_h >= 18:
                                    zoom_stack.append(
                                        (
                                            list(logical_region),
                                            None if zoomed_bg is None else zoomed_bg.copy(),
                                            zoom_level,
                                        )
                                    )
                                    logical_region = [
                                        true_cell_x0,
                                        true_cell_y0,
                                        true_cell_x1,
                                        true_cell_y1,
                                    ]
                                    try:
                                        cv2.imshow(
                                            overlay_win,
                                            np.zeros((screen_h, screen_w, 3), dtype=np.uint8),
                                        )
                                        cv2.waitKey(1)
                                        time.sleep(0.04)

                                        bg_img = pyautogui.screenshot(
                                            region=(
                                                true_cell_x0,
                                                true_cell_y0,
                                                cell_w,
                                                cell_h,
                                            )
                                        )
                                        bg_cv = cv2.cvtColor(
                                            np.array(bg_img), cv2.COLOR_RGB2BGR
                                        )
                                        bg_cv[np.all(bg_cv == (0, 0, 0), axis=-1)] = (1, 1, 1)
                                        zoomed_bg = cv2.resize(
                                            bg_cv, (screen_w, screen_h - HUD_HEIGHT_PX)
                                        )
                                    except Exception as e:
                                        print(f"[Zoom Screenshot Warning] {e}")
                                        zoomed_bg = None

                                    zoom_level += 1
                                    grid_row, grid_col = 1, 1
                                    dpad_armed = True
                                    held_dir = "CENTER"
                                    play_sound_async("ZOOM")
                                    print(
                                        f"[Left Wink -> Zoom Level {zoom_level}] Region: ({true_cell_x0}, {true_cell_y0}) to ({true_cell_x1}, {true_cell_y1})"
                                    )
                                last_action_t = now

                            elif action_class == 2:
                                norm_x0 = float(true_cell_x0) / max(1, screen_w)
                                norm_y0 = float(true_cell_y0) / max(1, screen_h)
                                norm_x1 = float(true_cell_x1) / max(1, screen_w)
                                norm_y1 = float(true_cell_y1) / max(1, screen_h)

                                print(
                                    f"[Right Wink -> Cell Click] Checking cell ({true_cell_x0}, {true_cell_y0})-({true_cell_x1}, {true_cell_y1})"
                                )
                                cv2.imshow(
                                    overlay_win, np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
                                )
                                cv2.waitKey(1)
                                time.sleep(0.025)

                                raw_win32_click(target_cx, target_cy, button="left")
                                kiosk.last_clicked_coords = (target_cx, target_cy)
                                kiosk.last_clicked_norm = (
                                    float(target_cx) / max(1, screen_w),
                                    float(target_cy) / max(1, screen_h),
                                )
                                cursor_hold_until = now + 0.65
                                play_sound_async("CLICK")

                                hwnd_overlay = force_window_topmost(overlay_win, hwnd_overlay)
                                reset_grid_to_center()
                                last_action_t = now

                                def _check_cell_or_ibeam(nx0, ny0, nx1, ny1):
                                    time.sleep(0.22)
                                    if mode != "EXTERNAL":
                                        return
                                    snapped = kiosk.find_input_inside_cell_or_focused(nx0, ny0, nx1, ny1)
                                    if snapped is not None:
                                        snx, sny, sidx, shref = snapped
                                        kiosk.last_clicked_norm = (snx, sny)
                                        kiosk.last_clicked_coords = (
                                            int(snx * screen_w),
                                            int(sny * screen_h),
                                        )
                                        kiosk.last_input_index = sidx
                                        if shref and not shref.startswith("file:"):
                                            kiosk.last_external_url = shref
                                        print(f"[Smart Typing] Snapped to Input #{sidx} at {kiosk.last_clicked_coords}! Opening Virtual Keyboard...")
                                        switch_mode("KEYBOARD")
                                    elif is_ibeam_cursor_active():
                                        print("[Smart Typing] Windows I-Beam Cursor Active! Opening Virtual Keyboard...")
                                        switch_mode("KEYBOARD")

                                threading.Thread(
                                    target=_check_cell_or_ibeam,
                                    args=(norm_x0, norm_y0, norm_x1, norm_y1),
                                    daemon=True,
                                ).start()

            # ==================================================================
            # RENDER 9-GRID OVERLAY WINDOW WHEN IN EXTERNAL WEBSITE MODE
            # ==================================================================
            if mode == "EXTERNAL" and not inject_in_progress and not overlay_visible:
                ensure_overlay_window()

            if mode == "EXTERNAL" and overlay_visible and not inject_in_progress:
                canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

                if not awake_state:
                    vx0, vy0, vx1, vy1 = visual_bounds
                    vcw, vch = (vx1 - vx0) / 3.0, (vy1 - vy0) / 3.0
                    for r in range(3):
                        for c in range(3):
                            x0, y0, x1, y1 = (
                                int(vx0 + c * vcw),
                                int(vy0 + r * vch),
                                int(vx0 + (c + 1) * vcw),
                                int(vy0 + (r + 1) * vch),
                            )
                            draw_dashed_rect(
                                canvas, (x0, y0), (x1, y1), (180, 180, 180), 1, 14
                            )

                    cx_scr = screen_w // 2
                    cv2.rectangle(
                        canvas, (cx_scr - 340, 18), (cx_scr + 340, 86), (12, 12, 12), -1
                    )
                    cv2.rectangle(
                        canvas, (cx_scr - 340, 18), (cx_scr + 340, 86), (0, 165, 255), 2
                    )
                    cv2.putText(
                        canvas,
                        f"9-GRID SLEEP STATE  |  Perform {3 - blink_progress} slow blink(s) to Wake",
                        (cx_scr - 315, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (0, 215, 255),
                        2,
                    )

                elif not is_calibrated:
                    canvas[:] = (16, 10, 7)
                    cx_scr, text_cy = screen_w // 2, screen_h // 2 + 150
                    cv2.rectangle(
                        canvas,
                        (cx_scr - 420, text_cy - 75),
                        (cx_scr + 420, text_cy + 75),
                        (12, 12, 12),
                        -1,
                    )
                    cv2.rectangle(
                        canvas,
                        (cx_scr - 420, text_cy - 75),
                        (cx_scr + 420, text_cy + 75),
                        (78, 59, 43),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        f"HANDS-FREE 9-POINT CALIBRATION ({current_calib_pt + 1}/9)",
                        (cx_scr - 310, text_cy - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 255, 180),
                        2,
                    )
                    cv2.putText(
                        canvas,
                        "Look at red center dot & BLINK ONCE (or Left Wink / SPACE)",
                        (cx_scr - 315, text_cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.68,
                        (200, 200, 200),
                        2,
                    )
                    if calib_msg:
                        cv2.putText(
                            canvas,
                            calib_msg,
                            (cx_scr - 240, text_cy + 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 80, 255),
                            2,
                        )

                    pix = (
                        int(calib_points[current_calib_pt][0]),
                        int(calib_points[current_calib_pt][1]),
                    )
                    for i in range(current_calib_pt):
                        pt_done = (int(calib_points[i][0]), int(calib_points[i][1]))
                        cv2.circle(canvas, pt_done, 9, (70, 160, 70), -1)
                        cv2.circle(canvas, pt_done, 3, (0, 0, 255), -1)

                    outer_col = (0, 255, 120) if calib_recording else (0, 140, 255)
                    cv2.circle(canvas, pix, 24, outer_col, -1)
                    cv2.circle(canvas, pix, 12, (25, 25, 25), 2)
                    cv2.circle(canvas, pix, 3, (0, 0, 255), -1)

                    if calib_recording:
                        cv2.putText(
                            canvas,
                            f"Capturing {len(calib_burst)}/{CALIB_SAMPLES_NEEDED}...",
                            (cx_scr - 125, text_cy + 105),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 255, 120),
                            2,
                        )

                else:
                    lx0, ly0, lx1, ly1 = logical_region
                    lcw, lch = (lx1 - lx0) / 3.0, (ly1 - ly0) / 3.0

                    if now >= cursor_hold_until:
                        if grid_row == -1:
                            target_cx = screen_w // 2
                            target_cy = 34
                        else:
                            target_cx = int(lx0 + (grid_col + 0.5) * lcw)
                            target_cy = int(ly0 + (grid_row + 0.5) * lch)
                        try:
                            pyautogui.moveTo(target_cx, target_cy, _pause=False)
                        except Exception:
                            pass

                    vx0, vy0, vx1, vy1 = visual_bounds
                    vcw, vch = (vx1 - vx0) / 3.0, (vy1 - vy0) / 3.0

                    if zoom_level > 1 and zoomed_bg is not None:
                        canvas[HUD_HEIGHT_PX:screen_h, 0:screen_w] = zoomed_bg

                    for r in range(3):
                        for c in range(3):
                            x0, y0, x1, y1 = (
                                int(vx0 + c * vcw),
                                int(vy0 + r * vch),
                                int(vx0 + (c + 1) * vcw),
                                int(vy0 + (r + 1) * vch),
                            )
                            is_active = (r == grid_row and c == grid_col)
                            if is_active:
                                cv2.rectangle(
                                    canvas,
                                    (x0 + 2, y0 + 2),
                                    (x1 - 2, y1 - 2),
                                    (0, 255, 255),
                                    4,
                                )
                            else:
                                draw_dashed_rect(
                                    canvas, (x0, y0), (x1, y1), (220, 220, 220), 1, 12
                                )

                    if zoom_level == 1:
                        cx_scr = screen_w // 2
                        kb_x0, kb_y0 = cx_scr - 230, 10
                        kb_x1, kb_y1 = cx_scr + 230, 58
                        kb_sel = (grid_row == -1)
                        cv2.rectangle(
                            canvas,
                            (kb_x0, kb_y0),
                            (kb_x1, kb_y1),
                            (24, 32, 48) if kb_sel else (15, 18, 24),
                            -1,
                        )
                        cv2.rectangle(
                            canvas,
                            (kb_x0, kb_y0),
                            (kb_x1, kb_y1),
                            (0, 255, 255) if kb_sel else (140, 160, 180),
                            4 if kb_sel else 2,
                        )
                        kb_label = (
                            "VIRTUAL KEYBOARD SELECTED (WINK TO OPEN)"
                            if kb_sel
                            else "^ LOOK UP FROM TOP ROW: OPEN VIRTUAL KEYBOARD ^"
                        )
                        cv2.putText(
                            canvas,
                            kb_label,
                            (cx_scr - (195 if kb_sel else 215), 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.54,
                            (0, 255, 255) if kb_sel else (225, 235, 245),
                            2,
                        )

                    mid_x = screen_w // 2
                    mid_y = HUD_HEIGHT_PX + (screen_h - HUD_HEIGHT_PX) // 2
                    ring_col = (0, 255, 140) if dpad_armed else (0, 180, 255)
                    cv2.circle(canvas, (mid_x, mid_y), 14, ring_col, 2)
                    cv2.circle(canvas, (mid_x, mid_y), 4, (0, 255, 255), -1)

                cv2.imshow(overlay_win, canvas)
                cv2.waitKey(1)
                hwnd_overlay = force_window_topmost(overlay_win, hwnd_overlay)

            # ==================================================================
            # BROADCAST LIVE TELEMETRY & STATE TO DASHBOARD & KEYBOARD
            # ==================================================================
            if (now - last_telemetry_t) >= 0.05:
                last_telemetry_t = now
                bpm_val = round(rolling_bpm, 1) if rolling_bpm else 0.0

                emit_ws_event(
                    {
                        "type": "telemetry",
                        "awake": awake_state,
                        "calibrated": is_calibrated,
                        "blink_progress": blink_progress,
                        "armed": dpad_armed,
                        "dir": direction,
                        "in_center": in_center_reset_zone,
                        "bpm": bpm_val,
                        "rpm": bpm_val,
                        "sos": sos_active,
                        "status": state_str,
                        "eyes_closed_sec": round(eyes_closed_sec, 1),
                    }
                )

                emit_ws_event(
                    {
                        "type": "state",
                        "awake": awake_state,
                        "calibrated": is_calibrated,
                        "armed": dpad_armed,
                        "blink_count": blink_progress,
                        "dir": direction,
                        "direction": "CENTER" if dpad_armed else direction,
                        "in_center": in_center_reset_zone,
                        "scores": {k: round(v, 2) for k, v in scores.items()},
                        "bpm": bpm_val,
                        "sos": sos_active,
                        "eyes_closed_sec": round(eyes_closed_sec, 1),
                    }
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        kiosk.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
