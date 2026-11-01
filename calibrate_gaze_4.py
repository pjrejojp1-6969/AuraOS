# =============================================================================
# AuraOS — 4_calibrate_gaze.py  (v4 — noisy-burst rejection + layer diagnostics)
# =============================================================================
import cv2, mediapipe as mp, joblib, os, time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np, pyautogui
from collections import deque

pyautogui.FAILSAFE = False
screen_w, screen_h = pyautogui.size()

# ------------------------------ tunables -------------------------------------
CAL_MARGIN_X    = int(screen_w * 0.03)
CAL_MARGIN_Y    = int(screen_h * 0.03)
BURST_FRAMES    = 30
MIN_EAR_SAMPLE  = 0.12
BURST_MAD_MAX   = 0.010      # reject burst if MAD > this (was silently accepted before)
BURST_RETRY_MAX = 3

DEADZONE_PIX    = 8
BIG_JUMP_PIX    = 260
EMA_ALPHA_NEAR  = 0.88
EMA_ALPHA_FAR   = 0.58
FIX_VAR_THRES   = 8e-5
SNAP_ALPHA      = 0.55
FIX_SNAP_PX     = 2

HISTORY_LEN     = 45
PRE_WINK_BACK   = 8
CLICK_COOLDOWN  = 1.4
CLICK_REARM     = 0.15
FREEZE_NORM_EAR = 0.70

HEAD_RATE_GATE  = np.deg2rad(2.5)
HEAD_DEADBAND   = np.deg2rad(5.0)
HEAD_BLEND_X    = 0.08 * screen_w
HEAD_BLEND_Y    = 0.10 * screen_h

RESIDUAL_CLAMP_FRAC = 0.25
OUTLIER_FACTOR      = 3.0

# ------------------------------ MediaPipe ------------------------------------
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=vision.RunningMode.VIDEO, num_faces=1,
    output_facial_transformation_matrixes=True, output_face_blendshapes=False,
)
detector = vision.FaceLandmarker.create_from_options(options)
click_model = joblib.load("ocular_ml_model.pkl") if os.path.exists("ocular_ml_model.pkl") else None

# ------------------------------ helpers --------------------------------------
def get_gaze_metrics(lm, w, h):
    def d(a, b):
        return np.hypot((lm[a].x - lm[b].x) * w, (lm[a].y - lm[b].y) * h)
    ear_l = (d(159, 145) + d(158, 153)) / (2.0 * d(33, 133) + 1e-6)
    ear_r = (d(386, 374) + d(385, 380)) / (2.0 * d(362, 263) + 1e-6)
    l_w = (lm[133].x - lm[33].x) + 1e-6
    r_w = (lm[263].x - lm[362].x) + 1e-6
    iris_x = ((lm[468].x - lm[33].x) / l_w + (lm[473].x - lm[362].x) / r_w) / 2.0
    iris_y = ((lm[468].y - lm[9].y) / l_w + (lm[473].y - lm[9].y) / r_w) / 2.0
    return ear_l, ear_r, iris_x, iris_y


def head_angles(mat):
    if mat is None: return 0.0, 0.0
    R = np.asarray(mat).reshape(4, 4)[:3, :3]
    return (float(np.arctan2(R[0, 2], R[2, 2])),
            float(np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))))


def extrap_interp(v, xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    if len(xs) == 1: return float(ys[0])
    if v <= xs[0]:
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return float(ys[0] + s * (v - xs[0]))
    if v >= xs[-1]:
        s = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return float(ys[-1] + s * (v - xs[-1]))
    return float(np.interp(v, xs, ys))


def bounded_assist(rel, deadband, gain):
    s = np.sign(rel); excess = abs(rel) - deadband
    if excess <= 0: return 0.0
    return float(s * gain * np.tanh(excess / deadband))

# ------------------------------ state ----------------------------------------
margin_x, margin_y = CAL_MARGIN_X, CAL_MARGIN_Y
cal_points = [
    (margin_x, margin_y), (screen_w // 2, margin_y), (screen_w - margin_x, margin_y),
    (margin_x, screen_h // 2), (screen_w // 2, screen_h // 2), (screen_w - margin_x, screen_h // 2),
    (margin_x, screen_h - margin_y), (screen_w // 2, screen_h - margin_y), (screen_w - margin_x, screen_h - margin_y),
]
cal_data, burst = [], []
cal_idx, is_burst, is_calibrated = 0, False, False
burst_retries = 0
last_burst_error = None

map_x_knots = map_y_knots = residual_map_raw = residual_map = None
residual_enabled = True
head_assist_enabled = True
center_signal_x = center_signal_y = 0.0
yaw_neutral = pitch_neutral = 0.0
apply_yaw_assist = apply_pitch_assist = False
sign_yaw = sign_pitch = 1.0
ear_open_l = ear_open_r = 0.28
ear_open_samples = []
cal_health = None
fit_report = {}

buffer = deque(maxlen=HISTORY_LEN)
raw_xy_hist = deque(maxlen=6)
pred_hist = deque(maxlen=30)
smooth_x, smooth_y = screen_w // 2, screen_h // 2
last_yaw, last_pitch = 0.0, 0.0
bias_x = bias_y = 0.0
wink_state = 'open'
last_click_time = last_open_time = time.monotonic()
reanchor_flash_until = last_var_warn = 0.0

# ------------------------------ fitting --------------------------------------
def fit_mapping():
    pts = np.array(cal_data, dtype=float)
    xs_t = sorted(set(pts[:, 4].tolist()))
    ys_t = sorted(set(pts[:, 5].tolist()))

    def knots(target_vals, in_col, tgt_col):
        ins, outs = [], []
        for tv in target_vals:
            sel = pts[pts[:, tgt_col] == tv]
            ins.append(float(np.median(sel[:, in_col]))); outs.append(float(tv))
        order = np.argsort(ins)
        return [ins[i] for i in order], [outs[i] for i in order]

    kx_in, kx_out = knots(xs_t, 0, 4)
    ky_in, ky_out = knots(ys_t, 1, 5)

    def mono(ins, outs):
        ins = np.asarray(ins, float); outs = np.asarray(outs, float)
        if len(ins) < 2 or np.any(np.diff(ins) <= 1e-9):
            return None, None, np.polyfit(ins, outs, 1), False
        return ins, outs, None, True

    fx_x, fy_x, poly_x, mono_x = mono(kx_in, kx_out)
    fx_y, fy_y, poly_y, mono_y = mono(ky_in, ky_out)

    mid = (screen_w // 2, screen_h // 2)
    sel = [d for d in cal_data if (d[4], d[5]) == mid]
    c_ix = float(np.median([d[0] for d in sel])) if sel else 0.0
    c_iy = float(np.median([d[1] for d in sel])) if sel else 0.0
    c_yaw = float(np.median([d[2] for d in sel])) if sel else 0.0
    c_pit = float(np.median([d[3] for d in sel])) if sel else 0.0

    yaw_by = {tx: float(np.median([d[2] for d in cal_data if d[4] == tx])) for tx in xs_t}
    pit_by = {ty: float(np.median([d[3] for d in cal_data if d[5] == ty])) for ty in ys_t}
    en_yaw = abs(yaw_by[xs_t[-1]] - yaw_by[xs_t[0]]) > np.deg2rad(1.5)
    en_pit = abs(pit_by[ys_t[-1]] - pit_by[ys_t[0]]) > np.deg2rad(1.5)
    s_yaw = 1.0 if yaw_by[xs_t[-1]] >= yaw_by[xs_t[0]] else -1.0
    s_pit = 1.0 if pit_by[ys_t[-1]] >= pit_by[ys_t[0]] else -1.0

    arr = np.array(ear_open_samples) if ear_open_samples else np.array([[0.28, 0.28]])
    e_l = float(np.median(arr[:, 0])); e_r = float(np.median(arr[:, 1]))

    return dict(
        x=(fx_x, fy_x, poly_x), y=(fx_y, fy_y, poly_y),
        c_signal=(c_ix, c_iy), head_neutral=(c_yaw, c_pit),
        en_yaw=en_yaw, en_pit=en_pit, s_yaw=s_yaw, s_pit=s_pit,
        ear_open=(e_l, e_r), mono_x=mono_x, mono_y=mono_y,
        kx_in=kx_in, ky_in=ky_in,
    )


def predict_axis(v, mapping):
    ins, outs, poly = mapping
    return float(np.polyval(poly, v)) if poly is not None else extrap_interp(v, ins, outs)


def build_residual_map():
    """Raw residuals + robust outlier suppression (median-based, not Gaussian)."""
    raw = {}
    for d in cal_data:
        ix, iy, yaw_c, pitch_c, tx, ty = d
        px = predict_axis(ix, map_x_knots); py = predict_axis(iy, map_y_knots)
        if apply_yaw_assist and head_assist_enabled:
            px += sign_yaw * bounded_assist(yaw_c - yaw_neutral, HEAD_DEADBAND, HEAD_BLEND_X)
        if apply_pitch_assist and head_assist_enabled:
            py += sign_pitch * bounded_assist(pitch_c - pitch_neutral, HEAD_DEADBAND, HEAD_BLEND_Y)
        rx, ry = tx - px, ty - py
        rx = float(np.clip(rx, -RESIDUAL_CLAMP_FRAC * screen_w, RESIDUAL_CLAMP_FRAC * screen_w))
        ry = float(np.clip(ry, -RESIDUAL_CLAMP_FRAC * screen_h, RESIDUAL_CLAMP_FRAC * screen_h))
        col = 0 if tx < screen_w // 2 - 10 else (2 if tx > screen_w // 2 + 10 else 1)
        row = 0 if ty < screen_h // 2 - 10 else (2 if ty > screen_h // 2 + 10 else 1)
        raw[(col, row)] = (rx, ry)

    # Robust outlier replacement: only replace cells that disagree with their
    # neighbours by more than OUTLIER_FACTOR× the local spread.
    clean = dict(raw)
    for k, v in raw.items():
        nbrs = [raw[n] for n in raw if n != k and abs(n[0] - k[0]) <= 1 and abs(n[1] - k[1]) <= 1]
        if len(nbrs) < 3: continue
        mx = float(np.median([n[0] for n in nbrs]))
        my = float(np.median([n[1] for n in nbrs]))
        sx = float(np.std([n[0] for n in nbrs])) + 1e-6
        sy = float(np.std([n[1] for n in nbrs])) + 1e-6
        if abs(v[0] - mx) > OUTLIER_FACTOR * sx or abs(v[1] - my) > OUTLIER_FACTOR * sy:
            clean[k] = (mx, my)
    return raw, clean


def apply_residual(px, py, res):
    if res is None or not residual_enabled: return px, py
    gx = [margin_x, screen_w // 2, screen_w - margin_x]
    gy = [margin_y, screen_h // 2, screen_h - margin_y]
    if px <= gx[1]: c0, cf = 0, (0.0 if px <= gx[0] else (px - gx[0]) / max(gx[1] - gx[0], 1e-6))
    else:           c0, cf = 1, (0.0 if px >= gx[2] else (px - gx[1]) / max(gx[2] - gx[1], 1e-6))
    cf = float(np.clip(cf, 0.0, 1.0))
    if py <= gy[1]: r0, rf = 0, (0.0 if py <= gy[0] else (py - gy[0]) / max(gy[1] - gy[0], 1e-6))
    else:           r0, rf = 1, (0.0 if py >= gy[2] else (py - gy[1]) / max(gy[2] - gy[1], 1e-6))
    rf = float(np.clip(rf, 0.0, 1.0))
    def g(c, r): return res.get((max(0, min(2, c)), max(0, min(2, r))), (0.0, 0.0))
    a00, a10, a01, a11 = g(c0, r0), g(c0 + 1, r0), g(c0, r0 + 1), g(c0 + 1, r0 + 1)
    w00, w10 = (1 - cf) * (1 - rf), cf * (1 - rf)
    w01, w11 = (1 - cf) * rf, cf * rf
    return (px + a00[0] * w00 + a10[0] * w10 + a01[0] * w01 + a11[0] * w11,
            py + a00[1] * w00 + a10[1] * w10 + a01[1] * w01 + a11[1] * w11)


def rebuild_from_cal_data():
    global map_x_knots, map_y_knots, residual_map_raw, residual_map, cal_health, is_calibrated
    global center_signal_x, center_signal_y, yaw_neutral, pitch_neutral
    global apply_yaw_assist, apply_pitch_assist, sign_yaw, sign_pitch
    global ear_open_l, ear_open_r, fit_report

    s = fit_mapping()
    map_x_knots, map_y_knots = s['x'], s['y']
    center_signal_x, center_signal_y = s['c_signal']
    yaw_neutral, pitch_neutral = s['head_neutral']
    apply_yaw_assist, apply_pitch_assist = s['en_yaw'], s['en_pit']
    sign_yaw, sign_pitch = s['s_yaw'], s['s_pit']
    ear_open_l, ear_open_r = s['ear_open']

    residual_map_raw, residual_map = build_residual_map()

    # Diagnostic report on residual behaviour per cell
    cell_report = {}
    for k in residual_map_raw:
        raw = residual_map_raw[k]; cln = residual_map[k]
        replaced = (abs(raw[0] - cln[0]) > 1 or abs(raw[1] - cln[1]) > 1)
        cell_report[k] = dict(raw=raw, cleaned=cln, replaced=replaced,
                              mag=float(np.hypot(*cln)))

    yaws = [d[2] for d in cal_data]; pitches = [d[3] for d in cal_data]
    yaw_spread = float(np.rad2deg(max(yaws) - min(yaws)))
    pitch_spread = float(np.rad2deg(max(pitches) - min(pitches)))

    issues = []
    if not s['mono_x']: issues.append("X non-monotonic -> LINEAR FALLBACK")
    if not s['mono_y']: issues.append("Y non-monotonic -> LINEAR FALLBACK")
    replaced = [k for k, v in cell_report.items() if v['replaced']]
    if replaced:
        issues.append(f"residual outliers replaced at {replaced}")
    if yaw_spread > 12: issues.append(f"yaw drift {yaw_spread:.0f}° during cal")
    if pitch_spread > 12: issues.append(f"pitch drift {pitch_spread:.0f}° during cal")
    big = [k for k, v in cell_report.items() if v['mag'] > 0.10 * max(screen_w, screen_h)]
    if big: issues.append(f"large residual (>10%) at {big}")

    cal_health = 'GOOD' if not issues else ('FAIR' if len(issues) == 1 else 'POOR')

    fit_report = dict(cell_report=cell_report, issues=issues,
                      yaw_spread=yaw_spread, pitch_spread=pitch_spread,
                      mono_x=s['mono_x'], mono_y=s['mono_y'],
                      kx=s['kx_in'], ky=s['ky_in'],
                      ear=(ear_open_l, ear_open_r))

    print("\n" + "=" * 68)
    print(f"[AuraOS] CALIBRATION QUALITY: {cal_health}")
    print(f"  monotonic: X={s['mono_x']} Y={s['mono_y']}   "
          f"yaw_drift={yaw_spread:.1f}° pitch_drift={pitch_spread:.1f}°")
    print(f"  EAR baseline L={ear_open_l:.3f} R={ear_open_r:.3f}")
    print("  residual per cell (col,row -> raw px | cleaned px | replaced):")
    for k in sorted(cell_report):
        r = cell_report[k]
        print(f"    {k}: raw=({r['raw'][0]:+6.1f},{r['raw'][1]:+6.1f})  "
              f"clean=({r['cleaned'][0]:+6.1f},{r['cleaned'][1]:+6.1f})  "
              f"{'*REPLACED*' if r['replaced'] else ''}")
    if issues:
        print("  ISSUES:")
        for it in issues: print(f"    - {it}")
    print("  keys: 't'=residual  'h'=head-assist  'p'=dump  'c'=recal  'q'=quit")
    print("=" * 68 + "\n")
    is_calibrated = True


def print_diagnostics():
    print("\n--- AuraOS DIAG ---")
    print(f"health={cal_health}  screen={screen_w}x{screen_h}")
    print(f"residual={'ON' if residual_enabled else 'OFF'}  "
          f"head_assist={'ON' if head_assist_enabled else 'OFF'}")
    if fit_report:
        print(f"knots_x={['%.4f' % k for k in fit_report['kx']]}  "
              f"knots_y={['%.4f' % k for k in fit_report['ky']]}")
        for k in sorted(fit_report['cell_report']):
            r = fit_report['cell_report'][k]
            print(f"  cell{k}: raw=({r['raw'][0]:+6.1f},{r['raw'][1]:+6.1f}) "
                  f"clean=({r['cleaned'][0]:+6.1f},{r['cleaned'][1]:+6.1f}) "
                  f"{'REPLACED' if r['replaced'] else ''}")
    for r in cal_data:
        print(f"  pt ix={r[0]:+.4f} iy={r[1]:+.4f} -> ({r[4]},{r[5]})")
    print("--- END ---\n")

# ------------------------------ loop -----------------------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("AuraOS Calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("AuraOS Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    frame = cv2.flip(frame, 1)
    fh, fw, _ = frame.shape

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect_for_video(mp_img, int(time.monotonic() * 1000))

    display = np.full((screen_h, screen_w, 3), 255, dtype=np.uint8)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('p'): print_diagnostics()
    if key == ord('t'):
        residual_enabled = not residual_enabled
        print(f"[AuraOS] residual: {'ON' if residual_enabled else 'OFF'}")
    if key == ord('h'):
        head_assist_enabled = not head_assist_enabled
        print(f"[AuraOS] head assist: {'ON' if head_assist_enabled else 'OFF'}")
    if key == ord('c'):
        is_calibrated = False; cal_idx = 0; is_burst = False
        cal_data.clear(); burst.clear(); ear_open_samples.clear()
        residual_map_raw = residual_map = None; cal_health = None
        burst_retries = 0; last_burst_error = None
        print("[AuraOS] recalibrating")

    if not results.face_landmarks:
        cv2.putText(display, "NO FACE — adjust position", (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow("AuraOS Calibration", display); continue

    face = results.face_landmarks[0]
    ear_l, ear_r, iris_x, iris_y = get_gaze_metrics(face, fw, fh)
    yaw, pitch = head_angles(
        results.facial_transformation_matrixes[0] if results.facial_transformation_matrixes else None)

    # =========== CALIBRATION ===========
    if not is_calibrated:
        pt_x, pt_y = cal_points[cal_idx]
        dot_color = (0, 200, 0) if is_burst else (0, 0, 255)
        if last_burst_error is not None and last_burst_error > BURST_MAD_MAX:
            dot_color = (0, 165, 255)  # orange = last burst was noisy
        cv2.circle(display, (pt_x, pt_y), 34, dot_color, -1)
        cv2.circle(display, (pt_x, pt_y), 7, (255, 255, 255), -1)

        prompt = f"Stare at bullseye -> SPACE ({cal_idx + 1}/9)"
        if burst_retries > 0:
            prompt += f"  retry {burst_retries}/{BURST_RETRY_MAX}"
        cv2.putText(display, prompt, (screen_w // 2 - 460, screen_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3)

        if last_burst_error is not None and last_burst_error > BURST_MAD_MAX:
            cv2.putText(display, f"Last burst NOISY (MAD={last_burst_error:.4f}) — hold still & retry",
                        (screen_w // 2 - 500, screen_h // 2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 120, 200), 2)

        if key == ord(' ') and not is_burst:
            is_burst, burst = True, []

        if is_burst:
            if ear_l > MIN_EAR_SAMPLE and ear_r > MIN_EAR_SAMPLE:
                burst.append((iris_x, iris_y, yaw, pitch))
                ear_open_samples.append((ear_l, ear_r))
            cv2.putText(display, f"Sampling: {len(burst)}/{BURST_FRAMES}",
                        (pt_x - 90, pt_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 0), 2)

            if len(burst) >= BURST_FRAMES:
                arr = np.array(burst); med = np.median(arr, axis=0)
                mad = float(np.median(np.abs(arr - med), axis=0).sum())
                last_burst_error = mad
                if mad > BURST_MAD_MAX and burst_retries < BURST_RETRY_MAX:
                    burst_retries += 1
                    is_burst = False; burst = []
                    print(f"[AuraOS] pt {cal_idx + 1}/9 NOISY (MAD={mad:.4f} > {BURST_MAD_MAX}) — retry")
                else:
                    cal_data.append((*med, pt_x, pt_y))
                    print(f"[AuraOS] pt {cal_idx + 1}/9 OK  "
                          f"iris=({med[0]:+.3f},{med[1]:+.3f}) "
                          f"yaw={np.rad2deg(med[2]):+.1f}° pitch={np.rad2deg(med[3]):+.1f}° "
                          f"MAD={mad:.4f}")
                    is_burst = False
                    cal_idx += 1
                    burst_retries = 0
                    if cal_idx >= 9:
                        rebuild_from_cal_data()
        cv2.imshow("AuraOS Calibration", display); continue

    # =========== ONLINE ===========
    buffer.append((ear_l, ear_r, iris_x, iris_y, yaw, pitch))
    raw_xy_hist.append((iris_x, iris_y))

    if key == ord('r') and len(buffer) >= 10:
        recent = list(buffer)[-12:]
        ix_r = float(np.median([b[2] for b in recent]))
        iy_r = float(np.median([b[3] for b in recent]))
        yaw_neutral = float(np.median([b[4] for b in recent]))
        pitch_neutral = float(np.median([b[5] for b in recent]))
        bias_x = center_signal_x - ix_r
        bias_y = center_signal_y - iy_r
        reanchor_flash_until = time.monotonic() + 1.2
        print(f"[AuraOS] re-anchored bias=({bias_x:+.4f},{bias_y:+.4f})")

    norm_ear = min(ear_l / max(ear_open_l, 1e-3), ear_r / max(ear_open_r, 1e-3))
    is_frozen = norm_ear < FREEZE_NORM_EAR

    if is_frozen:
        if wink_state == 'open':
            n_back = min(PRE_WINK_BACK + 2, len(buffer))
            pre = list(buffer)[-n_back]
            sx = predict_axis(pre[2] + bias_x, map_x_knots)
            sy = predict_axis(pre[3] + bias_y, map_y_knots)
            sx, sy = apply_residual(sx, sy, residual_map)
            smooth_x = int(np.clip(sx, 0, screen_w - 1))
            smooth_y = int(np.clip(sy, 0, screen_h - 1))
            pyautogui.moveTo(smooth_x, smooth_y, _pause=False)
            wink_state = 'frozen'
    else:
        if wink_state == 'frozen':
            wink_state = 'open'; last_open_time = time.monotonic()

        if len(raw_xy_hist) >= 5:
            arr = np.array(raw_xy_hist)
            ix_f, iy_f = float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        else:
            ix_f, iy_f = iris_x, iris_y

        pred_x = predict_axis(ix_f + bias_x, map_x_knots)
        pred_y = predict_axis(iy_f + bias_y, map_y_knots)
        base_x, base_y = pred_x, pred_y
        if apply_yaw_assist and head_assist_enabled:
            pred_x += sign_yaw * bounded_assist(yaw - yaw_neutral, HEAD_DEADBAND, HEAD_BLEND_X)
        if apply_pitch_assist and head_assist_enabled:
            pred_y += sign_pitch * bounded_assist(pitch - pitch_neutral, HEAD_DEADBAND, HEAD_BLEND_Y)
        after_head_x, after_head_y = pred_x, pred_y
        pred_x, pred_y = apply_residual(pred_x, pred_y, residual_map)
        pred_x = float(np.clip(pred_x, 0, screen_w - 1))
        pred_y = float(np.clip(pred_y, 0, screen_h - 1))

        pred_hist.append((pred_x, pred_y))
        if len(pred_hist) == 30 and np.var(pred_hist, axis=0).sum() > 10000 \
                and time.monotonic() - last_var_warn > 5.0:
            print("[AuraOS] WARN: prediction variance high — press 'r' to re-anchor")
            last_var_warn = time.monotonic()

        fix_locked = False
        if len(raw_xy_hist) >= 6 and np.var(np.array(raw_xy_hist), axis=0).sum() < FIX_VAR_THRES:
            fix_locked = True
            err = float(np.hypot(pred_x - smooth_x, pred_y - smooth_y))
            if err > FIX_SNAP_PX:
                smooth_x = int(round(SNAP_ALPHA * smooth_x + (1 - SNAP_ALPHA) * pred_x))
                smooth_y = int(round(SNAP_ALPHA * smooth_y + (1 - SNAP_ALPHA) * pred_y))
                smooth_x = max(0, min(screen_w - 1, smooth_x))
                smooth_y = max(0, min(screen_h - 1, smooth_y))
                pyautogui.moveTo(smooth_x, smooth_y, _pause=False)

        if not fix_locked:
            dist = float(np.hypot(pred_x - smooth_x, pred_y - smooth_y))
            if dist > DEADZONE_PIX:
                if dist > BIG_JUMP_PIX: alpha = EMA_ALPHA_FAR
                else:
                    t = (dist - DEADZONE_PIX) / (BIG_JUMP_PIX - DEADZONE_PIX)
                    alpha = EMA_ALPHA_NEAR + t * (EMA_ALPHA_FAR - EMA_ALPHA_NEAR)
                if abs(yaw - last_yaw) > HEAD_RATE_GATE or abs(pitch - last_pitch) > HEAD_RATE_GATE:
                    alpha = max(alpha, 0.96)
                smooth_x = int(alpha * smooth_x + (1 - alpha) * pred_x)
                smooth_y = int(alpha * smooth_y + (1 - alpha) * pred_y)
                smooth_x = max(0, min(screen_w - 1, smooth_x))
                smooth_y = max(0, min(screen_h - 1, smooth_y))
                pyautogui.moveTo(smooth_x, smooth_y, _pause=False)

    last_yaw, last_pitch = yaw, pitch

    if click_model is not None:
        l_w = (face[133].x - face[33].x) + 1e-6
        compat_iris_x = (face[468].x - face[33].x) / l_w
        action = click_model.predict([[ear_l, ear_r, compat_iris_x]])[0]
        now = time.monotonic()
        if (int(action) in (1, 2) and now - last_click_time > CLICK_COOLDOWN
                and now - last_open_time > CLICK_REARM):
            pyautogui.click(_pause=False); last_click_time = now

    health_color = {'GOOD': (0, 160, 0), 'FAIR': (0, 160, 200), 'POOR': (0, 0, 200)}.get(cal_health, (80, 80, 80))
    cv2.putText(display,
                f"AuraOS [{cal_health}]  't'=resid {'ON' if residual_enabled else 'OFF'}  "
                f"'h'=head {'ON' if head_assist_enabled else 'OFF'}  'p'=dump  'r'=reanchor  'c'=recal  'q'=quit",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, health_color, 2)
    cv2.putText(display,
                f"EAR L/R {ear_l:.2f}/{ear_r:.2f} norm={norm_ear:.2f} state={wink_state}",
                (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (60, 60, 60), 2)
    cv2.putText(display,
                f"bias=({bias_x:+.4f},{bias_y:+.4f})  "
                f"head=({np.rad2deg(yaw - yaw_neutral):+.1f}°,{np.rad2deg(pitch - pitch_neutral):+.1f}°)",
                (30, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 2)
    if time.monotonic() < reanchor_flash_until:
        cv2.putText(display, "RE-ANCHORED", (screen_w // 2 - 160, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 160, 0), 3)
    cv2.circle(display, (smooth_x, smooth_y), 15, (0, 200, 0), -1)
    cv2.circle(display, (smooth_x, smooth_y), 4, (255, 255, 255), -1)
    cv2.imshow("AuraOS Calibration", display)

cap.release()
cv2.destroyAllWindows()