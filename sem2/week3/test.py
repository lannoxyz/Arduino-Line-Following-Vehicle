"""
finale.py
Electrical Engineering Project

Action priority levels
1. Line following of colour blobs/coloured lines
2. Line following of black lines
3. Line recovery

task assign (RPi4 four cores):
  Core 0 — Flask Web 
  Core 1 — processing_loop for motor control, line following
  Core 2 — capture_loop from picamera
  Core 3 — inference_loop, ONNX model output obtain for symbol detection
"""

import time, sys, threading, os
from collections import deque
from flask import Flask, Response, jsonify
import cv2, numpy as np

# library import check
try:
    from picamera2 import Picamera2
except ImportError:
    print("error: sudo apt install python3-picamera2"); sys.exit(1)

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("error: sudo apt install python3-rpi.gpio"); sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("error: pip install onnxruntime --break-system-packages"); sys.exit(1)

# ═══════════════════════════════════════════════════════════════
#  Configurations
# ═══════════════════════════════════════════════════════════════

CAMERA_RESOLUTION = (640, 480) # highest at 30 fps
JPEG_QUALITY      = 60
FLIP_MODE         = -1

# Line Following
BINARY_THRESHOLD      = 80
DEAD_ZONE_PCT         = 0.05
SPIN_ZONE_PCT         = 0.20

# Speed
BASE_SPEED            = 20
MAX_SPEED             = 42
SPIN_SPEED            = 42

# Line Recovery
RECOVERY_ENABLED      = True
RECOVERY_SPEED        = 42
RECOVERY_TIMEOUT      = 3.0
LINE_LOST_THRESHOLD   = 100
HISTORY_DURATION      = 1.0
PIXEL_RATIO_THRESHOLD = 0.5

# Color Blob Following
COLOR_BLOB_SAT_THRESHOLD = 220   # HSV saturation threshold
COLOR_BLOB_MIN_AREA      = 100   # ignore all blob smaller than threshold
COLOR_BLOB_VAL_MIN       = 30    # HSV minimum brightness threshold

# GPIO pins define
PIN_IN1, PIN_IN2 = 27, 17
PIN_IN3, PIN_IN4 = 6, 5
PIN_ENA, PIN_ENB = 12, 13

# AI model parameters
ONNX_MODEL_PATH    = "model.onnx"
TFLITE_LABELS_PATH = "labels.txt"
ONNX_INPUT_SIZE    = 224

# Per-symbol Confidence Thresholds, how high to consider it detected
CONF_THRESHOLDS = {
    "arrowleft":   60.0,
    "arrowright":  60.0,
    "arrowup":     60.0,
    "warningsign": 60.0,
    "qrcode":      60.0,
    "thumb":       60.0,
    "recyclesign": 40.0,
    "buttonsign":  30.0,
}
CONF_THRESHOLD_DEFAULT = 50.0   # fallback for unlisted labels

ACTION_COOLDOWN_SEC = 7.0

PORT = 5000

# ═══════════════════════════════════════════════════════════════
#  Low-level Control
# ═══════════════════════════════════════════════════════════════

# gpio setup output
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [PIN_IN1, PIN_IN2, PIN_IN3, PIN_IN4, PIN_ENA, PIN_ENB]:
    GPIO.setup(pin, GPIO.OUT)

# 500Hz pwm frequency
pwm_a = GPIO.PWM(PIN_ENA, 500); pwm_a.start(0)
pwm_b = GPIO.PWM(PIN_ENB, 500); pwm_b.start(0)

def set_motors(left, right):
    """setup rotating speed of vehicle in left and right (-100 to 100)"""
    left  = max(-100.0, min(100.0, left))
    right = max(-100.0, min(100.0, right))
    GPIO.output(PIN_IN1, GPIO.HIGH if left  > 0 else GPIO.LOW)
    GPIO.output(PIN_IN2, GPIO.LOW  if left  > 0 else (GPIO.HIGH if left  < 0 else GPIO.LOW))
    GPIO.output(PIN_IN3, GPIO.HIGH if right > 0 else GPIO.LOW)
    GPIO.output(PIN_IN4, GPIO.LOW  if right > 0 else (GPIO.HIGH if right < 0 else GPIO.LOW))
    pwm_a.ChangeDutyCycle(abs(left))
    pwm_b.ChangeDutyCycle(abs(right))

def stop_motors():
    set_motors(0, 0)

# ═══════════════════════════════════════════════════════════════
#  Vision & Inference
# ═══════════════════════════════════════════════════════════════

print("Initializing Camera...")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": CAMERA_RESOLUTION, "format": "BGR888"}, # BGR format captured
    controls={"FrameDurationLimits": (33333, 33333)})) # 30 fps
picam2.start(); time.sleep(1)

_session = _input_name = None
_labels  = []

# load ONNX model
def _load_onnx():
    global _session, _labels, _input_name
    if not os.path.exists(TFLITE_LABELS_PATH): return False
    with open(TFLITE_LABELS_PATH) as f:
        _labels = [l.strip().split(None, 1)[-1] for l in f if l.strip()]
    if not os.path.exists(ONNX_MODEL_PATH): return False
    try:
        _session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        return True
    except Exception as e:
        print(f"Model Load Failed: {e}"); return False

# image processing for ONNX model
# BGR > resize > grayscale > RGB 3 channels > float > NHCW
def _onnx_classify(frame_bgr):
    resized = cv2.resize(frame_bgr, (ONNX_INPUT_SIZE, ONNX_INPUT_SIZE))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    rgb     = cv2.merge([gray, gray, gray])
    img_f   = (rgb.astype(np.float32) / 255.0 - 0.449) / 0.226
    blob    = np.transpose(img_f, (2, 0, 1))[np.newaxis, ...]
    try:
        scores = _session.run(None, {_input_name: blob})[0][0].astype(np.float32) # run ONNX model
        e_ = np.exp(scores - scores.max()) # convert scores into %
        probs = e_ / e_.sum()
        idx = int(np.argmax(probs)) # obtain class with highest confidence
        return _labels[idx], float(probs[idx] * 100)
    except:
        return "error", 0.0

# ═══════════════════════════════════════════════════════════════
#  Shared State, thread handling
# ═══════════════════════════════════════════════════════════════

frame_lock = threading.Lock() # locks thread, ensure only one thread can modify data
latest_raw_frame = None  # original frame 
latest_cam_frame = None  # processed frame

state_lock  = threading.Lock()
robot_state = { # core status of vehicle
    "status": "init", "direction": "straight", "error_pct": 0.0,
    "left_speed": 0.0, "right_speed": 0.0, "fps": 0.0, "black_ratio": 0.0,
    "line_lost_recovery": "inactive", "recovery_reason": "",
    "blob_mode": False  # whether blob tracking is used
}

symbol_lock = threading.Lock()
symbol_state = {"shape": "—", "confidence": 0.0, "action": "none", "flash": False} # core status for symbol detection

action_lock  = threading.Lock()
action_state = {"running": False, "permanent_stop": False} # whether if vehicle is moving or stopped

_action_cooldown = {}
frame_history = deque() # stores frame in deque

# line recovery when line is lost
history_lock  = threading.Lock()
recovery_lock = threading.Lock()
line_lost_state = {"is_recovering": False, "line_lost_time": None, "recovery_direction": None}

# ═══════════════════════════════════════════════════════════════
#  Performance Monitoring State
# ═══════════════════════════════════════════════════════════════

perf_lock = threading.Lock()
perf_state = {
    # Camera metrics
    "cam_capture_ms":      0.0,   # time to capture one frame (ms)
    "cam_fps":             0.0,   # actual camera capture fps
    "cam_resolution_w":    CAMERA_RESOLUTION[0],
    "cam_resolution_h":    CAMERA_RESOLUTION[1],
    "cam_frame_count":     0,     # total frames captured

    # Processing / line-following metrics
    "proc_fps":            0.0,   # processing loop fps
    "proc_loop_ms":        0.0,   # time for one processing iteration (ms)
    "proc_frame_count":    0,

    # Inference metrics
    "infer_fps":           0.0,   # inference calls per second
    "infer_ms":            0.0,   # time for one ONNX inference (ms)
    "preprocess_ms":       0.0,   # image pre-processing time (ms)
    "infer_frame_count":   0,

    # Pipeline latency
    "pipeline_latency_ms": 0.0,   # capture_timestamp → frame available in inference
    "encode_ms":           0.0,   # JPEG encode time for stream (ms)

    # Stream
    "stream_fps":          0.0,   # MJPEG stream fps served to browser
    "stream_frame_count":  0,
}

# rolling window for FPS calculations (stores timestamps)
_cam_ts_window    = deque(maxlen=60)
_proc_ts_window   = deque(maxlen=60)
_infer_ts_window  = deque(maxlen=60)
_stream_ts_window = deque(maxlen=60)

# timestamp injected into each frame for pipeline latency measurement
_frame_capture_ts_lock = threading.Lock()
_frame_capture_ts      = 0.0   # unix timestamp of last captured frame

def _fps_from_window(window):
    """Calculate fps from a deque of timestamps."""
    if len(window) < 2:
        return 0.0
    span = window[-1] - window[0]
    return (len(window) - 1) / span if span > 0 else 0.0

# ═══════════════════════════════════════════════════════════════
#  Color Blob Detection
# ═══════════════════════════════════════════════════════════════

def _detect_color_blob(frame_bgr):
    """
    Detect high-saturation color blobs in a frame and return the centroid and mask.

    Principle：
      - Convert into HSV space，so saturation is independant of lighting
      - S > COLOR_BLOB_SAT_THRESHOLD to filter dull colours like white/gray/black
      - Morphological cleanup
      - Use moments to calculate center of mass

    Returns:
      (cx, cy, sat_mask) — location of center in cx/cy, binary image showing detected areas in sat_mask
      (None, None, sat_mask) — no valid blob
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # filter and keep strong colours like red and yellow, will be masked in white
    sat_mask = cv2.inRange(
        hsv,
        (0,   COLOR_BLOB_SAT_THRESHOLD, COLOR_BLOB_VAL_MIN), # lower bound for HSV
        (180, 255,                      255) # upper bound
    )

    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN,  kernel)
    sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

    # compute center of mass of blob
    moments = cv2.moments(sat_mask) 
    if moments["m00"] > COLOR_BLOB_MIN_AREA:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy, sat_mask
  
    return None, None, sat_mask

# ═══════════════════════════════════════════════════════════════
#  Line Recovery
# ═══════════════════════════════════════════════════════════════

def add_frame_to_history(binary_image, timestamp):
    with history_lock:
        frame_history.append((timestamp, binary_image.copy()))
        cutoff = timestamp - HISTORY_DURATION
        while frame_history and frame_history[0][0] < cutoff: frame_history.popleft()

def analyze_pixel_distribution():
    with history_lock:
        if not frame_history: return None, 0.0
        l_px = r_px = 0
        w = frame_history[0][1].shape[1]
        for _, b in frame_history:
            l_px += cv2.countNonZero(b[:, :w//2])
            r_px += cv2.countNonZero(b[:, w//2:])
        total = l_px + r_px
        if total == 0: return None, 0.0
        r_ratio = r_px / total
        if r_ratio > PIXEL_RATIO_THRESHOLD: return "right", r_ratio - 0.5
        if r_ratio < (1.0 - PIXEL_RATIO_THRESHOLD): return "left", 0.5 - r_ratio
        return None, 0.0

def start_line_recovery():
    with recovery_lock:
        if not line_lost_state["is_recovering"]:
            dir_rec, conf = analyze_pixel_distribution()
            line_lost_state.update({
                "is_recovering": True,
                "line_lost_time": time.time(),
                "recovery_direction": dir_rec
            })

def stop_line_recovery():
    with recovery_lock: line_lost_state["is_recovering"] = False

def get_recovery_state():
    with recovery_lock:
        if not line_lost_state["is_recovering"]: return False, None, False
        elapsed = time.time() - line_lost_state["line_lost_time"]
        if elapsed > RECOVERY_TIMEOUT:
            line_lost_state["is_recovering"] = False
            return False, None, False
        return True, line_lost_state["recovery_direction"], True

# ═══════════════════════════════════════════════════════════════
#  Reaction System
# ═══════════════════════════════════════════════════════════════

def _set_action(desc):
    with symbol_lock: symbol_state["action"] = desc

def _is_in_cooldown(label):
    return (time.time() - _action_cooldown.get(label.lower(), 0)) < ACTION_COOLDOWN_SEC

def _mark_cooldown(label):
    _action_cooldown[label.lower()] = time.time()

def execute_action(label):
    key = label.lower()

    if key in ("thumb", "qrcode"):
        _set_action(f"flash_{key}")
        with symbol_lock: symbol_state["flash"] = True
        time.sleep(2.5)
        with symbol_lock: symbol_state["flash"] = False
        _set_action("none")
        _mark_cooldown(label)
        return

    if key in ("buttonsign", "warningsign"):
        with action_lock:
            action_state["permanent_stop"] = True
            action_state["running"] = False
        _set_action(f"{label} — STOPPED")
        stop_motors()
        print(f"[ACTION] Permanent stop triggered by {label}")
        return 

    with action_lock:
        if not action_state["running"]:
            action_state["running"] = True

    try:
        stop_motors()
        if key == "arrowup":
            _set_action("ArrowUp — Straight")
            set_motors(BASE_SPEED, BASE_SPEED); time.sleep(1.2)

        elif key in ("arrowleft", "arrowright"):
            direction = "left" if key == "arrowleft" else "right"
            _set_action(f"Arrow{direction.capitalize()} — Searching line...")

            stop_motors()
            set_motors(BASE_SPEED, BASE_SPEED); time.sleep(0.8)

            spin_l = -SPIN_SPEED if direction == "left" else  SPIN_SPEED
            spin_r =  SPIN_SPEED if direction == "left" else -SPIN_SPEED
            set_motors(spin_l, spin_r); time.sleep(0.5)

            LINE_CONFIRM_FRAMES = 3
            MAX_SEARCH_TIME     = 3.0
            found_count = 0
            t0 = time.time()
            while time.time() - t0 < MAX_SEARCH_TIME:
                set_motors(spin_l, spin_r)
                time.sleep(0.03)
                with frame_lock:
                    f = latest_cam_frame
                if f is not None:
                    gray   = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                    moments = cv2.moments(binary)
                    if moments["m00"] > LINE_LOST_THRESHOLD:
                        found_count += 1
                        if found_count >= LINE_CONFIRM_FRAMES:
                            break
                    else:
                        found_count = 0
            stop_motors()
            _set_action(f"Arrow{direction.capitalize()} — Done")

        elif key == "recyclesign":
            _set_action("Recycle — Spin 360")
            time.sleep(0.1)
            stop_motors()
            GROUND_SPIN_SPEED = int(SPIN_SPEED * 1.1)
            set_motors(GROUND_SPIN_SPEED, -GROUND_SPIN_SPEED)
            time.sleep(2.8)

    finally:
        with action_lock:
            action_state["running"] = False
        _set_action("none")
        _mark_cooldown(label)

# ═══════════════════════════════════════════════════════════════
#  Thread Loops
# ═══════════════════════════════════════════════════════════════

def capture_loop():
    global latest_cam_frame, _frame_capture_ts
    cam_f_cnt = 0
    cam_t_start = time.time()

    while True:
        try:
            t0 = time.time()
            frame = picam2.capture_array()
            capture_ms = (time.time() - t0) * 1000.0

            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE)

            now = time.time()
            with frame_lock:
                latest_cam_frame = frame
            with _frame_capture_ts_lock:
                _frame_capture_ts = now

            cam_f_cnt += 1
            _cam_ts_window.append(now)
            cam_fps = _fps_from_window(_cam_ts_window)

            with perf_lock:
                perf_state["cam_capture_ms"]   = round(capture_ms, 2)
                perf_state["cam_fps"]          = round(cam_fps, 1)
                perf_state["cam_frame_count"]  = cam_f_cnt
                perf_state["cam_resolution_w"] = frame.shape[1]
                perf_state["cam_resolution_h"] = frame.shape[0]

        except Exception:
            time.sleep(0.01)


def processing_loop():
    global latest_raw_frame
    f_cnt, t_start, cur_fps = 0, time.time(), 0.0
    proc_f_cnt = 0

    while True:
        loop_t0 = time.time()

        with action_lock:
            is_perm = action_state["permanent_stop"]
            is_run  = action_state["running"]

        with frame_lock: frame = latest_cam_frame
        if frame is None: time.sleep(0.01); continue

        h, w = frame.shape[:2]

        if is_perm:
            stop_motors()
            with frame_lock: latest_raw_frame = frame.copy()
            time.sleep(0.1); continue

        if is_run:
            disp = frame.copy()
            cv2.line(disp, (w//2, 0), (w//2, h-1), (255, 0, 0), 1)
            with frame_lock: latest_raw_frame = disp
            time.sleep(0.03); continue

        l_spd = r_spd = 0.0
        stat = drct = "no_line"
        err = 0.0
        rec_s = "inactive"; rec_r = ""
        cx = cy = -1
        b_ratio = 0.0
        blob_mode = False

        blob_cx, blob_cy, sat_mask = _detect_color_blob(frame)

        if blob_cx is not None:
            blob_mode = True
            stop_line_recovery()

            cx, cy = blob_cx, blob_cy
            err = (cx - w / 2) / (w / 2)
            abs_err = abs(err)

            if abs_err <= DEAD_ZONE_PCT:
                stat = "blob_straight"; drct = "straight"
                l_spd = r_spd = BASE_SPEED
            elif abs_err <= SPIN_ZONE_PCT:
                k = (abs_err - DEAD_ZONE_PCT) / (SPIN_ZONE_PCT - DEAD_ZONE_PCT)
                outer = BASE_SPEED + k * (MAX_SPEED - BASE_SPEED)
                if err > 0:
                    drct = "right"; stat = "blob_adjust_right"
                    l_spd = outer;      r_spd = BASE_SPEED
                else:
                    drct = "left";  stat = "blob_adjust_left"
                    l_spd = BASE_SPEED; r_spd = outer
            else:
                if err > 0:
                    drct = stat = "blob_spin_right"
                    l_spd =  SPIN_SPEED; r_spd = -SPIN_SPEED
                else:
                    drct = stat = "blob_spin_left"
                    l_spd = -SPIN_SPEED; r_spd =  SPIN_SPEED

            set_motors(l_spd, r_spd)

        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            moments  = cv2.moments(binary)
            b_ratio  = cv2.countNonZero(binary) / (h * w)
            has_line = moments["m00"] > LINE_LOST_THRESHOLD
            add_frame_to_history(binary, time.time())

            if has_line:
                stop_line_recovery()
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                err = (cx - w / 2) / (w / 2)
                abs_err = abs(err)

                if abs_err <= DEAD_ZONE_PCT:
                    stat = drct = "straight"
                    l_spd = r_spd = BASE_SPEED
                elif abs_err <= SPIN_ZONE_PCT:
                    k = (abs_err - DEAD_ZONE_PCT) / (SPIN_ZONE_PCT - DEAD_ZONE_PCT)
                    outer = BASE_SPEED + k * (MAX_SPEED - BASE_SPEED)
                    if err > 0:
                        drct = "right"; stat = "adjust_right"
                        l_spd = outer;      r_spd = BASE_SPEED
                    else:
                        drct = "left";  stat = "adjust_left"
                        l_spd = BASE_SPEED; r_spd = outer
                else:
                    if err > 0:
                        drct = stat = "spin_right"
                        l_spd =  SPIN_SPEED; r_spd = -SPIN_SPEED
                    else:
                        drct = stat = "spin_left"
                        l_spd = -SPIN_SPEED; r_spd =  SPIN_SPEED

                set_motors(l_spd, r_spd)

            else:
                if RECOVERY_ENABLED:
                    is_r, d, cont = get_recovery_state()
                    if not is_r:
                        start_line_recovery()
                        is_r, d, cont = get_recovery_state()
                    if cont and d:
                        rec_s = f"recovery_{d}"
                        l_spd, r_spd = (
                            (RECOVERY_SPEED, -RECOVERY_SPEED) if d == "right"
                            else (-RECOVERY_SPEED, RECOVERY_SPEED)
                        )
                        set_motors(l_spd, r_spd)
                    else:
                        stop_line_recovery()
                        stop_motors()
                else:
                    stop_motors()

            disp = frame.copy()
            cv2.line(disp, (w // 2, 0), (w // 2, h - 1), (255, 0, 0), 1)
            if has_line and cx != -1:
                cv2.circle(disp, (cx, cy), 10, (0, 0, 255), -1)
                cv2.line(disp, (w // 2, cy), (cx, cy), (0, 255, 0), 2)

        # FPS calculation
        f_cnt += 1
        proc_f_cnt += 1
        now = time.time()
        if now - t_start >= 1.0:
            cur_fps = f_cnt / (now - t_start)
            f_cnt = 0; t_start = now

        loop_ms = (now - loop_t0) * 1000.0
        _proc_ts_window.append(now)
        proc_fps = _fps_from_window(_proc_ts_window)

        with state_lock:
            robot_state.update({
                "status":             stat,
                "direction":          drct,
                "error_pct":          round(err * 100, 1),
                "left_speed":         l_spd,
                "right_speed":        r_spd,
                "fps":                round(cur_fps, 1),
                "black_ratio":        round(b_ratio * 100, 1),
                "line_lost_recovery": rec_s,
                "recovery_reason":    rec_r,
                "blob_mode":          blob_mode,
            })

        with perf_lock:
            perf_state["proc_fps"]         = round(proc_fps, 1)
            perf_state["proc_loop_ms"]     = round(loop_ms, 2)
            perf_state["proc_frame_count"] = proc_f_cnt

        with frame_lock:
            latest_raw_frame = disp


def inference_loop():
    infer_f_cnt = 0

    while True:
        with frame_lock: f = latest_cam_frame
        if f is None or _session is None: time.sleep(0.05); continue

        # measure pipeline latency: how old is this frame?
        with _frame_capture_ts_lock:
            cap_ts = _frame_capture_ts
        pipeline_latency_ms = (time.time() - cap_ts) * 1000.0 if cap_ts > 0 else 0.0

        # measure pre-process time separately
        t_pre = time.time()
        resized = cv2.resize(f, (ONNX_INPUT_SIZE, ONNX_INPUT_SIZE))
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        rgb     = cv2.merge([gray, gray, gray])
        img_f   = (rgb.astype(np.float32) / 255.0 - 0.449) / 0.226
        blob    = np.transpose(img_f, (2, 0, 1))[np.newaxis, ...]
        preprocess_ms = (time.time() - t_pre) * 1000.0

        # measure pure inference time
        t_inf = time.time()
        try:
            scores = _session.run(None, {_input_name: blob})[0][0].astype(np.float32)
            e_ = np.exp(scores - scores.max())
            probs = e_ / e_.sum()
            idx   = int(np.argmax(probs))
            label = _labels[idx]
            conf  = float(probs[idx] * 100)
        except Exception:
            label, conf = "error", 0.0
        infer_ms = (time.time() - t_inf) * 1000.0

        infer_f_cnt += 1
        now = time.time()
        _infer_ts_window.append(now)
        infer_fps = _fps_from_window(_infer_ts_window)

        with symbol_lock:
            symbol_state.update({"shape": label, "confidence": conf})

        with perf_lock:
            perf_state["infer_fps"]          = round(infer_fps, 1)
            perf_state["infer_ms"]           = round(infer_ms, 2)
            perf_state["preprocess_ms"]      = round(preprocess_ms, 2)
            perf_state["pipeline_latency_ms"]= round(pipeline_latency_ms, 2)
            perf_state["infer_frame_count"]  = infer_f_cnt

        threshold = CONF_THRESHOLDS.get(label.lower(), CONF_THRESHOLD_DEFAULT)
        if conf >= threshold:
            spawned = False
            with action_lock:
                busy = action_state["running"] or action_state["permanent_stop"]
                if not busy and not _is_in_cooldown(label):
                    if label.lower() not in ("thumb", "qrcode", "buttonsign", "warningsign"):
                        action_state["running"] = True
                    spawned = True
            if spawned:
                threading.Thread(target=execute_action, args=(label,), daemon=True).start()

        time.sleep(0.1)

# ═══════════════════════════════════════════════════════════════
#  Flask & Web
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>67 Car — Raw Data Monitor</title>
<style>
  /* 全局重置与极简黑色背景 */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { 
    background-color: #0d1117; 
    color: #c9d1d9; 
    font-family: 'Courier New', Courier, monospace; 
    padding: 20px; 
    display: flex; 
    gap: 30px; 
    height: 100vh;
    overflow: hidden;
  }
  
  /* 左侧视频流，强制缩小至 50% */
  #video_container { 
    flex: 0 0 50%; 
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  #video_container img { 
    width: 100%; 
    border: 2px solid #30363d; 
    border-radius: 4px;
  }

  /* 右侧数据排版 */
  #data_container { 
    flex: 1; 
    display: flex; 
    flex-direction: column; 
    gap: 15px; 
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap; /* 保持换行格式 */
  }

  h2 { font-size: 16px; border-bottom: 1px dashed #30363d; padding-bottom: 5px; color: #fff;}

  /* 数据分类颜色编码 */
  .color_symbol { color: #ff7b72; font-weight: bold; } /* 红色系：AI符号与警告 */
  .color_robot  { color: #3fb950; } /* 绿色系：电机与机器人状态 */
  .color_cam    { color: #79c0ff; } /* 蓝色系：相机硬件状态 */
  .color_onnx   { color: #ffa657; } /* 橙色系：推理引擎 */
  .color_pipe   { color: #d2a8ff; } /* 紫色系：管道与延迟 */
</style>
</head>
<body>

  <div id="video_container">
    <h2>◈ LIVE STREAM (50% SCALE)</h2>
    <img src="/video" alt="MJPEG Stream">
  </div>

  <div id="data_container">
    <h2>◈ RAW SYSTEM TELEMETRY</h2>
    
    <div id="out_symbol" class="color_symbol">Loading AI data...</div>
    <div id="out_robot" class="color_robot">Loading Robot status...</div>
    <div id="out_cam" class="color_cam">Loading Camera metrics...</div>
    <div id="out_onnx" class="color_onnx">Loading Inference data...</div>
    <div id="out_pipe" class="color_pipe">Loading Pipeline latency...</div>
  </div>

<script>
// 获取 DOM 节点
const el_symbol = document.getElementById('out_symbol');
const el_robot  = document.getElementById('out_robot');
const el_cam    = document.getElementById('out_cam');
const el_onnx   = document.getElementById('out_onnx');
const el_pipe   = document.getElementById('out_pipe');

// 浮点数格式化助手
const fmt = (v, dec=1) => typeof v === 'number' ? v.toFixed(dec) : v;

// 轮询拉取数据并更新 DOM
async function poll_data() {
  try {
    const [perf, sym, state] = await Promise.all([
      fetch('/api/perf').then(r => r.json()),
      fetch('/api/symbol').then(r => r.json()),
      fetch('/api/state').then(r => r.json()),
    ]);

    // 格式化输出字符串
    el_symbol.innerText = 
      `[AI TARGET]  Shape: ${sym.shape.padEnd(12)} | Conf: ${fmt(sym.confidence)}% | Action: ${sym.action}`;
      
    el_robot.innerText = 
      `[ROBOT]      Status: ${state.status.padEnd(15)} | Dir: ${state.direction.padEnd(10)}\n` +
      `             Error: ${fmt(state.error_pct)}% | Blob Mode: ${state.blob_mode}\n` +
      `             L_Spd: ${fmt(state.left_speed).padEnd(6)} | R_Spd: ${fmt(state.right_speed)}\n` +
      `             Proc FPS: ${fmt(perf.proc_fps).padEnd(5)} | Loop Latency: ${fmt(perf.proc_loop_ms)}ms`;

    el_cam.innerText = 
      `[CAMERA]     Res: ${perf.cam_resolution_w}x${perf.cam_resolution_h} | Cap FPS: ${fmt(perf.cam_fps)}\n` +
      `             Cap Latency: ${fmt(perf.cam_capture_ms)}ms | Frames: ${perf.cam_frame_count}`;

    el_onnx.innerText = 
      `[INFERENCE]  Infer FPS: ${fmt(perf.infer_fps)}\n` +
      `             Pre-proc: ${fmt(perf.preprocess_ms)}ms | ONNX Time: ${fmt(perf.infer_ms)}ms`;

    el_pipe.innerText = 
      `[PIPELINE]   E2E Latency: ${fmt(perf.pipeline_latency_ms)}ms\n` +
      `             JPEG Enc: ${fmt(perf.encode_ms)}ms | Stream FPS: ${fmt(perf.stream_fps)}`;

  } catch(e) {
    // 忽略网络抖动报错
  }
}

// 500ms 刷新率
setInterval(poll_data, 500);
poll_data();
</script>

</body>
</html>"""


@app.route('/video')
def video():
    def gen():
        stream_f_cnt = 0
        
        # 创建一个纯黑的备用错误帧，这样即使相机挂了，前端也不会一直转圈
        # create a blank error frame to prevent browser hanging
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "NO SIGNAL / CAMERA ERROR", (80, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, error_buf = cv2.imencode('.jpg', error_frame)
        error_bytes = error_buf.tobytes()

        while True:
            with frame_lock: 
                current_frame = latest_raw_frame
            
            # 如果核心处理线程没有输出画面，主动抛出错误帧
            if current_frame is None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + error_bytes + b'\r\n'
                time.sleep(0.5) # 降低刷新率，节省资源
                continue

            # 正常编码逻辑
            t_enc = time.time()
            _, img_buf = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            encode_ms = (time.time() - t_enc) * 1000.0

            stream_f_cnt += 1
            now = time.time()
            _stream_ts_window.append(now)
            
            # 由于在局部函数中，调用全局作用域的函数不需要声明 global
            stream_fps = _fps_from_window(_stream_ts_window)

            with perf_lock:
                perf_state["encode_ms"]          = round(encode_ms, 2)
                perf_state["stream_fps"]         = round(stream_fps, 1)
                perf_state["stream_frame_count"] = stream_f_cnt

            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img_buf.tobytes() + b'\r\n'
            time.sleep(0.04) # 限制最高帧率

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/symbol')
def api_symbol():
    with symbol_lock: return jsonify(dict(symbol_state))

@app.route('/api/state')
def api_state():
    with state_lock: return jsonify(dict(robot_state))

@app.route('/api/perf')
def api_perf():
    with perf_lock: return jsonify(dict(perf_state))

# ═══════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    _load_onnx()
    import ctypes

    def _set_affinity(mask):
        try:
            ctypes.CDLL('libc.so.6').sched_setaffinity(
                0, ctypes.sizeof(ctypes.c_ulong),
                ctypes.byref(ctypes.c_ulong(mask))
            )
        except:
            pass

    def _spawn(target, mask, name):
        def wrap(): _set_affinity(mask); target()
        threading.Thread(target=wrap, name=name, daemon=True).start()

    _spawn(capture_loop,    0b0100, "cap")   # Core 2
    _spawn(processing_loop, 0b0010, "line")  # Core 1
    _spawn(inference_loop,  0b1000, "inf")   # Core 3
    _set_affinity(0b0001)                    # Flask -> Core 0

    try:
        app.run(host='0.0.0.0', port=PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        stop_motors(); picam2.stop(); GPIO.cleanup()