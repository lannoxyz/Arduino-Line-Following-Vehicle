"""
robot.py — 巡线 + 色块跟随 + 符号检测 + 反应系统
Electrical Engineering Project

巡线优先级:
  ① 高饱和度色块跟随 (S > 220, area > 100px) — 最高优先级
  ② 黑线跟随 (标准模式)
  ③ 丢线恢复 (历史帧方向旋转)

线程分配 (RPi4 四核):
  Core 0 — Flask Web 服务
  Core 1 — 巡线视觉处理 + 电机控制 (processing_loop)
  Core 2 — 相机采集 (capture_loop)
  Core 3 — ONNX 推理 + 反应触发 (inference_loop)
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
        # m00 = total white pixels/colour blob kept
        # m10 = total pixel value in x axis
        # m01 = total pixel value in y axis
        return cx, cy, sat_mask
  
    return None, None, sat_mask

# ═══════════════════════════════════════════════════════════════
#  Line Recovery
# pixel analysation when line is lost, guess on where line should be
# binary image is flipped so the black line represented in white pixels
# count number of white pixels in each left and right side and make comparison
# ═══════════════════════════════════════════════════════════════

# store captured frame in the latest second
def add_frame_to_history(binary_image, timestamp):
    with history_lock:
        frame_history.append((timestamp, binary_image.copy()))
        cutoff = timestamp - HISTORY_DURATION
        while frame_history and frame_history[0][0] < cutoff: frame_history.popleft()

# analyse frame stored
# separate frame into left and right side from the middle
# count and compare number of white pixels on each side ( binary image fed is inverted )
# return left/right depending on whichever side has more white pixels
def analyze_pixel_distribution():
    with history_lock:
        if not frame_history: return None, 0.0
        l_px = r_px = 0
        w = frame_history[0][1].shape[1]
        for _, b in frame_history:
            l_px += cv2.countNonZero(b[:, :w//2]) # left side
            r_px += cv2.countNonZero(b[:, w//2:]) # right side
        total = l_px + r_px # total
        if total == 0: return None, 0.0
        r_ratio = r_px / total
        if r_ratio > PIXEL_RATIO_THRESHOLD: return "right", r_ratio - 0.5
        if r_ratio < (1.0 - PIXEL_RATIO_THRESHOLD): return "left", 0.5 - r_ratio
        return None, 0.0

# trigger line recovery when line is lost
def start_line_recovery():
    with recovery_lock:
        if not line_lost_state["is_recovering"]:
            dir_rec, conf = analyze_pixel_distribution()
            line_lost_state.update({
                "is_recovering": True,
                "line_lost_time": time.time(),
                "recovery_direction": dir_rec
            })

# stop line recovery when line is found
def stop_line_recovery():
    with recovery_lock: line_lost_state["is_recovering"] = False

# obtain current recovery status
def get_recovery_state():
    with recovery_lock:
        if not line_lost_state["is_recovering"]: return False, None, False
        elapsed = time.time() - line_lost_state["line_lost_time"]
        if elapsed > RECOVERY_TIMEOUT:
            line_lost_state["is_recovering"] = False
            return False, None, False
        return True, line_lost_state["recovery_direction"], True

# ═══════════════════════════════════════════════════════════════
#  Reaction System, how should vehicle react to detected symbols
# ═══════════════════════════════════════════════════════════════

def _set_action(desc):
    with symbol_lock: symbol_state["action"] = desc # update status for debug

def _is_in_cooldown(label):
    return (time.time() - _action_cooldown.get(label.lower(), 0)) < ACTION_COOLDOWN_SEC # cooldown 

def _mark_cooldown(label):
    _action_cooldown[label.lower()] = time.time()

def execute_action(label):
    key = label.lower()

    # 1. Thumb/QR, show 
    if key in ("thumb", "qrcode"):
        _set_action(f"flash_{key}")
        with symbol_lock: symbol_state["flash"] = True
        time.sleep(2.5)
        with symbol_lock: symbol_state["flash"] = False
        _set_action("none")
        _mark_cooldown(label)
        return

    # 2. Button/Warning, stop
    if key in ("buttonsign", "warningsign"):
        with action_lock:
            action_state["permanent_stop"] = True
            action_state["running"] = False
        _set_action(f"{label} — STOPPED")
        stop_motors()
        print(f"[ACTION] Permanent stop triggered by {label}")
        return 

    # 3. Arrow/Recycle
    with action_lock:
        if not action_state["running"]:
            action_state["running"] = True

    try:
        stop_motors()  # reset from past reactions
        if key == "arrowup":
            _set_action("ArrowUp — Straight")
            set_motors(BASE_SPEED, BASE_SPEED); time.sleep(1.2) # travel forward for 1.2s

        elif key in ("arrowleft", "arrowright"):
            direction = "left" if key == "arrowleft" else "right"
            _set_action(f"Arrow{direction.capitalize()} — Searching line...")

            # 1. forward for a small distance before rotating
            stop_motors()
            set_motors(BASE_SPEED, BASE_SPEED); time.sleep(0.8)

            # 2. left/right rotation for 0.5s towards intended pathway
            spin_l = -SPIN_SPEED if direction == "left" else  SPIN_SPEED
            spin_r =  SPIN_SPEED if direction == "left" else -SPIN_SPEED
            set_motors(spin_l, spin_r); time.sleep(0.5)

            # 3. begin to search for line and continue line following
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

      # recycle sign, rotate 360 degrees
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

# continuous feed from picamera
def capture_loop():
    global latest_cam_frame
    while True:
        try:
            frame = picam2.capture_array()
            if FLIP_MODE is not None: frame = cv2.flip(frame, FLIP_MODE)
            with frame_lock: latest_cam_frame = frame
        except: time.sleep(0.01)

# processing loop
def processing_loop():
    global latest_raw_frame
    f_cnt, t_start, cur_fps = 0, time.time(), 0.0

    while True:
        # acquire action lock
        with action_lock:
            is_perm = action_state["permanent_stop"]
            is_run  = action_state["running"]
          
        # get latest feed from camera
        with frame_lock: frame = latest_cam_frame
        if frame is None: time.sleep(0.01); continue # pause for 10ms if no frame captured

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

        # ── default configuration and status ──────────────────────────────────
        l_spd = r_spd = 0.0
        stat = drct = "no_line"
        err = 0.0
        rec_s = "inactive"; rec_r = ""
        cx = cy = -1
        b_ratio = 0.0
        blob_mode = False

        # ══════════════════════════════════════════════════════
        #  Colour blob detection and follow - first priority
        # ══════════════════════════════════════════════════════
        blob_cx, blob_cy, sat_mask = _detect_color_blob(frame)

        if blob_cx is not None:
            blob_mode = True
            stop_line_recovery()  # stop line recovery when blob within vision

            cx, cy = blob_cx, blob_cy
            err = (cx - w / 2) / (w / 2) # calculate error from colour blob
            abs_err = abs(err)

            # 0% ~ 5% of error：dead zone, proceed forward
            if abs_err <= DEAD_ZONE_PCT:
                stat = "blob_straight"; drct = "straight"
                l_spd = r_spd = BASE_SPEED
              
            # 5% ~ 40% of error：increased speed on one side of motor, the other motor speed remain constant
            elif abs_err <= SPIN_ZONE_PCT:
                k = (abs_err - DEAD_ZONE_PCT) / (SPIN_ZONE_PCT - DEAD_ZONE_PCT)
                outer = BASE_SPEED + k * (MAX_SPEED - BASE_SPEED)
                if err > 0: # slight off towards left, increase speed on right motor
                    drct = "right"; stat = "blob_adjust_right"
                    l_spd = outer;      r_spd = BASE_SPEED
                else: # slight off towards right, increase speed on left motor
                    drct = "left";  stat = "blob_adjust_left"
                    l_spd = BASE_SPEED; r_spd = outer
            else:# 40% ~ 100%：spin zone, clockwise/anti clockwise rotation 
                if err > 0:
                    drct = stat = "blob_spin_right"
                    l_spd =  SPIN_SPEED; r_spd = -SPIN_SPEED
                else:
                    drct = stat = "blob_spin_left"
                    l_spd = -SPIN_SPEED; r_spd =  SPIN_SPEED

            set_motors(l_spd, r_spd)

        else:
            # ══════════════════════════════════════════════════
            #  black line following - second priority
            # ══════════════════════════════════════════════════
          # image processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # BGR > grayscale
            _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255,
                                      cv2.THRESH_BINARY_INV) # binary conversion + invert to use moments and compute center of mass
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) # morphological cleanup
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
  
            moments  = cv2.moments(binary)
            b_ratio  = cv2.countNonZero(binary) / (h * w)
            has_line = moments["m00"] > LINE_LOST_THRESHOLD # number of white pixel is less than threshold -> line considered lost
            add_frame_to_history(binary, time.time())

            if has_line:
                stop_line_recovery()
              # calculate center of mass
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                err = (cx - w / 2) / (w / 2)
                abs_err = abs(err)
              
                # same line following logic as colour blob for black lines
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
                # ══════════════════════════════════════════════
                #  line recovery - third priority
                # ══════════════════════════════════════════════
                if RECOVERY_ENABLED:
                    is_r, d, cont = get_recovery_state()
                    if not is_r:
                        start_line_recovery()
                        is_r, d, cont = get_recovery_state()
                    if cont and d:
                        rec_s = f"recovery_{d}"
                        l_spd, r_spd = (
                            (RECOVERY_SPEED, -RECOVERY_SPEED) if d == "right" # rotate on the spot to search for line
                            else (-RECOVERY_SPEED, RECOVERY_SPEED)
                        )
                        set_motors(l_spd, r_spd)
                    else:
                        stop_line_recovery()
                        stop_motors()
                else:
                    stop_motors()

            # draw and display middle vertical line, circle for center of mass, another horizontal line between circle and vertical line
            disp = frame.copy()
            cv2.line(disp, (w // 2, 0), (w // 2, h - 1), (255, 0, 0), 1)
            if has_line and cx != -1:
                cv2.circle(disp, (cx, cy), 10, (0, 0, 255), -1)
                cv2.line(disp, (w // 2, cy), (cx, cy), (0, 255, 0), 2)

        # ── FPS calculation ────────────────────────────────────────────
        f_cnt += 1
        if time.time() - t_start >= 1.0:
            cur_fps = f_cnt / (time.time() - t_start)
            f_cnt = 0; t_start = time.time()

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

        with frame_lock:
            latest_raw_frame = disp

def inference_loop():
    while True:
        with frame_lock: f = latest_cam_frame
        if f is None or _session is None: time.sleep(0.05); continue

        label, conf = _onnx_classify(f)
        with symbol_lock: symbol_state.update({"shape": label, "confidence": conf})

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
    return """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>67 Car</title>
    <style>
        body { margin:0; background:#000; color:#fff; font-family:monospace; display:flex; flex-direction:column; align-items:center; padding:16px; }
        img { width:100%; max-width:640px; border-radius:8px; border: 2px solid #333; }
        #result { margin-top:15px; text-align:center; }
        #shape { font-size:32px; font-weight:bold; color: #00d2ff; }
        #action { font-size:18px; color:#ff9f43; margin-top:8px; }
        #blob_badge { margin-top:6px; font-size:14px; color:#00ffcc; opacity:0; transition:0.3s; }
        #alert_msg { margin-top:10px; font-size:20px; color:#4cd137; opacity:0; transition:0.3s; }
        .flashing { animation: flash 0.25s 6; }
        @keyframes flash { 0%,100% {opacity:1;} 50% {opacity:0.2;} }
    </style></head><body>
    <img src="/video">
    <div id="result">
        <div id="shape">—</div>
        <div id="conf"></div>
        <div id="action"></div>
        <div id="blob_badge">● BLOB MODE</div>
        <div id="alert_msg">Detected!</div>
    </div>
    <script>
        let prev_flash = false;
        setInterval(async () => {
            const sym = await fetch('/api/symbol').then(r => r.json());
            document.getElementById('shape').textContent = sym.shape;
            document.getElementById('conf').textContent = sym.confidence.toFixed(1) + '%';
            document.getElementById('action').textContent = sym.action !== 'none' ? sym.action : '';

            const alert = document.getElementById('alert_msg');
            if(sym.flash && !prev_flash) {
                alert.textContent = sym.shape + " DETECTED!";
                alert.style.opacity = 1;
                document.getElementById('shape').classList.add('flashing');
                setTimeout(() => {
                    alert.style.opacity = 0;
                    document.getElementById('shape').classList.remove('flashing');
                }, 3000);
            }
            prev_flash = sym.flash;

            const st = await fetch('/api/state').then(r => r.json());
            const badge = document.getElementById('blob_badge');
            badge.style.opacity = st.blob_mode ? 1 : 0;
        }, 300);
    </script></body></html>"""

@app.route('/video')
def video():
    def gen():
        while True:
            with frame_lock: f = latest_raw_frame
            if f is not None:
                _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
            time.sleep(0.04)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/symbol')
def api_symbol():
    with symbol_lock: return jsonify(dict(symbol_state))

@app.route('/api/state')
def api_state():
    with state_lock: return jsonify(dict(robot_state))

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

  # assign different task to each core in raspberry pi
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
