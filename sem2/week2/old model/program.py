"""
UNMC Parking System — Raspberry Pi 4 Edition
=============================================
Hardware map (BCM numbering):
  Camera  : OV5647 via CSI FPC ribbon  →  picamera2 (native, no IP needed)
  OLED    : SSD1306 128×64  I2C        →  SDA=GPIO2  (Pin 3)
                                           SCL=GPIO3  (Pin 5)
  Servo   : SG90 gate       PWM        →  GPIO18     (Pin 12) [hardware PWM0]
  Parking : 4× square-wave sensors     →  GPIO17 (Pin11) slot0
                                           GPIO27 (Pin13) slot1
                                           GPIO22 (Pin15) slot2
                                           GPIO23 (Pin16) slot3

Install dependencies once:
  sudo apt install -y python3-picamera2 pigpio python3-pigpio
  sudo systemctl enable pigpiod && sudo systemctl start pigpiod
  pip install flask onnxruntime opencv-python-headless luma.oled pillow
"""

import cv2
import threading
import time
import numpy as np
from flask import Flask, Response, jsonify, render_template
from datetime import datetime
import onnxruntime as ort

from picamera2 import Picamera2

import pigpio

from luma.core.interface.serial import i2c as luma_i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import ImageFont

# =============================================================================
# Configuration
# =============================================================================

CAM_WIDTH        = 640
CAM_HEIGHT       = 480
CAM_FPS          = 30
CAM_FLIP         = 0
CAM_ROTATE       = cv2.ROTATE_90_CLOCKWISE

VEHICLE_CLASSES  = [2, 3, 5, 7]   # car, motorcycle, bus, truck
YOLO_IMGSZ       = 640
YOLO_INTERVAL_S  = 0.2
STREAM_QUALITY   = 60

GATE_CONFIRM_S   = 1.0
GATE_OPEN_S      = 5.0

SERVO_GPIO       = 18
SERVO_OPEN_US    = 1500
SERVO_CLOSE_US   = 500

PARK_GPIOS       = [17, 27, 22, 23]
PARK_COUNT       = 4
FREQ_SAMPLE_S    = 0.5
FREQ_THRESHOLD   = 100.0
BASELINE_ALPHA   = 0.15

OLED_I2C_PORT    = 1
OLED_I2C_ADDR    = 0x3C
OLED_WIDTH       = 128
OLED_HEIGHT      = 64

# =============================================================================
# Shared state
# =============================================================================

lock            = threading.Lock()
raw_frame       = None
processed_frame = None

system_state = {
    "parking":    [False] * PARK_COUNT,
    "freq":       [0.0]   * PARK_COUNT,
    "servo":      "Closed",
    "cam_online": False,
    "sub_online": True,
    "last_seen":  "Never",
}

# Gate FSM — only touched by yolo_runner thread
_gate_state        = "IDLE"
_gate_detect_since = None
_gate_opened_at    = None

# pigpio
pi = None

# Parking counters — written by pigpio callbacks, read by sampler thread
_pulse_count   = [0] * PARK_COUNT
_pulse_lock    = threading.Lock()
_current_freq  = [0.0]  * PARK_COUNT
_baseline_freq = [-1.0] * PARK_COUNT  # -1 = uninitialised

# Slot occupied — ONLY set by web API, never by GPIO sampler
_slot_occupied = [False] * PARK_COUNT

# OLED
oled_dev       = None
_oled_override = ""
_oled_until    = 0.0

# =============================================================================
# pigpio rising-edge callbacks
# =============================================================================

def _make_cb(idx):
    def _cb(gpio, level, tick):
        with _pulse_lock:
            _pulse_count[idx] += 1
    return _cb

# =============================================================================
# Servo
# =============================================================================

def servo_set_open():
    if pi and pi.connected:
        pi.set_servo_pulsewidth(SERVO_GPIO, SERVO_OPEN_US)
    print("[SERVO] Open")
    _oled_flash("WELCOME")

def servo_set_close():
    if pi and pi.connected:
        pi.set_servo_pulsewidth(SERVO_GPIO, SERVO_CLOSE_US)
    print("[SERVO] Closed")

# =============================================================================
# OLED
# =============================================================================

def _oled_flash(msg: str):
    global _oled_override, _oled_until
    _oled_override = msg
    _oled_until    = time.time() + GATE_OPEN_S + 0.5

def _try_font(size: int):
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()

def oled_loop():
    global oled_dev
    try:
        serial   = luma_i2c(port=OLED_I2C_PORT, address=OLED_I2C_ADDR)
        oled_dev = ssd1306(serial, width=OLED_WIDTH, height=OLED_HEIGHT)
        print("[OLED] Ready")
    except Exception as e:
        print(f"[OLED] Init failed: {e}")
        return

    f_big = _try_font(18)
    f_sm  = _try_font(10)

    while True:
        try:
            now_t = time.time()
            with canvas(oled_dev) as draw:
                if now_t < _oled_until and _oled_override:
                    msg  = _oled_override
                    bbox = draw.textbbox((0, 0), msg, font=f_big)
                    w    = bbox[2] - bbox[0]
                    draw.text(((OLED_WIDTH - w) // 2, 20), msg,
                              fill="white", font=f_big)
                else:
                    dt       = datetime.now()
                    time_str = dt.strftime("%H:%M:%S")
                    date_str = dt.strftime("%d %b %Y")

                    draw.text((10, 2),  "UNMC PARKING", fill="white", font=f_sm)
                    draw.line([(0, 13), (127, 13)], fill="white")
                    draw.text((8,  16), time_str,    fill="white", font=f_big)
                    draw.line([(0, 38), (127, 38)], fill="white")
                    draw.text((22, 42), date_str,    fill="white", font=f_sm)

                    # Slot status — follows _slot_occupied (web-controlled)
                    slot_str = "  ".join(
                        "X" if _slot_occupied[i] else "O"
                        for i in range(PARK_COUNT)
                    )
                    draw.text((4, 54), slot_str, fill="white", font=f_sm)
        except Exception as e:
            print(f"[OLED] Draw error: {e}")

        time.sleep(1.0)

# =============================================================================
# Parking sampler — freq measurement ONLY, never touches _slot_occupied
# =============================================================================

def parking_sampler():
    while True:
        time.sleep(FREQ_SAMPLE_S)

        with _pulse_lock:
            counts = list(_pulse_count)
            for i in range(PARK_COUNT):
                _pulse_count[i] = 0

        for i in range(PARK_COUNT):
            freq = counts[i] / FREQ_SAMPLE_S
            _current_freq[i] = freq

            if _baseline_freq[i] < 0:
                _baseline_freq[i] = freq
                continue

            delta = freq - _baseline_freq[i]
            if abs(delta) >= FREQ_THRESHOLD:
                _baseline_freq[i] = freq
            else:
                _baseline_freq[i] += BASELINE_ALPHA * delta

        # Only update freq in system_state — parking state is web-controlled
        with lock:
            system_state["freq"] = [round(f, 1) for f in _current_freq]

# =============================================================================
# YOLO — ONNX Runtime
# =============================================================================

_COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]

print("[YOLO] Loading yolov8n.onnx ...")
_sess_opts = ort.SessionOptions()
_sess_opts.intra_op_num_threads = 4
_sess_opts.inter_op_num_threads = 2
_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_session = ort.InferenceSession(
    "yolov8n.onnx",
    sess_options=_sess_opts,
    providers=["CPUExecutionProvider"]
)
_input_name  = _session.get_inputs()[0].name
_CONF_THRESH = 0.20
print("[YOLO] ONNX Runtime ready")


def _preprocess(bgr: np.ndarray):
    img = cv2.resize(bgr, (YOLO_IMGSZ, YOLO_IMGSZ))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0)


def _postprocess(outputs, orig_h, orig_w):
    preds    = outputs[0][0].T                        # [8400, 84]
    boxes    = preds[:, :4]
    cls_prob = preds[:, 4:]
    cls_ids  = np.argmax(cls_prob, axis=1)
    confs    = cls_prob[np.arange(len(cls_ids)), cls_ids]

    mask = (confs >= _CONF_THRESH) & np.isin(cls_ids, VEHICLE_CLASSES)
    boxes, confs, cls_ids = boxes[mask], confs[mask], cls_ids[mask]

    sx = orig_w / YOLO_IMGSZ
    sy = orig_h / YOLO_IMGSZ
    x1 = ((boxes[:, 0] - boxes[:, 2] / 2) * sx).astype(int)
    y1 = ((boxes[:, 1] - boxes[:, 3] / 2) * sy).astype(int)
    x2 = ((boxes[:, 0] + boxes[:, 2] / 2) * sx).astype(int)
    y2 = ((boxes[:, 1] + boxes[:, 3] / 2) * sy).astype(int)
    return list(zip(x1, y1, x2, y2, confs, cls_ids))


def run_cv_pipeline(frame: np.ndarray):
    if frame is None:
        return None, False
    bgr = frame if (frame.ndim == 3 and frame.shape[2] == 3) \
          else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    orig_h, orig_w = bgr.shape[:2]
    outputs    = _session.run(None, {_input_name: _preprocess(bgr)})
    detections = _postprocess(outputs, orig_h, orig_w)
    vehicle_detected = False
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        label = f"{_COCO_NAMES[cls_id]} {conf:.2f}"
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(bgr, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vehicle_detected = True
    return bgr, vehicle_detected

# =============================================================================
# Gate FSM
# =============================================================================

def _gate_fsm_tick(vehicle_detected: bool):
    global _gate_state, _gate_detect_since, _gate_opened_at
    now = time.time()

    if _gate_state == "IDLE":
        if vehicle_detected:
            _gate_state        = "CONFIRMING"
            _gate_detect_since = now
            print("[GATE] Vehicle detected — confirming...")

    elif _gate_state == "CONFIRMING":
        if vehicle_detected:
            if now - _gate_detect_since >= GATE_CONFIRM_S:
                _gate_state     = "OPEN"
                _gate_opened_at = now
                servo_set_open()
                print("[GATE] Confirmed → OPEN")
        else:
            _gate_state        = "IDLE"
            _gate_detect_since = None
            print("[GATE] Lost before confirm — back to IDLE")

    elif _gate_state == "OPEN":
        if now - _gate_opened_at >= GATE_OPEN_S:
            _gate_state        = "IDLE"
            _gate_detect_since = None
            _gate_opened_at    = None
            servo_set_close()
            print("[GATE] Cooldown done — back to IDLE")

    with lock:
        system_state["servo"] = "Open" if _gate_state == "OPEN" else "Closed"


def trigger_gate():
    global _gate_state, _gate_opened_at, _gate_detect_since
    _gate_state        = "OPEN"
    _gate_opened_at    = time.time()
    _gate_detect_since = None
    servo_set_open()
    with lock:
        system_state["servo"] = "Open"
    print("[GATE] Manual trigger")

# =============================================================================
# Camera grabber + YOLO runner
# =============================================================================

_latest_raw  = None
_latest_lock = threading.Lock()

def camera_grabber():
    global raw_frame, _latest_raw
    picam    = Picamera2()
    frame_us = int(1_000_000 / CAM_FPS)
    config   = picam.create_video_configuration(
        main={"size": (CAM_WIDTH, CAM_HEIGHT), "format": "BGR888"},
        controls={"FrameDurationLimits": (frame_us, frame_us)},
    )
    picam.configure(config)
    picam.start()
    time.sleep(2.0)
    print(f"[CAM] OV5647 started {CAM_WIDTH}x{CAM_HEIGHT} @ {CAM_FPS} fps")

    while True:
        try:
            frame = picam.capture_array()
            if frame is not None:
                if CAM_FLIP is not None:
                    frame = cv2.flip(frame, CAM_FLIP)
                if CAM_ROTATE is not None:
                    frame = cv2.rotate(frame, CAM_ROTATE)
                frame = frame.copy()
                with lock:
                    raw_frame = frame
                    system_state["cam_online"] = True
                    system_state["last_seen"]  = datetime.now().strftime("%H:%M:%S")
                with _latest_lock:
                    _latest_raw = frame
        except Exception as e:
            print(f"[CAM] Error: {e}")
            with lock:
                system_state["cam_online"] = False
            time.sleep(2.0)


def yolo_runner():
    global processed_frame
    last_run = 0.0
    while True:
        now = time.time()
        if now - last_run < YOLO_INTERVAL_S:
            time.sleep(0.01)
            continue
        last_run = now
        with _latest_lock:
            frame = _latest_raw
        if frame is None:
            time.sleep(0.05)
            continue
        try:
            proc, vehicle_detected = run_cv_pipeline(frame)
            _gate_fsm_tick(vehicle_detected)
            with lock:
                processed_frame = (proc if proc is not None else frame).copy()
        except Exception as e:
            print(f"[YOLO] Error: {e}")

# =============================================================================
# MJPEG stream
# =============================================================================

def stream_generator(feed: str):
    while True:
        with lock:
            f = raw_frame if feed == "raw" else processed_frame
        if f is not None:
            ok, jpeg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, STREAM_QUALITY])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + jpeg.tobytes()
                    + b"\r\n"
                )
        time.sleep(0.033)

# =============================================================================
# Flask
# =============================================================================

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video/raw")
def video_raw():
    return Response(stream_generator("raw"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video/processed")
def video_processed():
    return Response(stream_generator("processed"),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    with lock:
        state = dict(system_state)
    return jsonify(state)

@app.route("/api/gate/<gate_type>")
def api_gate(gate_type):
    trigger_gate()
    return jsonify({"ok": True})

@app.route("/api/parking/<int:slot>/<int:state>")
def api_parking_set(slot, state):
    if 0 <= slot < PARK_COUNT:
        occupied = bool(state)
        _slot_occupied[slot] = occupied
        with lock:
            system_state["parking"] = list(_slot_occupied)
        _oled_flash(f"A{slot+1} {'OCC' if occupied else 'FREE'}")
        print(f"[PARK] Slot {slot+1} manually → {'OCCUPIED' if occupied else 'VACANT'}")
    return jsonify({"ok": True})

# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    pi = pigpio.pi()
    if not pi.connected:
        print("[pigpio] WARNING: pigpiod not running — GPIO disabled")
        print("         Fix: sudo systemctl start pigpiod")
        pi = None
    else:
        print("[pigpio] Connected")
        pi.set_servo_pulsewidth(SERVO_GPIO, SERVO_CLOSE_US)
        for idx, gpio in enumerate(PARK_GPIOS):
            pi.set_mode(gpio, pigpio.INPUT)
            pi.set_pull_up_down(gpio, pigpio.PUD_DOWN)
            pi.callback(gpio, pigpio.RISING_EDGE, _make_cb(idx))
        print(f"[PARK] Monitoring BCM GPIOs {PARK_GPIOS}")

    threading.Thread(target=oled_loop,       daemon=True).start()
    threading.Thread(target=parking_sampler, daemon=True).start()
    threading.Thread(target=camera_grabber,  daemon=True).start()
    threading.Thread(target=yolo_runner,     daemon=True).start()

    print("[Flask] http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
