"""
symbol_detector.py
use: python3 symbol_detector.py
Dashboard: http://<RPi_IP>:5001

relies on: picamera2, opencv-python, flask, numpy, onnxruntime
installation: pip install onnxruntime --break-system-packages
"""

# import libraries
import time, sys, threading, os
from flask import Flask, Response, jsonify
import cv2, numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    print("error: sudo apt install python3-picamera2"); sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("error: pip install onnxruntime --break-system-packages"); sys.exit(1)

# ═══════════════════════════════════════════════════════════════
#  Configuration, parameters
# ═══════════════════════════════════════════════════════════════

RESOLUTION         = (640, 480) # 640 x 480p, highest for 30fps
JPEG_QUALITY       = 60
FLIP_MODE          = -1

ONNX_MODEL_PATH    = "model.onnx"
TFLITE_LABELS_PATH = "labels.txt"
ONNX_CONF_THRESH   = 0.6
ONNX_INPUT_SIZE    = 224

SHAPE_MIN_AREA     = 800
SHAPE_AREA_RATIO   = 0.02
COOLDOWN_SEC       = 5.0

PORT               = 5001

# ═══════════════════════════════════════════════════════════════
#  setup camera
# ═══════════════════════════════════════════════════════════════

print("camera startup...")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "BGR888"}, # capture frame in bgr format
    controls={"FrameDurationLimits": (16666, 16666)}))
picam2.start(); time.sleep(1)
print(f"camera ready  {RESOLUTION}")

# ═══════════════════════════════════════════════════════════════
#  ONNX Runtime 
# ═══════════════════════════════════════════════════════════════

_session    = None
_labels     = []
_input_name = None

def _load_onnx():
    """load onnx model resource file"""
    global _session, _labels, _input_name

    # load labels file
    if not os.path.exists(TFLITE_LABELS_PATH):
        print(f"[ONNX] unable to load labels file: {TFLITE_LABELS_PATH}"); return False
    with open(TFLITE_LABELS_PATH) as f:
        _labels = [l.strip().split(None, 1)[-1] for l in f if l.strip()]
    print(f"[ONNX] 标签: {_labels}")

    # load ONNX model
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"[ONNX] unable to locate model file: {ONNX_MODEL_PATH}"); return False
    try:
        # only CPU on raspberry pi, load CPUProvider
        providers = ["CPUExecutionProvider"]
        _session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
        _input_name = _session.get_inputs()[0].name
        input_shape = _session.get_inputs()[0].shape
        print(f"[ONNX] model import success  input={_input_name}  shape={input_shape}")
        print(f"[ONNX] onnxruntime {ort.__version__}")
        return True
    except Exception as e:
        print(f"[ONNX] failed to load model: {e}")
        _session = None
        return False

def _onnx_classify(frame_bgr):
    """use ONNX Runtime to identify symbol/shape，return (label, confidence)。"""
    if _session is None:
        return "unknown", 0.0

    # resize → RGB → float32 → [0,1] → NCHW
    # float 32 conversion is nesessary since calculation will involve decimal
    # framework expects NCHW, number of image, channels, height, width
    resized = cv2.resize(frame_bgr, (ONNX_INPUT_SIZE, ONNX_INPUT_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_f   = rgb.astype(np.float32) / 255.0          # [0, 1]
    # NCHW: (1, 3, H, W), 1 image at a time, 3 channels R G B
    blob    = np.transpose(img_f, (2, 0, 1))[np.newaxis, ...]

    try:
        outputs = _session.run(None, {_input_name: blob})
        scores  = outputs[0][0].astype(np.float32)    # (num_classes,)
    except Exception as e:
        print(f"[ONNX] 推理失败: {e}")
        return "unknown", 0.0

    # Softmax
    e     = np.exp(scores - scores.max())
    probs = e / e.sum()

    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    label = _labels[idx] if idx < len(_labels) else f"class{idx}"
    return label, conf

# ═══════════════════════════════════════════════════════════════
#  Contour detection
# ═══════════════════════════════════════════════════════════════

# capture frame in bgr, convert into grayscale and into binary
def _get_contour(frame_bgr):
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 8)
    # filtering, canny edge detection, morphological cleaning, find contours
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 90)
    edges = cv2.dilate(edges,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                       iterations=2)
    cnts_e, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    # fill contours larger than 200px
    if cv2.contourArea(c) > 200:
        # Draw contour filled solid white
        cv2.drawContours(filled, [c], -1, 255, -1)

    # Merge adaptive threshold, filled contours to get the best of both
    fg = cv2.bitwise_or(adapt, filled)

    # Remove small noise
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))

    # Find only outermost contours in the cleaned binary image
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours too small to be a valid shape
    valid = [c for c in cnts if cv2.contourArea(c) >= SHAPE_MIN_AREA]

    # No valid shape found — return empty result
    if not valid:
        return None, None, 0.0, 0.0

    # Pick the largest contour as the primary shape candidate
    best  = max(valid, key=cv2.contourArea)
    area  = cv2.contourArea(best)
    perim = cv2.arcLength(best, True)

    # Circularity score: perfect circle = 1.0, square ≈ 0.785, irregular < 0.5
    circ = min(1.0, (4 * np.pi * area) / (perim ** 2)) if perim > 0 else 0.0

    # Simplify contour into polygon — len(approx) gives corner count
    approx = cv2.approxPolyDP(best, 0.02 * perim, True)

    # Fraction of total frame the shape occupies (0.0 - 1.0)
    # Small ratio = shape is far, large ratio = shape is close
    ratio = area / (frame_bgr.shape[0] * frame_bgr.shape[1])

    return best, approx, circ, ratio

def _contour_classify(best_c, approx, circ):
    vl = len(approx) if approx is not None else 0
    if   circ > 0.82: return "circle"
    elif vl == 3:      return "triangle"
    elif vl == 4:
        _, (rw, rh), _ = cv2.minAreaRect(best_c)
        return "square" if max(rw, rh) / (min(rw, rh) + 1e-5) < 1.2 else "rectangle"
    elif vl == 5:      return "pentagon"
    elif vl == 6:      return "hexagon"
    else:
        ha = cv2.contourArea(cv2.convexHull(best_c))
        return "star" if cv2.contourArea(best_c) / (ha + 1e-5) < 0.75 else "polygon"

# ═══════════════════════════════════════════════════════════════
# thread lock to handle shared camera access for ONNX and contour detection
# ═══════════════════════════════════════════════════════════════

state_lock = threading.Lock()
det_state  = {
    "status":      "scanning",
    "shape_name":  "",
    "shape_conf":  "",
    "shape_score": 0.0,
    "mode":        "contour",
    "fps":         0.0,
    "resolution":  f"{RESOLUTION[0]}x{RESOLUTION[1]}",
    "history":     [],
}

frame_lock   = threading.Lock()
latest_frame = None
latest_debug = None

_live = {
    "contour": None, "approx": None, "circ": 0.0,
    "area_ratio": 0.0, "last_trigger": 0.0,
    "identifying": False,
}

# ═══════════════════════════════════════════════════════════════
#  identify shapes and symbol
# ═══════════════════════════════════════════════════════════════

def _identify_bg(frame_bgr, contour, approx, circ):
    onnx_label, conf = _onnx_classify(frame_bgr)

    name   = onnx_label
    detail = f"onnx | conf={conf:.2f}"
    score  = conf
    mode   = "onnx"

    ts = time.strftime("%H:%M:%S")
    print(f"[Symbol] {name}  ({detail})")

    with state_lock:
        det_state["status"]      = "detected"
        det_state["shape_name"]  = name
        det_state["shape_conf"]  = detail
        det_state["shape_score"] = round(score * 100, 1)
        det_state["mode"]        = mode
        det_state["history"].append({
            "time": ts, "shape": name,
            "conf": detail, "score": round(score * 100, 1)
        })
        if len(det_state["history"]) > 20:
            det_state["history"].pop(0)

    _live["identifying"] = False

# ═══════════════════════════════════════════════════════════════
#  processing loop
# ═══════════════════════════════════════════════════════════════

def _put_text(img, text, pos, scale=0.46, color=(0, 255, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   1, cv2.LINE_AA)

def processing_loop():
    global latest_frame, latest_debug
    fps_c, fps_t, fps = 0, time.time(), 0.0

    while True:
        try:
            # capture frame
            now   = time.time()
            frame = picam2.capture_array()
            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE) # flip frame if upside down
            h, w = frame.shape[:2]

            # call function contour detection, and return variables
            best, approx, circ, ratio = _get_contour(frame)
            _live["contour"]    = best
            _live["approx"]     = approx
            _live["circ"]       = circ
            _live["area_ratio"] = ratio

            in_cooldown = (now - _live["last_trigger"]) < COOLDOWN_SEC
            triggered   = (best is not None) and (ratio >= SHAPE_AREA_RATIO)

            # print current status for debugging
            if triggered and not in_cooldown and not _live["identifying"]:
                _live["last_trigger"] = now
                _live["identifying"]  = True
                with state_lock:
                    det_state["status"]     = "identifying"
                    det_state["shape_name"] = "..."

                threading.Thread(
                    target=_identify_bg,
                    args=(frame.copy(), best, approx, circ),
                    daemon=True).start()

            fps_c += 1
            if now - fps_t >= 1.0:
                fps = fps_c / (now - fps_t); fps_c = 0; fps_t = now
            with state_lock:
                det_state["fps"] = round(fps, 1)

            # overlay, display footage and drawed contours for debugging, all for flask server
            disp = frame.copy()

            if best is not None:
                cv2.drawContours(disp, [best], -1, (255, 255, 255), 2)
                x, y, bw, bh = cv2.boundingRect(best)
                cv2.rectangle(disp, (x, y), (x+bw, y+bh), (255, 255, 255), 1)
                if approx is not None:
                    for pt in approx:
                        px, py = pt[0]
                        cv2.circle(disp, (px, py), 5, (255, 255, 255), -1)
                        cv2.circle(disp, (px, py), 3, (0, 0, 0), -1)
                M = cv2.moments(best)
                if M["m00"] > 0:
                    mcx = int(M["m10"] / M["m00"])
                    mcy = int(M["m01"] / M["m00"])
                    cv2.line(disp, (mcx-10, mcy), (mcx+10, mcy), (255,255,255), 1)
                    cv2.line(disp, (mcx, mcy-10), (mcx, mcy+10), (255,255,255), 1)

            with state_lock:
                s_name  = det_state["shape_name"]
                s_conf  = det_state["shape_conf"]
                s_score = det_state["shape_score"]
                s_stat  = det_state["status"]
                s_mode  = det_state["mode"]

            panel_w = 190
            ov = disp.copy()
            cv2.rectangle(ov, (0, 0), (panel_w, h), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.45, disp, 0.55, 0, disp)
            cv2.line(disp, (panel_w, 0), (panel_w, h), (80, 80, 80), 1)

            stat_col = {"scanning": (160,160,160), "identifying": (80,200,200),
                        "detected": (80,255,80)}.get(s_stat, (160,160,160))
            mode_col = (100,200,255) if s_mode == "onnx" else (200,200,200)

            info = [
                ("SYMBOL DETECT",  (200,200,200), 0.44, True),
                (None, None, 0, False),
                (f"Mode  : {s_mode.upper()}", mode_col, 0.38, False),
                (f"State : {s_stat.upper()}", stat_col,  0.42, False),
                (None, None, 0, False),
                (f"Shape : {s_name.upper() if s_name else '---'}", (255,255,255), 0.48, False),
                (f"Score : {s_score:.0f}%", (200,200,200), 0.42, False),
                (None, None, 0, False),
            ]
            if best is not None:
                verts = len(approx) if approx is not None else 0
                info += [
                    (f"Corners:{verts}",         (200,200,200), 0.42, False),
                    (f"Circ.  :{circ:.2f}",       (200,200,200), 0.42, False),
                    (f"Area   :{ratio*100:.1f}%", (200,200,200), 0.42, False),
                ]
            info += [(None, None, 0, False), ("Detail:", (160,160,160), 0.38, False)]
            if s_conf:
                for p in [s_conf[i:i+22] for i in range(0, min(len(s_conf), 66), 22)]:
                    info.append((f"  {p}", (160,160,160), 0.36, False))
            info += [
                (None, None, 0, False),
                (f"FPS : {fps:.1f}", (160,160,160), 0.38, False),
            ]

            y_cur = 18
            for item in info:
                txt, col, sc, bold = item
                if txt is None: y_cur += 6; continue
                if bold:
                    cv2.putText(disp, txt, (8, y_cur), cv2.FONT_HERSHEY_SIMPLEX, sc, (0,0,0), 3, cv2.LINE_AA)
                    cv2.putText(disp, txt, (8, y_cur), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)
                else:
                    _put_text(disp, txt, (8, y_cur), sc, col)
                y_cur += 17

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur  = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 30, 90)
            debug = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            if best is not None:
                cv2.drawContours(debug, [best], -1, (0, 255, 0), 2)
            _put_text(debug, "EDGE DEBUG", (8, 22), 0.5, (0, 255, 0))

            with frame_lock:
                latest_frame = disp
                latest_debug = debug

        except Exception as e:
            print(f"[Loop] {e}"); time.sleep(0.05)

# ═══════════════════════════════════════════════════════════════
#  Flask server and web interface setup
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)

def _enc(f):
    ok, buf = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else None

def _stream(src):
    while True:
        with frame_lock: f = src()
        if f is not None:
            d = _enc(f)
            if d:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + d + b'\r\n'
        time.sleep(0.01)

@app.route('/')
def index():
    return """<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<title>Symbol Detector</title><style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#f5f5f5;color:#111;font-family:-apple-system,sans-serif;font-size:14px;padding:20px}
h1{font-size:16px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;
   margin-bottom:20px;border-bottom:1px solid #ddd;padding-bottom:10px}
.video-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
.vc{background:#fff;border:1px solid #e0e0e0;border-radius:4px;overflow:hidden}
.vc .lbl{font-size:11px;font-weight:500;text-transform:uppercase;letter-spacing:.06em;
  color:#888;padding:8px 12px;border-bottom:1px solid #eee}
.vc img{width:100%;display:block}
.big-result{background:#fff;border:1px solid #e0e0e0;border-radius:4px;
  padding:20px 24px;margin-bottom:20px;display:flex;align-items:center;gap:20px}
.big-result.active{border-color:#111}
.br-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#999;width:70px;flex-shrink:0}
.br-name{font-size:32px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;flex:1}
.br-score{font-size:13px;font-weight:600;color:#555;width:80px;text-align:right}
.br-mode{font-size:11px;color:#999;width:80px;text-align:right}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:8px;margin-bottom:20px}
.stat{background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:10px 12px}
.stat .sl{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#999;margin-bottom:4px}
.stat .sv{font-size:18px;font-weight:600;color:#111}
.sec{background:#fff;border:1px solid #e0e0e0;border-radius:4px;padding:12px 14px;margin-bottom:20px}
.sec-t{font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:#999;margin-bottom:10px}
.history-row{display:flex;align-items:center;gap:10px;padding:6px 0;
  border-bottom:1px solid #f0f0f0;font-size:12px}
.history-row:last-child{border-bottom:none}
.ht{color:#bbb;width:52px;flex-shrink:0}
.hn{font-weight:600;color:#111;width:100px;text-transform:uppercase}
.hc{color:#999;flex:1;font-size:11px}
.hs{color:#555;font-weight:600;width:40px;text-align:right}
.empty{color:#ccc;font-size:12px;padding:4px 0}
</style></head><body>
<h1>Symbol Detector</h1>
<div class="video-row">
  <div class="vc"><div class="lbl">Camera + Detection</div><img src="/video_frame"></div>
  <div class="vc"><div class="lbl">Edge Debug</div><img src="/video_debug"></div>
</div>
<div class="big-result" id="bigres">
  <div class="br-lbl">Detected</div>
  <div class="br-name" id="bn">—</div>
  <div class="br-score" id="bsc"></div>
  <div class="br-mode" id="bmd"></div>
</div>
<div class="stat-grid">
  <div class="stat"><div class="sl">Status</div><div class="sv" id="ss">—</div></div>
  <div class="stat"><div class="sl">Mode</div><div class="sv" id="sm">—</div></div>
  <div class="stat"><div class="sl">FPS</div><div class="sv" id="sf">—</div></div>
  <div class="stat"><div class="sl">Resolution</div><div class="sv" id="sr">—</div></div>
</div>
<div class="sec">
  <div class="sec-t">Detection history (latest first)</div>
  <div id="hist"><div class="empty">Nothing detected yet</div></div>
</div>
<script>
async function r(){
  try{
    const s=await(await fetch('/api/state')).json();
    document.getElementById('ss').textContent=s.status;
    document.getElementById('sm').textContent=s.mode;
    document.getElementById('sf').textContent=s.fps;
    document.getElementById('sr').textContent=s.resolution;
    const br=document.getElementById('bigres');
    br.className='big-result'+(s.status==='detected'?' active':'');
    document.getElementById('bn').textContent=s.shape_name||'—';
    document.getElementById('bsc').textContent=s.shape_score?s.shape_score+'%':'';
    document.getElementById('bmd').textContent=s.mode?s.mode.toUpperCase():'';
    const h=s.history||[];
    const hel=document.getElementById('hist');
    if(!h.length){hel.innerHTML='<div class="empty">Nothing detected yet</div>';}
    else{
      hel.innerHTML=[...h].reverse().map(x=>
        '<div class="history-row">'+
        '<span class="ht">'+x.time+'</span>'+
        '<span class="hn">'+x.shape+'</span>'+
        '<span class="hc">'+x.conf+'</span>'+
        '<span class="hs">'+x.score+'%</span>'+
        '</div>'
      ).join('');
    }
  }catch(e){}
}
r();setInterval(r,300);
</script></body></html>"""

@app.route('/video_frame')
def video_frame():
    return Response(_stream(lambda: latest_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_debug')
def video_debug():
    return Response(_stream(lambda: latest_debug),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def api_state():
    with state_lock:
        return jsonify(dict(det_state))

# ═══════════════════════════════════════════════════════════════
#  Activation
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    _load_onnx()
    threading.Thread(target=processing_loop, daemon=True).start()
    print("symbol detection activate")
    try:
        print(f"Dashboard: http://0.0.0.0:{PORT}")
        app.run(host='0.0.0.0', port=PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        picam2.stop()
        print("shut down successful.")
