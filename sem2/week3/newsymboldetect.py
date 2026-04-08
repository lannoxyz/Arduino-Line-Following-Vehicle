"""
symbol_detector.py — ONNX 持续推理
Dashboard: http://<RPi_IP>:5001
"""
import time, sys, threading, os
from flask import Flask, Response, jsonify
import cv2, numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    print("错误: sudo apt install python3-picamera2"); sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    print("错误: pip install onnxruntime --break-system-packages"); sys.exit(1)

# ═══════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════

RESOLUTION      = (640, 480)
JPEG_QUALITY    = 60
FLIP_MODE       = -1

ONNX_MODEL_PATH    = "model.onnx"
TFLITE_LABELS_PATH = "labels.txt"
ONNX_INPUT_SIZE    = 224

PORT = 5001

# ═══════════════════════════════════════════════════════
#  相机
# ═══════════════════════════════════════════════════════

print("初始化相机...")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "BGR888"},
    controls={"FrameDurationLimits": (16666, 16666)}))
picam2.start(); time.sleep(1)
print(f"相机就绪  {RESOLUTION}")

# ═══════════════════════════════════════════════════════
#  ONNX Runtime
# ═══════════════════════════════════════════════════════

_session = _input_name = None
_labels  = []

def _load_onnx():
    global _session, _labels, _input_name
    if not os.path.exists(TFLITE_LABELS_PATH):
        print(f"[ONNX] 找不到标签文件: {TFLITE_LABELS_PATH}"); return False
    with open(TFLITE_LABELS_PATH) as f:
        _labels = [l.strip().split(None, 1)[-1] for l in f if l.strip()]
    print(f"[ONNX] 标签: {_labels}")
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"[ONNX] 找不到模型文件: {ONNX_MODEL_PATH}"); return False
    try:
        _session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        print(f"[ONNX] 模型加载成功  input={_input_name}  shape={_session.get_inputs()[0].shape}")
        return True
    except Exception as e:
        print(f"[ONNX] 模型加载失败: {e}"); return False

def _onnx_classify(frame_bgr):
    """返回 (label, confidence_percent)，始终返回结果。"""
    resized = cv2.resize(frame_bgr, (ONNX_INPUT_SIZE, ONNX_INPUT_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob    = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
    try:
        scores = _session.run(None, {_input_name: blob})[0][0].astype(np.float32)
    except Exception as e:
        print(f"[ONNX] 推理失败: {e}"); return "error", 0.0
    e_    = np.exp(scores - scores.max())
    probs = e_ / e_.sum()
    idx   = int(np.argmax(probs))
    conf  = float(probs[idx])
    label = _labels[idx] if idx < len(_labels) else f"class{idx}"
    return label, round(conf * 100, 1)

# ═══════════════════════════════════════════════════════
#  共享状态
# ═══════════════════════════════════════════════════════

state_lock = threading.Lock()
det_state  = {"shape": "—", "confidence": 0.0}

frame_lock   = threading.Lock()
latest_frame = None

# ═══════════════════════════════════════════════════════
#  推理线程（独立循环，持续运行）
# ═══════════════════════════════════════════════════════

def inference_loop():
    """独立线程：持续从最新帧推理，不阻塞采集线程。"""
    while True:
        with frame_lock:
            f = latest_frame
        if f is None or _session is None:
            time.sleep(0.05); continue

        label, conf = _onnx_classify(f)
        print(f"[ONNX] {label}  {conf:.1f}%")
        with state_lock:
            det_state["shape"]      = label
            det_state["confidence"] = conf

# ═══════════════════════════════════════════════════════
#  采集线程
# ═══════════════════════════════════════════════════════

def capture_loop():
    global latest_frame
    while True:
        try:
            frame = picam2.capture_array()
            if FLIP_MODE is not None:
                frame = cv2.flip(frame, FLIP_MODE)
            with frame_lock:
                latest_frame = frame
        except Exception as e:
            print(f"[Capture] {e}"); time.sleep(0.05)

# ═══════════════════════════════════════════════════════
#  Flask
# ═══════════════════════════════════════════════════════

app = Flask(__name__)

def _stream():
    while True:
        with frame_lock:
            f = latest_frame
        if f is not None:
            ok, buf = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'
        time.sleep(0.033)

@app.route('/')
def index():
    return """<!DOCTYPE html><html><head><meta charset="UTF-8">
<title>Symbol Detector</title>
<style>
  body { margin: 0; background: #000; color: #fff; font-family: monospace;
         display: flex; flex-direction: column; align-items: center; padding: 16px; }
  img  { width: 100%; max-width: 640px; display: block; }
  #result { margin-top: 12px; text-align: center; }
  #shape  { font-size: 32px; font-weight: bold; text-transform: uppercase; }
  #conf   { font-size: 16px; color: #aaa; margin-top: 4px; }
</style></head><body>
<img src="/video">
<div id="result">
  <div id="shape">—</div>
  <div id="conf"></div>
</div>
<script>
setInterval(async () => {
  try {
    const s = await (await fetch('/api/state')).json();
    document.getElementById('shape').textContent = s.shape || '—';
    document.getElementById('conf').textContent  = s.confidence ? s.confidence + '%' : '';
  } catch(e) {}
}, 300);
</script></body></html>"""

@app.route('/video')
def video():
    return Response(_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def api_state():
    with state_lock:
        return jsonify(dict(det_state))

# ═══════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    _load_onnx()
    threading.Thread(target=capture_loop,   daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    print(f"Dashboard: http://0.0.0.0:{PORT}")
    try:
        app.run(host='0.0.0.0', port=PORT, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n关闭中...")
    finally:
        picam2.stop()
        print("已安全关闭。")