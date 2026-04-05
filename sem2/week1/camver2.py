import cv2
import numpy as np
import threading
import time
from flask import Flask, Response, request, jsonify
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

# ===============================
# 手写 PID 类
# ===============================
class PID:
    def __init__(self, kp, ki, kd, sample_time=0.02, limit_min=-0.15, limit_max=0.15):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sample_time = sample_time
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.integral = 0
        self.prev_error = 0

    def compute(self, error):
        p = self.kp * error
        self.integral += error * self.sample_time
        i = self.ki * self.integral
        d = self.kd * (error - self.prev_error) / self.sample_time
        self.prev_error = error
        out = p + i + d
        return max(self.limit_min, min(self.limit_max, out))

# ===============================
# 硬件设置
# ===============================
SERVO_PIN = 17
SERVO_MIN = -1
SERVO_MAX = 1
factory = PiGPIOFactory()
servo = Servo(SERVO_PIN, pin_factory=factory)

# ===============================
# PID 默认值
# ===============================
pid = PID(kp=0.04, ki=0.001, kd=0.015, sample_time=0.02, limit_min=-0.15, limit_max=0.15)

# ===============================
# 全局状态
# ===============================
frame = None
running = True
servo_active = True
current_servo_angle = 0
last_pid_out = 0
last_servo_update = 0
lock = threading.Lock()

# ===============================
# 摄像头线程
# ===============================
def camera_thread():
    global frame, running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 25)
    while running:
        ret, img = cap.read()
        if ret:
            with lock:
                frame = img.copy()
    cap.release()

# ===============================
# 图像处理 + PID + Servo
# ===============================
def process_thread():
    global frame, current_servo_angle, last_pid_out, last_servo_update, servo_active
    while running:
        with lock:
            if frame is None:
                continue
            img = frame.copy()
        h, w, _ = img.shape
        cx = w // 2

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

        # 找到最亮点
        _, _, _, max_loc = cv2.minMaxLoc(gray)
        target_x = max_loc[0]

        error = (cx - target_x) / cx  # -1~1

        pid_out_raw = pid.compute(error)
        alpha = 0.3
        pid_out = alpha * pid_out_raw + (1 - alpha) * last_pid_out
        last_pid_out = pid_out

        now = time.time()
        if servo_active and now - last_servo_update >= 0.05:
            step = np.clip(pid_out, -0.04, 0.04)
            new_angle = current_servo_angle + step
            new_angle = max(SERVO_MIN, min(SERVO_MAX, new_angle))
            servo.value = new_angle
            current_servo_angle = new_angle
            last_servo_update = now

        time.sleep(0.001)

# ===============================
# Flask 服务器
# ===============================
app = Flask(__name__)

html_page = """
<!DOCTYPE html>
<html>
<head>
<title>Servo Line Follower</title>
<style>
body { background: #121212; color:white; font-family:sans-serif; text-align:center; }
.panel { display:inline-block; margin:10px; }
img { width:320px; border:2px solid #fff; }
input[type=range] { width:100%; }
button { width:100%; padding:10px; font-size:16px; margin-top:5px;}
</style>
</head>
<body>
<h2>Servo PID Tuner</h2>
<div class="panel">
<h3>Gray</h3>
<img src="/stream_gray">
</div>
<div class="panel">
<h3>Binary</h3>
<img src="/stream_binary">
</div>
<div class="panel">
<button id="toggleBtn" onclick="toggleServo()">STOP SERVO</button><br>
<label>P: <span id="p_val">0.04</span></label>
<input type="range" min="0" max="0.15" step="0.001" value="0.04" id="p_slider" oninput="updatePID()"><br>
<label>I: <span id="i_val">0.001</span></label>
<input type="range" min="0" max="0.01" step="0.0001" value="0.001" id="i_slider" oninput="updatePID()"><br>
<label>D: <span id="d_val">0.015</span></label>
<input type="range" min="0" max="0.05" step="0.001" value="0.015" id="d_slider" oninput="updatePID()"><br>
</div>
<script>
let servoActive = true;
function toggleServo(){
    servoActive = !servoActive;
    fetch('/api/set_servo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({active:servoActive})});
    document.getElementById("toggleBtn").innerText = servoActive?"STOP SERVO":"START SERVO";
}
function updatePID(){
    const p = document.getElementById("p_slider").value;
    const i = document.getElementById("i_slider").value;
    const d = document.getElementById("d_slider").value;
    document.getElementById("p_val").innerText=p;
    document.getElementById("i_val").innerText=i;
    document.getElementById("d_val").innerText=d;
    fetch(`/api/set_pid?p=${p}&i=${i}&d=${d}`);
}
</script>
</body>
</html>
"""

# ===============================
# 视频流生成器
# ===============================
def gen_gray():
    global frame
    while True:
        with lock:
            if frame is None:
                time.sleep(0.03)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, buffer = cv2.imencode(".jpg", gray)
            frame_bytes = buffer.tobytes()
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')
        time.sleep(0.03)

def gen_binary():
    global frame
    while True:
        with lock:
            if frame is None:
                time.sleep(0.03)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
            _, buffer = cv2.imencode(".jpg", binary)
            frame_bytes = buffer.tobytes()
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+frame_bytes+b'\r\n')
        time.sleep(0.03)

# ===============================
# Flask 路由
# ===============================
@app.route("/")
def index():
    return html_page

@app.route("/stream_gray")
def stream_gray():
    return Response(gen_gray(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream_binary")
def stream_binary():
    return Response(gen_binary(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/set_servo", methods=['POST'])
def set_servo():
    global servo_active
    data = request.json
    servo_active = data.get("active", True)
    return jsonify({"status":"ok"})

@app.route("/api/set_pid")
def set_pid():
    global pid
    try:
        p = float(request.args.get("p", pid.kp))
        i = float(request.args.get("i", pid.ki))
        d = float(request.args.get("d", pid.kd))
        pid.kp = p
        pid.ki = i
        pid.kd = d
    except:
        pass
    return jsonify({"status":"ok"})

# ===============================
# 主程序
# ===============================
if __name__=="__main__":
    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=process_thread, daemon=True).start()
    print("Server running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
