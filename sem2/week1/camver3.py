import time
import sys
import threading
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory

# 检查 Picamera2
try:
    from picamera2 import Picamera2
except ImportError:
    print("错误: 未安装 Picamera2。sudo apt install python3-picamera2")
    sys.exit(1)

app = Flask(__name__)

# ================= 1. 用户调试区域 =================
RESOLUTION = (640, 480)
BLACK_THRESHOLD = 70      # 黑线阈值
ROI_HEIGHT_RATIO = 0.8    # 屏幕下半部分
FLIP_MODE = -1            # 图像翻转模式

# H-Bridge Motor GPIO
IN1, IN2 = 27, 17  # 左轮
IN3, IN4 = 6, 5    # 右轮
ENA, ENB = 12, 13  # PWM 控制
PWM_FREQ = 1000

# ================= 2. 全局状态 =================
control_state = {
    "pwm_speed": 0.4,       # 初始速度 0~1
    "center_threshold": 40   # 中心容忍偏差
}

output_frame = None
lock = threading.Lock()

# ================= 3. 硬件初始化 =================
print("初始化电机和PWM...")
factory = None
try:
    factory = PiGPIOFactory()
    print("成功连接 pigpiod (硬件PWM)")
except:
    print("警告: 未检测到 pigpiod，使用软件PWM")

# 左轮
in1 = DigitalOutputDevice(IN1, pin_factory=factory)
in2 = DigitalOutputDevice(IN2, pin_factory=factory)
ena = PWMOutputDevice(ENA, frequency=PWM_FREQ, pin_factory=factory)

# 右轮
in3 = DigitalOutputDevice(IN3, pin_factory=factory)
in4 = DigitalOutputDevice(IN4, pin_factory=factory)
enb = PWMOutputDevice(ENB, frequency=PWM_FREQ, pin_factory=factory)

# 停止小车函数
def stop_car():
    in1.off(); in2.off(); in3.off(); in4.off()
    ena.value = 0; enb.value = 0

stop_car()

# 初始化相机
print("初始化相机...")
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "BGR888"},
)
picam2.configure(config)
picam2.start()

# ================= 4. 图像处理 + 小车控制线程 =================
def process_thread():
    global output_frame, control_state

    center_x = RESOLUTION[0] // 2

    while True:
        try:
            frame = picam2.capture_array()
            frame = cv2.flip(frame, FLIP_MODE)

            h, w, _ = frame.shape
            roi_h = int(h * ROI_HEIGHT_RATIO)
            roi_y = h - roi_h
            roi = frame[roi_y:h, :]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

            # 找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 默认停止
            stop_car()
            status_text = "No Line"
            color_status = (0, 0, 255)

            if len(contours) > 0:
                c = max(contours, key=cv2.contourArea)
                if cv2.contourArea(c) > 50:
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 画线和中心点
                        cv2.drawContours(roi, [c], -1, (0, 255, 0), 2)
                        cv2.circle(roi, (cx, cy), 8, (0, 0, 255), -1)
                        cv2.line(roi, (center_x, 0), (center_x, roi_h), (255, 255, 0), 2)

                        error = cx - center_x
                        threshold = control_state["center_threshold"]
                        pwm_speed = control_state["pwm_speed"]

                        # 判断偏左偏右
                        if abs(error) <= threshold:
                            # 中间 → 前进
                            in1.on(); in2.off(); ena.value = pwm_speed
                            in3.on(); in4.off(); enb.value = pwm_speed
                            status_text = "Forward"
                            color_status = (0, 255, 0)
                        elif error < -threshold:
                            # 线偏左 → 原地左转
                            in1.off(); in2.on(); ena.value = pwm_speed
                            in3.on(); in4.off(); enb.value = pwm_speed
                            status_text = "Turn Left"
                            color_status = (255, 255, 0)
                        elif error > threshold:
                            # 线偏右 → 原地右转
                            in1.on(); in2.off(); ena.value = pwm_speed
                            in3.off(); in4.on(); enb.value = pwm_speed
                            status_text = "Turn Right"
                            color_status = (255, 0, 255)

            # OSD 显示
            osd_line = f"Status: {status_text} | Speed: {control_state['pwm_speed']:.2f} | Thr: {control_state['center_threshold']}"
            cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
            cv2.putText(frame, osd_line, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)

            with lock:
                output_frame = frame.copy()

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.01)

# ================= 5. Flask 视频流 + 控制接口 =================
def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)

@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Line Follower Control</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #121212; color: #e0e0e0; font-family: sans-serif; text-align: center; padding: 10px; }
            .container { max-width: 640px; margin: 0 auto; }
            img { width: 100%; border: 2px solid #333; }
            .panel { background: #1e1e1e; padding: 20px; margin-top: 15px; text-align: left; }
            input[type=range] { width: 100%; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Line Follower Control</h2>
            <img src="/video_feed">
            <div class="panel">
                <label>Speed: <span id="speed_val">0.5</span></label>
                <input type="range" id="speed_slider" min="0" max="1" step="0.01" value="0.5" oninput="updateSpeed()">
                <br><br>
                <label>Center Threshold: <span id="thr_val">20</span></label>
                <input type="range" id="thr_slider" min="5" max="100" step="1" value="20" oninput="updateThr()">
            </div>
        </div>
        <script>
            function updateSpeed(){
                const val = document.getElementById("speed_slider").value;
                document.getElementById("speed_val").innerText = val;
                fetch(`/api/set_speed?value=${val}`);
            }
            function updateThr(){
                const val = document.getElementById("thr_slider").value;
                document.getElementById("thr_val").innerText = val;
                fetch(`/api/set_thr?value=${val}`);
            }
        </script>
    </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/set_speed")
def set_speed():
    try:
        val = float(request.args.get("value", 0.5))
        control_state["pwm_speed"] = max(0, min(1, val))
    except: pass
    return jsonify({"status": "ok"})

@app.route("/api/set_thr")
def set_thr():
    try:
        val = int(request.args.get("value", 20))
        control_state["center_threshold"] = val
    except: pass
    return jsonify({"status": "ok"})

# ================= 6. 启动 =================
if __name__ == "__main__":
    t = threading.Thread(target=process_thread, daemon=True)
    t.start()
    print("Server running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
