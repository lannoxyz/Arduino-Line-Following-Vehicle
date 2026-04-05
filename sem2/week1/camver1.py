import time
import sys
import threading
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

# 检查 Picamera2
try:
    from picamera2 import Picamera2
except ImportError:
    print("错误: 未安装 Picamera2。sudo apt install python3-picamera2")
    sys.exit(1)

app = Flask(__name__)

# ================= 1. 用户调试区域 (修改这里) =================

# --- 图像识别参数 ---
RESOLUTION = (640, 480)
# [修改] 黑色阈值调低到 40。只有很黑的线才会被识别，灰色的不识别。
BLACK_THRESHOLD = 50    
# [修改] ROI 高度比例改为 0.66 (占屏幕 2/3)
ROI_HEIGHT_RATIO = 0.60 
FLIP_MODE = 1

# --- 舵机参数 ---
SERVO_PIN = 18
SERVO_MIN = -90
SERVO_MAX = 90

# [关键修复] PID 方向控制
# 如果舵机往反方向跑（看到线在左边，却往右转），请把这里的 1 改成 -1
PID_DIRECTION = 1 

# ================= 2. 全局状态 =================
control_state = {
    "servo_active": True,  
    "p_gain": 1,        
    "d_gain": 1         
}

output_frame = None
lock = threading.Lock()
fps_display = 0
current_servo_angle = 0

# ================= 3. 硬件初始化 =================
print("正在初始化舵机...")
factory = None
try:
    factory = PiGPIOFactory()
    print("成功连接 pigpiod (硬件PWM)")
except:
    print("警告: 未检测到 pigpiod，使用软件 PWM (可能会抖动)")

servo = AngularServo(SERVO_PIN, min_angle=SERVO_MIN, max_angle=SERVO_MAX, pin_factory=factory)
# 初始归位
servo.angle = 0

print("正在初始化相机...")
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": RESOLUTION, "format": "BGR888"},
    controls={
        "FrameDurationLimits": (11000, 16666),
        "AnalogueGain": 8.0,
        "ExposureTime": 8000
    }
)
picam2.configure(config)
picam2.start()

# ================= 4. 视觉处理线程 =================
def process_thread():
    global output_frame, current_servo_angle, fps_display, control_state
    
    last_error = 0
    center_x = RESOLUTION[0] // 2
    frame_cnt = 0
    start_time = time.time()
    
    print("视觉线程已启动，等待图像...")
    
    while True:
        try:
            # 参数快照
            current_p = control_state["p_gain"]
            current_d = control_state["d_gain"]
            is_active = control_state["servo_active"]

            # 1. 获取图像
            frame = picam2.capture_array()
            
            # 2. 图像翻转 (根据你的安装情况，0=上下翻转, -1=旋转180度)
            # 如果左右方向反了，可能需要把这里改成 -1
            frame = cv2.flip(frame, -1) 
            
            # 3. ROI 裁剪 (根据比例切出底部 2/3)
            h, w, _ = frame.shape
            roi_h = int(h * ROI_HEIGHT_RATIO) 
            roi_y = h - roi_h # 计算起始 Y 坐标
            roi = frame[roi_y:h, :] 
            
            # 4. 二值化处理
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # 只有比 40 更黑的像素才会变成白色 (255)
            _, thresh = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            status_text = "Idle"
            color_status = (100, 100, 100)
            
            # --- 画出浅蓝色参考中线 (贯穿整个 ROI) ---
            # 参数: 图像, 起点, 终点, 颜色(BGR), 线宽
            cv2.line(roi, (center_x, 0), (center_x, roi_h), (255, 255, 0), 2)

            if len(contours) > 0:
                # 找到最大黑块
                c = max(contours, key=cv2.contourArea)
                
                # 如果面积太小（比如只是个噪点），忽略
                if cv2.contourArea(c) > 100:
                    status_text = "Tracking"
                    color_status = (0, 255, 0)
                    
                    # 画出识别到的黑线轮廓
                    cv2.drawContours(roi, [c], -1, (0, 255, 0), 2)
                    
                    M = cv2.moments(c)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 画出黑线中心红点
                        cv2.circle(roi, (cx, cy), 8, (0, 0, 255), -1)

                        # --- PID 计算 ---
                        error = center_x - cx
                        
                        # [BUG修复] 乘上方向系数
                        pid_out = ((error * current_p) + ((error - last_error) * current_d)) * PID_DIRECTION
                        last_error = error
                        
                        if is_active:
                            new_angle = current_servo_angle + pid_out
                            # 限制范围
                            new_angle = max(SERVO_MIN, min(SERVO_MAX, new_angle))
                            
                            # 执行动作
                            servo.angle = new_angle
                            current_servo_angle = new_angle
                            
                            # 调试信息: 如果你发现舵机不动，看终端有没有这行字
                            # print(f"Err: {error} | Out: {pid_out:.2f} | Ang: {new_angle:.1f}")
                else:
                    status_text = "Noise Ignored"
            else:
                status_text = "No Line"
                color_status = (0, 0, 255)

            # OSD 显示
            frame_cnt += 1
            if time.time() - start_time >= 1.0:
                fps_display = frame_cnt
                frame_cnt = 0
                start_time = time.time()
            
            osd_line1 = f"FPS: {fps_display} | Ang: {int(current_servo_angle)}"
            osd_line2 = f"P:{current_p} D:{current_d} Thr:{BLACK_THRESHOLD}"
            
            cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.putText(frame, osd_line1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, osd_line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)

            with lock:
                output_frame = frame.copy()
                
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.01)

# ================= 5. Flask API =================

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.04)

@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Line Follower Tuner V2</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { background-color: #121212; color: #e0e0e0; font-family: sans-serif; text-align: center; padding: 10px; }
            .container { max-width: 640px; margin: 0 auto; }
            img { width: 100%; border: 2px solid #333; }
            .panel { background: #1e1e1e; padding: 20px; margin-top: 15px; text-align: left; }
            input[type=range] { width: 100%; }
            button { width: 100%; padding: 15px; font-size: 18px; font-weight: bold; cursor: pointer; }
            .btn-on { background-color: #cf6679; color: white; }
            .btn-off { background-color: #03dac6; color: black; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>视觉调试 (阈值: 40)</h2>
            <img src="/video_feed">
            <div class="panel">
                <button id="toggleBtn" class="btn-on" onclick="toggleServo()">STOP SERVO</button>
                <br><br>
                <label>P Gain: <span id="p_val">0.04</span></label>
                <input type="range" id="p_slider" min="0" max="0.15" step="0.001" value="0.04" oninput="updatePID()">
                <br>
                <label>D Gain: <span id="d_val">0.05</span></label>
                <input type="range" id="d_slider" min="0" max="0.15" step="0.001" value="0.05" oninput="updatePID()">
            </div>
        </div>
        <script>
            let servoActive = true;
            function toggleServo() {
                servoActive = !servoActive;
                const btn = document.getElementById("toggleBtn");
                fetch('/api/set_servo', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({active: servoActive})
                });
                if (servoActive) { btn.innerText = "STOP SERVO"; btn.className = "btn-on"; } 
                else { btn.innerText = "START SERVO"; btn.className = "btn-off"; }
            }
            function updatePID() {
                const p = document.getElementById("p_slider").value;
                const d = document.getElementById("d_slider").value;
                document.getElementById("p_val").innerText = p;
                document.getElementById("d_val").innerText = d;
                fetch(`/api/set_pid?p=${p}&d=${d}`);
            }
        </script>
    </body>
    </html>
    """

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/set_servo", methods=['POST'])
def set_servo():
    data = request.json
    control_state["servo_active"] = data.get("active", False)
    return jsonify({"status": "ok"})

@app.route("/api/set_pid")
def set_pid():
    try:
        p = float(request.args.get('p', 0.04))
        d = float(request.args.get('d', 0.05))
        control_state["p_gain"] = p
        control_state["d_gain"] = d
    except ValueError: pass
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    t = threading.Thread(target=process_thread)
    t.daemon = True
    t.start()
    print("服务器已启动: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
