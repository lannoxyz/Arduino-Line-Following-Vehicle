import time
import sys
import threading
from flask import Flask, Response
import cv2
import numpy as np

# 尝试引入 Picamera2，如果报错则提示安装
try:
    from picamera2 import Picamera2
except ImportError:
    print("错误: 找不到 Picamera2 库。请运行: sudo apt install python3-picamera2")
    sys.exit(1)

app = Flask(__name__)

# ================= 配置区域 =================
# 1. 分辨率与帧率
# 推荐: (1280, 720) -> 清晰且流畅 (约 40fps)
# 极速: (640, 480)  -> 画面较糊但能跑满 60fps+
RESOLUTION = (1000, 600)

# 2. 画面翻转
# 0 = 垂直翻转, 1 = 水平翻转, -1 = 旋转180度
FLIP_MODE = -1 

# 3. 紫色修复强度 (OV5647 NoIR 版本专用)
# 1.0 = 不修改
# 小于 1.0 = 减弱该颜色
# 大于 1.0 = 增强该颜色
# 如果画面还是紫，把 RED_GAIN 和 BLUE_GAIN 调低 (例如 0.6)
RED_GAIN = 1   # 红色通道增益
BLUE_GAIN = 1  # 蓝色通道增益
GREEN_GAIN = 1 # 绿色通道增益 (通常不动)

# ================= 相机初始化 =================
print("正在初始化相机...")
picam2 = Picamera2()
 
# 配置相机硬件
config = picam2.create_video_configuration(
    main={
        "size": RESOLUTION,
        "format": "BGR888" # 直接输出 OpenCV 格式
    },
    controls={
        # 强制设置帧率范围 (微秒), 16666 60fps
        "FrameDurationLimits": (16666, 16666),
        # 可以在这里锁定白平衡，防止颜色乱跳
        # "AwbMode": "auto" 
    }
)
picam2.configure(config)
picam2.start()
print(f"相机已启动! 分辨率: {RESOLUTION}")

# ================= 图像处理逻辑 =================
def process_frame(image):
    """
    在这里集中处理所有图像效果
    """
    # 1. 翻转画面
    if FLIP_MODE is not None:
        image = cv2.flip(image, FLIP_MODE)

    # 2. 修复紫色 (颜色通道加权)
    # 原理：分离通道 -> 乘以系数 -> 限制范围 -> 合并
    # 注意：这会消耗一些 CPU，但比复杂的矩阵运算快
    if RED_GAIN != 1.0 or BLUE_GAIN != 1.0:
        # 分离通道 (OpenCV 默认是 BGR 顺序)
        b, g, r = cv2.split(image)
        
        # 使用 numpy 进行快速矩阵乘法
        if BLUE_GAIN != 1.0:
            b = cv2.multiply(b, BLUE_GAIN)
        if RED_GAIN != 1.0:
            r = cv2.multiply(r, RED_GAIN)
        if GREEN_GAIN != 1.0:
            g = cv2.multiply(g, GREEN_GAIN)

        # 合并回图像
        image = cv2.merge((b, g, r))

    # 3. (可选) 在画面上添加信息
    cv2.putText(image, f"RPi Cam: {RESOLUTION[0]}x{RESOLUTION[1]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image

def generate_frames():
    while True:
        try:
            # 从 Picamera2 获取最新一帧 (非阻塞)
            frame = picam2.capture_array()

            # 处理图像 (翻转、调色)
            frame = process_frame(frame)

            # 编码为 JPEG
            # quality=60 是平衡点，画质尚可，延迟极低
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Frame Error: {e}")
            time.sleep(0.01)

# ================= Flask 路由 =================
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>Raspberry Pi Robot View</title>
        <style>
            body { background-color: #1a1a1a; color: white; text-align: center; font-family: sans-serif; }
            h1 { margin-top: 20px; }
            img { border: 5px solid #444; border-radius: 10px; max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Robot Camera Live Feed</h1>
        <p>Status: Online | Mode: High FPS</p>
        <img src="/video_feed">
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 0.0.0.0 允许局域网访问
    # threaded=True 允许多人同时观看 (虽然会卡)
    print("服务器已启动: http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)