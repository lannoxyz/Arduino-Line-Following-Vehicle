"""
line_follower.py — 纯巡线，无图形检测 (三段式线性控制 + 智能丢线恢复)
用法: python3 line_follower.py
Dashboard: http://<RPi_IP>:5000

改进功能: 智能丢线恢复机制 (基于像素分布)
- 缓存最近1秒内的帧数据
- 丢线时分析左右两侧黑色像素分布
- 判断黑线更可能在哪一边，朝该方向旋转修正
"""

import time
import sys
import threading
from collections import deque
from flask import Flask, Response, jsonify
import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:
    print("错误: sudo apt install python3-picamera2")
    sys.exit(1)

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("错误: sudo apt install python3-rpi.gpio")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# 核心配置参数
# ═══════════════════════════════════════════════════════════════

camera_resolution = (640, 480)
jpeg_quality = 60
flip_mode = -1

binary_threshold = 100

# 误差区间定义 (按照 Lanno 的要求)
dead_zone_pct = 0.03  # 0-10%: 直行死区
spin_zone_pct = 0.18  # 40%以上: 原地旋转极限区

# 电机GPIO针脚配置
pin_in1, pin_in2 = 27, 17
pin_in3, pin_in4 = 6, 5
pin_ena, pin_enb = 12, 13

# 速度参数
base_speed = 40   # 基础直行速度
max_speed = 60   # 单边轮最大加速上限
spin_speed = 45   # 原地旋转时的速度

# 丢线恢复参数 (改进版本 - 基于像素分布)
line_lost_recovery_enabled = True     # 启用丢线恢复机制
line_lost_recovery_speed = 45         # 丢线恢复时的旋转速度
line_lost_recovery_timeout = 2.0      # 恢复尝试的最大时间（秒）
line_lost_check_threshold = 100       # 丢线判定阈值（黑色像素数）
recovery_history_duration = 1.0       # 缓存历史帧的时间长度（秒）
pixel_ratio_threshold = 0.5          # 判断左右倾向的阈值 (0.5=平均, <0.5=更多在左, >0.5=更多在右)


# ═══════════════════════════════════════════════════════════════
# GPIO & 电机控制模块
# ═══════════════════════════════════════════════════════════════

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in [pin_in1, pin_in2, pin_in3, pin_in4, pin_ena, pin_enb]:
    GPIO.setup(pin, GPIO.OUT)

# 设置 PWM 频率为 500Hz
pwm_a = GPIO.PWM(pin_ena, 500)
pwm_b = GPIO.PWM(pin_enb, 500)

pwm_a.start(0)
pwm_b.start(0)


def set_motors(left_power, right_power):
    """
    设置左右电机速度，范围 -100 到 100
    """
    left_power = max(-100.0, min(100.0, left_power))
    
    GPIO.output(pin_in1, GPIO.HIGH if left_power > 0 else GPIO.LOW)
    GPIO.output(pin_in2, GPIO.LOW if left_power > 0 else (GPIO.HIGH if left_power < 0 else GPIO.LOW))
    pwm_a.ChangeDutyCycle(abs(left_power))

    right_power = max(-100.0, min(100.0, right_power))
    
    GPIO.output(pin_in3, GPIO.HIGH if right_power > 0 else GPIO.LOW)
    GPIO.output(pin_in4, GPIO.LOW if right_power > 0 else (GPIO.HIGH if right_power < 0 else GPIO.LOW))
    pwm_b.ChangeDutyCycle(abs(right_power))


def stop_motors():
    """安全停止所有电机"""
    set_motors(0, 0)


# ═══════════════════════════════════════════════════════════════
# 相机初始化
# ═══════════════════════════════════════════════════════════════

print("初始化相机...")

picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size": camera_resolution, "format": "BGR888"},
        controls={"FrameDurationLimits": (16666, 16666)}
    )
)
picam2.start()
time.sleep(1)

print("相机就绪")


# ═══════════════════════════════════════════════════════════════
# 状态机与全局变量
# ═══════════════════════════════════════════════════════════════

state_lock = threading.Lock()

robot_state = {
    "status": "init",
    "direction": "straight",
    "error_pct": 0.0,
    "left_speed": 0.0,
    "right_speed": 0.0,
    "fps": 0.0,
    "resolution": f"{camera_resolution[0]}x{camera_resolution[1]}",
    "black_ratio": 0.0,
    "line_lost_recovery": "inactive",
    "recovery_reason": ""
}

frame_lock = threading.Lock()
latest_raw_frame = None
latest_binary_frame = None

# 丢线恢复机制的全局变量 (改进版本 - 基于像素分布)
recovery_lock = threading.Lock()

# 历史帧缓存 - 存储最近 N 秒的二值化帧和时间戳
history_lock = threading.Lock()
frame_history = deque()  # 存储 (timestamp, binary_image) 元组

line_lost_state = {
    "is_recovering": False,        # 是否正在恢复
    "line_lost_time": None,        # 丢线开始时间
    "recovery_direction": None,    # 恢复方向: "left", "right", 或 None
}


# ═══════════════════════════════════════════════════════════════
# 智能像素分布分析函数 (改进版本核心)
# ═══════════════════════════════════════════════════════════════

def add_frame_to_history(binary_image, timestamp):
    """
    将二值化帧添加到历史缓存
    """
    with history_lock:
        frame_history.append((timestamp, binary_image.copy()))
        
        # 清理超过指定时间的历史帧
        cutoff_time = timestamp - recovery_history_duration
        while frame_history and frame_history[0][0] < cutoff_time:
            frame_history.popleft()


def analyze_pixel_distribution():
    """
    分析历史帧中的像素分布，判断黑线更可能在左还是右
    返回: (direction, confidence)
    direction: "left", "right", 或 None
    confidence: 0.0-1.0 之间的置信度
    """
    with history_lock:
        if not frame_history:
            return None, 0.0
        
        # 合并最近历史帧
        height, width = None, None
        left_pixels = 0
        right_pixels = 0
        
        for timestamp, binary_image in frame_history:
            if height is None:
                height, width = binary_image.shape[:2]
            
            # 左半边
            left_half = binary_image[:, :width//2]
            left_pixels += cv2.countNonZero(left_half)
            
            # 右半边
            right_half = binary_image[:, width//2:]
            right_pixels += cv2.countNonZero(right_half)
        
        # 计算比例
        total_pixels = left_pixels + right_pixels
        
        if total_pixels == 0:
            return None, 0.0
        
        right_ratio = right_pixels / total_pixels  # 右侧像素比例
        
        # 判断方向和置信度
        if right_ratio > pixel_ratio_threshold:
            # 右侧黑线更多
            direction = "right"
            confidence = right_ratio - 0.5  # 越接近 1.0，置信度越高
        elif right_ratio < (1.0 - pixel_ratio_threshold):
            # 左侧黑线更多
            direction = "left"
            confidence = (0.5 - right_ratio)  # 越接近 0.0，置信度越高
        else:
            # 左右均衡，不确定
            direction = None
            confidence = 0.0
        
        return direction, confidence


def start_line_recovery():
    """
    启动丢线恢复机制
    分析历史帧，判断应该朝哪个方向旋转
    """
    with recovery_lock:
        if not line_lost_state["is_recovering"]:
            # 分析像素分布
            direction, confidence = analyze_pixel_distribution()
            
            line_lost_state["is_recovering"] = True
            line_lost_state["line_lost_time"] = time.time()
            line_lost_state["recovery_direction"] = direction
            
            if direction:
                print(f"[恢复] 丢线，检测到黑线更可能在 {direction}，置信度: {confidence:.2%}")
            else:
                print(f"[恢复] 丢线，左右像素分布不确定，停止恢复")


def stop_line_recovery():
    """
    停止丢线恢复机制
    """
    with recovery_lock:
        line_lost_state["is_recovering"] = False
        line_lost_state["line_lost_time"] = None
        line_lost_state["recovery_direction"] = None


def get_recovery_state():
    """
    获取当前恢复状态
    返回: (is_recovering, direction, should_continue)
    """
    with recovery_lock:
        if not line_lost_state["is_recovering"]:
            return False, None, False
        
        elapsed = time.time() - line_lost_state["line_lost_time"]
        direction = line_lost_state["recovery_direction"]
        
        # 超时检查
        if elapsed > line_lost_recovery_timeout:
            line_lost_state["is_recovering"] = False
            return False, None, False
        
        return True, direction, True


def perform_recovery_rotation(direction):
    """
    执行恢复旋转
    """
    if direction == "right":
        # 黑线在右边，向右转
        left_speed = line_lost_recovery_speed
        right_speed = -line_lost_recovery_speed
    elif direction == "left":
        # 黑线在左边，向左转
        left_speed = -line_lost_recovery_speed
        right_speed = line_lost_recovery_speed
    else:
        # 不确定方向，停止
        left_speed = 0
        right_speed = 0
    
    return left_speed, right_speed


# ═══════════════════════════════════════════════════════════════
# 核心视觉处理与控制线程
# ═══════════════════════════════════════════════════════════════

def processing_loop():
    global latest_raw_frame, latest_binary_frame

    frame_count = 0
    time_start = time.time()
    current_fps = 0.0

    while True:
        # 抓取图像
        frame = picam2.capture_array()

        if flip_mode is not None:
            frame = cv2.flip(frame, flip_mode)

        height, width = frame.shape[:2]

        # 灰度化与二值化
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(
            gray_image,
            binary_threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # 计算当前时间戳和图像矩
        current_time = time.time()
        moments = cv2.moments(binary_image)
        black_ratio = cv2.countNonZero(binary_image) / (height * width)
        has_line = moments["m00"] > line_lost_check_threshold

        # 添加当前帧到历史缓存
        add_frame_to_history(binary_image, current_time)

        # 初始化速度与状态变量
        left_speed = 0
        right_speed = 0
        status_text = "no_line"
        direction_text = "no_line"
        error_val = 0
        recovery_status = "inactive"
        recovery_reason = ""

        if has_line:
            # 线已找到，停止恢复
            stop_line_recovery()
            
            # 计算质心
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            # 归一化误差范围: -1.0 到 1.0
            error_val = (center_x - width / 2) / (width / 2)
            abs_error = abs(error_val)

            # ---------------------------------------------------
            # 核心逻辑：三段式控制
            # ---------------------------------------------------
            
            # 1. 0% ~ 10%：死区直行
            if abs_error <= dead_zone_pct:
                status_text = "straight"
                direction_text = "straight"
                left_speed = base_speed
                right_speed = base_speed

            # 2. 10% ~ 40%：单边线性加速
            elif abs_error <= spin_zone_pct:
                # 计算比例系数 (0.0 -> 1.0)
                k_ratio = (abs_error - dead_zone_pct) / (spin_zone_pct - dead_zone_pct)
                
                # 外侧轮线性加速，内侧轮保持基础速度
                outer_speed = base_speed + k_ratio * (max_speed - base_speed)
                inner_speed = base_speed

                if error_val < 0:
                    # 偏右，需向右转
                    direction_text = "right"
                    status_text = "adjust_right"
                    left_speed = outer_speed
                    right_speed = inner_speed
                else:
                    # 偏左，需向左转
                    direction_text = "left"
                    status_text = "adjust_left"
                    left_speed = inner_speed
                    right_speed = outer_speed

            # 3. 40% ~ 100%：极限区原地旋转
            else:
                if error_val < 0:
                    # 严重偏右，向右原地旋转
                    direction_text = "spin_right"
                    status_text = "spin_right"
                    left_speed = spin_speed
                    right_speed = -spin_speed
                else:
                    # 严重偏左，向左原地旋转
                    direction_text = "spin_left"
                    status_text = "spin_left"
                    left_speed = -spin_speed
                    right_speed = spin_speed

            # 执行电机控制
            set_motors(left_speed, right_speed)

        else:
            # 黑线丢失
            if line_lost_recovery_enabled:
                is_recovering, direction, should_continue = get_recovery_state()
                
                if not is_recovering:
                    # 第一次检测到丢线，启动恢复
                    start_line_recovery()
                    is_recovering, direction, should_continue = get_recovery_state()
                
                if should_continue and direction:
                    # 继续恢复
                    recovery_status = f"recovery_{direction}"
                    recovery_reason = f"pixel_dist_{direction}"
                    left_speed, right_speed = perform_recovery_rotation(direction)
                    status_text = f"line_lost_recovery_{direction}"
                    direction_text = f"recovery_{direction}"
                    set_motors(left_speed, right_speed)
                else:
                    # 恢复失败或超时，停止
                    stop_line_recovery()
                    stop_motors()
                    recovery_status = "inactive"
                    recovery_reason = "recovery_failed"
            else:
                # 恢复机制禁用，立即停止
                stop_motors()
                recovery_status = "inactive"
            
            center_x = -1
            center_y = -1

        # 计算帧率
        frame_count += 1
        time_now = time.time()
        if time_now - time_start >= 1:
            current_fps = frame_count / (time_now - time_start)
            frame_count = 0
            time_start = time_now

        # 更新状态字典
        with state_lock:
            robot_state.update({
                "status": status_text,
                "direction": direction_text,
                "error_pct": round(error_val * 100, 1),
                "left_speed": round(left_speed, 1),
                "right_speed": round(right_speed, 1),
                "fps": round(current_fps, 1),
                "black_ratio": round(black_ratio * 100, 1),
                "line_lost_recovery": recovery_status,
                "recovery_reason": recovery_reason
            })

        # 图像绘制与推流准备
        raw_display = frame.copy()
        cv2.line(raw_display, (width // 2, 0), (width // 2, height - 1), (255, 0, 0), 1)

        if has_line:
            cv2.circle(raw_display, (center_x, center_y), 10, (0, 0, 255), -1)
            cv2.line(raw_display, (width // 2, center_y), (center_x, center_y), (0, 255, 0), 2)

        bin_display = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        cv2.line(bin_display, (width // 2, 0), (width // 2, height - 1), (0, 255, 0), 1)

        with frame_lock:
            latest_raw_frame = raw_display
            latest_binary_frame = bin_display


# ═══════════════════════════════════════════════════════════════
# Flask Web 端推流
# ═══════════════════════════════════════════════════════════════

app = Flask(__name__)

def encode_frame(frame_data):
    """将图像编码为JPEG格式"""
    success, buffer = cv2.imencode(
        '.jpg',
        frame_data,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    return buffer.tobytes() if success else None


def stream_generator(source_function):
    """生成视频流的生成器"""
    while True:
        with frame_lock:
            current_frame = source_function()

        if current_frame is not None:
            encoded_data = encode_frame(current_frame)
            if encoded_data:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded_data + b'\r\n')
        time.sleep(0.01)


@app.route('/')
def index():
    return """
    <html>
    <head>
    <title>Line Follower Dashboard</title>
    </head>
    <body>
    <h2>Raw Camera Feed</h2>
    <img src="/video_raw">
    <h2>Binary Vision</h2>
    <img src="/video_binary">
    </body>
    </html>
    """

@app.route('/video_raw')
def video_raw():
    return Response(
        stream_generator(lambda: latest_raw_frame),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/video_binary')
def video_binary():
    return Response(
        stream_generator(lambda: latest_binary_frame),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/api/state')
def api_state():
    with state_lock:
        return jsonify(dict(robot_state))


# ═══════════════════════════════════════════════════════════════
# 程序启动入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # 启动视觉与控制后台线程
    threading.Thread(
        target=processing_loop,
        daemon=True
    ).start()

    print("处理线程已启动")
    print(f"丢线恢复机制: {'启用 (基于像素分布)' if line_lost_recovery_enabled else '禁用'}")
    print(f"历史缓存时长: {recovery_history_duration}秒")
    print(f"像素比例阈值: {pixel_ratio_threshold}")

    try:
        print("Dashboard运行在: http://0.0.0.0:5000")
        # 启动Flask服务器
        app.run(
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n收到退出信号，关闭中...")
    finally:
        # 安全清理资源
        stop_motors()
        picam2.stop()
        GPIO.cleanup()
        print("已安全关闭硬件资源")
