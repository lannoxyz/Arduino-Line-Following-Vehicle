"""
line_follower.py
use: python3 line_follower.py
Dashboard: http://<RPi_IP>:5000
"""

# import libraries
import time
import sys
import threading
from collections import deque
from flask import Flask, Response, jsonify
import cv2
import numpy as np

# setup picamera
try:
    from picamera2 import Picamera2
except ImportError:
    print("error: sudo apt install python3-picamera2")
    sys.exit(1)

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("error: sudo apt install python3-rpi.gpio")
    sys.exit(1)


# parameters, threshold values declaration
camera_resolution = (640, 480)
jpeg_quality = 60
flip_mode = -1

binary_threshold = 150

# size of zones for line following
dead_zone_pct = 0.03  # within 3% of error, travel straight
spin_zone_pct = 0.18  # 82%

# define gpio pins
pin_in1, pin_in2 = 27, 17
pin_in3, pin_in4 = 6, 5
pin_ena, pin_enb = 12, 13

# speed parameters
base_speed = 40   # base speed at dead zone
max_speed = 60   # max speed on one side motor between spin zone and dead zone
spin_speed = 45   # speed at spin zone, motor in opposite rotation

# parameters when line is lost
line_lost_recovery_enabled = True     
line_lost_recovery_speed = 45         # rotating speed to search for line
line_lost_recovery_timeout = 2.0      # time given to search for line
line_lost_check_threshold = 100       # line lost when number of black pixel < 100
recovery_history_duration = 1.0       # duration of "memory" for line recovery
pixel_ratio_threshold = 0.5           # more black pixels on the left if <0.5, right if > 0.5


# ═══════════════════════════════════════════════════════════════
# GPIO & setup
# ═══════════════════════════════════════════════════════════════

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in [pin_in1, pin_in2, pin_in3, pin_in4, pin_ena, pin_enb]:
    GPIO.setup(pin, GPIO.OUT) # set gpio output pins

# 设置 PWM 频率为 500Hz
pwm_a = GPIO.PWM(pin_ena, 500) # 500Hz pwm frequency for smoothest performance
pwm_b = GPIO.PWM(pin_enb, 500)

pwm_a.start(0)
pwm_b.start(0)


def set_motors(left_power, right_power):
    """
  setup motor speed in terms of % duty cycle
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
    set_motors(0, 0) # function to shut down motors


# ═══════════════════════════════════════════════════════════════
# picam configuration
# ═══════════════════════════════════════════════════════════════

print("camera configuration...")

picam2 = Picamera2()
picam2.configure(
    picam2.create_video_configuration(
        main={"size": camera_resolution, "format": "BGR888"},
        controls={"FrameDurationLimits": (16666, 16666)}
    )
)
picam2.start()
time.sleep(1)

print("camera is ready")


# ═══════════════════════════════════════════════════════════════
# state of vehicle, ensure only one thread runs and lock the rest
# threading.Lock() to lock selected thread
# ═══════════════════════════════════════════════════════════════

state_lock = threading.Lock()

# state of vehicle
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
latest_raw_frame = None # to hold the latest frame captured
latest_binary_frame = None # to hold to latest frame converted to binary

# line recovery state, search for line
recovery_lock = threading.Lock()

# deque to store all binary frames captured in the latest second
history_lock = threading.Lock()
frame_history = deque()  

# state of line recovery
line_lost_state = {
    "is_recovering": False,       
    "line_lost_time": None,       
    "recovery_direction": None,   
}


# ═══════════════════════════════════════════════════════════════
# pixel analysation when line is lost, guess on where line should be
# binary image is flipped so the black line represented in white pixels
# count number of white pixels in each left and right side and make comparison
# ═══════════════════════════════════════════════════════════════

def add_frame_to_history(binary_image, timestamp):
    """
    store binary images as timestamp
    """
    with history_lock:
        frame_history.append((timestamp, binary_image.copy()))
        
        # cleaning of all images that has been stored for > 1s
        cutoff_time = timestamp - recovery_history_duration
        while frame_history and frame_history[0][0] < cutoff_time:
            frame_history.popleft()


def analyze_pixel_distribution():
    """
    analyse pixels in binary image
    make an educated guess, whether or not line should be on the left or right
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

            
        # count number of white pixels
            # left side of frame
            left_half = binary_image[:, :width//2]
            left_pixels += cv2.countNonZero(left_half)
            
            # right side of frame
            right_half = binary_image[:, width//2:]
            right_pixels += cv2.countNonZero(right_half)
    
        total_pixels = left_pixels + right_pixels # total number of white pixels in frame
        
        if total_pixels == 0:
            return None, 0.0
        
        right_ratio = right_pixels / total_pixels  # ratio of white pixels in right side
        
        # decide direction of rotation to recover line
        if right_ratio > pixel_ratio_threshold:
            # rotate right if more pixels on right side
            direction = "right"
            confidence = right_ratio - 0.5  # 越接近 1.0，置信度越高
        elif right_ratio < (1.0 - pixel_ratio_threshold):
            # rotate left otherwise
            direction = "left"
            confidence = (0.5 - right_ratio)  # confidence higher as value approaches 0
        else:
            # balanced ratio on left and right side
            direction = None
            confidence = 0.0
        
        return direction, confidence


def start_line_recovery():
    """
    begin line recovery, trigger pixel analysis, start timer
    """
    with recovery_lock:
        if not line_lost_state["is_recovering"]:

            direction, confidence = analyze_pixel_distribution()
            
            line_lost_state["is_recovering"] = True
            line_lost_state["line_lost_time"] = time.time()
            line_lost_state["recovery_direction"] = direction

            # printing of infomation for debugging
            if direction:
                print(f"[恢复] 丢线，检测到黑线更可能在 {direction}，置信度: {confidence:.2%}")
            else:
                print(f"[恢复] 丢线，左右像素分布不确定，停止恢复")


def stop_line_recovery():
    """
    stop line recovery when line is found, resume line following
    """
    with recovery_lock:
        line_lost_state["is_recovering"] = False
        line_lost_state["line_lost_time"] = None
        line_lost_state["recovery_direction"] = None


def get_recovery_state():
    """
    state of recovery
    is line lost? 
    is the car still search for line?
    """
    with recovery_lock:
        if not line_lost_state["is_recovering"]:
            return False, None, False
        
        elapsed = time.time() - line_lost_state["line_lost_time"]
        direction = line_lost_state["recovery_direction"]
        
        # timeout of 2s
        if elapsed > line_lost_recovery_timeout:
            line_lost_state["is_recovering"] = False
            return False, None, False
        
        return True, direction, True


def perform_recovery_rotation(direction):
    """
    rotate motors to move vehicle in decided direction
    """
    if direction == "right":
        left_speed = line_lost_recovery_speed
        right_speed = -line_lost_recovery_speed
        
    elif direction == "left":
        
        left_speed = -line_lost_recovery_speed
        right_speed = line_lost_recovery_speed
        
    else:
        
        left_speed = 0
        right_speed = 0
    
    return left_speed, right_speed


# perform line following in loop

def processing_loop():
    global latest_raw_frame, latest_binary_frame

    frame_count = 0
    time_start = time.time()
    current_fps = 0.0

    while True:
        # capture frame from picam
        frame = picam2.capture_array()

        if flip_mode is not None:
            frame = cv2.flip(frame, flip_mode)

        height, width = frame.shape[:2]

       # converting capture frame
        # bgr to grayscale and to binary
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(
            gray_image,
            binary_threshold,
            255,
            cv2.THRESH_BINARY_INV # invert binary image to use moments() and calculate mass center
        )

        # cleaning and filter, remove small white/black holes due to lighting etc
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # tracking of current frame and status
        current_time = time.time()
        moments = cv2.moments(binary_image)
        black_ratio = cv2.countNonZero(binary_image) / (height * width) # named black for minimal confusion, function counts number of white pixels
        has_line = moments["m00"] > line_lost_check_threshold

        add_frame_to_history(binary_image, current_time)

        # setup default speed and status on startup
        left_speed = 0
        right_speed = 0
        status_text = "no_line"
        direction_text = "no_line"
        error_val = 0
        recovery_status = "inactive"
        recovery_reason = ""

        if has_line:
            # line has been found, stop line recovery
            stop_line_recovery()
            
            # calculate center of mass of white pixels in inverted binery image
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            # m00 = total white pixels
            # m10 = sum of pixel values in the x axis
            # m01 = sum of pixel values in the y axis

            # error: how far off is the line from the center? varies from -1.0-1.0
            error_val = (center_x - width / 2) / (width / 2)
            abs_error = abs(error_val)

            # ---------------------------------------------------
            # line following logic in 3 zones
            # ---------------------------------------------------
            
            # 0% ~ 3% of error：dead zone, proceed forward
            if abs_error <= dead_zone_pct:
                status_text = "straight"
                direction_text = "straight"
                left_speed = base_speed
                right_speed = base_speed

            # 3% ~ 40% of error：increased speed on one side of motor, the other motor speed remain constant
            elif abs_error <= spin_zone_pct:
                  # linear increase from base speed-max speed depending on how far off is error
                k_ratio = (abs_error - dead_zone_pct) / (spin_zone_pct - dead_zone_pct)
                outer_speed = base_speed + k_ratio * (max_speed - base_speed)
                inner_speed = base_speed

                if error_val < 0:
                    # slight off towards right, increase speed on left motor
                    direction_text = "right"
                    status_text = "adjust_right"
                    left_speed = outer_speed
                    right_speed = inner_speed
                else:
                    # slight off towards left, increase speed on right motor
                    direction_text = "left"
                    status_text = "adjust_left"
                    left_speed = inner_speed
                    right_speed = outer_speed

            # 3. 40% ~ 100%：spin zone, clockwise/anti clockwise rotation 
            else:
                if error_val < 0:
                    # far off towards left, rotate towards left
                    direction_text = "spin_right"
                    status_text = "spin_right"
                    left_speed = spin_speed
                    right_speed = -spin_speed
                else:
                    # far off towards right, rotate towards left
                    direction_text = "spin_left"
                    status_text = "spin_left"
                    left_speed = -spin_speed
                    right_speed = spin_speed
                    
            set_motors(left_speed, right_speed)

        else:
            # line lost from frame, enable line recovery
            if line_lost_recovery_enabled:
                is_recovering, direction, should_continue = get_recovery_state()
                
                if not is_recovering:
                    start_line_recovery()
                    is_recovering, direction, should_continue = get_recovery_state()
                
                if should_continue and direction:
                    # proceed to search for line
                    recovery_status = f"recovery_{direction}"
                    recovery_reason = f"pixel_dist_{direction}"
                    left_speed, right_speed = perform_recovery_rotation(direction)
                    status_text = f"line_lost_recovery_{direction}"
                    direction_text = f"recovery_{direction}"
                    set_motors(left_speed, right_speed)
                else:
                 # timeout, stop motors
                    stop_line_recovery()
                    stop_motors()
                    recovery_status = "inactive"
                    recovery_reason = "recovery_failed"
            else:
                stop_motors()
                recovery_status = "inactive"
            
            center_x = -1
            center_y = -1

        # calculate fps
        frame_count += 1
        time_now = time.time()
        if time_now - time_start >= 1:
            current_fps = frame_count / (time_now - time_start)
            frame_count = 0
            time_start = time_now

        # update status dictionary
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

        # copy captured frame, draw line and circle for debugging
        # line represent center of frame
        # circle represent center of mass of black line in frame
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


# setup flask server and web interface
# web interface will display raw footage from picamera, and same footage converted into binary, along with circle and line drawn for center of mass and center of frame
app = Flask(__name__)

def encode_frame(frame_data):
    """frame presented in jpeg"""
    success, buffer = cv2.imencode(
        '.jpg',
        frame_data,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    )
    return buffer.tobytes() if success else None


def stream_generator(source_function):
    """generating web live streaming"""
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



# final startup, print current status, web interface address
if __name__ == '__main__':
    threading.Thread(
        target=processing_loop,
        daemon=True
    ).start()

    print("line following activate")
    print(f"丢线恢复机制: {'启用 (基于像素分布)' if line_lost_recovery_enabled else '禁用'}")
    print(f"历史缓存时长: {recovery_history_duration}秒")
    print(f"像素比例阈值: {pixel_ratio_threshold}")

    try:
        print("Dashboard: http://0.0.0.0:5000")
        # activate flask sever
        app.run(
            host='0.0.0.0',
            port=5000,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n收到退出信号，关闭中...")
    finally:
        # safely shut down system
        stop_motors()
        picam2.stop()
        GPIO.cleanup()
        print("all hardware successfully shut down")
