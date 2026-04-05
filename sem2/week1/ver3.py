from flask import Flask, render_template_string, request
import RPi.GPIO as GPIO
import time

# =========================================
# 1. GPIO Setup & Hardware Configuration
# =========================================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Pin Definitions
IN1, IN2 = 17, 27   # Left Motor
IN3, IN4 = 5, 6     # Right Motor
ENA, ENB = 12, 13   # Speed Control (PWM)

# Initialize Pins
for p in [IN1, IN2, IN3, IN4, ENA, ENB]:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, 0)

# PWM Initialization
current_freq = 100
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0)
pwmB.start(0)

# Default Start Speed
current_speed_pwm = 200
action_lock = False

# =========================================
# 2. Motor Logic Functions
# =========================================
def set_duty_cycle(val_0_to_255):
    duty = max(0, min(100, (val_0_to_255 / 255) * 100))
    pwmA.ChangeDutyCycle(duty)
    pwmB.ChangeDutyCycle(duty)

def forward():
    GPIO.output(IN1, 0); GPIO.output(IN2, 1)
    GPIO.output(IN3, 0); GPIO.output(IN4, 1)

def backward():
    GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    GPIO.output(IN3, 1); GPIO.output(IN4, 0)

def turn_left():
    GPIO.output(IN1, 0); GPIO.output(IN2, 1)
    GPIO.output(IN3, 1); GPIO.output(IN4, 0)

def turn_right():
    GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    GPIO.output(IN3, 0); GPIO.output(IN4, 1)

def stop_motors():
    GPIO.output(IN1, 0); GPIO.output(IN2, 0)
    GPIO.output(IN3, 0); GPIO.output(IN4, 0)

# =========================================
# 3. Flask Application & UI
# =========================================
app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RPi Precision Rover Pro</title>
    <style>
        body { background-color: #f0f2f5; color: #333; font-family: 'Segoe UI', Roboto, sans-serif; text-align: center; padding: 10px; }
        h2 { color: #2c3e50; margin-bottom: 5px; }
        
        .container { max-width: 600px; margin: 0 auto; }
        
        .control-group { 
            margin: 15px 0; padding: 20px; 
            background: white; border-radius: 15px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); 
            transition: transform 0.2s;
        }
        
        /* 两个滑块并排的样式 */
        .dual-slider-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .slider-label { font-weight: bold; color: #555; width: 80px; text-align: left; }
        .slider-wrapper { flex-grow: 1; margin: 0 10px; }
        .slider { -webkit-appearance: none; width: 100%; height: 10px; border-radius: 5px; background: #e9ecef; outline: none; }
        .slider::-webkit-slider-thumb { -webkit-appearance: none; width: 22px; height: 22px; border-radius: 50%; background: #007BFF; cursor: pointer; border: 2px solid #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        
        /* 不同的滑块颜色以示区分 */
        #speedSlider::-webkit-slider-thumb { background: #28a745; } /* Green for m/s */
        
        .value-box { 
            width: 70px; padding: 5px; 
            border: 1px solid #ced4da; border-radius: 5px; 
            text-align: center; font-weight: bold; color: #007BFF; 
        }

        .calc-panel { background: #e8f4ff; border: 1px solid #b6d4fe; }
        .input-row { display: flex; justify-content: center; gap: 20px; align-items: center; margin: 15px 0; }
        .input-group { display: flex; flex-direction: column; align-items: center; }
        .input-group label { font-size: 0.85em; color: #666; margin-bottom: 5px; }
        .input-group input { 
            padding: 10px; width: 90px; text-align: center; 
            border: 2px solid #bdc3c7; border-radius: 8px; font-size: 1.1em; 
            transition: border-color 0.3s;
        }
        .input-group input:focus { border-color: #007BFF; outline: none; }

        .btn-main { 
            background-color: #007BFF; color: white; 
            padding: 12px 30px; border: none; border-radius: 8px; 
            font-size: 16px; margin: 5px; cursor: pointer; 
            box-shadow: 0 4px 0 #0056b3; 
            transition: all 0.1s; 
        }
        .btn-main:active { transform: translateY(4px); box-shadow: 0 0 0 #0056b3; }
        .btn-turn { background-color: #6c757d; box-shadow: 0 4px 0 #545b62; }
        .btn-turn:active { transform: translateY(4px); box-shadow: 0 0 0 #545b62; }

        .status-bar { font-size: 0.9em; color: #888; margin-top: 20px; font-family: monospace; }
        .lock-icon { display: none; color: #dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <h2>🚀 RPi 物理引擎终端</h2>
        <div class="status-bar">
            Status: <span id="status">Online</span> <span id="lock" class="lock-icon">🔒 BUSY</span>
        </div>

        <div class="control-group">
            <h3>⚙️ 动力耦合系统 (PWM ⇌ Speed)</h3>
            
            <div class="dual-slider-row">
                <span class="slider-label">PWM</span>
                <div class="slider-wrapper">
                    <input type="range" min="0" max="255" value="{{ pwm }}" class="slider" id="pwmSlider" oninput="handlePwmChange(this.value)">
                </div>
                <input type="number" id="pwmInput" class="value-box" value="{{ pwm }}" readonly>
            </div>

            <div class="dual-slider-row">
                <span class="slider-label">Speed</span>
                <div class="slider-wrapper">
                    <input type="range" min="0" max="0.80" step="0.01" value="0" class="slider" id="speedSlider" oninput="handleSpeedChange(this.value)">
                </div>
                <input type="text" id="speedInput" class="value-box" style="color: #28a745;" readonly> <span style="font-size:0.8em">m/s</span>
            </div>
        </div>

        <div class="control-group calc-panel">
            <h3>📏 动作规划 (Dist ⇌ Time)</h3>
            
            <div class="input-row">
                <div class="input-group">
                    <label>距离 (meters)</label>
                    <input type="number" id="distInput" step="0.01" placeholder="m" oninput="calculateTimeFromDist()">
                </div>
                <div style="font-size: 1.5em; color: #007BFF;">⇌</div>
                <div class="input-group">
                    <label>时间 (seconds)</label>
                    <input type="number" id="timeInput" step="0.1" value="1.0" placeholder="s" oninput="calculateDistFromTime()">
                </div>
            </div>

            <div>
                <button class="btn-main btn-turn" onclick="triggerTurn('left')">⬅️ 左转 (Time)</button>
                <button class="btn-main" onclick="triggerForward()">⬆️ 执行直行 (Forward)</button>
                <button class="btn-main btn-turn" onclick="triggerTurn('right')">➡️ 右转 (Time)</button>
            </div>
        </div>

        <div class="control-group">
            <h3>🎮 键盘驾驶 (WASD)</h3>
            <p style="font-size:0.8em; color:#999;">按键即走，松开即停</p>
        </div>
    </div>

    <script>
        // ===========================================
        // 1. ⚡ 3-Point Calibration System (三点校准)
        // ===========================================
        // 请在此处输入你实测的三个真实数据点
        // 现在的逻辑是：低速段和高速段会有不同的“斜率”，比单一线性更准。
        
        const CALIB = {
            low:  { pwm: 100, speed: 0.28 }, // 实测点 1 (低速)
            mid:  { pwm: 150, speed: 0.4 }, // 实测点 2 (中速)
            high: { pwm: 200, speed: 0.5 }, // 实测点 3 (高速)
            deadzone: 90                      // 电机死区 (低于此PWM不转)
        };

        // 内部自动计算两段斜率 (无需手动修改)
        const slope_low = (CALIB.mid.speed - CALIB.low.speed) / (CALIB.mid.pwm - CALIB.low.pwm);
        const slope_high = (CALIB.high.speed - CALIB.mid.speed) / (CALIB.high.pwm - CALIB.mid.pwm);
        
        // 最大理论速度预估 (用于滑块上限)
        const MAX_SPEED_EST = CALIB.high.speed + ((255 - CALIB.high.pwm) * slope_high);

        // 初始化变量
        let currentSpeed_ms = 0; 

        // ===========================================
        // 2. 🧮 Advanced Coupling Logic (分段计算)
        // ===========================================

        // PWM -> Speed (根据所在的区间选择不同的公式)
        function pwmToSpeed(pwm) {
            if (pwm < CALIB.deadzone) return 0;
            
            if (pwm <= CALIB.low.pwm) {
                // 极低速区间 (死区到低速点，假设线性)
                let s = (pwm - CALIB.deadzone) * (CALIB.low.speed / (CALIB.low.pwm - CALIB.deadzone));
                return Math.max(0, s);
            } 
            else if (pwm <= CALIB.mid.pwm) {
                // 区间 A: Low -> Mid
                return CALIB.low.speed + (pwm - CALIB.low.pwm) * slope_low;
            } 
            else {
                // 区间 B: Mid -> High -> Max
                return CALIB.mid.speed + (pwm - CALIB.mid.pwm) * slope_high;
            }
        }

        // Speed -> PWM (反向计算)
        function speedToPwm(speed) {
            if (speed <= 0) return 0;

            if (speed <= CALIB.low.speed) {
                // 极低速反算
                let ratio = speed / CALIB.low.speed;
                return CALIB.deadzone + (ratio * (CALIB.low.pwm - CALIB.deadzone));
            }
            else if (speed <= CALIB.mid.speed) {
                // 区间 A 反算
                return CALIB.low.pwm + (speed - CALIB.low.speed) / slope_low;
            }
            else {
                // 区间 B 反算
                return CALIB.mid.pwm + (speed - CALIB.mid.speed) / slope_high;
            }
        }

        // ===========================================
        // 3. UI Interaction Handlers
        // ===========================================

        function handlePwmChange(val) {
            let pwm = parseInt(val);
            document.getElementById("pwmInput").value = pwm;
            
            // 核心：计算对应速度
            let spd = pwmToSpeed(pwm);
            currentSpeed_ms = spd;
            
            // 更新 UI
            document.getElementById("speedSlider").value = spd.toFixed(3);
            document.getElementById("speedInput").value = spd.toFixed(3);

            fetch("/set_speed?val=" + pwm);
            calculateTimeFromDist(); // 保持距离不变，更新时间
        }

        function handleSpeedChange(val) {
            let spd = parseFloat(val);
            currentSpeed_ms = spd;
            document.getElementById("speedInput").value = spd.toFixed(3);

            // 核心：反算对应 PWM
            let pwm = Math.round(speedToPwm(spd));
            
            // 边界保护
            pwm = Math.max(0, Math.min(255, pwm));

            document.getElementById("pwmSlider").value = pwm;
            document.getElementById("pwmInput").value = pwm;

            fetch("/set_speed?val=" + pwm);
            calculateTimeFromDist();
        }

        function calculateTimeFromDist() {
            let dist = parseFloat(document.getElementById("distInput").value);
            if (isNaN(dist)) return;

            if (currentSpeed_ms <= 0.001) {
                document.getElementById("timeInput").value = "---";
                return;
            }
            // Time = Dist / Speed
            let time = dist / currentSpeed_ms;
            document.getElementById("timeInput").value = time.toFixed(2);
        }

        function calculateDistFromTime() {
            let time = parseFloat(document.getElementById("timeInput").value);
            if (isNaN(time)) return;

            // Dist = Speed * Time
            let dist = currentSpeed_ms * time;
            document.getElementById("distInput").value = dist.toFixed(2);
        }

        // ===========================================
        // 4. Initialization & Event Listeners
        // ===========================================
        
        window.onload = function() {
            // 设置速度滑块的最大值 (根据校准数据动态调整)
            document.getElementById("speedSlider").max = MAX_SPEED_EST.toFixed(2);
            
            // 初始化计算
            let startPwm = document.getElementById("pwmSlider").value;
            handlePwmChange(startPwm);
            
            // 默认距离填入
            if(currentSpeed_ms > 0) {
                 document.getElementById("distInput").value = (currentSpeed_ms * 1.0).toFixed(2);
            } else {
                 document.getElementById("distInput").value = "0.50";
            }
        };

        // 键盘控制逻辑
        let keys = {};
        document.addEventListener("keydown", (e) => {
            if (keys[e.key.toLowerCase()]) return;
            keys[e.key.toLowerCase()] = true;
            fetch("/manual_drive?act=down&key=" + e.key.toLowerCase());
        });
        document.addEventListener("keyup", (e) => {
            keys[e.key.toLowerCase()] = false;
            fetch("/manual_drive?act=up&key=" + e.key.toLowerCase());
        });

        // 按钮控制逻辑
        function setBusy(isBusy) {
            document.getElementById("lock").style.display = isBusy ? "inline" : "none";
        }

        function triggerTurn(direction) {
            let duration = document.getElementById("timeInput").value;
            if(duration <= 0 || isNaN(duration) || duration == "---") { alert("Time invalid!"); return; }

            setBusy(true);
            document.getElementById("status").innerText = "Turning " + direction + "...";
            fetch(`/timed_turn?dir=${direction}&t=${duration}`)
                .then(r => r.text()).then(() => {
                    document.getElementById("status").innerText = "Online";
                    setBusy(false);
                });
        }

        function triggerForward() {
            let duration = document.getElementById("timeInput").value;
            if(duration <= 0 || isNaN(duration) || duration == "---") { 
                alert("Increase speed or set time!"); return; 
            }
            setBusy(true);
            let dist = document.getElementById("distInput").value;
            document.getElementById("status").innerText = `Forward ${dist}m...`;
            fetch(`/timed_forward?t=${duration}`)
                .then(r => r.text()).then(() => {
                    document.getElementById("status").innerText = "Online";
                    setBusy(false);
                });
        }
    </script>
</body>
</html>
"""

# =========================================
# 4. Backend Routes
# =========================================
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, pwm=current_speed_pwm, freq=current_freq)

@app.route("/set_speed")
def route_set_speed():
    global current_speed_pwm
    if action_lock: return "locked"
    try:
        val = int(request.args.get("val", 0))
        current_speed_pwm = max(0, min(255, val))
        set_duty_cycle(current_speed_pwm)
        return "speed_ok"
    except ValueError:
        return "error"

@app.route("/manual_drive")
def route_manual():
    global action_lock
    if action_lock: return "locked"

    action = request.args.get("act")
    key = request.args.get("key")

    if action == "down":
        if key == 'w': forward()
        elif key == 's': backward()
        elif key == 'a': turn_left()
        elif key == 'd': turn_right()
        set_duty_cycle(current_speed_pwm)
    elif action == "up":
        if key in ['w', 's', 'a', 'd']:
            stop_motors()
    return "ok"

@app.route("/timed_turn")
def route_timed_turn():
    global action_lock
    if action_lock: return "busy"

    direction = request.args.get("dir")
    try:
        duration = float(request.args.get("t", 1.0))
    except ValueError:
        return "invalid_time"

    action_lock = True
    if direction == "left": turn_left()
    elif direction == "right": turn_right()
    
    set_duty_cycle(current_speed_pwm)
    time.sleep(duration)
    stop_motors()
    action_lock = False
    return "turn_complete"

@app.route("/timed_forward")
def route_timed_forward():
    global action_lock
    if action_lock: return "busy"

    try:
        duration = float(request.args.get("t", 1.0))
    except ValueError:
        return "invalid_time"

    action_lock = True
    forward()
    set_duty_cycle(current_speed_pwm)
    time.sleep(duration)
    stop_motors()
    action_lock = False
    return "forward_complete"

# =========================================
# 5. Run
# =========================================
if __name__ == "__main__":
    try:
        # Note: 'threaded=True' helps with multiple simultaneous requests (like slider dragging)
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        pwmA.stop()
        pwmB.stop()
        GPIO.cleanup()
        print("\nSystem Offline.")
