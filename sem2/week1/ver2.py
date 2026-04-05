from flask import Flask, render_template_string, request
import RPi.GPIO as GPIO
import time

# ===== GPIO Setup =====
GPIO.setmode(GPIO.BCM)

# Direction pins
in1, in2 = 17, 27   # LEFT MOTOR
in3, in4 = 5, 6     # RIGHT MOTOR

# Speed pins
ena, enb = 12, 13

for p in [in1, in2, in3, in4]:
    GPIO.setup(p, GPIO.OUT)
    GPIO.output(p, 0)

GPIO.setup(ena, GPIO.OUT)
GPIO.setup(enb, GPIO.OUT)

pwmA = GPIO.PWM(ena, 1500)
pwmB = GPIO.PWM(enb, 1500)
pwmA.start(0)
pwmB.start(0)

current_speed = 80
current_freq = 1500
override_lock = False

# 车轮参数
wheel_circumference = 0.195  # m
pwm100_speed_rps = 2          # pwm100时轮子2转/s
pwm100_mps = pwm100_speed_rps * wheel_circumference  # m/s

# ===== 基本动作 =====
def set_speed(val):
    duty = max(0, min(100, val / 255 * 100))
    pwmA.ChangeDutyCycle(duty)
    pwmB.ChangeDutyCycle(duty)

def forward(): GPIO.output(in1,0); GPIO.output(in2,1); GPIO.output(in3,0); GPIO.output(in4,1)
def backward(): GPIO.output(in1,1); GPIO.output(in2,0); GPIO.output(in3,1); GPIO.output(in4,0)
def left(): GPIO.output(in1,0); GPIO.output(in2,1); GPIO.output(in3,1); GPIO.output(in4,0)
def right(): GPIO.output(in1,1); GPIO.output(in2,0); GPIO.output(in3,0); GPIO.output(in4,1)
def stop(): GPIO.output(in1,0); GPIO.output(in2,0); GPIO.output(in3,0); GPIO.output(in4,0)

# ===== Flask App =====
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>小车控制</title>
<style>
body { text-align:center; font-family:Arial; }
h2 { margin-top:20px; }
.slider {
  width: 80%;
  height: 40px;
  -webkit-appearance: none;
  appearance: none;
  border-radius: 10px;
  background: linear-gradient(90deg, green, yellow, red);
  outline: none;
}
.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 35px; height: 35px;
  background: #fff; border-radius:50%; cursor:pointer;
}
.small-slider {
  width:60%; height:20px;
}
.button { font-size:20px; padding:15px 25px; margin:8px; }
.fire { background:red; color:white; font-size:26px; padding:20px 40px; border-radius:12px; margin-top:15px; }
.display-box {
  border:1px solid #333; padding:6px 12px; display:inline-block; width:80px; margin-left:10px;
  font-weight:bold;
}
</style>
</head>
<body>

<h2>🚗 小车全功能控制面板</h2>
<p>W A S D 控制移动；松开停止<br>速度与 PWM 频率由下方滑条调节</p>

<h3>速度控制 (0-255)</h3>
<input type="range" min="0" max="255" value="{{speed}}" class="slider" id="speedBar" oninput="updateSpeed(this.value)">
<span class="display-box" id="speedVal">{{speed}}</span>

<h3>PWM 频率控制 (100Hz-1000Hz)</h3>
<input type="range" min="100" max="1000" value="{{freq}}" class="small-slider" id="freqBar" oninput="updateFreq(this.value)">
<span class="display-box" id="freqVal">{{freq}}</span>

<h3>当前小车速度 (m/s)</h3>
<span class="display-box" id="mpsVal">{{ "%.2f"|format(mps) }}</span>

<script>
let keys = {};
function updateSpeed(v){
    document.getElementById("speedVal").innerText = v;
    fetch("/speed?val=" + v);
    let mps = (v/100*0.39).toFixed(2);
    document.getElementById("mpsVal").innerText = mps;
}
function updateFreq(v){
    document.getElementById("freqVal").innerText = v;
    fetch("/freq?val=" + v);
}

document.addEventListener("keydown",(e)=>{
    if(keys[e.key]) return;
    keys[e.key]=true;
    fetch("/keydown?key="+e.key);
});
document.addEventListener("keyup",(e)=>{
    keys[e.key]=false;
    fetch("/keyup?key="+e.key);
});

// Rotation buttons
function sendBtn(cmd){ fetch("/rotate?cmd="+cmd); }

// Fire
function fire(){ fetch("/fire"); }
</script>

<h3>旋转控制</h3>
<button class="button" onclick="sendBtn('l45')">左转30°</button>
<button class="button" onclick="sendBtn('l90')">左转60°</button>
<button class="button" onclick="sendBtn('l180')">左转90°</button><br>
<button class="button" onclick="sendBtn('r45')">右转30°</button>
<button class="button" onclick="sendBtn('r90')">右转60°</button>
<button class="button" onclick="sendBtn('r180')">右转90°</button>

<h2>🔥 发射模式</h2>
<button class="fire" onclick="fire()">发射！！！🚀</button>

</body>
</html>
"""

@app.route("/")
def index():
    mps = current_speed / 100 * pwm100_mps
    return render_template_string(HTML, speed=current_speed, freq=current_freq, mps=mps)

@app.route("/speed")
def speed():
    global current_speed, override_lock
    if override_lock: return "locked"
    current_speed = int(request.args.get("val",80))
    set_speed(current_speed)
    return "ok"

@app.route("/freq")
def freq():
    global current_freq
    val = int(request.args.get("val",1500))
    val = max(100, min(1000, val))
    current_freq = val
    pwmA.ChangeFrequency(val)
    pwmB.ChangeFrequency(val)
    return "ok"

@app.route("/keydown")
def keydown():
    global override_lock
    if override_lock: return "locked"
    key = request.args.get("key")
    if key=="w": forward(); set_speed(current_speed)
    elif key=="s": backward(); set_speed(current_speed)
    elif key=="a": left(); set_speed(current_speed)
    elif key=="d": right(); set_speed(current_speed)
    return "ok"

@app.route("/keyup")
def keyup():
    global override_lock
    if override_lock: return "locked"
    key = request.args.get("key")
    if key in ["w","a","s","d"]: stop()
    return "ok"

@app.route("/rotate")
def rotate():
    global override_lock
    if override_lock: return "locked"
    cmd = request.args.get("cmd")
    mapping = {"45":0.40,"90":0.80,"180":1.20}
    if cmd.startswith("l"):
        angle=cmd[1:]; left(); set_speed(current_speed)
        time.sleep(mapping[angle]); stop()
    elif cmd.startswith("r"):
        angle=cmd[1:]; right(); set_speed(current_speed)
        time.sleep(mapping[angle]); stop()
    return "ok"

@app.route("/fire")
def fire():
    global override_lock
    override_lock = True
    forward()
    seq = [(0,100),(50,150),(100,200),(150,255)]
    for start,end in seq:
        for spd in range(start,end+1,5):
            set_speed(spd)
            time.sleep(2/(end-start))
    for spd in range(255,-1,-5):
        set_speed(spd)
        time.sleep(3/255)
    stop()
    override_lock = False
    return "fired!"

if __name__=="__main__":
    try:
        app.run(host="0.0.0.0",port=5000)
    finally:
        pwmA.stop(); pwmB.stop(); GPIO.cleanup()
