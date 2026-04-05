import RPi.GPIO as GPIO
import time

# ====== GPIO 引脚 ======
# 左轮
IN1 = 11
IN2 = 13
ENA = 32

# 右轮
IN3 = 29
IN4 = 31
ENB = 33

# ====== GPIO 设置 ======
GPIO.setmode(GPIO.BOARD)  # 使用板子引脚编号
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# ====== PWM 设置 ======
left_pwm = GPIO.PWM(ENA, 500)   # 100 Hz
right_pwm = GPIO.PWM(ENB, 500)
left_pwm.start(50)
right_pwm.start(50)

# ====== 前进函数 ======
def forward(duration=30, speed=100):

    GPIO.output(ENA, 255)
    GPIO.output(ENB, 255)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    time.sleep(duration)
    print("Stopped")

# ====== 主程序 ======
try:
    forward()
finally:
    left_pwm.stop()
    right_pwm.stop()
    GPIO.cleanup()

