import RPi.GPIO as GPIO
import time

# ====== GPIO pins ======
# left motor
IN1 = 11
IN2 = 13
ENA = 32

# right motor
IN3 = 29
IN4 = 31
ENB = 33

# ====== GPIO configuration ======
GPIO.setmode(GPIO.BOARD)  # following board mode on pins naming
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

# ====== PWM setup ======
left_pwm = GPIO.PWM(ENA, 500)   # 500Hz pwm frequency
right_pwm = GPIO.PWM(ENB, 500)
left_pwm.start(50) # 50% duty cycle
right_pwm.start(50)

# ====== forward ======
def forward(duration=30, speed=100):

    GPIO.output(ENA, 255)
    GPIO.output(ENB, 255)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW) # left motor
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW) # right motor
    time.sleep(duration)
    print("Stopped")

# ====== call function ======
try:
    forward()
finally:
    left_pwm.stop()
    right_pwm.stop()
    GPIO.cleanup()

