
#include <LiquidCrystal.h>
#include <Wire.h>
#include <MPU6050_light.h>

// lcd mpu pins on arduino
LiquidCrystal lcd(8, 9, 4, 5, 6, 7);
MPU6050 mpu(Wire);

// motor pins on arduino
#define L_PWM   11
#define R_PWM   3
#define L_IN1   A1
#define L_IN2   2
#define R_IN1   12
#define R_IN2   13

// ramp detection
bool uphillTriggered = false;
unsigned long angleStartTime = 0;   // timer for when uphill angle is detected/reached
float angleFiltered = 0;
float angleMax = 0;

const float ANGLE_THRESHOLD = 6.0;      // angle at which anglestarttime timer will start
const unsigned long HOLD_TIME = 1000;   // time at which angle detected > angle threshold

// motor setup
void setMotor(int Lspd, int Rspd) {
    if (Lspd >= 0) { digitalWrite(L_IN1, HIGH); digitalWrite(L_IN2, LOW); }
    else { digitalWrite(L_IN1, LOW); digitalWrite(L_IN2, HIGH); Lspd = -Lspd; }

    if (Rspd >= 0) { digitalWrite(R_IN1, HIGH); digitalWrite(R_IN2, LOW); }
    else { digitalWrite(R_IN1, LOW); digitalWrite(R_IN2, HIGH); Rspd = -Rspd; }

    analogWrite(L_PWM, constrain(Lspd, 0, 255));
    analogWrite(R_PWM, constrain(Rspd, 0, 255));
}

//  obtain angle value from mpu
float readAngleX() {
    mpu.update();//update current angle value
    return mpu.getAngleX();//return the updated value
}

// update lcd to display angle
void updateLCD() {
    lcd.setCursor(0, 0);
    lcd.print("Angle:");
    lcd.print(angleFiltered);

    lcd.setCursor(0, 1);
    lcd.print("Max:");
    lcd.print(angleMax);
}

// startup
void setup() {
    lcd.begin(16, 2);

    Wire.begin();
    mpu.begin();// initialize the I²C-bus
    mpu.setFilterGyroCoef(0.95);   // 95% data from the gyroscope, can be changed
    mpu.calcOffsets();    // calibrate, obtain a default angle        
    delay(500); // time given for calcOffsets() to complete, car does not move

    pinMode(L_PWM, OUTPUT);
    pinMode(R_PWM, OUTPUT);
    pinMode(L_IN1, OUTPUT);
    pinMode(L_IN2, OUTPUT);
    pinMode(R_IN1, OUTPUT);
    pinMode(R_IN2, OUTPUT);

    lcd.print("TOUCH MORE GRASS");
    delay(1000);
    lcd.clear();

    setMotor(160, 130);   // set motor to run in straight line, different speed here to adjust for the issue with motor distributed, they run in different speed
}

// actions performed in a loop
void loop() {
    // get raw angle
    float raw = readAngleX();// defining the raw value
    angleFiltered = 0.3 * angleFiltered + 0.7 * raw;
    // new angle now carries 30% of the old value, 70% of updated value from readAngleX() 

    // record the maximum angle (for easier recording of the ramp angle)
    if (angleFiltered > angleMax) angleMax = angleFiltered;

    // ramp detection and climbing
    if (!uphillTriggered) {
        if (abs(angleFiltered) > ANGLE_THRESHOLD) {
            if (angleStartTime == 0) {
                angleStartTime = millis();  // start timer
            } //when timer > 1s, set motors to max
            else if (millis() - angleStartTime >= HOLD_TIME) {
                uphillTriggered = true;
                setMotor(255,255); //set both motors to max speed to climb ramp when ramp is detected
            }
        } else {
            angleStartTime = 0;  // reset timer when angle detected , threshold angle
        }
    }

    // action on the ramp
if (uphillTriggered) {

    if (abs(angleFiltered) <= ANGLE_THRESHOLD) {

        // stop 4 seconds
        delay(100);
        setMotor(0, 0);
        for (int i = 0; i < 4; i++) {
            updateLCD();
            delay(1000);
        }

        // rotate 360 degrees
        setMotor(255, -255);
        delay(3900);

        // slowly going down the ramp
        setMotor(0, 0); delay(1000);
        setMotor(100, 100); delay(1500);

        // stop the motors and complete task
        setMotor(0, 0);
        while (1);
    }
}

    updateLCD();
    delay(10);
}
