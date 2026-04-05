#include <Arduino.h>
#include <Wire.h>

// ultrasonic pins
#define TRIG A3
#define ECHO A2


// Motor pins
#define L_PWM 11
#define R_PWM 3
#define L_IN1 A1
#define L_IN2 2
#define R_IN1 12
#define R_IN2 13

int stopDistance = 50; // distance at which vehicle rotates when reached
void setMotor(int Lspd, int Rspd); //declare function

// function to control left and right motor speed
void setMotor(int Lspd, int Rspd) {

  // left motor
  if (Lspd >= 0) {
    digitalWrite(L_IN1, HIGH);
    digitalWrite(L_IN2, LOW);
  } else {
    digitalWrite(L_IN1, LOW);
    digitalWrite(L_IN2, HIGH);
    Lspd = -Lspd;
  }

  // right motor
  if (Rspd >= 0) {
    digitalWrite(R_IN1, HIGH);
    digitalWrite(R_IN2, LOW);
  } else {
    digitalWrite(R_IN1, LOW);
    digitalWrite(R_IN2, HIGH);
    Rspd = -Rspd;
  }

  analogWrite(L_PWM, constrain(Lspd, 0, 255));
  analogWrite(R_PWM, constrain(Rspd, 0, 255));
}

// get distance between sensor and obstacle
long getDistance() {
    digitalWrite(TRIG, LOW);//pulse emitted, low for 2 micrseconds
    delayMicroseconds(2);
    digitalWrite(TRIG, HIGH);//high for 10 microseconds
    delayMicroseconds(10);
    digitalWrite(TRIG, LOW);//return to low

    long duration = pulseIn(ECHO, HIGH, 25000); // duration at which obstacle is detected 25ms
    long distance = duration * 0.034 / 2;       // speed of sound =  340m/s → 0.034 cm/us
    return distance;
}

// startup
void setup() {
  pinMode(TRIG, OUTPUT);//signal sent to trig to emit pulse
  pinMode(ECHO, INPUT);//signal received from echo to determine range

  pinMode(L_PWM, OUTPUT);
  pinMode(R_PWM, OUTPUT);
  pinMode(L_IN1, OUTPUT);
  pinMode(L_IN2, OUTPUT);
  pinMode(R_IN1, OUTPUT);
  pinMode(R_IN2, OUTPUT);
  delay(2000);
  setMotor(150, 150); //vehicle travel in straight line
}

// rotate when obstacle, done in loop
void loop() {
  long d = getDistance();

    if (d > 0 && d < stopDistance) {
        setMotor(255,-255);     // obstacle detected -> rotate 360 degrees until no obstacle detected
        delay(1000);
    } else {
        setMotor(150, 150); // continue to travel in straight line once no more obstacles detected
    }

    delay(10);
}
