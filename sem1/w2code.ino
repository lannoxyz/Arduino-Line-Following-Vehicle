#include <LiquidCrystal.h>

// lcd pins on arduino
LiquidCrystal lcd(8, 9, 4, 5, 6, 7);

// light sensor pins on arduino
#define IR_LEFT   A3
#define IR_MID    A2
#define IR_RIGHT  A1

// encoder pins on arduino
#define ENC_LEFT  A5   // PCINT21
#define ENC_RIGHT A4   // PCINT20

// motor pins on arduino
#define L_PWM   11
#define R_PWM   3
#define L_IN1   1
#define L_IN2   2
#define R_IN1   12
#define R_IN2   13

// declare vehicle constant values
const int baseSpeed = 180;     
const float circumference = 23.5;   
const float encoderPPR = 40.0; 

// global variables
volatile long leftCount = 0;
volatile long rightCount = 0;

unsigned long startTime = 0;
unsigned long allWhiteTimer = 0;
unsigned long allBlackTimer = 0;

float distance = 0.0;
bool stoppedForever = false;

// function declare
void readIRSensors(bool &L, bool &M, bool &R);
void motorStopHard();
void setMotor(int Lspd, int Rspd);
int speedTrim();
void updateLCD();
float updateDist();

// function to stop and shut down all motors
void motorStopHard() {
  analogWrite(L_PWM, 0);
  analogWrite(R_PWM, 0);
  digitalWrite(L_IN1, LOW);
  digitalWrite(L_IN2, LOW);
  digitalWrite(R_IN1, LOW);
  digitalWrite(R_IN2, LOW);
  stoppedForever = true;
}

// left and right encoders count
volatile bool lastA4 = HIGH;
volatile bool lastA5 = HIGH;

ISR(PCINT1_vect) {
  bool currA4 = digitalRead(A4);
  bool currA5 = digitalRead(A5);

  if (lastA4 == LOW && currA4 == HIGH) leftCount++;
  if (lastA5 == LOW && currA5 == HIGH) rightCount++;

  lastA4 = currA4;
  lastA5 = currA5;
}

// function to control speed of left and right motors easily
void setMotor(int Lspd, int Rspd) {

  //left motors
  if (Lspd >= 0) {
    digitalWrite(L_IN1, HIGH);
    digitalWrite(L_IN2, LOW);
  } else {
    digitalWrite(L_IN1, LOW);
    digitalWrite(L_IN2, HIGH);
    Lspd = -Lspd;
  }

  // right motors
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

// speed trim function
int speedTrim() {
  long diff = rightCount - leftCount;
  return constrain(diff * 2, -60, 60);
}

// function to read outputs of light sensors
void readIRSensors(bool &L, bool &M, bool &R) {
  L = (digitalRead(IR_LEFT)  == LOW);
  M = (digitalRead(IR_MID)   == HIGH);
  R = (digitalRead(IR_RIGHT) == LOW);
}

// distance calculation
float updateDist() {
  float rev = ((leftCount + rightCount) / 2.0) / encoderPPR;
  distance = rev * circumference;
  return distance;
}

// update lcd display
void updateLCD() {
  lcd.setCursor(0, 0);
  lcd.print("Time: ");
  lcd.print((millis() - startTime) / 1000);
  lcd.print("s       ");

  lcd.setCursor(0, 1);
  lcd.print("Dist: ");
  lcd.print(distance);
  lcd.print("cm      ");
}

// startup the vehicle
void setup() {
  Serial.begin(9600);
  lcd.begin(16, 2);

  pinMode(IR_LEFT,  INPUT_PULLUP);
  pinMode(IR_MID,   INPUT_PULLUP);
  pinMode(IR_RIGHT, INPUT_PULLUP);

  pinMode(ENC_LEFT,  INPUT_PULLUP);
  pinMode(ENC_RIGHT, INPUT_PULLUP);

  pinMode(L_PWM, OUTPUT);
  pinMode(R_PWM, OUTPUT);
  pinMode(L_IN1, OUTPUT);
  pinMode(L_IN2, OUTPUT);
  pinMode(R_IN1, OUTPUT);
  pinMode(R_IN2, OUTPUT);

  PCICR |= (1 << PCIE1);
  PCMSK1 |= (1 << PCINT20) | (1 << PCINT21);

  startTime = millis();

  lcd.print("Line Car Ready!");
  delay(1000);
}

// actions done in a loop
void loop() {
  bool L, M, R;
  readIRSensors(L, M, R);

  // shut down motors
  if (stoppedForever) {
    motorStopHard();
    return;
  }

      // stop motors when all 3 sensor detects black lines
  if (L && M && R) {
    motorStopHard();
    return;
  }

// clockwise rotate 360 degrees when all 3 sensor detects white for more than 200ms
if (!L && !M && !R) {
    if (allWhiteTimer == 0) allWhiteTimer = millis();  // start timer whenever all sensor detects white

    if (millis() - allWhiteTimer > 200) {      
        setMotor(255, -255);   // rotate 360 degrees
        updateLCD();
        return;
    }
} else {
    allWhiteTimer = 0;   // reset timer when any of the sensor detects black
}

  // speed trim and adjustments on motors
  int trim = speedTrim();
  int Lspd = baseSpeed - trim;
  int Rspd = baseSpeed + trim;

  // line following
  if (M) {
    // middle gives high output
    // middle detect black
    setMotor(Lspd, Rspd);
  } 
  else if ((M && L) || (L && !M && !R)) {
    // middle high left low, OR left and middle low right high
    // black line on left, increase right motor speed
    setMotor(Lspd - 160, Rspd + 140);
  }
  else if ((M && R) || (!L && !M && R)) {
    // middle high right low, OR right and middle low left high
    // black line on right, increase left motor speed
    setMotor(Lspd + 140 , Rspd - 160);
  }

  // update display on lcd
  updateDist();
  updateLCD();
  delay(1);
}
