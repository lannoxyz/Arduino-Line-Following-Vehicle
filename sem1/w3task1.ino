
#include <LiquidCrystal.h>

// lcd pins on arduino
LiquidCrystal lcd(8, 9, 4, 5, 6, 7);

// assign sensor pins
#define IR_LEFT   0
#define IR_MID    A3
#define IR_RIGHT  A0

// assign encoder pins
#define ENC_LEFT  A1  // PCINT21
#define ENC_RIGHT A2   // PCINT20

// assign pins on arduino
#define L_PWM   11
#define R_PWM   3
#define L_IN1   1
#define L_IN2   2
#define R_IN1   12
#define R_IN2   13

//declare vehicle variables
const int baseSpeed = 90;
const float circumference = 39;
const float encoderPPR = 40.0;
volatile long leftCount = 0;
volatile long rightCount = 0;

unsigned long startTime = 0;
unsigned long allWhiteTimer = 0;
unsigned long allBlackTimer = 0;

float distance = 0.0;
bool stoppedForever = false;

volatile bool lastA1 = HIGH;
volatile bool lastA2 = HIGH;
bool pause = true;

// function declare
void readIRSensors(bool &L, bool &M, bool &R);
void motorStopHard();
void setMotor(int Lspd, int Rspd);
int speedTrim();
void updateLCD();
float updateDist();

// force stop and shut down motor (used when all 3 sensor detects black)
void motorStopHard() {
  analogWrite(L_PWM, 0);
  analogWrite(R_PWM, 0);
  digitalWrite(L_IN1, LOW);
  digitalWrite(L_IN2, LOW);
  digitalWrite(R_IN1, LOW);
  digitalWrite(R_IN2, LOW);
  stoppedForever = true;
}

// encoder read
volatile bool lastA4 = HIGH;
volatile bool lastA5 = HIGH;

ISR(PCINT1_vect) {
  bool currA1 = digitalRead(ENC_LEFT);
  bool currA2 = digitalRead(ENC_RIGHT);

  if (lastA1 == LOW && currA1 == HIGH) leftCount++;
  if (lastA2 == LOW && currA2 == HIGH) rightCount++;

  lastA1 = currA1;
  lastA2 = currA2;
}


// function to easily set speed of left and right motors
void setMotor(int Lspd, int Rspd) {

  // left
  if (Lspd >= 0) {
    digitalWrite(L_IN1, HIGH);
    digitalWrite(L_IN2, LOW);
  } else {
    digitalWrite(L_IN1, LOW);
    digitalWrite(L_IN2, HIGH);
    Lspd = -Lspd;
  }

  // right
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

// vehicle speed trim 
int speedTrim() {
  long diff = rightCount - leftCount;
  return constrain(diff * 2, -70, 70);
}

// Function to read light sensors
void readIRSensors(bool &L, bool &M, bool &R) {
  L = (digitalRead(IR_LEFT)  == LOW);
  M = (digitalRead(IR_MID)   == HIGH);
  R = (digitalRead(IR_RIGHT) == LOW);
}

// Calculate distance
float updateDist() {
  float rev = ((leftCount + rightCount) / 2.0) / encoderPPR;
  distance = rev * circumference;
  return distance;
}

// update lcd
void updateLCD() {
  lcd.setCursor(0, 0);
  lcd.print("Time: ");
  lcd.print((millis() - startTime) / 1000);
  lcd.print("s       ");

  updateDist();
  lcd.setCursor(0, 1);
  lcd.print("Dist: ");
  lcd.print(distance);
  lcd.print("cm      ");
}

// motor setup
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

  // 启用 PCINT1 (A0-A5)
  PCICR |= (1 << PCIE1);
  PCMSK1 |= (1 << PCINT9) | (1 << PCINT10);

  startTime = millis();

  lcd.print("Line Car Ready!");
  delay(1000);
}

// actions performed in a loop
void loop() {
  bool L, M, R;
  readIRSensors(L, M, R); //stop for 2 seconds when distance travelled = 330cm
  if (pause && distance>=330.0) { 
      setMotor(0,0); 
      delay(1000); 
      updateLCD();
      delay(1000); 
      updateLCD(); 
      pause=false;
      return; 
    }

  // shut down motors
  if (stoppedForever) {
    motorStopHard();
    return;
  }

      // shut down motors if all sensor detects black
  if (L && M && R) {
    motorStopHard();
    return;
  }

// 360 degree right turn when all sensor detects white for more than 200ms
if (!L && !M && !R) {
    if (allWhiteTimer == 0) allWhiteTimer = millis();  // 记录开始时间

    if (millis() - allWhiteTimer > 200) {      
        setMotor(200, -200);   // left speed 200 right speed -200
        updateLCD();
        return;
    }
} else {
    allWhiteTimer = 0;   // reset 300ms timer whenever black lines are detected
}


  // left and right motor speed after trim
  int trim = speedTrim();
  int Lspd = baseSpeed - trim;
  int Rspd = baseSpeed + trim;

  // ----------- 线路跟随逻辑 -----------
  if (M) {
    // black line middle
    setMotor(Lspd, Rspd);
  } 
  else if ((M && L) || (L && !M && !R)) {
    // black line on left - increase speed on right motor
    setMotor(Lspd - 200, Rspd + 190 );
  }
  else if ((M && R) || (!L && !M && R)){
    // black line on right - increase speed on left motor
    setMotor(Lspd + 100, Rspd - 150);
  }

  // update displays
  updateDist();
  updateLCD();
  delay(1);
}
