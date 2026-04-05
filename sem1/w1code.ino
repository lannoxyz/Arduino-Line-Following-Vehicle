#include <LiquidCrystal.h>

//pins on lcd
LiquidCrystal lcd(8, 9, 4, 5, 6, 7);

// Motor pins
const int IN1 = 1, IN2 = 2, IN3 = 12, IN4 = 13;
const int ENA = 11; // left wheel
const int ENB = 3;  // right wheel

unsigned long startMillis;
int secondsElapsed = 0;


//starting the vehicle
void setup() {
  lcd.begin(16, 2);
  lcd.print("Initializing...");
  delay(2000);
  
//all in pins setup to output since signals are sent out to the motor driver
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);

  moveStraight();
  //timer starts when vehicle starts
  startMillis = millis();
}

//function to move in straight line
void moveStraight() {
  digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH); digitalWrite(IN4, LOW);
}

//function to stop motor at 10s
void stopMotors() {
  //low output, no movement
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
}


void loop() {
  secondsElapsed = (millis() - startMillis) / 1000;

  // display on lcdeeeeee
  lcd.setCursor(0, 0);
  lcd.print("Time:");
  lcd.print(secondsElapsed);
  lcd.print("s ");

//stop motors at 10s
  if (secondsElapsed > 10) {
    stopMotors();
    lcd.clear();
    lcd.print("10 sec goal done");
    lcd.setCursor(0, 1);
    lcd.print("TOUCH MORE GRASS");
    while (1);
  }
}
