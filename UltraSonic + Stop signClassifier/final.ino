#include <Servo.h>

Servo myservo;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 140;    // variable to store the servo position
int redPin = 3;
int bluePin = 5;
int greenPin = 6;
int sec = 1000

void setup() {
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);  
}

void loop() {
  myservo.write(40);
  color();
  for (pos = 40; pos <= 140; pos += 1) {
    // in steps of 1 degree
    myservo.write(pos);
    delay(20);
  }
  delay(sec*10);
  for (pos = 140; pos >= 40; pos -= 1) {
    myservo.write(pos);
    delay(40);
  }
  delay(sec);
}

void color(){
  setColor(0, 255, 255);  // Red
  delay(sec*3);
  setColor(255, 255, 0);  // Green
  delay(sec*10);
}

void setColor(int red, int green, int blue)
{
  analogWrite(redPin, red);
  analogWrite(greenPin, green);
  analogWrite(bluePin, blue);  
}

