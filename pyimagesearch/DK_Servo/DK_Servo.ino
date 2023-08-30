#include <Servo.h>
Servo myservo;
void setup() {
  // put your setup code here, to run once:
  myservo.attach(3);
  Serial.begin(115200);
}

void loop() {
  // put your main code here, to run repeatedly:

  String dulieu = "";
  dulieu = Serial.readStringUntil('\r');
  if (dulieu == "dong")
  {
    myservo.write(0);
  }
    if (dulieu == "mo")
  {
    myservo.write(180);
  }
}
