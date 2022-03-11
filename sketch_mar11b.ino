int data;

#include <Servo.h>
#include <SoftwareSerial.h>

#define RxD 3
#define TxD 2

SoftwareSerial BTserial(RxD, TxD);

int pos = 0;

Servo servo_9;
Servo servo_8;

void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT);
  digitalWrite(13, LOW);
  BTserial.begin(9600);
}

void loop() {
  while (BTserial.available()) {
    data = BTserial.read();
  }
  if (data == '1') {
    servo_9.attach(9, 500, 2500);
    servo_8.attach(8, 500, 2500);
    digitalWrite(13, HIGH);
    for (pos = 0; pos <= 180; pos += 1) {
      // tell servo to go to position in variable 'pos'
      servo_9.write(pos);
      servo_8.write(pos);
      // wait 15 ms for servo to reach the position
      delay(15); // Wait for 15 millisecond(s)
    }
    for (pos = 180; pos >= 0; pos -= 1) {
      // tell servo to go to position in variable 'pos'
      servo_9.write(pos);
      servo_8.write(pos);
      // wait 15 ms for servo to reach the position
      delay(15); // Wait for 15 millisecond(s)
    }
  }
  else if (data == '0') {
    digitalWrite(13, LOW);
    servo_9.detach();
    servo_8.detach();
    pos = 0;
  }
}
