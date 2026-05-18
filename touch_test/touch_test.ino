#include "CapacitiveSensorR4.h" // to use Arduino Uno R3, change this to Paul Bagder's Capacitive Sensing Library
 
const int SEND_PIN = 19;
const int SAMPLES = 30;
 
CapacitiveSensor cs_1 = CapacitiveSensor(SEND_PIN, 23);
CapacitiveSensor cs_2 = CapacitiveSensor(SEND_PIN, 22);
CapacitiveSensor cs_3 = CapacitiveSensor(SEND_PIN, 1);
CapacitiveSensor cs_4 = CapacitiveSensor(SEND_PIN, 5);
CapacitiveSensor cs_5 = CapacitiveSensor(SEND_PIN, 2);
CapacitiveSensor cs_6 = CapacitiveSensor(SEND_PIN, 3);
CapacitiveSensor cs_7 = CapacitiveSensor(SEND_PIN, 4);
 
 
void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
}
 
void loop() {
   //resistor should be pin 5 (start => in node)
  long sensorValue1 = cs_1.capacitiveSensor(SAMPLES);
  //long sensorValue2 = cs_2.capacitiveSensor(SAMPLES);
  //long sensorValue3 = cs_3.capacitiveSensor(SAMPLES);
  //long sensorValue4 = cs_4.capacitiveSensor(SAMPLES);
  long sensorValue5 = cs_5.capacitiveSensor(SAMPLES);
  //long sensorValue6 = cs_6.capacitiveSensor(SAMPLES);
  long sensorValue7 = cs_7.capacitiveSensor(SAMPLES);
  
  //Serial.print("S6,");
  Serial.print(sensorValue7); Serial.print(",");
  Serial.print(sensorValue1); Serial.print(",");
  //Serial.print(sensorValue2); Serial.print(",");
  //Serial.print(sensorValue3); Serial.print(",");
  //Serial.print(sensorValue4); Serial.print(",");
  Serial.println(sensorValue5);
  //Serial.print(sensorValue6); Serial.print(",");
}
 