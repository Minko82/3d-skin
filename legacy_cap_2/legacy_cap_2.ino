#include "CapacitiveSensorR4.h" // to use Arduino Uno R3, change this to Paul Bagder's Capacitive Sensing Library

const int SEND_PIN = 19;
const int SAMPLES = 1;

CapacitiveSensor cs_1 = CapacitiveSensor(SEND_PIN, 23);
CapacitiveSensor cs_2 = CapacitiveSensor(SEND_PIN, 22);
CapacitiveSensor cs_3 = CapacitiveSensor(SEND_PIN, 1);
CapacitiveSensor cs_4 = CapacitiveSensor(SEND_PIN, 5);
CapacitiveSensor cs_5 = CapacitiveSensor(SEND_PIN, 2);
CapacitiveSensor cs_6 = CapacitiveSensor(SEND_PIN, 3);
CapacitiveSensor cs_7 = CapacitiveSensor(SEND_PIN, 4);

void setup() {
  Serial.begin(115200);
  Serial.println("Arduino has started!");
  cs_1.set_CS_Timeout_Millis(200);
  cs_2.set_CS_Timeout_Millis(200);
  cs_3.set_CS_Timeout_Millis(200);
  cs_4.set_CS_Timeout_Millis(200);
  cs_5.set_CS_Timeout_Millis(200);
  cs_6.set_CS_Timeout_Millis(200);
  cs_7.set_CS_Timeout_Millis(200);
}

void loop() {
   //resistor should be pin 5 (start => in node)
  long sensorValue1 = cs_1.capacitiveSensorRaw(SAMPLES);
  long sensorValue2 = cs_2.capacitiveSensorRaw(SAMPLES);
  long sensorValue3 = cs_3.capacitiveSensorRaw(SAMPLES);
  long sensorValue4 = cs_4.capacitiveSensorRaw(SAMPLES);
  long sensorValue5 = cs_5.capacitiveSensorRaw(SAMPLES);
  long sensorValue6 = cs_6.capacitiveSensorRaw(SAMPLES);
  long sensorValue7 = cs_7.capacitiveSensorRaw(SAMPLES);
  
  //Serial.print("S6,");
  Serial.print(sensorValue1); Serial.print(",");
  Serial.print(sensorValue2); Serial.print(",");
  Serial.print(sensorValue3); Serial.print(",");
  Serial.print(sensorValue4); Serial.print(",");
  Serial.print(sensorValue5); Serial.print(",");
  Serial.print(sensorValue6); Serial.print(",");
  Serial.println(sensorValue7);
}

/*
#include "CapacitiveSensorR4.h" // to use Arduino Uno R3, change this to Paul Bagder's Capacitive Sensing Library
#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>

const int SEND_PIN = 19;
const int SAMPLES = 30;

CapacitiveSensor cs_1 = CapacitiveSensor(SEND_PIN, 23);
CapacitiveSensor cs_2 = CapacitiveSensor(SEND_PIN, 22);
CapacitiveSensor cs_3 = CapacitiveSensor(SEND_PIN, 1);
CapacitiveSensor cs_4 = CapacitiveSensor(SEND_PIN, 5);
CapacitiveSensor cs_5 = CapacitiveSensor(SEND_PIN, 2);
CapacitiveSensor cs_6 = CapacitiveSensor(SEND_PIN, 3);
CapacitiveSensor cs_7 = CapacitiveSensor(SEND_PIN, 4);

// --- TOF CONFIGURATION ---
#define NUM_TOF_SENSORS 4
const uint8_t muxPorts[NUM_TOF_SENSORS] = {0, 2, 4, 6};
SparkFun_VL53L5CX imagers[NUM_TOF_SENSORS];
QWIICMUX i2cMux;

void setup() {
  Serial.begin(115200);

  Wire.begin(SDA, SCL);
  Wire.setClock(1000000);

  Wire.beginTransmission(0x70);
  if (Wire.endTransmission() != 0) {
    Wire.end();
    Wire.begin(6, 7);
    Wire.setClock(1000000);
  }

  if (!i2cMux.begin()) {
    for (uint8_t muxAddr = 0x70; muxAddr <= 0x77; muxAddr++) {
      Wire.beginTransmission(muxAddr);
      if (Wire.endTransmission() == 0 && i2cMux.begin(muxAddr, Wire)) break;
    }
  }

  for (uint8_t i = 0; i < NUM_TOF_SENSORS; i++) {
    i2cMux.setPort(muxPorts[i]);
    if (imagers[i].begin()) {
      imagers[i].setPowerMode(SF_VL53L5CX_POWER_MODE::SLEEP);
    }
    i2cMux.disablePort(muxPorts[i]);
  }
}

void loop() {
  // resistor should be pin 5 (start => in node)
  long sensorValue1 = cs_1.capacitiveSensor(SAMPLES);
  // long sensorValue2 = cs_2.capacitiveSensor(SAMPLES);
  long sensorValue3 = cs_3.capacitiveSensor(SAMPLES);
  // long sensorValue4 = cs_4.capacitiveSensor(SAMPLES);
  long sensorValue5 = cs_5.capacitiveSensor(SAMPLES);
  // long sensorValue6 = cs_6.capacitiveSensor(SAMPLES);
  // long sensorValue7 = cs_7.capacitiveSensor(SAMPLES);

  //Serial.print("S6,");
  Serial.print(sensorValue1); Serial.print(",");
  // Serial.print(sensorValue2); Serial.print(",");
  Serial.print(sensorValue3); Serial.print(",");
  // Serial.print(sensorValue4); Serial.print(",");
  // Serial.print(sensorValue5); Serial.print(",");
  // Serial.print(sensorValue6); Serial.print(",");
  Serial.println(sensorValue5);
}
*/
