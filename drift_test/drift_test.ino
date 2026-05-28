#include "CapacitiveSensorESP32.h"

// ── Pin mapping (ESP32-C6) ──────────────────────────────────────────────────
// Update these to match your physical wiring.
// Avoid GPIO8/9 (boot strapping) and GPIO12/13 (USB D+/D- on USB-CDC boards).
const int SEND_PIN = 19;
const int SAMPLES  = 30;

CapacitiveSensor cs_1 = CapacitiveSensor(SEND_PIN, 23);
CapacitiveSensor cs_5 = CapacitiveSensor(SEND_PIN,  2);
CapacitiveSensor cs_7 = CapacitiveSensor(SEND_PIN,  4);
// ───────────────────────────────────────────────────────────────────────────

void setup() {
  Serial.begin(9600);
}

void loop() {
  long sensorValue7 = cs_7.capacitiveSensor(SAMPLES);
  long sensorValue1 = cs_1.capacitiveSensor(SAMPLES);
  long sensorValue5 = cs_5.capacitiveSensor(SAMPLES);

  Serial.print(sensorValue7); Serial.print(",");
  Serial.print(sensorValue1); Serial.print(",");
  Serial.println(sensorValue5);
}
