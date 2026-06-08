// tof_serial.ino
// --------------
// Reads 3 × VL53L5CX sensors through a Qwiic MUX and streams:
//     dist0,dist1,dist2\n      (mm; 65535 = no valid reading)
//
// Libraries required (Arduino Library Manager):
//   - SparkFun VL53L5CX Arduino Library
//   - SparkFun I2C Mux Arduino Library

#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <Arduino.h>

#define NUM_SENSORS  3
#define IMAGE_WIDTH  8
#define NUM_ZONES    (IMAGE_WIDTH * IMAGE_WIDTH)
#define INVALID_MM   4000
#define BAUD_RATE    115200

SparkFun_VL53L5CX imagers[NUM_SENSORS];
QWIICMUX          i2cMux;
VL53L5CX_ResultsData measurementData;

// ── median of valid zones ─────────────────────────────────────────────
uint16_t medianZones(VL53L5CX_ResultsData &data) {
  uint16_t valid[NUM_ZONES];
  uint8_t  count = 0;
  for (uint8_t i = 0; i < NUM_ZONES; i++) {
    uint16_t d = (uint16_t)data.distance_mm[i];
    uint8_t  s = data.target_status[i];
    if ((s == 5 || s == 9) && d < INVALID_MM) {
      valid[count++] = d;
    }
  }
  if (count == 0) return 65535;
  for (uint8_t i = 1; i < count; i++) {
    uint16_t key = valid[i];
    int8_t   j   = i - 1;
    while (j >= 0 && valid[j] > key) { valid[j+1] = valid[j]; j--; }
    valid[j+1] = key;
  }
  return valid[count / 2];
}

// ── I2C scan — prints every responding address ────────────────────────
void i2cScan() {
  Serial.println("  I2C scan:");
  uint8_t found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("    0x%02X\n", addr);
      found++;
    }
  }
  if (found == 0) Serial.println("    (nothing)");
}

// ── try to init mux on default pins then on 6/7 ──────────────────────
bool initMux() {
  // --- default pins ---
  Wire.begin();
  Wire.setClock(1000000);
  Serial.printf("Trying default pins (SDA=%d SCL=%d):\n", SDA, SCL);
  i2cScan();
  for (uint8_t addr = 0x70; addr <= 0x77; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0 && i2cMux.begin(addr, Wire)) {
      Serial.printf("  Mux at 0x%02X — OK\n", addr);
      return true;
    }
  }

  // --- Qwiic / ESP32-C6 pins 6/7 ---
  Serial.println("Not found. Trying SDA=6 SCL=7:");
  Wire.end();
  Wire.begin(6, 7);
  Wire.setClock(1000000);
  i2cScan();
  for (uint8_t addr = 0x70; addr <= 0x77; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0 && i2cMux.begin(addr, Wire)) {
      Serial.printf("  Mux at 0x%02X on pins 6/7 — OK\n", addr);
      return true;
    }
  }

  return false;
}

void setup() {
  Serial.begin(BAUD_RATE);
  delay(2000);
  Serial.println("=== TOF Serial Boot ===");

  if (!initMux()) {
    Serial.println("ERROR: Mux not found on any pin combo. Check wiring. Halting.");
    while (1) { delay(1000); }
  }

  // Show which mux ports have a sensor
  Serial.println("Scanning mux ports for sensors:");
  for (uint8_t port = 0; port < 8; port++) {
    i2cMux.setPort(port);
    Wire.beginTransmission(0x29);
    if (Wire.endTransmission() == 0) {
      Serial.printf("  Sensor at mux port %d\n", port);
    }
  }

  // Sensors are on mux ports 0, 2, 6  (J3=port0, J4=port2, J5=port6)
  const uint8_t MUX_PORTS[NUM_SENSORS] = {0, 2, 6};

  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    i2cMux.setPort(MUX_PORTS[i]);
    Serial.printf("Init sensor %d (mux port %d)... ", i, i);
    if (!imagers[i].begin()) {
      Serial.println("FAILED.");
      Serial.printf("  >> No sensor on mux port %d. Check J%d connector.\n", i, i+3);
      while (1) { delay(1000); }
    }
    imagers[i].setResolution(IMAGE_WIDTH * IMAGE_WIDTH);
    imagers[i].setRangingFrequency(15);
    imagers[i].startRanging();
    Serial.println("OK");
  }

  Serial.println("All sensors running. Streaming dist0,dist1,dist2...");
}

void loop() {
  const uint8_t MUX_PORTS[NUM_SENSORS] = {0, 2, 6};
  uint16_t distances[NUM_SENSORS];
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    i2cMux.setPort(MUX_PORTS[i]);
    while (!imagers[i].isDataReady()) { delay(5); }
    imagers[i].getRangingData(&measurementData);
    distances[i] = medianZones(measurementData);
  }
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    Serial.print(distances[i]);
    if (i < NUM_SENSORS - 1) Serial.print(",");
  }
  Serial.println();
}
