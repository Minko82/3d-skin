#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "credentials.hpp"
#include <Arduino.h>

#define NUM_SENSORS 4
#define START_ID 0
#define IMAGE_WIDTH 8

SparkFun_VL53L5CX imagers[NUM_SENSORS];
QWIICMUX i2cMux;

VL53L5CX_ResultsData measurementData;

WiFiUDP udp;

// Packet layout: [1 byte sensor ID] [64 uint16_t filtered] [64 uint16_t raw]
uint8_t full_pkt[1 + (sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH) * 2];

void error(const char* msg) {
  Serial.println(msg);
  Serial.flush();
  while (1) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(1000);
    digitalWrite(LED_BUILTIN, LOW);
    delay(1000);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  delay(3000);
  Serial.println("=== BOOT ===");
  Serial.flush();
  Serial.println("VL53L5CX 3-Sensor Mux - ESP32-C6 Mini");
  Serial.flush();

  // --- WiFi ---
  Serial.println("Starting WiFi...");
  Serial.flush();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("\nWiFi connected!");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  Serial.print("MAC Address: ");
  Serial.println(WiFi.macAddress());
  Serial.print("RSSI: ");
  Serial.print(WiFi.RSSI());
  Serial.println(" dBm");
  Serial.flush();

  udp.begin(UDP_PORT);
  WiFi.setSleep(false);

  // --- I2C + Mux + Sensors ---
  // Try multiple common Qwiic pin combos: default, then 6/7
  Serial.println("Starting I2C...");
  Serial.printf("Default I2C SDA=%d SCL=%d -- trying those first\n", SDA, SCL);
  Wire.begin(SDA, SCL);
  Wire.setClock(1000000);

  // Quick check if anything responds on default pins
  Wire.beginTransmission(0x70);
  uint8_t muxCheck = Wire.endTransmission();
  if (muxCheck != 0) {
    Serial.println("Nothing on default pins. Trying SDA=6 SCL=7 (common Qwiic)...");
    Wire.end();
    Wire.begin(6, 7);
    Wire.setClock(1000000);
  }

  Serial.println("I2C initialized.");
  Serial.flush();

  // Full I2C bus scan
  Serial.println("--- I2C bus scan ---");
  uint8_t devicesFound = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    uint8_t err = Wire.endTransmission();
    if (err == 0) {
      Serial.printf("  Found device at 0x%02X\n", addr);
      devicesFound++;
    }
  }
  Serial.printf("Scan complete: %d device(s) found.\n", devicesFound);
  Serial.flush();

  Serial.println("Checking mux at default address 0x70...");
  Serial.flush();

  if (i2cMux.begin() == false) {
    Serial.println("Mux not found at 0x70. Trying alternate addresses...");
    Serial.flush();
    bool muxFound = false;
    for (uint8_t muxAddr = 0x70; muxAddr <= 0x77; muxAddr++) {
      Wire.beginTransmission(muxAddr);
      if (Wire.endTransmission() == 0) {
        Serial.printf("  Mux responding at 0x%02X, trying begin()...\n", muxAddr);
        Serial.flush();
        if (i2cMux.begin(muxAddr, Wire)) {
          Serial.printf("  Mux initialized at 0x%02X!\n", muxAddr);
          muxFound = true;
          break;
        }
      }
    }
    if (!muxFound) {
      error("Mux not detected on any address (0x70-0x77). Check wiring. Freezing...");
    }
  }
  Serial.println("Mux detected.");
  Serial.flush();

  // Scan each mux port
  for (uint8_t port = 0; port < 8; port++) {
    i2cMux.setPort(port);
    Serial.printf("Mux port %d devices:", port);
    bool any = false;
    for (uint8_t addr = 1; addr < 127; addr++) {
      Wire.beginTransmission(addr);
      if (Wire.endTransmission() == 0) {
        Serial.printf(" 0x%02X", addr);
        any = true;
      }
    }
    if (!any) Serial.print(" (none)");
    Serial.println();
  }
  Serial.flush();

  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    Serial.printf("Initializing sensor %d on mux port %d...\n", i, i);
    Serial.flush();

    i2cMux.setPort(i);

    if (imagers[i].begin() == false) {
      char buf[64];
      snprintf(buf, sizeof(buf), "Sensor %d not found on port %d. Check wiring. Freezing...", i, i);
      error(buf);
    }

    imagers[i].setResolution(IMAGE_WIDTH * IMAGE_WIDTH);  // 8x8
    imagers[i].setRangingFrequency(15);

    uint32_t resolution = sqrt(imagers[i].getResolution());
    Serial.printf("Sensor %d initialized: %dx%d @ %d Hz\n",
                  i, resolution, resolution, imagers[i].getRangingFrequency());
    Serial.flush();

    imagers[i].startRanging();
  }

  digitalWrite(LED_BUILTIN, HIGH);
  Serial.println("All sensors running.");
}

void send_data(uint8_t id) {
  full_pkt[0] = id + START_ID;

  // Filtered data: replace invalid readings (status 255) with 4000
  uint16_t filtered[IMAGE_WIDTH * IMAGE_WIDTH];
  std::transform(
    std::begin(measurementData.distance_mm),
    std::end(measurementData.distance_mm),
    std::begin(measurementData.target_status),
    std::begin(filtered),
    [](const auto &dist, const auto &status) {
      return (status == 255) ? static_cast<uint16_t>(4000) : static_cast<uint16_t>(dist);
    }
  );

  // Pack: [id][filtered 128 bytes][raw 128 bytes]
  memcpy(full_pkt + 1, filtered, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH);
  memcpy(full_pkt + 1 + (sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH),
         measurementData.distance_mm,
         sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH);

  udp.beginPacket(UDP_HOST, UDP_PORT);
  udp.write(full_pkt, sizeof(full_pkt));
  udp.endPacket();
}

void loop() {
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    i2cMux.setPort(i);

    while (!imagers[i].isDataReady()) {
      delay(5);
    }

    imagers[i].getRangingData(&measurementData);
    send_data(i);
  }
}