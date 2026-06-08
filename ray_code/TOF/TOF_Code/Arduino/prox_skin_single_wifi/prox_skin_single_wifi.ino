#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "credentials.hpp"
#include <Arduino.h>

SparkFun_VL53L5CX imager;

VL53L5CX_ResultsData measurementData;

#define IMAGE_WIDTH 8
#define SENSOR_ID 0

WiFiUDP udp;

// Packet layout: [1 byte sensor ID] [64 uint16_t filtered] [64 uint16_t raw]
uint8_t full_pkt[1 + (sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH) * 2];

void error() {
  while (1) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(1000);
    digitalWrite(LED_BUILTIN, LOW);
    delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("VL53L5CX Single Sensor - ESP32-C6 Mini");

  // --- WiFi ---
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

  udp.begin(UDP_PORT);
  WiFi.setSleep(false);

  // --- I2C + Sensor ---
  pinMode(LED_BUILTIN, OUTPUT);

  Wire.begin();
  Wire.setClock(1000000); // VL53L5CX supports up to 1 MHz I2C

  if (imager.begin() == false) {
    Serial.println("Sensor not found. Check wiring. Freezing...");
    error();
  }

  imager.setResolution(IMAGE_WIDTH * IMAGE_WIDTH); // 8x8
  imager.setRangingFrequency(15);

  uint32_t resolution = sqrt(imager.getResolution());
  Serial.printf("Sensor initialized: %dx%d @ %d Hz\n", resolution, resolution, imager.getRangingFrequency());

  imager.startRanging();
  digitalWrite(LED_BUILTIN, HIGH);
}

void send_data() {
  full_pkt[0] = SENSOR_ID;

  // Filtered data: replace invalid readings (status 255) with 4000
  uint16_t filtered[(IMAGE_WIDTH * IMAGE_WIDTH)];
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
  while (!imager.isDataReady()) {
    delay(5);
  }

  imager.getRangingData(&measurementData);
  send_data();
}
