#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "credentials.hpp"
#include <Arduino.h>


#define CAP_SENDER 19 
#define NUM_CAP_SENSORS 3
const uint8_t capReceivers[NUM_CAP_SENSORS] = {23, 1, 2};
long baselines[NUM_CAP_SENSORS];

#define NUM_TOF_SENSORS 3
#define START_ID 0
#define IMAGE_WIDTH 8

const uint8_t muxPorts[NUM_TOF_SENSORS] = {0,2,6};

SparkFun_VL53L5CX imagers[NUM_TOF_SENSORS];
QWIICMUX i2cMux;
VL53L5CX_ResultsData measurementData;

int latest_tof_averages[NUM_TOF_SENSORS] = {0, 0, 0};

WiFiUDP udp;
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


long readCapSensor(uint8_t sendPin, uint8_t receivePin) {
  long totalTime = 0;
  for (int i = 0; i < 30; i++) {
    pinMode(sendPin, OUTPUT);
    pinMode(receivePin, OUTPUT);
    digitalWrite(sendPin, LOW);
    digitalWrite(receivePin, LOW);
    delayMicroseconds(10);
    
    pinMode(receivePin, INPUT);
    digitalWrite(sendPin, HIGH);
    
    unsigned long start_time = micros();
    while(digitalRead(receivePin) == LOW && (micros() - start_time) < 4000) {}
    totalTime += (micros() - start_time);
  }
  return totalTime;
}

void send_data(uint8_t id) {
  full_pkt[0] = id + START_ID;

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

  memcpy(full_pkt + 1, filtered, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH);
  memcpy(full_pkt + 1 + (sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH),
         measurementData.distance_mm,
         sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH);

  udp.beginPacket(UDP_HOST, UDP_PORT);
  udp.write(full_pkt, sizeof(full_pkt));
  udp.endPacket();
}
void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  delay(3000);
  Serial.println("=== BOOT ===");
  Serial.flush();

  // --- 1. WiFi Setup ---
  Serial.println("Starting WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  udp.begin(UDP_PORT);
  WiFi.setSleep(false);

  // --- 2. I2C & ToF Setup ---
  Wire.begin(SDA, SCL);
  Wire.setClock(1000000);

  Wire.beginTransmission(0x70);
  if (Wire.endTransmission() != 0) {
    Wire.end();
    Wire.begin(6, 7);
    Wire.setClock(1000000);
  }

  if (i2cMux.begin() == false) {
    bool muxFound = false;
    for (uint8_t muxAddr = 0x70; muxAddr <= 0x77; muxAddr++) {
      Wire.beginTransmission(muxAddr);
      if (Wire.endTransmission() == 0) {
        if (i2cMux.begin(muxAddr, Wire)) {
          muxFound = true;
          break;
        }
      }
    }
    if (!muxFound) error("Mux not detected. Freezing...");
  }
  
  Serial.println("Mux detected. Initializing ToF sensors...");
  for (uint8_t i = 0; i < NUM_TOF_SENSORS; i++) {
    uint8_t port = muxPorts[i];
    i2cMux.setPort(port);

    if (imagers[i].begin() == false) {
      char buf[64];
      snprintf(buf, sizeof(buf), "Sensor %d not found on port %d.", i, port);
      error(buf);
    }
    imagers[i].setResolution(IMAGE_WIDTH * IMAGE_WIDTH); 
    imagers[i].setRangingFrequency(15);
    imagers[i].startRanging();
  }

  // --- 3. Capacitive Touch Calibration ---
  Serial.println("Calibrating Capacitive Pads... DO NOT TOUCH!");
  for(int i = 0; i < NUM_CAP_SENSORS; i++) { 
    baselines[i] = readCapSensor(CAP_SENDER, capReceivers[i]); 
    yield();
  }

  digitalWrite(LED_BUILTIN, HIGH);
  Serial.println("System Fully Online.");
}

// ==========================================
// --- MAIN LOOP ---
// ==========================================
unsigned long lastCapRead = 0;

void loop() {
  //Read TOF sensors
  for (uint8_t i = 0; i < NUM_TOF_SENSORS; i++) {
    i2cMux.setPort(muxPorts[i]); 

    if (imagers[i].isDataReady()) {
      imagers[i].getRangingData(&measurementData);
      send_data(i); 

      long sum = 0;
      int valid_zones = 0;
      for(int j = 0; j < 64; j++) {
        if(measurementData.target_status[j] != 255) {
          sum += measurementData.distance_mm[j];
          valid_zones++;
        }
      }
      latest_tof_averages[i] = (valid_zones > 0) ? (sum / valid_zones) : 4000;
    }
  }

  if (millis() - lastCapRead >= 50) {
    lastCapRead = millis();
    
    Serial.print("CAP: ");
    for (int i = 0; i < NUM_CAP_SENSORS; i++) {
      long val = readCapSensor(CAP_SENDER, capReceivers[i]) - baselines[i];
      Serial.print(val);
      Serial.print(" | ");
    }
    
    Serial.print("  TOF(avg mm): ");
    for (int i = 0; i < NUM_TOF_SENSORS; i++) {
      Serial.print(latest_tof_averages[i]);
      Serial.print(" | ");
    }
    Serial.println();
  }

  delay(1); 
}