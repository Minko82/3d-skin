#include <Wire.h>
#include <SparkFun_VL53L5CX_Library.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include "credentials.hpp"
#include <Arduino.h>
#include <algorithm> 

#define START_ID 0
#define IMAGE_WIDTH 8

// Total number of sensors
#define NUM_SENSORS 5 

// I2C Addresses
#define MUX_MAIN_ADDR 0x70
#define MUX_SUB_ADDR  0x71
#define MUX_SUB2_ADDR 0x72 

// Port of PCB in which the Muxes are connected to
/*Port 0 = Socket J3
Port 2 = Socket J4
Port 4 = Socket J5
Port 6 = Socket J6*/

#define SUB_MUX_PORT  0    
#define SUB2_MUX_PORT 2    

struct SensorConfig {
  uint8_t muxNum; 
  uint8_t port;   
};

//Sensor Mapping
// Order by sensor number
//First Number = Mux number: 0 = Main Mux(PCB),1 = Sub Mux 1, 2 = Sub Mux 2                     
//Second Number = Port number in the mux.
//Example:
/*
SensorConfig sensorMap[NUM_SENSORS] = {
  {0, 0}, // Sensor 0: Main Mux, Port 0
  {1, 2}, // Sensor 1: Sub Mux 1 (0x71), Port 2 
  {2, 1}, // Sensor 2: Sub Mux 2 (0x72), Port 1
};*/

SensorConfig sensorMap[NUM_SENSORS] = {
  {1, 0},
  {2, 3},
  {2, 2},
  {1, 3},
  {2, 1}
};

SparkFun_VL53L5CX imagers[NUM_SENSORS];
QWIICMUX muxMain;

VL53L5CX_ResultsData measurementData;
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

void setAdafruitMuxPort(uint8_t addr, uint8_t port) {
  Wire.beginTransmission(addr);
  Wire.write(1 << port); 
  Wire.endTransmission();
}


void routeToSensor(uint8_t index) {
  if (sensorMap[index].muxNum == 1) {
    muxMain.setPort(SUB_MUX_PORT);
    setAdafruitMuxPort(MUX_SUB_ADDR, sensorMap[index].port);
  } 
  else if (sensorMap[index].muxNum == 2) {
    muxMain.setPort(SUB2_MUX_PORT);
    setAdafruitMuxPort(MUX_SUB2_ADDR, sensorMap[index].port);
  } 
  else {
    muxMain.setPort(sensorMap[index].port);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  delay(3000);
  Serial.println("=== BOOT ===");

  //WiFi
  Serial.println("Starting WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("\nWiFi connected!");
  Serial.println(WiFi.localIP());

  udp.begin(UDP_PORT);
  WiFi.setSleep(false);

  //I2C Initialization
  Serial.println("Starting I2C on pins 6 (SDA) and 7 (SCL)...");
  Wire.begin(6, 7); 
  Wire.setClock(1000000);

  //Mux Initialization
  Serial.println("Initializing Onboard Main Mux (0x70)...");
  if (!muxMain.begin(MUX_MAIN_ADDR, Wire)) {
    error("Main Mux not found at 0x70. Hardware issue. Freezing...");
  }
  
  muxMain.setPort(SUB_MUX_PORT);
  Wire.beginTransmission(MUX_SUB_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("WARNING: Sub Mux 1 not found at 0x71.");
  } else {
    Serial.println("Sub Mux 1 found at 0x71.");
  }

  muxMain.setPort(SUB2_MUX_PORT);
  Wire.beginTransmission(MUX_SUB2_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.println("WARNING: Sub Mux 2 not found at 0x72.");
  } else {
    Serial.println("Sub Mux 2 found at 0x72.");
  }

  Serial.println("All multiplexers initialized successfully.");

  //Sensor Initialization
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    Serial.printf("Initializing sensor %d on Mux Num %d, Port %d...\n", 
                  i, 
                  sensorMap[i].muxNum, 
                  sensorMap[i].port);

    routeToSensor(i);

    if (!imagers[i].begin()) {
      char buf[64];
      snprintf(buf, sizeof(buf), "Sensor %d failed to initialize. Freezing...", i);
      error(buf);
    }

    imagers[i].setResolution(IMAGE_WIDTH * IMAGE_WIDTH); 
    imagers[i].setRangingFrequency(15);
    imagers[i].startRanging();
  }

  digitalWrite(LED_BUILTIN, HIGH);
  Serial.println("All sensors running.");
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

void loop() {
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    routeToSensor(i);

    while (!imagers[i].isDataReady()) {
      delay(5);
    }

    imagers[i].getRangingData(&measurementData);
    send_data(i);
  }
}