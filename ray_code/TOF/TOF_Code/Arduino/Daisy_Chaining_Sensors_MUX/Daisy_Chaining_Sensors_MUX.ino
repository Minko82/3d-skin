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
#define NUM_SENSORS 7 

// I2C Addresses
#define MUX_MAIN_ADDR 0x70 // Hybrid PCB or main MUX
#define MUX_SUB_ADDR  0x71 // EXTERNAL MUX address

// Port of PCB in which the Mux is connected to(0, 2, 4, or 6)
#define SUB_MUX_PORT  6    

struct SensorConfig {
  bool onSubMux; // true if connected to the external second mux
  uint8_t port;  // The port number on its respective mux
};

/*
//Reference Code Left here if you need to add a mux to a mux or more sensors to the PCB
SensorConfig sensorMap[NUM_SENSORS] = {
  {false, 0}, // Sensor 0: PCB/Main Mux, Port 0 (J3)
  {false, 2}, // Sensor 1: PCB/Main Mux, Port 2 (J4)
  {false, 4}, // Sensor 2: PCB/Main Mux, Port 4 (J5)
  {true,  0}, // Sensor 3: External Mux, Port 0
  {true,  1}, // Sensor 4: External Mux, Port 1
  {true,  7}  // Sensor 5: External Mux, Port 2
};
*/
SensorConfig sensorMap[NUM_SENSORS] = {
  {true,  0}, // Sensor 3: External Mux, Port 0
  {true,  1}, // Sensor 4: External Mux, Port 1
  {true,  2},  // Sensor 5: External Mux, Port 2
  {true,  3},  // Sensor 5: External Mux, Port 3
  {true,  4},  // Sensor 5: External Mux, Port 4
  {true,  5},  // Sensor 5: External Mux, Port 5
  {true,  6}  // Sensor 5: External Mux, Port 6  
};

SparkFun_VL53L5CX imagers[NUM_SENSORS];
QWIICMUX muxMain;
QWIICMUX muxSub;

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

// --- Routing Helper Function ---
void routeToSensor(uint8_t index) {
  if (sensorMap[index].onSubMux) {
    // 1. Tell PCB/Main MUX to open the channel to the external mux socket
    muxMain.setPort(SUB_MUX_PORT);
    // 2. Tell PCB/Main MUX to open the channel to the specific sensor
    muxSub.setPort(sensorMap[index].port);
  } else {
    // Tell PCB/Main MUX to open the channel to the sensor directly
    muxMain.setPort(sensorMap[index].port);
  }
}

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  delay(3000);
  Serial.println("=== BOOT ===");
  Serial.println("Self-Cap Hybrid V1 - Cascaded Mux");

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
  
  // Open the port where the external mux is connected
  muxMain.setPort(SUB_MUX_PORT);
  
  Serial.println("Initializing External Sub Mux (0x71)...");
  if (!muxSub.begin(MUX_SUB_ADDR, Wire)) {
    error("Sub Mux not found at 0x71. Check A0 jumper on external board. Freezing...");
  }

  Serial.println("Both multiplexers initialized successfully.");

  // --- Sensor Initialization ---
  for (uint8_t i = 0; i < NUM_SENSORS; i++) {
    Serial.printf("Initializing sensor %d on %s Mux, Port %d...\n", 
                  i, 
                  sensorMap[i].onSubMux ? "External" : "Main Onboard", 
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