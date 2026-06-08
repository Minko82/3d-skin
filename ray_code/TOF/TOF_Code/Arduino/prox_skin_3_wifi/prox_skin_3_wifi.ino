#include <Wire.h>

#include <SparkFun_VL53L5CX_Library.h>
#include <SparkFun_I2C_Mux_Arduino_Library.h> //Click here to get the library: http://librarymanager/All#SparkFun_I2C_Mux
#include <WiFi.h>
#include <WiFiUdp.h>
#include "credentials.hpp"
#include <Arduino.h>

#define NUM_SENSORS 3

#define START_ID 0

SparkFun_VL53L5CX imagers[NUM_SENSORS];
QWIICMUX i2cMux;

VL53L5CX_ResultsData measurementData;

#define IMAGE_WIDTH 8

//this is used to ensure the correct frame is being read
#define FRAME_HEADER_1 0xA5
#define FRAME_HEADER_2 0x5A

WiFiUDP udp;

uint8_t pkt[1 + (sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH)];
uint8_t full_pkt [1 + (sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH)*2];
// Sends data as 3 blocks within 1 array: 0th index is sensor index, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH is actual sensor data
// Second sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH block is data modified based on if it is a valid point or not


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
Serial.println("HIRO Group ToF Imager Sensor");

WiFi.begin(WIFI_SSID, WIFI_PASS);

while (WiFi.status() != WL_CONNECTED) {
delay(100);
Serial.println("wifi");
}
Serial.println("udp packet");
Serial.println(String(udp.endPacket()));

Serial.println("\nWiFi connected!");
Serial.print("IP Address: ");
Serial.println(WiFi.localIP());

Serial.print("MAC Address: ");
Serial.println(WiFi.macAddress());

Serial.print("SSID: ");
Serial.println(WiFi.SSID());

Serial.print("Signal strength (RSSI): ");
Serial.print(WiFi.RSSI());
Serial.println(" dBm");

Serial.print("Gateway IP: ");
Serial.println(WiFi.gatewayIP());

Serial.print("DNS IP: ");
Serial.println(WiFi.dnsIP());
udp.begin(UDP_PORT);
WiFi.setSleep(false);

pinMode(LED_BUILTIN, OUTPUT);

Wire.begin(); //This resets I2C bus to 100kHz
Wire.setClock(1000000); //Sensor has max I2C freq of 1MHz

if (i2cMux.begin() == false) {
Serial.println("Mux not detected. Freezing...");
error();
}

for (uint8_t i = 0; i < NUM_SENSORS; i++) {
Serial.printf("Initializing sensor: %d.\n", i);

i2cMux.setPort(i);

if (imagers[i].begin() == false) {
Serial.printf("Unable to initialize sensor: %d, freezing.\n", i);

error();
}

imagers[i].setResolution(IMAGE_WIDTH * IMAGE_WIDTH);

imagers[i].setRangingFrequency(15);

uint32_t resolution = sqrt(imagers[i].getResolution());

Serial.printf("Sensor: %d initialized with resolution %dx%d and frequency %d\n", i, resolution, resolution, imagers[i].getRangingFrequency());

imagers[i].startRanging();
}

digitalWrite(LED_BUILTIN, HIGH);
}

void printMatrix(int16_t* data) {
Serial.println("Distance Matrix (mm):");
for (int row = 0; row < IMAGE_WIDTH; row++) {
for (int col = 0; col < IMAGE_WIDTH; col++) {
Serial.printf("%6d ", data[row * IMAGE_WIDTH + col]);
}
Serial.println();
}
Serial.println();
}

template <class OutputIter, class UnaryFunction>
void apply_pointwise(OutputIter first, OutputIter last, UnaryFunction f)
{
std::transform(first, last, first, f);
}

//4000 - invalid data
//0 - valid data


void send_data(uint8_t id) {

pkt[0] = id + START_ID;
full_pkt[0] = id + START_ID;

//if you set this to uint8_t, no weird zeroes. but then the sent packaged data is wrong.
uint16_t final_pkt[(sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH)]; // Data that is invalid is shown as 4000, valid data is shown as actual sensor value
std::transform(std::begin( measurementData.distance_mm),
std::end( measurementData.distance_mm),
std::begin(measurementData.target_status),
std::begin(final_pkt),
[](const auto &pkt_val, const auto &status){
Serial.print(status);
Serial.print(" ");
return ((status == 255)/* || (pkt_val > 300)*/) ? static_cast<uint16_t>(4000) : static_cast<uint16_t>(pkt_val);
});
Serial.println("");
// This sends data that is only valid. Invalid points are sent as 4000 (4mm).
memcpy(full_pkt + 1, final_pkt, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH); // This sends actual data
memcpy(full_pkt + 1 + (sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH), measurementData.distance_mm, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH);
// Valid points are sent as the reported value in mm from the sensor.

//print only the first sensor's output
if (id==0){
//printMatrix(measurementData.distance_mm);
size_t len = sizeof( measurementData.distance_mm) / sizeof( measurementData.distance_mm[0]);
//for (size_t i = 0; i < len; i++){
//Serial.print(sensor_status[i]);
//Serial.print(pkt[i]);
//Serial.print(measurementData[i]);
//Serial.print(" ");
//}
//Serial.println("");
}

//memcpy(pkt + 1, measurementData.distance_mm, sizeof(uint16_t) * IMAGE_WIDTH * IMAGE_WIDTH); // this only sends the actual sensor data

udp.beginPacket(UDP_HOST, UDP_PORT);
udp.write(full_pkt, sizeof(full_pkt));
udp.endPacket();


// Serial.print("hello from" + uint8_t(id));
}

void loop() {
for (uint8_t i = 0; i < NUM_SENSORS; i++) {
// Serial.printf("Getting data from sensor: %d.\n", i);

i2cMux.setPort(i);

while (!imagers[i].isDataReady()) {
delay(5); //Small delay between polling
}

imagers[i].getRangingData(&measurementData);

send_data(i);

/*
//unfinished code below
//add frame header and checksum
uint8_t frameData[128]; // 64 values × 2 bytes each
uint8_t checksum = 0;

// Copy signed 16-bit data correctly using memcpy
for (int i = 0; i < 64; i++) {
int16_t value = measurementData.distance_mm[i]; // Read as int16_t
memcpy(&frameData[2 * i], &value, sizeof(int16_t)); // Preserve sign

//use XOR checksum
checksum ^= frameData[2 * i]; // Update checksum
checksum ^= frameData[2 * i + 1]; // Update checksum
}
delay(10); // Prevent buffer overflow
*/
//Serial.println(checksum);
//Serial.println(PORT);

}
}