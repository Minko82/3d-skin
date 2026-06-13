/*
 * Cap_Skin_Firmware.ino
 * ---------------------
 * Firmware for the 3-node self-capacitive skin demo.
 *
 * Board:   Arduino Uno R4 (Minima / WiFi).
 *          Uses the bundled CapacitiveSensorR4 library (Uno R3 is NOT supported
 *          by this lib — swap in Paul Badger's CapacitiveSensor library instead).
 *
 * Wiring:  One shared SEND pin (19) drives the sensor electrodes through a
 *          high-value resistor (~1-10 MOhm). Each node has its own RECEIVE pin.
 *          The demo only uses Nodes 1, 5 and 7 (pins 23, 2, 4).
 *
 * Output:  One line per loop, three comma-separated integers, at 9600 baud:
 *
 *              <Node1>,<Node5>,<Node7>
 *
 *          This order and baud rate match Demo/Cap_Visualizer.py exactly.
 */

#include "CapacitiveSensorR4.h"

// ─── configuration ──────────────────────────────────────────────────────────
const int SEND_PIN = 19;   // shared drive pin for all electrodes
const int SAMPLES  = 30;   // samples averaged per reading (higher = smoother/slower)

// Receive pins for the three demo nodes.
const int RX_NODE_1 = 23;
const int RX_NODE_5 = 2;
const int RX_NODE_7 = 4;

// ─── sensors ────────────────────────────────────────────────────────────────
CapacitiveSensor cs_1 = CapacitiveSensor(SEND_PIN, RX_NODE_1);
CapacitiveSensor cs_5 = CapacitiveSensor(SEND_PIN, RX_NODE_5);
CapacitiveSensor cs_7 = CapacitiveSensor(SEND_PIN, RX_NODE_7);

void setup() {
  Serial.begin(9600);   // must match Cap_Visualizer.py (BAUD_DEFAULT = 9600)

  // Cap a stuck/over-coupled channel so one bad node can't stall the whole loop.
  cs_1.set_CS_Timeout_Millis(200);
  cs_5.set_CS_Timeout_Millis(200);
  cs_7.set_CS_Timeout_Millis(200);
}

void loop() {
  long sensorValue1 = cs_1.capacitiveSensor(SAMPLES);
  long sensorValue5 = cs_5.capacitiveSensor(SAMPLES);
  long sensorValue7 = cs_7.capacitiveSensor(SAMPLES);

  // Order matters: Node 1, Node 5, Node 7 — matches the visualizer.
  Serial.print(sensorValue1); Serial.print(",");
  Serial.print(sensorValue5); Serial.print(",");
  Serial.println(sensorValue7);
}
