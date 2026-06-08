// ─────────────────────────────────────────────────────────────────────────────
//  I2C_Pin_Scanner — finds which GPIO pins the I2C mux/sensors are wired to.
//  Flash this once, open Serial Monitor @ 115200, and read the result.
//  It reports every (SDA, SCL) pair that finds the TCA9548A mux (0x70) or any
//  I2C device, so you know exactly what to put in Wire.begin(SDA, SCL).
// ─────────────────────────────────────────────────────────────────────────────

#include <Wire.h>

// Candidate GPIOs to test on an ESP32-C6 Dev Module.
const int CANDIDATES[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23};
const int NCAND = sizeof(CANDIDATES) / sizeof(CANDIDATES[0]);

const byte MUX_ADDR = 0x70;
const byte MLX_ADDR = 0x0C;

void scanPair(int sda, int scl) {
  Wire.end();
  delay(5);
  if (!Wire.begin(sda, scl)) return;
  Wire.setClock(100000);
  delay(5);

  bool foundMux = false, foundMlx = false;
  String others = "";

  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      if (addr == MUX_ADDR)      foundMux = true;
      else if (addr == MLX_ADDR) foundMlx = true;
      else { others += "0x"; others += String(addr, HEX); others += " "; }
    }
  }

  if (foundMux || foundMlx || others.length() > 0) {
    Serial.print(">>> SDA=");
    Serial.print(sda);
    Serial.print(" SCL=");
    Serial.print(scl);
    Serial.print("  => ");
    if (foundMux) Serial.print("[MUX 0x70 FOUND] ");
    if (foundMlx) Serial.print("[MLX 0x0C FOUND] ");
    if (others.length()) { Serial.print("others: "); Serial.print(others); }
    Serial.println();
  }
}

void setup() {
  Serial.begin(115200);
  delay(3000);
  Serial.println("\n=== I2C Pin Scanner ===");
  Serial.println("Testing GPIO pairs for the mux (0x70) / MLX (0x0C)…\n");

  for (int a = 0; a < NCAND; a++) {
    for (int b = 0; b < NCAND; b++) {
      if (a == b) continue;
      scanPair(CANDIDATES[a], CANDIDATES[b]);
    }
  }

  Serial.println("\n=== Scan complete ===");
  Serial.println("Use the SDA/SCL pair marked [MUX 0x70 FOUND] in the firmware.");
}

void loop() {
  delay(1000);
}
