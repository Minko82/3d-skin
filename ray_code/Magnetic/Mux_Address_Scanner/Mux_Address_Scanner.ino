// ─────────────────────────────────────────────────────────────────────────────
//  Mux_Address_Scanner — lists every I2C device on every mux channel.
//  Reveals the real topology: how many boards (mux channels) and how many
//  magnetometer chips per board (I2C addresses).  Flash once, read the output.
// ─────────────────────────────────────────────────────────────────────────────

#include <Wire.h>

#define MUX_ADDRESS 0x70

void setMuxChannel(byte channel) {
  Wire.beginTransmission(MUX_ADDRESS);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void disableMux() {
  Wire.beginTransmission(MUX_ADDRESS);
  Wire.write(0x00);   // deselect all channels
  Wire.endTransmission();
}

void setup() {
  Serial.begin(115200);
  delay(3000);

  // This board uses SDA=6 SCL=7 (confirmed earlier).
  Wire.end();
  delay(5);
  Wire.begin(6, 7);
  Wire.setClock(100000);
  delay(5);

  Serial.println("\n=== Mux + Address Scanner ===");

  Wire.beginTransmission(MUX_ADDRESS);
  if (Wire.endTransmission() != 0) {
    Serial.println("No mux at 0x70 — check wiring.");
    return;
  }
  Serial.println("Mux found at 0x70.\n");

  for (byte ch = 0; ch < 8; ch++) {
    setMuxChannel(ch);
    delay(5);

    String found = "";
    int count = 0;
    // Scan the addresses a magnetometer board is likely to use.
    for (byte addr = 0x08; addr <= 0x1F; addr++) {
      Wire.beginTransmission(addr);
      if (Wire.endTransmission() == 0) {
        found += "0x";
        if (addr < 16) found += "0";
        found += String(addr, HEX);
        found += " ";
        count++;
      }
    }

    Serial.print("Channel ");
    Serial.print(ch);
    Serial.print(": ");
    if (count == 0) Serial.println("(empty)");
    else {
      Serial.print(count);
      Serial.print(" chip(s) -> ");
      Serial.println(found);
    }
    disableMux();
  }

  Serial.println("\n=== Scan complete ===");
  Serial.println("Tell me: how many channels have chips, and which addresses each shows.");
}

void loop() {
  delay(1000);
}
