// ─────────────────────────────────────────────────────────────────────────────
//  Magnetic Skin Firmware — MLX90393 grid via TCA9548A I2C mux
//  ESP32-C6 (native USB), I2C on SDA=6 / SCL=7.
//
//  Topology (from the Mux_Address_Scanner):
//    Each populated mux channel = one BOARD with 5 chips at addresses
//    0x0C (centre) + 0x10,0x11,0x12,0x13 (corners).
//
//  Global sensor id = channel * 5 + chipIndex, so:
//    board on channel 0 → CH0..CH4,  channel 1 → CH5..CH9, etc.
//
//  Streams one line per cycle:
//    CH0: X:.. Y:.. Z:..   |   CH1: ...   | ...
//  which Demo/Mag_3D_Visualizer.py parses directly.
// ─────────────────────────────────────────────────────────────────────────────

#include <Wire.h>
#include "Adafruit_MLX90393.h"

#define MUX_ADDRESS 0x70

const byte CHIP_ADDRS[5] = {0x0C, 0x10, 0x11, 0x12, 0x13};  // centre + 4 corners
const byte CHIPS_PER_BOARD = 5;

bool  muxDetected = false;
byte  numChannels = 0;
byte  activeChannels[8];

Adafruit_MLX90393 sensors[8 * CHIPS_PER_BOARD];   // indexed by ch*5 + chip
bool  chipOk[8 * CHIPS_PER_BOARD];

void setMuxChannel(byte channel) {
  if (!muxDetected) return;
  Wire.beginTransmission(MUX_ADDRESS);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void configChip(int idx) {
  sensors[idx].setGain(MLX90393_GAIN_2_5X);
  sensors[idx].setResolution(MLX90393_X, MLX90393_RES_17);
  sensors[idx].setResolution(MLX90393_Y, MLX90393_RES_17);
  sensors[idx].setResolution(MLX90393_Z, MLX90393_RES_16);
  // Fast settings for low latency (adaptive thresholds tolerate the extra noise)
  sensors[idx].setOversampling(MLX90393_OSR_1);
  sensors[idx].setFilter(MLX90393_FILTER_2);
}

void setup() {
  Serial.begin(115200);
  delay(3000); // Wait for ESP32-C6 native USB

  // I2C on GPIO6/7 (confirmed with I2C_Pin_Scanner)
  Wire.end();
  delay(5);
  Wire.begin(6, 7);
  Wire.setClock(400000);
  delay(5);

  // STEP 1: find the mux
  Wire.beginTransmission(MUX_ADDRESS);
  if (Wire.endTransmission() == 0) {
    muxDetected = true;
    Serial.println("Multiplexer detected at 0x70");
  } else {
    Serial.println("No Multiplexer found — check wiring (SDA=6 SCL=7).");
    while (1);
  }

  // STEP 2: find which channels have a board.
  // Probe ALL 5 chip addresses — a board counts as present if ANY chip ACKs,
  // so a flaky/slow centre chip can't make us skip the whole board.
  delay(50);   // let sensors finish powering up before probing
  for (byte ch = 0; ch < 8; ch++) {
    setMuxChannel(ch);
    byte found = 0;
    for (byte j = 0; j < CHIPS_PER_BOARD; j++) {
      Wire.beginTransmission(CHIP_ADDRS[j]);
      if (Wire.endTransmission() == 0) found++;
    }
    if (found > 0) {
      activeChannels[numChannels++] = ch;
      Serial.print("Board found on channel "); Serial.print(ch);
      Serial.print(" ("); Serial.print(found); Serial.println(" chips)");
    }
  }
  Serial.print("Total boards: ");
  Serial.println(numChannels);

  // STEP 3: init every chip on every board (retry up to 3× per chip)
  for (byte c = 0; c < numChannels; c++) {
    byte ch = activeChannels[c];
    setMuxChannel(ch);
    for (byte j = 0; j < CHIPS_PER_BOARD; j++) {
      int idx = ch * CHIPS_PER_BOARD + j;
      bool ok = false;
      for (byte attempt = 0; attempt < 3 && !ok; attempt++) {
        ok = sensors[idx].begin_I2C(CHIP_ADDRS[j], &Wire);
        if (!ok) delay(20);
      }
      chipOk[idx] = ok;
      if (ok) configChip(idx);
      Serial.print("Init ch"); Serial.print(ch);
      Serial.print(" addr0x"); Serial.print(CHIP_ADDRS[j], HEX);
      Serial.print(" (CH"); Serial.print(idx); Serial.print(")");
      Serial.print(" status="); Serial.println(ok ? 0 : 1);
    }
  }

  Serial.println("Streaming...");
}

void loop() {
  if (numChannels == 0) return;

  float x, y, z;
  for (byte c = 0; c < numChannels; c++) {
    byte ch = activeChannels[c];
    setMuxChannel(ch);                   // select this board once

    // PIPELINE: kick off all 5 chips on this board so they convert in parallel,
    // wait once, then read them all. ~5× faster than read-one-at-a-time.
    for (byte j = 0; j < CHIPS_PER_BOARD; j++) {
      int idx = ch * CHIPS_PER_BOARD + j;
      if (chipOk[idx]) sensors[idx].startSingleMeasurement();
    }
    delay(8);   // conversion time for OSR_1 / FILTER_2 (all chips at once)

    for (byte j = 0; j < CHIPS_PER_BOARD; j++) {
      int idx = ch * CHIPS_PER_BOARD + j;
      Serial.print("CH"); Serial.print(idx); Serial.print(": ");
      if (chipOk[idx] && sensors[idx].readMeasurement(&x, &y, &z)) {
        Serial.print("X:"); Serial.print(x, 1); Serial.print(" ");
        Serial.print("Y:"); Serial.print(y, 1); Serial.print(" ");
        Serial.print("Z:"); Serial.print(z, 1); Serial.print("   |   ");
      } else {
        // read failed — try to re-init this chip for next cycle
        chipOk[idx] = sensors[idx].begin_I2C(CHIP_ADDRS[j], &Wire);
        if (chipOk[idx]) configChip(idx);
        Serial.print("[FAIL]   |   ");
      }
    }
  }
  Serial.println();
  // no extra delay — the 3 boards × 8 ms conversion already paces it (~30 Hz)
}
