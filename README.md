# 3d-skin

## Test 2: Repeatability

The capacitive sketch in [legacy_cap 2.ino](/D:/Machines%20virtueles/3d-skin/legacy_cap%202/legacy_cap%202.ino) now supports a serial-triggered repeatability harness for a 6-cycle center-touch test.

### Capacitive sensor:
SparkFun Qwiic Pocket Development Board - ESP32-C6
link to the sensor: https://www.sparkfun.com/sparkfun-qwiic-pocket-development-board-esp32-c6.html

### Skin:
Link to the paper: https://hiro-group.ronc.one/papers/2026_Kohlbrenner_ICRA.pdf

### Operator workflow

1. Upload [legacy_cap 2.ino](/D:/Machines%20virtueles/3d-skin/legacy_cap%202/legacy_cap%202.ino) to the board and open the serial monitor at `115200` if you want to watch the device output.
2. Use a standardized `50 g` weight.
3. Do not touch the skin during the 2-second calibration phase.
4. When the LED is solid ON, place the weight on the center of the skin and leave it there for the full 5-second contact phase.
5. When the LED turns OFF, remove the weight and leave the skin untouched for the full 5-second release phase.
6. Repeat until the run finishes. The full run takes about 1 minute.

### PC logger

Install the Python dependency:

```powershell
python -m pip install -r tools/requirements.txt
```

Run the logger against the Arduino serial port:

```powershell
python tools/repeatability_logger.py --port COM3
```

The logger sends `TEST2_START`, waits for the `T2_RESULT` record, and writes:

- `artifacts/repeatability/test2_<timestamp>_samples.csv`
- `artifacts/repeatability/test2_<timestamp>_cycles.csv`
- `artifacts/repeatability/test2_<timestamp>_summary.json`

Each sample row now includes raw and baseline-normalized values for all 7 capacitive sensors, plus the `center_aggregate` used for repeatability scoring.

If you need to stop a run early, send `TEST2_ABORT` over serial.
