"""
drift_logger.py
---------------
Reads sensor data from the drift_test ESP32-C6 sketch over serial and saves
it to a single timestamped CSV in drift_test/artifacts/.

drift_test output format (one line per loop iteration):
    sensorValue7,sensorValue1,sensorValue5

Usage:
    python drift_test/drift_logger.py --port /dev/cu.usbmodem1101
    python drift_test/drift_logger.py --port COM3 --duration-hours 8
    python drift_test/drift_logger.py --port /dev/ttyUSB0 --output my_run.csv

Press Ctrl+C at any time to stop recording early.
"""

import argparse
import csv
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: 'pyserial' is not installed.")
    print("  Run:  pip install pyserial")
    sys.exit(1)


DEFAULT_BAUD          = 9600
DEFAULT_DURATION_HRS  = 8.0
PROGRESS_INTERVAL_SEC = 300.0  # status line every 5 min


def find_esp32_port():
    """Return the first likely ESP32 serial port, or None."""
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        desc = (p.description or "").lower()
        mfr  = (p.manufacturer or "").lower()
        if any(kw in desc or kw in mfr for kw in
               ("cp210", "ch340", "ch9102", "ftdi", "esp", "usbserial", "acm")):
            return p.device
    for pattern in (
        "/dev/tty.usbmodem*",
        "/dev/tty.usbserial*",
        "/dev/tty.wchusbserial*",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log drift_test ESP32-C6 data to a single CSV."
    )
    parser.add_argument(
        "--port", "-p",
        default=None,
        help="Serial port (e.g. /dev/ttyUSB0 or COM3). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate (default: {DEFAULT_BAUD}, must match drift_test.ino).",
    )
    parser.add_argument(
        "--duration-hours", "-d",
        type=float,
        default=DEFAULT_DURATION_HRS,
        help=f"Stop recording after this many hours (default: {DEFAULT_DURATION_HRS}).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path. Defaults to drift_test/artifacts/drift_YYYYMMDD_HHMMSS.csv",
    )
    return parser.parse_args()


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    args = parse_args()

    port = args.port
    if port is None:
        port = find_esp32_port()
        if port is None:
            print("ERROR: Could not auto-detect an ESP32 serial port.")
            print("  Connect your ESP32-C6 and try again, or specify --port manually.")
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    output = args.output
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = Path(__file__).parent / "artifacts"
        output    = str(out_dir / f"drift_{timestamp}.csv")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    duration_sec = args.duration_hours * 3600.0

    print("=" * 55)
    print("  drift_test → CSV logger")
    print("=" * 55)
    print(f"  Port     : {port}")
    print(f"  Baud     : {args.baud}")
    print(f"  Output   : {output}")
    print(f"  Duration : {args.duration_hours}h")
    print()
    print("  Do NOT touch the skin during this test.")
    print()

    try:
        ser = serial.Serial(port, args.baud, timeout=2)
        time.sleep(0.5)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"ERROR: Could not open serial port '{port}'.\n  {e}")
        sys.exit(1)

    row_count     = 0
    start_time    = None
    last_flush    = 0.0
    last_progress = 0.0

    with open(output, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "elapsed_ms", "sensor_1", "sensor_5", "sensor_7"])

        try:
            while True:
                if start_time and (time.time() - start_time) >= duration_sec:
                    print(f"\nDuration of {args.duration_hours}h reached. Stopping.")
                    break

                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 3:
                    print(f"  [skip] Unexpected line: {line!r}")
                    continue

                try:
                    s7, s1, s5 = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    print(f"  [skip] Non-integer values: {line!r}")
                    continue

                now = time.time()
                ts  = datetime.now().isoformat(timespec="milliseconds")

                if start_time is None:
                    start_time = now
                    print("Recording…  (Ctrl+C to stop early)\n")

                elapsed_ms = round((now - start_time) * 1000, 1)
                elapsed_s  = now - start_time

                writer.writerow([ts, elapsed_ms, s1, s5, s7])
                row_count += 1

                if elapsed_s - last_flush >= 5.0:
                    csvfile.flush()
                    last_flush = elapsed_s

                if elapsed_s - last_progress >= PROGRESS_INTERVAL_SEC:
                    print(
                        f"  [{format_elapsed(elapsed_s)}] "
                        f"{row_count} rows  |  "
                        f"s1={s1}  s5={s5}  s7={s7}"
                    )
                    last_progress = elapsed_s

        except KeyboardInterrupt:
            print("\n\nStopped early (Ctrl+C).")
        finally:
            csvfile.flush()
            ser.close()

    print(f"\nDone. {row_count} rows saved to: {output}")


if __name__ == "__main__":
    main()
