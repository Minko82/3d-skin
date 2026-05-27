"""
touch_test_logger.py
--------------------
Reads capacitive sensor data streaming from the touch_test Arduino sketch
over a serial port and saves it to a timestamped CSV file.

touch_test output format (one line per loop iteration):
    sensorValue1,sensorValue5,sensorValue7

Usage:
    python touch_test_logger.py                        # auto-detect port
    python touch_test_logger.py --port /dev/ttyUSB0   # specify port (Linux/Mac)
    python touch_test_logger.py --port COM3            # specify port (Windows)
    python touch_test_logger.py --port /dev/ttyACM0 --output my_run.csv
    python touch_test_logger.py --duration 30          # stop after 30 seconds

Press Ctrl+C at any time to stop recording early.
"""

import argparse
import csv
import glob
import os
import sys
import time
from datetime import datetime

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: 'pyserial' is not installed.")
    print("  Run:  pip install pyserial")
    sys.exit(1)


# --------------------------------------------------------------------------- #
#  Port auto-detection
# --------------------------------------------------------------------------- #

def find_arduino_port():
    """Return the first likely Arduino serial port, or None if not found."""
    ports = list(serial.tools.list_ports.comports())
    candidates = []
    for p in ports:
        desc = (p.description or "").lower()
        mfr  = (p.manufacturer or "").lower()
        if any(kw in desc or kw in mfr for kw in ("arduino", "ch340", "cp210", "ftdi", "usbserial", "acm")):
            candidates.append(p.device)
    if candidates:
        return candidates[0]
    # Fallback glob patterns — macOS first, then Linux
    for pattern in (
        "/dev/tty.usbmodem*",   # macOS (Arduino Uno R4, Mega, etc.)
        "/dev/tty.usbserial*",  # macOS (CH340 / FTDI clones)
        "/dev/tty.wchusbserial*",
        "/dev/ttyACM*",         # Linux
        "/dev/ttyUSB*",         # Linux
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


# --------------------------------------------------------------------------- #
#  Main logger
# --------------------------------------------------------------------------- #

def log_touch_test(port, baud, output_path, duration):
    print(f"  Port     : {port}")
    print(f"  Baud     : {baud}")
    print(f"  Output   : {output_path}")
    print(f"  Duration : {'unlimited (Ctrl+C to stop)' if duration is None else f'{duration}s'}")
    print()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    row_count = 0
    start_time = None

    try:
        ser = serial.Serial(port, baud, timeout=2)
        print(f"Serial port opened. Waiting for data…\n")
        # Flush any stale bytes in the buffer
        time.sleep(0.5)
        ser.reset_input_buffer()
    except serial.SerialException as e:
        print(f"ERROR: Could not open serial port '{port}'.\n  {e}")
        sys.exit(1)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["timestamp", "elapsed_ms", "sensor_1", "sensor_5", "sensor_7"])

        try:
            while True:
                # Enforce duration limit
                if start_time and duration and (time.time() - start_time) >= duration:
                    print(f"\nDuration of {duration}s reached. Stopping.")
                    break

                raw = ser.readline()
                if not raw:
                    continue  # timeout with no data

                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 3:
                    # Skip malformed lines (e.g. partial startup noise)
                    print(f"  [skip] Unexpected line: {line!r}")
                    continue

                try:
                    s1, s5, s7 = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    print(f"  [skip] Non-integer values: {line!r}")
                    continue

                now = time.time()
                ts = datetime.now().isoformat(timespec="milliseconds")

                if start_time is None:
                    start_time = now
                    print("Recording…  (Ctrl+C to stop)\n")

                elapsed_ms = round((now - start_time) * 1000, 1)
                writer.writerow([ts, elapsed_ms, s1, s5, s7])
                csvfile.flush()          # write immediately so data isn't lost on Ctrl+C
                row_count += 1

                # Live feedback every 50 rows
                if row_count % 50 == 0:
                    print(f"  {row_count:>6} rows  |  last: {ts}  |  s1={s1}  s5={s5}  s7={s7}")

        except KeyboardInterrupt:
            print("\n\nStopped by user (Ctrl+C).")
        finally:
            ser.close()

    print(f"\nDone. {row_count} rows saved to: {output_path}")


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Log touch_test Arduino data to CSV."
    )
    parser.add_argument(
        "--port", "-p",
        default=None,
        help="Serial port (e.g. /dev/ttyACM0 or COM3). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=9600,
        help="Baud rate (default: 9600, must match touch_test.ino).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path. Defaults to artifacts/touch_test/touch_YYYYMMDD_HHMMSS.csv",
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=None,
        help="Stop recording after this many seconds (default: run until Ctrl+C).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve port
    port = args.port
    if port is None:
        port = find_arduino_port()
        if port is None:
            print("ERROR: Could not auto-detect an Arduino serial port.")
            print("  Connect your Arduino and try again, or specify --port manually.")
            sys.exit(1)
        print(f"Auto-detected port: {port}")

    # Resolve output path
    output = args.output
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(script_dir, "..", "artifacts", "touch_test")
        output = os.path.join(artifacts_dir, f"touch_{timestamp}.csv")

    print("=" * 55)
    print("  touch_test → CSV logger")
    print("=" * 55)
    log_touch_test(port, args.baud, output, args.duration)


if __name__ == "__main__":
    main()
