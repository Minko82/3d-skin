"""
serial_monitor.py  —  quick serial sniffer with ESP32-C6 USB-reconnect support.
Run this, then press RESET on the ESP32. It will survive the USB dropout.

Usage:
    python Demo/serial_monitor.py
    python Demo/serial_monitor.py --port /dev/cu.usbmodem11101
"""
import sys, glob, time, argparse
try:
    import serial, serial.tools.list_ports
except ImportError:
    sys.exit("pip install pyserial")


def find_port(prefer=None):
    """Return the first matching USB-serial port, optionally preferring a specific one."""
    if prefer and glob.glob(prefer):
        return prefer
    for p in serial.tools.list_ports.comports():
        d = (p.description or "").lower()
        m = (p.manufacturer or "").lower()
        if any(k in d or k in m for k in
               ("cp210", "ch340", "ch9102", "ftdi", "esp", "usbserial", "acm", "usbmodem")):
            return p.device
    for pat in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*",
                "/dev/tty.usbmodem*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def wait_for_port(prefer=None, timeout=10.0):
    """Block until the port (re)appears after a USB dropout. Returns port name."""
    print("Waiting for ESP32 USB port to appear…", end="", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        p = find_port(prefer)
        if p:
            print(f"  found {p}")
            return p
        print(".", end="", flush=True)
        time.sleep(0.4)
    print()
    return None


p = argparse.ArgumentParser()
p.add_argument("--port",  "-p", default=None)
p.add_argument("--baud",  "-b", type=int, default=115200)
p.add_argument("--lines", "-n", type=int, default=0,
               help="Stop after N non-empty lines (0 = stream forever until Ctrl-C).")
args = p.parse_args()

# ── show all available ports first ──────────────────────────────────
all_ports = list(serial.tools.list_ports.comports())
if all_ports:
    print("Available ports:")
    for pp in all_ports:
        print(f"  {pp.device:<28} {pp.description}")
else:
    print("No serial ports visible yet — press RESET on the ESP32 and wait.")

port = args.port or find_port()
if not port:
    port = wait_for_port(args.port, timeout=15)
if not port:
    sys.exit("No port found after waiting. Check the USB cable.")

print(f"\nOpening {port} @ {args.baud} …")
try:
    ser = serial.Serial(port, args.baud, timeout=2)
except serial.SerialException as e:
    sys.exit(f"Could not open port: {e}")

time.sleep(1.5)
ser.reset_input_buffer()
limit = args.lines if args.lines > 0 else float("inf")
print("Connected. Streaming"
      + (f" {args.lines} lines." if args.lines > 0 else " forever (Ctrl-C to stop)."))
print(">>> If nothing appears, press RESET on the ESP32 now <<<")
print("─" * 60)

count = 0
buf = ""
try:
    while count < limit:
        try:
            chunk = ser.read(ser.in_waiting or 1)
        except serial.SerialException:
            # USB dropped (ESP32 reset) — wait for it to come back
            ser.close()
            print("\n[USB dropout detected — waiting for reconnect…]")
            port = wait_for_port(port, timeout=12)
            if not port:
                print("Port did not reappear. Exiting.")
                break
            time.sleep(1.5)           # let ESP32 finish booting
            ser = serial.Serial(port, args.baud, timeout=2)
            ser.reset_input_buffer()
            buf = ""
            print("[Reconnected. Reading…]\n")
            continue

        if chunk:
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if line:
                    print(repr(line))
                    count += 1
                    if count >= limit:
                        break
except KeyboardInterrupt:
    print("\n[Stopped by user]")

try:
    ser.close()
except Exception:
    pass
print("─" * 60)
print("Done. Paste the output above to fix the parser if needed.")
