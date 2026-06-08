#!/usr/bin/env python3
"""
Control VL53L5CX sensors on ESP32 boards via UDP commands.

Commands are sent to port 5050 on each ESP32. If no --esp-ip is given,
the script auto-discovers ESP32 addresses by briefly listening for data
packets on the sensor UDP ports (5005-5009).

Usage:
    python sensor_control.py --off              # stop all sensors on all boards
    python sensor_control.py --on               # start all sensors on all boards
    python sensor_control.py --off 0            # stop sensor 0 on all boards
    python sensor_control.py --on 1             # start sensor 1 on all boards
    python sensor_control.py --status           # query sensor states
    python sensor_control.py --off --esp-ip 192.168.1.42   # target specific board
"""

import argparse
import select
import socket
import sys
import time

CMD_PORT = 5050
DATA_PORTS = [5005, 5006, 5007, 5008, 5009]
DISCOVER_TIMEOUT = 3.0


def discover_esp32s(timeout=DISCOVER_TIMEOUT):
    """Listen on data ports to discover ESP32 source IPs."""
    socks = []
    for port in DATA_PORTS:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setblocking(False)
            s.bind(("0.0.0.0", port))
            socks.append(s)
        except OSError:
            pass

    if not socks:
        print("Could not bind to any data ports for discovery.")
        return set()

    print(f"Discovering ESP32 boards for {timeout:.0f}s (listening on ports {DATA_PORTS})...")
    ips = set()
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0)
        readable, _, _ = select.select(socks, [], [], min(remaining, 0.5))
        for s in readable:
            try:
                _, addr = s.recvfrom(4096)
                ips.add(addr[0])
            except BlockingIOError:
                pass

    for s in socks:
        s.close()

    if ips:
        print(f"Found {len(ips)} board(s): {', '.join(sorted(ips))}")
    else:
        print("No ESP32 boards discovered. Use --esp-ip to specify manually.")
    return ips


def send_command(ip, cmd, timeout=2.0):
    """Send a command to an ESP32 and return the response."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    try:
        sock.sendto(cmd.encode(), (ip, CMD_PORT))
        try:
            data, _ = sock.recvfrom(256)
            return data.decode().strip()
        except socket.timeout:
            return None
    finally:
        sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="Control VL53L5CX sensors on ESP32 boards via UDP",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--on", nargs="?", const="ALL", metavar="SENSOR_ID",
                       help="Start ranging (all sensors, or a specific ID)")
    group.add_argument("--off", nargs="?", const="ALL", metavar="SENSOR_ID",
                       help="Stop ranging (all sensors, or a specific ID)")
    group.add_argument("--status", action="store_true",
                       help="Query sensor states")
    parser.add_argument("--esp-ip", action="append", default=[],
                        help="ESP32 IP address (repeatable; skips auto-discovery)")
    parser.add_argument("--discover-timeout", type=float, default=DISCOVER_TIMEOUT,
                        help=f"Auto-discovery listen time in seconds (default: {DISCOVER_TIMEOUT})")
    args = parser.parse_args()

    if args.esp_ip:
        targets = set(args.esp_ip)
    else:
        targets = discover_esp32s(args.discover_timeout)
        if not targets:
            sys.exit(1)

    if args.status:
        cmd = "STATUS"
    elif args.on is not None:
        cmd = "START_ALL" if args.on == "ALL" else f"START:{args.on}"
    else:
        cmd = "STOP_ALL" if args.off == "ALL" else f"STOP:{args.off}"

    for ip in sorted(targets):
        print(f"  {ip} <- {cmd} ... ", end="", flush=True)
        resp = send_command(ip, cmd)
        if resp:
            print(resp)
        else:
            print("(no response — board may be unreachable)")


if __name__ == "__main__":
    main()
