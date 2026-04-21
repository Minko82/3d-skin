from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

import serial


CHANNEL_COUNT = 7
CALIBRATION_SECONDS = 2.0
PHASE_SECONDS = 5.0
SETTLE_SECONDS = 1.0
DEFAULT_CYCLES = 6
DEFAULT_BAUD = 115200

PHASE_CALIBRATION = "CALIBRATION"
PHASE_CONTACT = "CONTACT"
PHASE_RELEASE = "RELEASE"
PHASE_COMPLETE = "COMPLETE"


@dataclass
class SampleRecord:
  run_ms: int
  cycle: int
  phase: str
  phase_elapsed_ms: int
  raw: list[int]
  aggregate: int | None


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Record Test 2 repeatability data from the reverted capacitive sketch."
  )
  parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/cu.usbmodem1101")
  parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Serial baud rate (default: {DEFAULT_BAUD})")
  parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES, help=f"Number of contact/release cycles (default: {DEFAULT_CYCLES})")
  parser.add_argument(
      "--out-dir",
      default=str(Path("artifacts") / "repeatability"),
      help="Artifact output directory",
  )
  return parser.parse_args()


def parse_sample_line(line: str) -> list[int] | None:
  parts = [part.strip() for part in line.split(",")]
  if len(parts) != CHANNEL_COUNT:
    return None

  try:
    return [int(part) for part in parts]
  except ValueError:
    return None


def current_phase(run_seconds: float, cycles: int) -> tuple[str, int, float]:
  if run_seconds < CALIBRATION_SECONDS:
    return PHASE_CALIBRATION, 0, run_seconds

  test_seconds = run_seconds - CALIBRATION_SECONDS
  cycle_span = PHASE_SECONDS * 2.0
  total_test_seconds = cycles * cycle_span
  if test_seconds >= total_test_seconds:
    return PHASE_COMPLETE, 0, 0.0

  cycle = int(test_seconds // cycle_span) + 1
  cycle_offset = test_seconds % cycle_span
  if cycle_offset < PHASE_SECONDS:
    return PHASE_CONTACT, cycle, cycle_offset
  return PHASE_RELEASE, cycle, cycle_offset - PHASE_SECONDS


def mean(values: list[int | float]) -> float:
  return sum(values) / len(values) if values else 0.0


def detect_live_channels(samples: list[SampleRecord]) -> list[int]:
  first_contact_samples = [
      sample
      for sample in samples
      if sample.cycle == 1 and sample.phase == PHASE_CONTACT
  ]
  return [
      index
      for index in range(CHANNEL_COUNT)
      if any(sample.raw[index] >= 0 for sample in first_contact_samples)
  ]


def compute_aggregate(raw: list[int], live_channels: list[int], detection_complete: bool) -> int | None:
  tracked_channels = live_channels if detection_complete else list(range(CHANNEL_COUNT))
  valid_values = [raw[index] for index in tracked_channels if raw[index] >= 0]
  return round(mean(valid_values)) if valid_values else None


def write_samples_csv(path: Path, samples: list[SampleRecord]) -> None:
  fieldnames = ["run_ms", "cycle", "phase", "phase_elapsed_ms"]
  fieldnames.extend(f"raw_ch{i}" for i in range(1, CHANNEL_COUNT + 1))
  fieldnames.append("all_pad_aggregate")

  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for sample in samples:
      row = {
          "run_ms": sample.run_ms,
          "cycle": sample.cycle,
          "phase": sample.phase,
          "phase_elapsed_ms": sample.phase_elapsed_ms,
          "all_pad_aggregate": sample.aggregate,
      }
      for index, value in enumerate(sample.raw, start=1):
        row[f"raw_ch{index}"] = value
      writer.writerow(row)


def build_cycle_metrics(samples: list[SampleRecord], cycles: int, live_channels: list[int]) -> list[dict[str, object]]:
  cycle_rows: list[dict[str, object]] = []
  has_live_channels = bool(live_channels)

  for cycle in range(1, cycles + 1):
    cycle_samples = [sample for sample in samples if sample.cycle == cycle]
    contact_samples = [
        sample.aggregate
        for sample in cycle_samples
        if sample.phase == PHASE_CONTACT and sample.phase_elapsed_ms >= int(SETTLE_SECONDS * 1000)
        and sample.aggregate is not None
    ]
    release_samples = [
        sample
        for sample in cycle_samples
        if sample.phase == PHASE_RELEASE and sample.phase_elapsed_ms >= int(SETTLE_SECONDS * 1000)
    ]

    contact_mean = mean(contact_samples)
    contact_samples_count = len(contact_samples)
    release_timeout_rate = (
        sum(1 for sample in release_samples if sample.aggregate is None) / len(release_samples)
        if release_samples
        else 0.0
    )
    valid = has_live_channels and bool(contact_samples) and release_timeout_rate >= 0.5

    cycle_rows.append({
        "cycle": cycle,
        "contact_mean": round(contact_mean, 3),
        "contact_samples_count": contact_samples_count,
        "release_timeout_rate": round(release_timeout_rate, 3),
        "valid": valid,
    })

  return cycle_rows


def write_cycles_csv(path: Path, cycles: list[dict[str, object]]) -> None:
  fieldnames = ["cycle", "contact_mean", "contact_samples_count", "release_timeout_rate", "valid"]
  with path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(cycles)


def write_summary_json(path: Path, summary: dict[str, object]) -> None:
  with path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")


def print_phase_prompt(phase: str, cycle: int) -> None:
  if phase == PHASE_CALIBRATION:
    print("[phase] CALIBRATION: do not touch the skin.")
  elif phase == PHASE_CONTACT:
    print(f"[phase] CONTACT cycle={cycle}: place the grounded 50 g metal target now.")
  elif phase == PHASE_RELEASE:
    print(f"[phase] RELEASE cycle={cycle}: remove the grounded target now.")


def main() -> int:
  args = parse_args()
  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  samples_path = out_dir / f"test2_{timestamp}_samples.csv"
  cycles_path = out_dir / f"test2_{timestamp}_cycles.csv"
  summary_path = out_dir / f"test2_{timestamp}_summary.json"

  try:
    ser = serial.Serial(args.port, args.baud, timeout=0.2)
  except serial.SerialException as exc:
    print(f"Serial error: {exc}", file=sys.stderr)
    return 1

  print(f"Opened {args.port} at {args.baud} baud.")
  print("Listening to raw capacitive stream. Test 2 timing is host-driven.")

  time.sleep(2.0)
  ser.reset_input_buffer()

  samples: list[SampleRecord] = []
  live_channels: list[int] = []
  live_channel_detection_complete = False

  start_time = time.monotonic()
  last_phase = None
  last_cycle = 0

  try:
    while True:
      run_seconds = time.monotonic() - start_time
      phase, cycle, phase_elapsed = current_phase(run_seconds, args.cycles)

      if phase != last_phase:
        if last_phase == PHASE_CONTACT and last_cycle == 1 and not live_channel_detection_complete:
          live_channels = detect_live_channels(samples)
          dead_channels = [index for index in range(CHANNEL_COUNT) if index not in live_channels]
          print(
              "[detect] Live channels after cycle 1 contact: "
              f"{[index + 1 for index in live_channels]} "
              f"(dead: {[index + 1 for index in dead_channels]})"
          )
          if not live_channels:
            print("Detection error: no live channels responded during cycle 1 contact.", file=sys.stderr)
          live_channel_detection_complete = True

        print_phase_prompt(phase, cycle)
        last_phase = phase
        last_cycle = cycle
        if phase == PHASE_COMPLETE:
          break

      line = ser.readline().decode("utf-8", errors="ignore").strip()
      if not line:
        continue

      sample = parse_sample_line(line)
      if sample is None:
        continue

      aggregate = compute_aggregate(sample, live_channels, live_channel_detection_complete)
      samples.append(SampleRecord(
          run_ms=round(run_seconds * 1000),
          cycle=cycle,
          phase=phase,
          phase_elapsed_ms=round(phase_elapsed * 1000),
          raw=sample,
          aggregate=aggregate,
      ))
  finally:
    ser.close()

  cycle_rows = build_cycle_metrics(samples, args.cycles, live_channels)
  contact_means = [float(row["contact_mean"]) for row in cycle_rows if bool(row["valid"])]
  median_contact = round(median(contact_means), 3) if contact_means else 0.0
  band_abs = max(5, round(median_contact * 0.05)) if contact_means else 0
  passed = bool(contact_means) and len(contact_means) == args.cycles
  if passed:
    passed = all(abs(float(row["contact_mean"]) - median_contact) <= band_abs for row in cycle_rows)

  summary = {
      "passed": passed,
      "cycles_requested": args.cycles,
      "valid_cycles": len(contact_means),
      "live_channels": [index + 1 for index in live_channels],
      "median_contact": median_contact,
      "band_abs": band_abs,
      "samples_path": str(samples_path),
      "cycles_path": str(cycles_path),
  }

  write_samples_csv(samples_path, samples)
  write_cycles_csv(cycles_path, cycle_rows)
  write_summary_json(summary_path, summary)

  print(f"Samples written to {samples_path}")
  print(f"Cycles written to {cycles_path}")
  print(f"Summary written to {summary_path}")
  live_channels_display = "[" + ",".join(str(index + 1) for index in live_channels) + "]"
  print(
      "Result: "
      f"{'PASS' if passed else 'FAIL'} | "
      f"median_contact={median_contact} | "
      f"band_abs={band_abs} | "
      f"valid_cycles={len(contact_means)}/{args.cycles} | "
      f"live_channels={live_channels_display}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
