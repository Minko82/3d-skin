from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path


CHANNEL_COUNT = 7
DEFAULT_BAUD = 115200


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Record Test 1 drift data from the capacitive sketch."
  )
  parser.add_argument("--port", required=True, help="Serial port, e.g. COM3 or /dev/cu.usbmodem1101")
  parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Serial baud rate (default: {DEFAULT_BAUD})")
  parser.add_argument(
      "--duration-hours",
      type=float,
      default=8.0,
      help="Total capture duration in hours (default: 8.0)",
  )
  parser.add_argument(
      "--checkpoint-interval-min",
      type=float,
      default=30.0,
      help="Minutes between checkpoint starts (default: 30.0)",
  )
  parser.add_argument(
      "--checkpoint-window-sec",
      type=float,
      default=60.0,
      help="Checkpoint window size in seconds (default: 60.0)",
  )
  parser.add_argument(
      "--ghost-rate-threshold",
      type=float,
      default=0.05,
      help="Ghost-rate failure threshold (default: 0.05)",
  )
  parser.add_argument(
      "--out-dir",
      default=str(Path("artifacts") / "drift"),
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


def mean(values: list[int | float]) -> float:
  return sum(values) / len(values) if values else 0.0


def format_elapsed(seconds: float) -> str:
  total_seconds = max(0, int(seconds))
  hours = total_seconds // 3600
  minutes = (total_seconds % 3600) // 60
  secs = total_seconds % 60
  return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def checkpoint_minute_value(start_sec: float) -> int | float:
  minutes = start_sec / 60.0
  if float(minutes).is_integer():
    return int(minutes)
  return round(minutes, 3)


def build_checkpoint_rows(checkpoint: dict[str, object]) -> list[dict[str, object]]:
  window_samples = checkpoint["samples"]
  rows: list[dict[str, object]] = []
  for channel_index in range(CHANNEL_COUNT):
    raw_values = [sample[channel_index] for sample in window_samples]
    live_values = [value for value in raw_values if value >= 0]
    sample_count = len(raw_values)
    live_count = len(live_values)
    ghost_rate = (live_count / sample_count) if sample_count else 0.0
    ghost_mean = round(mean(live_values), 1) if live_values else None
    rows.append({
        "checkpoint_minute": checkpoint["minute"],
        "channel": channel_index + 1,
        "sample_count": sample_count,
        "live_count": live_count,
        "ghost_rate": round(ghost_rate, 3),
        "ghost_mean": ghost_mean,
    })
  return rows


def open_due_checkpoints(
    elapsed_seconds: float,
    checkpoint_times: list[float],
    checkpoint_window_sec: float,
    duration_seconds: float,
    next_checkpoint_index: int,
    open_checkpoints: list[dict[str, object]],
) -> int:
  while next_checkpoint_index < len(checkpoint_times) and elapsed_seconds >= checkpoint_times[next_checkpoint_index]:
    start_sec = checkpoint_times[next_checkpoint_index]
    minute_value = checkpoint_minute_value(start_sec)
    open_checkpoints.append({
        "index": next_checkpoint_index + 1,
        "minute": minute_value,
        "start_sec": start_sec,
        "close_sec": min(start_sec + checkpoint_window_sec, duration_seconds),
        "samples": [],
        "truncated": start_sec + checkpoint_window_sec > duration_seconds,
    })
    print(
        f"[t={format_elapsed(elapsed_seconds)}] "
        f"checkpoint {next_checkpoint_index + 1}/{len(checkpoint_times)} window open "
        f"(minute={minute_value})"
    )
    next_checkpoint_index += 1
  return next_checkpoint_index


def close_finished_checkpoints(
    elapsed_seconds: float,
    open_checkpoints: list[dict[str, object]],
    checkpoint_writer: csv.DictWriter,
    samples_handle,
    checkpoints_handle,
    completed_checkpoints: list[dict[str, object]],
    forced: bool = False,
    interrupted: bool = False,
) -> list[dict[str, object]]:
  remaining_checkpoints: list[dict[str, object]] = []
  for checkpoint in open_checkpoints:
    should_close = forced or elapsed_seconds >= checkpoint["close_sec"]
    if not should_close:
      remaining_checkpoints.append(checkpoint)
      continue

    rows = build_checkpoint_rows(checkpoint)
    for row in rows:
      checkpoint_writer.writerow(row)

    checkpoint_summary = {
        "minute": checkpoint["minute"],
        "rows": rows,
    }
    completed_checkpoints.append(checkpoint_summary)

    channel_summary = " ".join(
        f"ch{row['channel']} ghost={row['ghost_rate']:.2f}"
        for row in rows
    )
    suffix = ""
    if checkpoint["truncated"]:
      suffix = " [truncated]"
    elif interrupted:
      suffix = " [interrupted]"
    print(
        f"[t={format_elapsed(elapsed_seconds)}] "
        f"checkpoint {checkpoint['index']} closed: {channel_summary}{suffix}"
    )

    checkpoints_handle.flush()
    samples_handle.flush()

  return remaining_checkpoints


def main() -> int:
  args = parse_args()
  try:
    import serial
  except ModuleNotFoundError:
    print("Runtime error: pyserial is not installed in this Python environment.", file=sys.stderr)
    return 1

  if args.duration_hours <= 0:
    print("Argument error: --duration-hours must be > 0.", file=sys.stderr)
    return 1
  if args.checkpoint_interval_min <= 0:
    print("Argument error: --checkpoint-interval-min must be > 0.", file=sys.stderr)
    return 1
  if args.checkpoint_window_sec <= 0:
    print("Argument error: --checkpoint-window-sec must be > 0.", file=sys.stderr)
    return 1
  if args.ghost_rate_threshold < 0:
    print("Argument error: --ghost-rate-threshold must be >= 0.", file=sys.stderr)
    return 1

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  samples_path = out_dir / f"test1_{timestamp}_samples.csv"
  checkpoints_path = out_dir / f"test1_{timestamp}_checkpoints.csv"
  summary_path = out_dir / f"test1_{timestamp}_summary.json"

  try:
    ser = serial.Serial(args.port, args.baud, timeout=0.2)
  except serial.SerialException as exc:
    print(f"Serial error: {exc}", file=sys.stderr)
    return 1

  duration_seconds = args.duration_hours * 3600.0
  checkpoint_interval_sec = args.checkpoint_interval_min * 60.0
  checkpoint_count = max(1, math.ceil(duration_seconds / checkpoint_interval_sec))
  checkpoint_times = [index * checkpoint_interval_sec for index in range(checkpoint_count)]

  print(f"Opened {args.port} at {args.baud} baud.")
  print("Listening to raw capacitive stream. Test 1 drift timing is host-driven.")

  time.sleep(2.0)
  ser.reset_input_buffer()

  completed_checkpoints: list[dict[str, object]] = []
  open_checkpoints: list[dict[str, object]] = []
  next_checkpoint_index = 0
  interrupted = False
  last_samples_flush = 0.0
  start = time.monotonic()

  try:
    with (
        samples_path.open("w", newline="", encoding="utf-8") as samples_handle,
        checkpoints_path.open("w", newline="", encoding="utf-8") as checkpoints_handle,
    ):
      samples_writer = csv.writer(samples_handle)
      checkpoint_writer = csv.DictWriter(
          checkpoints_handle,
          fieldnames=[
              "checkpoint_minute",
              "channel",
              "sample_count",
              "live_count",
              "ghost_rate",
              "ghost_mean",
          ],
      )
      samples_writer.writerow(["run_ms", *[f"raw_ch{i}" for i in range(1, CHANNEL_COUNT + 1)]])
      checkpoint_writer.writeheader()

      while True:
        elapsed_seconds = time.monotonic() - start
        next_checkpoint_index = open_due_checkpoints(
            elapsed_seconds,
            checkpoint_times,
            args.checkpoint_window_sec,
            duration_seconds,
            next_checkpoint_index,
            open_checkpoints,
        )
        open_checkpoints = close_finished_checkpoints(
            elapsed_seconds,
            open_checkpoints,
            checkpoint_writer,
            samples_handle,
            checkpoints_handle,
            completed_checkpoints,
        )

        if elapsed_seconds >= duration_seconds:
          break

        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not line:
          continue

        sample = parse_sample_line(line)
        if sample is None:
          continue

        sample_elapsed = time.monotonic() - start
        if sample_elapsed > duration_seconds:
          continue

        next_checkpoint_index = open_due_checkpoints(
            sample_elapsed,
            checkpoint_times,
            args.checkpoint_window_sec,
            duration_seconds,
            next_checkpoint_index,
            open_checkpoints,
        )
        samples_writer.writerow([round(sample_elapsed * 1000), *sample])

        for checkpoint in open_checkpoints:
          if checkpoint["start_sec"] <= sample_elapsed < checkpoint["close_sec"]:
            checkpoint["samples"].append(sample)

        if sample_elapsed - last_samples_flush >= 5.0:
          samples_handle.flush()
          last_samples_flush = sample_elapsed

      final_elapsed = min(time.monotonic() - start, duration_seconds)
      open_checkpoints = close_finished_checkpoints(
          final_elapsed,
          open_checkpoints,
          checkpoint_writer,
          samples_handle,
          checkpoints_handle,
          completed_checkpoints,
          forced=True,
      )
      samples_handle.flush()
      checkpoints_handle.flush()
  except KeyboardInterrupt:
    interrupted = True
    try:
      with (
          samples_path.open("a", newline="", encoding="utf-8") as samples_handle,
          checkpoints_path.open("a", newline="", encoding="utf-8") as checkpoints_handle,
      ):
        checkpoint_writer = csv.DictWriter(
            checkpoints_handle,
            fieldnames=[
                "checkpoint_minute",
                "channel",
                "sample_count",
                "live_count",
                "ghost_rate",
                "ghost_mean",
            ],
        )
        final_elapsed = min(time.monotonic() - start, duration_seconds)
        next_checkpoint_index = open_due_checkpoints(
            final_elapsed,
            checkpoint_times,
            args.checkpoint_window_sec,
            duration_seconds,
            next_checkpoint_index,
            open_checkpoints,
        )
        open_checkpoints = close_finished_checkpoints(
            final_elapsed,
            open_checkpoints,
            checkpoint_writer,
            samples_handle,
            checkpoints_handle,
            completed_checkpoints,
            forced=True,
            interrupted=True,
        )
        samples_handle.flush()
        checkpoints_handle.flush()
    finally:
      pass
  finally:
    ser.close()

  channels_flagged_set: set[int] = set()
  worst_channel: int | None = None
  worst_ghost_rate = 0.0
  worst_checkpoint_minute: int | float | None = None
  any_checkpoint_samples = False

  for checkpoint in completed_checkpoints:
    for row in checkpoint["rows"]:
      ghost_rate = float(row["ghost_rate"])
      channel = int(row["channel"])
      minute = row["checkpoint_minute"]
      sample_count = int(row["sample_count"])
      if sample_count > 0:
        any_checkpoint_samples = True
      if ghost_rate > args.ghost_rate_threshold:
        channels_flagged_set.add(channel)

      if (
          sample_count > 0
          and (
              worst_channel is None
              or ghost_rate > worst_ghost_rate
              or (ghost_rate == worst_ghost_rate and minute < worst_checkpoint_minute)
          )
      ):
        worst_channel = channel
        worst_ghost_rate = ghost_rate
        worst_checkpoint_minute = minute

  channels_flagged = sorted(channels_flagged_set)
  passed = len(channels_flagged) == 0
  if not any_checkpoint_samples:
    worst_channel = None
    worst_ghost_rate = 0.0
    worst_checkpoint_minute = None

  summary = {
      "passed": passed,
      "duration_hours": args.duration_hours,
      "checkpoint_interval_min": args.checkpoint_interval_min,
      "checkpoint_window_sec": args.checkpoint_window_sec,
      "ghost_rate_threshold": args.ghost_rate_threshold,
      "checkpoints_recorded": len(completed_checkpoints),
      "channels_flagged": channels_flagged,
      "worst_channel": worst_channel,
      "worst_ghost_rate": worst_ghost_rate,
      "worst_checkpoint_minute": worst_checkpoint_minute,
      "samples_path": str(samples_path),
      "checkpoints_path": str(checkpoints_path),
  }
  if interrupted:
    summary["interrupted"] = True

  with summary_path.open("w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
    handle.write("\n")

  print(f"Samples written to {samples_path}")
  print(f"Checkpoints written to {checkpoints_path}")
  print(f"Summary written to {summary_path}")
  if passed:
    print(
        "Result: "
        f"PASS | duration={args.duration_hours}h | "
        f"checkpoints={len(completed_checkpoints)} | "
        f"channels_flagged={channels_flagged}"
    )
  else:
    print(
        "Result: "
        f"FAIL | duration={args.duration_hours}h | "
        f"checkpoints={len(completed_checkpoints)} | "
        f"channels_flagged={channels_flagged} | "
        f"worst=ch{worst_channel}@minute={worst_checkpoint_minute} rate={worst_ghost_rate:.2f}"
    )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
