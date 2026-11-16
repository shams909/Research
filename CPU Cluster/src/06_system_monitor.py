"""
CPU Cluster Lab — Script 06: System Monitor
============================================
PURPOSE:
    Monitors CPU usage, RAM, and network throughput on the current node
    during training. Run this in a SEPARATE terminal alongside training
    to collect system-level metrics for your research paper.

WHAT IT TRACKS (every N seconds):
    - CPU usage % per core + overall
    - RAM used / total (GB)
    - Network bytes sent/received per second (MB/s)
    - Saves to results/sysmon_<hostname>_<timestamp>.csv

RUN (in a separate terminal while training is running):
    source .venv/bin/activate
    python3 src/06_system_monitor.py                  # default: every 2s
    python3 src/06_system_monitor.py --interval 1     # every 1 second
    python3 src/06_system_monitor.py --duration 300   # run for 5 minutes

STOP: Ctrl+C — data already saved to CSV
"""

import os
import sys
import csv
import time
import socket
import signal
import argparse
import datetime

import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Handle Ctrl+C gracefully
_running = True
def _stop(sig, frame):
    global _running
    _running = False
    print("\n  [monitor] Stopped — data saved")
signal.signal(signal.SIGINT, _stop)


def get_net_speed(prev_counters, interval):
    """Calculate network MB/s since last check."""
    curr = psutil.net_io_counters()
    sent_mbs = (curr.bytes_sent - prev_counters.bytes_sent) / interval / 1e6
    recv_mbs = (curr.bytes_recv - prev_counters.bytes_recv) / interval / 1e6
    return curr, round(sent_mbs, 3), round(recv_mbs, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Sampling interval in seconds (default: 2)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Total monitoring duration in seconds (default: until Ctrl+C)")
    args = parser.parse_args()

    hostname = socket.gethostname()
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, f"sysmon_{hostname}_{ts}.csv")

    # System info
    cpu_count = psutil.cpu_count(logical=True)
    ram_total = psutil.virtual_memory().total / 1e9

    print()
    print("=" * 65)
    print("  SYSTEM MONITOR — CPU Cluster Lab")
    print(f"  Node     : {hostname}")
    print(f"  CPUs     : {cpu_count} logical cores")
    print(f"  RAM      : {ram_total:.1f} GB")
    print(f"  Interval : {args.interval}s")
    print(f"  Output   : {csv_path}")
    print(f"  Stop     : Ctrl+C")
    print("=" * 65)
    print(f"\n  {'Time':>8}  {'CPU%':>6}  {'RAM(GB)':>8}  {'RAM%':>6}  "
          f"{'Net↑MB/s':>10}  {'Net↓MB/s':>10}")
    print("  " + "-" * 55)

    fieldnames = [
        "timestamp", "elapsed_s",
        "cpu_pct", "ram_used_gb", "ram_pct",
        "net_sent_mbs", "net_recv_mbs",
    ]

    prev_net = psutil.net_io_counters()
    start    = time.time()
    rows     = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while _running:
            # Check duration limit
            elapsed = time.time() - start
            if args.duration and elapsed >= args.duration:
                break

            time.sleep(args.interval)

            # Collect metrics
            cpu_pct      = psutil.cpu_percent(interval=None)
            vm           = psutil.virtual_memory()
            ram_used     = vm.used / 1e9
            ram_pct      = vm.percent
            prev_net, sent_mbs, recv_mbs = get_net_speed(prev_net, args.interval)
            ts_now       = datetime.datetime.now().strftime("%H:%M:%S")
            elapsed      = round(time.time() - start, 1)

            row = {
                "timestamp"   : ts_now,
                "elapsed_s"   : elapsed,
                "cpu_pct"     : cpu_pct,
                "ram_used_gb" : round(ram_used, 2),
                "ram_pct"     : ram_pct,
                "net_sent_mbs": sent_mbs,
                "net_recv_mbs": recv_mbs,
            }
            writer.writerow(row)
            f.flush()

            print(f"  {ts_now:>8}  {cpu_pct:>5.1f}%  {ram_used:>6.2f} GB  "
                  f"{ram_pct:>5.1f}%  {sent_mbs:>10.3f}  {recv_mbs:>10.3f}")

    print(f"\n  Saved {csv_path}")
    print("  Run src/05_plot_results.py to include system metrics in figures")


if __name__ == "__main__":
    main()
