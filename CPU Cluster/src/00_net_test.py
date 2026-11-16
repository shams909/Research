"""
CPU Cluster Lab — Script 00: Network Benchmark
===============================================
PURPOSE:
    Measures raw network bandwidth and latency between cluster nodes using
    PyTorch AllReduce operations. Run this before any training to understand
    your network's capability and document it for your research paper.

WHY THIS MATTERS:
    The Apple Silicon paper (base paper) uses Thunderbolt 5 (~120 GB/s).
    Your setup uses Gigabit Ethernet (~0.1 GB/s).
    This script quantifies that gap — which IS your research finding.

WHAT IT MEASURES:
    - Effective AllReduce bandwidth (MB/s) at different tensor sizes
    - Per-operation latency (ms)
    - Efficiency vs theoretical Gigabit Ethernet limit (940 MB/s)

OUTPUT:
    - Console table of results
    - CSV saved to results/net_test_<N>nodes_<timestamp>.csv

RUN:
    Node 01:  bash launch_master.sh 00_net_test.py
    Node 02:  bash launch_worker.sh 00_net_test.py 1
"""

import os
import sys
import csv
import time
import socket
import datetime

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def print_header(rank, world_size):
    if rank != 0:
        return
    print()
    print("=" * 65)
    print("  NETWORK BENCHMARK — CPU Cluster Lab")
    print(f"  Nodes     : {world_size}")
    print(f"  Backend   : {config.BACKEND}")
    print(f"  Master    : {config.MASTER_ADDR}:{config.MASTER_PORT}")
    print(f"  This node : {socket.gethostname()}")
    print("=" * 65)
    print(f"  {'Tensor Size':>12}  {'Bandwidth':>12}  {'Latency':>10}  {'Efficiency':>10}")
    print(f"  {'(MB)':>12}  {'(MB/s)':>12}  {'(ms/op)':>10}  {'vs GbE':>10}")
    print("  " + "-" * 50)


def run_benchmark(rank, world_size):
    # Tensor sizes to test: from small (gradient steps) to large (full model)
    sizes_mb = [0.1, 1, 10, 50, 100, 250, 500]
    WARMUP   = 5
    ITERS    = 20
    results  = []
    GBE_MAX  = 940.0  # Theoretical Gigabit Ethernet ceiling (MB/s)

    for size_mb in sizes_mb:
        num_floats = int(size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
        tensor = torch.ones(num_floats, dtype=torch.float32)

        # Warmup (not measured)
        for _ in range(WARMUP):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()

        # Timed measurement
        start = time.perf_counter()
        for _ in range(ITERS):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / ITERS) * 1000
        # AllReduce on N nodes: each node sends/receives 2*(N-1)/N * size
        effective_mb   = size_mb * (2 * (world_size - 1) / world_size)
        bandwidth_mbs  = (effective_mb * ITERS) / elapsed
        efficiency_pct = (bandwidth_mbs / GBE_MAX) * 100

        results.append({
            "size_mb"       : size_mb,
            "bandwidth_mbs" : round(bandwidth_mbs,  2),
            "latency_ms"    : round(avg_latency_ms, 3),
            "efficiency_pct": round(efficiency_pct, 1),
            "world_size"    : world_size,
        })

        if rank == 0:
            print(f"  {size_mb:>12.1f}  {bandwidth_mbs:>12.1f}  "
                  f"{avg_latency_ms:>10.2f}  {efficiency_pct:>9.1f}%")

    return results


def save_results(results, world_size):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(
        config.RESULTS_DIR, f"net_test_{world_size}nodes_{ts}.csv"
    )
    with open(fname, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    return fname


def main():
    dist.init_process_group(backend=config.BACKEND)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    print_header(rank, world_size)
    dist.barrier()

    results = run_benchmark(rank, world_size)

    if rank == 0:
        avg_bw = sum(r["bandwidth_mbs"] for r in results) / len(results)
        print("  " + "-" * 50)
        print(f"\n  Average bandwidth : {avg_bw:.1f} MB/s")
        print(f"  Theoretical GbE   : 940.0 MB/s")
        print(f"  Overall efficiency: {avg_bw/940*100:.1f}%")
        fname = save_results(results, world_size)
        print(f"\n  Results saved → {fname}")
        print()
        print("=" * 65)
        print("  NETWORK TEST COMPLETE")
        print("=" * 65)
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
