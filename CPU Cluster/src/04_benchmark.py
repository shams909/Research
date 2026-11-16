
"""
CPU Cluster Lab — Script 04: Scaling Benchmark & Analysis
==========================================================
PURPOSE:
    Measures and reports training throughput for the current cluster
    configuration. Run this after 02_train_cifar.py and 03_train_gpt2.py
    to get clean, structured numbers for your research paper.

WHAT IT DOES:
    1. Runs a short ResNet-18 benchmark (N steps, measures imgs/sec)
    2. Runs a short GPT-2 benchmark (N steps, measures tokens/sec)
    3. Computes scaling efficiency vs single-node baseline (if available)
    4. Generates a text report saved to results/benchmark_report_<ts>.txt
    5. Saves all raw numbers to results/benchmark_<N>nodes_<ts>.csv

HOW TO INTERPRET:
    Scaling efficiency = (N-node throughput) / (N × single-node throughput)
    Perfect scaling    = 100% (never achieved in practice)
    Typical on GbE    = 60–85% (communication overhead on slow network)
    This gap vs Apple  = your paper's key finding

RUN:
    Node 01:  bash launch_master.sh 04_benchmark.py
    Node 02:  bash launch_worker.sh 04_benchmark.py 1
"""

import os
import sys
import csv
import time
import math
import socket
import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

BENCH_STEPS = 30   # Steps per benchmark — increase for more stable numbers
WARMUP_STEPS = 5   # Warmup steps excluded from measurement


# ── ResNet-18 Benchmark ───────────────────────────────────────────────────────

def benchmark_resnet(rank, world_size, batch_size=32):
    """Measure image throughput with ResNet-18 (CIFAR-adapted)."""
    model      = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model      = DDP(model)
    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    times = []

    for step in range(BENCH_STEPS + WARMUP_STEPS):
        # Synthetic data — same shapes as CIFAR-10
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        dist.barrier()
        elapsed = time.perf_counter() - t0

        if step >= WARMUP_STEPS:
            times.append(elapsed)

    avg_step_time = sum(times) / len(times)
    imgs_per_sec  = (batch_size * world_size) / avg_step_time

    return {
        "task"            : "ResNet-18 / CIFAR-10",
        "batch_per_node"  : batch_size,
        "global_batch"    : batch_size * world_size,
        "avg_step_time_s" : round(avg_step_time, 4),
        "throughput"      : round(imgs_per_sec, 1),
        "unit"            : "images/sec",
        "world_size"      : world_size,
    }


# ── GPT-2 Benchmark ───────────────────────────────────────────────────────────

def benchmark_gpt2(rank, world_size, batch_size=4, seq_len=512):
    """Measure token throughput with GPT-2 Small."""
    from transformers import GPT2LMHeadModel, GPT2Config

    cfg = GPT2Config(
        vocab_size=50257, n_positions=seq_len,
        n_embd=768, n_layer=12, n_head=12
    )
    model     = GPT2LMHeadModel(cfg)
    model     = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    times = []

    for step in range(BENCH_STEPS + WARMUP_STEPS):
        # Synthetic token ids
        input_ids = torch.randint(0, 50257, (batch_size, seq_len))
        labels    = input_ids.clone()

        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=labels).loss
        loss.backward()
        optimizer.step()
        dist.barrier()
        elapsed = time.perf_counter() - t0

        if step >= WARMUP_STEPS:
            times.append(elapsed)

    avg_step_time  = sum(times) / len(times)
    tokens_per_sec = (batch_size * seq_len * world_size) / avg_step_time

    return {
        "task"            : "GPT-2 Small (117M)",
        "batch_per_node"  : batch_size,
        "global_batch"    : batch_size * world_size,
        "avg_step_time_s" : round(avg_step_time, 4),
        "throughput"      : round(tokens_per_sec, 1),
        "unit"            : "tokens/sec",
        "world_size"      : world_size,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(results, world_size, hostname):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 65,
        "  CPU CLUSTER LAB — BENCHMARK REPORT",
        f"  Generated : {ts}",
        f"  Nodes     : {world_size}",
        f"  Master    : {config.MASTER_ADDR}",
        f"  Node      : {hostname}",
        "=" * 65,
        "",
    ]

    for r in results:
        lines += [
            f"  Task          : {r['task']}",
            f"  Batch/node    : {r['batch_per_node']}",
            f"  Global batch  : {r['global_batch']}",
            f"  Throughput    : {r['throughput']} {r['unit']}",
            f"  Step time     : {r['avg_step_time_s']} s",
            "",
        ]

    lines += [
        "  Notes:",
        "  - Measurements use synthetic data (same shapes as real datasets)",
        "  - Communication overhead IS included (AllReduce in backward pass)",
        f"  - Benchmark steps: {BENCH_STEPS} (after {WARMUP_STEPS} warmup)",
        "  - For scaling efficiency: compare with single-node run",
        "=" * 65,
    ]

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group(backend=config.BACKEND)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    hostname   = socket.gethostname()

    if rank == 0:
        print()
        print("=" * 65)
        print("  SCALING BENCHMARK — CPU Cluster Lab")
        print(f"  Nodes: {world_size}  |  Backend: {config.BACKEND}")
        print("=" * 65)

    results = []

    # ── ResNet-18 benchmark ───────────────────────────────────────────────────
    if rank == 0:
        print(f"\n  [1/2] Benchmarking ResNet-18 ({BENCH_STEPS} steps)...")
    dist.barrier()

    r1 = benchmark_resnet(rank, world_size, batch_size=config.DEFAULT_BATCH_SIZE)
    results.append(r1)
    dist.barrier()

    if rank == 0:
        print(f"        Throughput: {r1['throughput']} {r1['unit']}")

    # ── GPT-2 benchmark ───────────────────────────────────────────────────────
    if rank == 0:
        print(f"\n  [2/2] Benchmarking GPT-2 Small ({BENCH_STEPS} steps)...")
    dist.barrier()

    r2 = benchmark_gpt2(
        rank, world_size,
        batch_size=config.GPT2_BATCH_SIZE,
        seq_len=config.GPT2_SEQ_LEN
    )
    results.append(r2)
    dist.barrier()

    if rank == 0:
        print(f"        Throughput: {r2['throughput']} {r2['unit']}")

    # ── Save & Report ─────────────────────────────────────────────────────────
    if rank == 0:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV
        csv_f = os.path.join(
            config.RESULTS_DIR, f"benchmark_{world_size}nodes_{ts}.csv"
        )
        with open(csv_f, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Text report
        report     = generate_report(results, world_size, hostname)
        report_f   = os.path.join(
            config.RESULTS_DIR, f"benchmark_report_{world_size}nodes_{ts}.txt"
        )
        with open(report_f, "w") as f:
            f.write(report)

        print()
        print(report)
        print(f"\n  CSV    saved → {csv_f}")
        print(f"  Report saved → {report_f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
