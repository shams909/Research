"""
CPU Cluster Lab — Script 04: Scaling Benchmark & Communication Profiling
========================================================================
PURPOSE:
    Measures training throughput, scaling efficiency, and fine-grained
    step-time decomposition (Forward, Backward, AllReduce, Optimizer)
    across commodity CPU clusters over physical Fast Ethernet (100 Mbps).

CAPABILITIES:
    1. Direct timing instrumentation:
       - Forward pass duration (ms)
       - Local backward computation duration (ms)
       - Collective AllReduce synchronization duration (ms)
       - Optimizer step duration (ms)
    2. Two Scientific Benchmark Modes:
       - Mode 'micro' : Fixed-Microbatch K-Sweep (K ∈ {1, 2, 4, 8, 16})
       - Mode 'global': Fixed-Global-Batch K-Sweep (K ∈ {1, 2, 4, 8, 16})
       - Mode 'single': Single benchmark run (K from config or --k)
    3. Multi-trial statistical rigor:
       - Runs N trials per configuration (default: 3)
       - Automatically computes Mean (μ) and Standard Deviation (σ)
    4. Structured CSV storage:
       - results/02_sweep_fixed_microbatch/
       - results/03_sweep_fixed_global_batch/
       - results/04_step_time_breakdown/
       - results/benchmark_<N>nodes_<ts>.csv (backward compatible)

RUN:
    Single Run:
        bash launch_master.sh 04_benchmark.py 2 --mode single --k 4

    Fixed-Microbatch Sweep (Exp 2):
        bash launch_master.sh 04_benchmark.py 2 --mode micro --trials 3

    Fixed-Global-Batch Sweep (Exp 3):
        bash launch_master.sh 04_benchmark.py 2 --mode global --trials 3

    Both Sweeps sequentially:
        bash launch_master.sh 04_benchmark.py 2 --mode both --trials 3
"""

import os
import sys
import csv
import time
import math
import socket
import datetime
import argparse
import statistics

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet18

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── ResNet-18 Benchmark Engine ────────────────────────────────────────────────

def benchmark_resnet(rank, world_size, batch_size=32, accum_steps=1, bench_steps=8, warmup_steps=2, trial_id=1):
    """
    Measures ResNet-18 image throughput and decomposes step time into:
    Forward compute, Backward compute, AllReduce sync, and Optimizer update.
    """
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = DDP(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    step_times = []
    forward_times = []
    backward_times = []
    allreduce_times = []
    optimizer_times = []

    total_steps = bench_steps + warmup_steps
    total_imgs = batch_size * accum_steps * world_size

    for step in range(total_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        fwd_step = 0.0
        bwd_step = 0.0
        ar_step  = 0.0

        for micro in range(accum_steps):
            x = torch.randn(batch_size, 3, 32, 32)
            y = torch.randint(0, 10, (batch_size,))

            # 1. Forward timing
            tf0 = time.perf_counter()
            out = model(x)
            loss = criterion(out, y) / accum_steps
            fwd_step += (time.perf_counter() - tf0)

            # 2. Backward & Communication timing
            if micro < accum_steps - 1 and world_size > 1:
                tb0 = time.perf_counter()
                with model.no_sync():
                    loss.backward()
                bwd_step += (time.perf_counter() - tb0)
            else:
                ts0 = time.perf_counter()
                loss.backward()
                if world_size > 1:
                    dist.barrier()
                sync_elapsed = time.perf_counter() - ts0

                # Separate local backward pass from AllReduce barrier wait
                if accum_steps > 1 and bwd_step > 0:
                    avg_local_bwd = bwd_step / (accum_steps - 1)
                    bwd_step += avg_local_bwd
                    ar_step = max(sync_elapsed - avg_local_bwd, 0.0)
                else:
                    ar_step = sync_elapsed

        # 3. Optimizer timing
        to0 = time.perf_counter()
        optimizer.step()
        opt_step = time.perf_counter() - to0

        elapsed = time.perf_counter() - t0

        if step >= warmup_steps:
            step_times.append(elapsed)
            forward_times.append(fwd_step)
            backward_times.append(bwd_step)
            allreduce_times.append(ar_step)
            optimizer_times.append(opt_step)

        if rank == 0:
            step_tput = total_imgs / max(elapsed, 1e-4)
            tag = "warmup" if step < warmup_steps else f"bench #{trial_id}"
            print(f"        [{tag:>8} {step+1:>2}/{total_steps}] Step: {elapsed:>6.2f}s | "
                  f"Fwd: {fwd_step*1000:>5.1f}ms | Bwd: {bwd_step*1000:>5.1f}ms | "
                  f"Sync: {ar_step*1000:>6.1f}ms | Speed: {step_tput:>5.1f} img/s", flush=True)

    avg_step_s   = statistics.mean(step_times) if step_times else 0.0
    throughput   = total_imgs / max(avg_step_s, 1e-4)

    return {
        "model"             : "ResNet-18",
        "task"              : "ResNet-18 / CIFAR-10",
        "accum_steps"       : accum_steps,
        "batch_per_node"    : batch_size,
        "effective_node_b"  : batch_size * accum_steps,
        "global_batch"      : total_imgs,
        "trial_id"          : trial_id,
        "avg_step_time_s"   : round(avg_step_s, 4),
        "throughput"        : round(throughput, 1),
        "unit"              : "images/sec",
        "world_size"        : world_size,
        "forward_ms"        : round(statistics.mean(forward_times) * 1000, 2),
        "backward_ms"       : round(statistics.mean(backward_times) * 1000, 2),
        "allreduce_ms"      : round(statistics.mean(allreduce_times) * 1000, 2),
        "optimizer_ms"      : round(statistics.mean(optimizer_times) * 1000, 2),
    }


# ── GPT-2 Small Benchmark Engine ──────────────────────────────────────────────

def benchmark_gpt2(rank, world_size, batch_size=4, seq_len=512, accum_steps=1, bench_steps=8, warmup_steps=2, trial_id=1):
    """
    Measures GPT-2 Small token throughput and decomposes step time into:
    Forward compute, Backward compute, AllReduce sync, and Optimizer update.
    """
    from transformers import GPT2LMHeadModel, GPT2Config

    cfg = GPT2Config(
        vocab_size=50257, n_positions=seq_len,
        n_embd=768, n_layer=12, n_head=12
    )
    model = GPT2LMHeadModel(cfg)
    model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    step_times = []
    forward_times = []
    backward_times = []
    allreduce_times = []
    optimizer_times = []

    total_steps  = bench_steps + warmup_steps
    total_tokens = batch_size * seq_len * accum_steps * world_size

    for step in range(total_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        fwd_step = 0.0
        bwd_step = 0.0
        ar_step  = 0.0

        for micro in range(accum_steps):
            input_ids = torch.randint(0, 50257, (batch_size, seq_len))
            labels    = input_ids.clone()

            # 1. Forward timing
            tf0 = time.perf_counter()
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / accum_steps
            fwd_step += (time.perf_counter() - tf0)

            # 2. Backward & Communication timing
            if micro < accum_steps - 1 and world_size > 1:
                tb0 = time.perf_counter()
                with model.no_sync():
                    loss.backward()
                bwd_step += (time.perf_counter() - tb0)
            else:
                ts0 = time.perf_counter()
                loss.backward()
                if world_size > 1:
                    dist.barrier()
                sync_elapsed = time.perf_counter() - ts0

                # Separate local backward pass from AllReduce barrier wait
                if accum_steps > 1 and bwd_step > 0:
                    avg_local_bwd = bwd_step / (accum_steps - 1)
                    bwd_step += avg_local_bwd
                    ar_step = max(sync_elapsed - avg_local_bwd, 0.0)
                else:
                    ar_step = sync_elapsed

        # 3. Optimizer timing
        to0 = time.perf_counter()
        optimizer.step()
        opt_step = time.perf_counter() - to0

        elapsed = time.perf_counter() - t0

        if step >= warmup_steps:
            step_times.append(elapsed)
            forward_times.append(fwd_step)
            backward_times.append(bwd_step)
            allreduce_times.append(ar_step)
            optimizer_times.append(opt_step)

        if rank == 0:
            step_tput = total_tokens / max(elapsed, 1e-4)
            tag = "warmup" if step < warmup_steps else f"bench #{trial_id}"
            print(f"        [{tag:>8} {step+1:>2}/{total_steps}] Step: {elapsed:>6.2f}s | "
                  f"Fwd: {fwd_step*1000:>5.1f}ms | Bwd: {bwd_step*1000:>5.1f}ms | "
                  f"Sync: {ar_step*1000:>6.1f}ms | Speed: {step_tput:>5.1f} tok/s", flush=True)

    avg_step_s     = statistics.mean(step_times) if step_times else 0.0
    tokens_per_sec = total_tokens / max(avg_step_s, 1e-4)

    return {
        "model"             : "GPT-2 Small",
        "task"              : "GPT-2 Small (117M)",
        "accum_steps"       : accum_steps,
        "batch_per_node"    : batch_size,
        "effective_node_b"  : batch_size * accum_steps,
        "global_batch"      : total_tokens,
        "trial_id"          : trial_id,
        "avg_step_time_s"   : round(avg_step_s, 4),
        "throughput"        : round(tokens_per_sec, 1),
        "unit"              : "tokens/sec",
        "world_size"        : world_size,
        "forward_ms"        : round(statistics.mean(forward_times) * 1000, 2),
        "backward_ms"       : round(statistics.mean(backward_times) * 1000, 2),
        "allreduce_ms"      : round(statistics.mean(allreduce_times) * 1000, 2),
        "optimizer_ms"      : round(statistics.mean(optimizer_times) * 1000, 2),
    }


# ── Aggregate Statistics Across Trials ────────────────────────────────────────

def aggregate_trials(trials_list):
    """Computes mean and std dev across multiple trials for a single configuration."""
    tputs = [r["throughput"] for r in trials_list]
    steps = [r["avg_step_time_s"] for r in trials_list]
    fwds  = [r["forward_ms"] for r in trials_list]
    bwds  = [r["backward_ms"] for r in trials_list]
    ars   = [r["allreduce_ms"] for r in trials_list]
    opts  = [r["optimizer_ms"] for r in trials_list]

    summary = dict(trials_list[0])
    summary["trials_count"]       = len(trials_list)
    summary["throughput_mean"]    = round(statistics.mean(tputs), 2)
    summary["throughput_std"]     = round(statistics.stdev(tputs), 2) if len(tputs) > 1 else 0.0
    summary["step_time_mean_s"]   = round(statistics.mean(steps), 4)
    summary["step_time_std_s"]    = round(statistics.stdev(steps), 4) if len(steps) > 1 else 0.0
    summary["forward_ms_mean"]    = round(statistics.mean(fwds), 2)
    summary["backward_ms_mean"]   = round(statistics.mean(bwds), 2)
    summary["allreduce_ms_mean"]  = round(statistics.mean(ars), 2)
    summary["optimizer_ms_mean"]  = round(statistics.mean(opts), 2)

    return summary


# ── Text Report Generator ─────────────────────────────────────────────────────

def generate_report(summaries, world_size, hostname, mode):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 70,
        "  CPU CLUSTER LAB — EMPIRICAL SCALING BENCHMARK REPORT",
        f"  Generated : {ts}",
        f"  Cluster   : {world_size} Nodes  |  Master: {config.MASTER_ADDR}",
        f"  Mode      : {mode.upper()}  |  Reporter: {hostname}",
        "=" * 70,
        "",
        f"  {'Model / Task':<22} {'K':<4} {'Global Batch':<14} {'Step Time (s)':<16} {'Throughput (μ ± σ)':<20}",
        "  " + "-" * 66,
    ]

    for s in summaries:
        tput_str = f"{s['throughput_mean']} ± {s['throughput_std']} {s['unit']}"
        time_str = f"{s['step_time_mean_s']:.3f} s"
        lines.append(f"  {s['model']:<22} {s['accum_steps']:<4} {s['global_batch']:<14} {time_str:<16} {tput_str:<20}")

    lines += [
        "",
        "  Timing Breakdown (ms):",
        f"  {'Model':<16} {'K':<4} {'Forward (ms)':<14} {'Backward (ms)':<14} {'AllReduce (ms)':<16} {'Optimizer (ms)':<14}",
        "  " + "-" * 66,
    ]

    for s in summaries:
        lines.append(f"  {s['model']:<16} {s['accum_steps']:<4} {s['forward_ms_mean']:<14.1f} {s['backward_ms_mean']:<14.1f} {s['allreduce_ms_mean']:<16.1f} {s['optimizer_ms_mean']:<14.1f}")

    lines += [
        "=" * 70,
        "  All measurements collected on bare-metal hardware.",
        "  AllReduce represents blocking collective network synchronization.",
        "=" * 70,
    ]

    return "\n".join(lines)


# ── Master Orchestrator ───────────────────────────────────────────────────────

def run_experiment_suite(rank, world_size, mode, k_list, num_trials, bench_steps, warmup_steps, model_filter):
    """
    Executes a structured sweep across models and K values.
    Saves raw trial logs, summary tables, and timing decomposition.
    """
    hostname = socket.gethostname()
    all_raw_results = []
    all_summaries   = []

    models_to_test = []
    if model_filter in ["all", "resnet"]:
        models_to_test.append("resnet")
    if model_filter in ["all", "gpt2"]:
        models_to_test.append("gpt2")

    for model_name in models_to_test:
        for k in k_list:
            trial_results = []

            # Determine batch sizing based on mode
            if model_name == "resnet":
                if mode == "global":
                    # Lock global batch at 64 images
                    b_per_node = max(1, 64 // (world_size * k))
                else:
                    # Fixed microbatch mode: 32 per node
                    b_per_node = config.DEFAULT_BATCH_SIZE

                if rank == 0:
                    print(f"\n>>> [ResNet-18 | {mode.upper()}] Benchmarking K={k} (microbatch={b_per_node}, global={b_per_node * k * world_size})...")

                for trial in range(1, num_trials + 1):
                    dist.barrier()
                    r = benchmark_resnet(
                        rank, world_size,
                        batch_size=b_per_node,
                        accum_steps=k,
                        bench_steps=bench_steps,
                        warmup_steps=warmup_steps,
                        trial_id=trial
                    )
                    r["mode"] = mode
                    trial_results.append(r)
                    all_raw_results.append(r)

            elif model_name == "gpt2":
                if mode == "global":
                    # Lock global batch at 32 sequences (16,384 tokens)
                    b_per_node = max(1, 32 // (world_size * k))
                else:
                    # Fixed microbatch mode: 4 per node
                    b_per_node = config.GPT2_BATCH_SIZE

                if rank == 0:
                    print(f"\n>>> [GPT-2 Small | {mode.upper()}] Benchmarking K={k} (microbatch={b_per_node}, global={b_per_node * k * world_size * config.GPT2_SEQ_LEN} tokens)...")

                for trial in range(1, num_trials + 1):
                    dist.barrier()
                    r = benchmark_gpt2(
                        rank, world_size,
                        batch_size=b_per_node,
                        seq_len=config.GPT2_SEQ_LEN,
                        accum_steps=k,
                        bench_steps=bench_steps,
                        warmup_steps=warmup_steps,
                        trial_id=trial
                    )
                    r["mode"] = mode
                    trial_results.append(r)
                    all_raw_results.append(r)

            if rank == 0 and trial_results:
                summary = aggregate_trials(trial_results)
                summary["mode"] = mode
                all_summaries.append(summary)
                print(f"    --> K={k} Summary: {summary['throughput_mean']} ± {summary['throughput_std']} {summary['unit']} | Step: {summary['step_time_mean_s']:.3f}s | AllReduce: {summary['allreduce_ms_mean']:.1f}ms")

    # Save results on rank 0
    if rank == 0 and all_raw_results:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Directory routing
        if mode == "micro":
            target_dir = os.path.join(config.RESULTS_DIR, "02_sweep_fixed_microbatch")
        elif mode == "global":
            target_dir = os.path.join(config.RESULTS_DIR, "03_sweep_fixed_global_batch")
        else:
            target_dir = config.RESULTS_DIR

        raw_dir = os.path.join(target_dir, "raw_trials")
        os.makedirs(raw_dir, exist_ok=True)

        timing_dir = os.path.join(config.RESULTS_DIR, "04_step_time_breakdown")
        os.makedirs(timing_dir, exist_ok=True)

        # 2. Save Raw Trials CSV
        raw_csv_path = os.path.join(raw_dir, f"raw_trials_{world_size}nodes_{ts}.csv")
        with open(raw_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_raw_results[0].keys())
            writer.writeheader()
            writer.writerows(all_raw_results)

        # 3. Save Summary CSV (Mean ± Std)
        summary_csv_path = os.path.join(target_dir, f"summary_{mode}_{world_size}nodes_{ts}.csv")
        latest_summary   = os.path.join(target_dir, f"summary_{mode}.csv")
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)
        with open(latest_summary, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)

        # 4. Save Timing Decomposition Table
        timing_csv_path = os.path.join(timing_dir, f"timing_decomposition_{mode}_{world_size}nodes_{ts}.csv")
        latest_timing   = os.path.join(timing_dir, f"timing_decomposition_{mode}.csv")
        with open(timing_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)
        with open(latest_timing, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)

        # 5. Backward-compatible classic CSV
        classic_csv = os.path.join(config.RESULTS_DIR, f"benchmark_{world_size}nodes_{ts}.csv")
        with open(classic_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_summaries[0].keys())
            writer.writeheader()
            writer.writerows(all_summaries)

        # 6. Text report
        report_text = generate_report(all_summaries, world_size, hostname, mode)
        report_path = os.path.join(target_dir, f"report_{mode}_{world_size}nodes_{ts}.txt")
        with open(report_path, "w") as f:
            f.write(report_text)

        print()
        print(report_text)
        print(f"\n  ✓ Summary CSV : {summary_csv_path}")
        print(f"  ✓ Raw Trials  : {raw_csv_path}")
        print(f"  ✓ Timings     : {timing_csv_path}")
        print(f"  ✓ Text Report : {report_path}\n")


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CPU Cluster Lab Benchmark & Communication Profiling")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "micro", "global", "both"],
                        help="Benchmark mode: 'single' (standard run), 'micro' (fixed microbatch K-sweep), 'global' (fixed global batch K-sweep), 'both'")
    parser.add_argument("--model", type=str, default="all", choices=["all", "resnet", "gpt2"],
                        help="Workload filter: 'all', 'resnet', or 'gpt2'")
    parser.add_argument("--k", type=int, default=None,
                        help="Accumulation steps for single mode (default: config.GRADIENT_ACCUM_STEPS)")
    parser.add_argument("--k_list", type=str, default="1,2,4,8,16",
                        help="Comma-separated K values for sweeps (default: '1,2,4,8,16')")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of repetitions per configuration (default: 3)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Measured benchmark steps per trial (default: 8 for sweep, 30 for single)")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Warmup steps excluded from measurement (default: 2)")

    args = parser.parse_args()

    dist.init_process_group(backend=config.BACKEND)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print()
        print("=" * 70)
        print("  CPU CLUSTER LAB — BENCHMARK & COMMUNICATION PROFILER")
        print(f"  Cluster Size : {world_size} Nodes  |  Backend: {config.BACKEND}")
        print(f"  Run Mode     : {args.mode.upper()}  |  Trials/Config: {args.trials}")
        print("=" * 70)

    # Resolve K list
    if args.mode == "single":
        single_k = args.k if args.k is not None else getattr(config, "GRADIENT_ACCUM_STEPS", 4)
        k_list = [single_k]
        bench_steps = args.steps if args.steps is not None else 30
        warmup_steps = args.warmup
        modes_to_run = ["single"]
    else:
        k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
        bench_steps = args.steps if args.steps is not None else 8
        warmup_steps = args.warmup
        modes_to_run = ["micro", "global"] if args.mode == "both" else [args.mode]

    for m in modes_to_run:
        dist.barrier()
        run_experiment_suite(
            rank, world_size,
            mode=m,
            k_list=k_list,
            num_trials=args.trials,
            bench_steps=bench_steps,
            warmup_steps=warmup_steps,
            model_filter=args.model
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
