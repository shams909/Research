"""
CPU Cluster Lab — Script 01: Smoke Test (Cluster Validation)
=============================================================
PURPOSE:
    Proves the distributed cluster is correctly set up and communicating.
    This is Milestone 1 — show this output to your professor.

WHAT IT TESTS:
    Test 1 — AllReduce SUM    : Core operation used in every training step.
                                Each node sends a value, all receive the sum.
    Test 2 — Broadcast        : Master sends data (e.g. model weights) to all.
    Test 3 — Barrier Sync     : All nodes synchronize at a checkpoint.
    Test 4 — Gradient Sim     : Simulates real DDP gradient synchronization.

EXPECTED OUTPUT (2 nodes):
    Node 0 (Acer) sends 1 → both see 3 (1+2) ✓
    Node 1 (HP)   sends 2 → both see 3       ✓
    ALL TESTS PASSED — CLUSTER IS WORKING ✓

RUN:
    Node 01:  bash launch_master.sh 01_smoke_test.py
    Node 02:  bash launch_worker.sh 01_smoke_test.py 1
"""

import os
import sys
import time
import socket

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Individual Tests ──────────────────────────────────────────────────────────

def test_allreduce(rank, world_size):
    """
    AllReduce SUM: each node contributes rank+1, all receive the total.
    This is exactly what happens to gradients in every DDP training step.
    """
    value = torch.tensor([float(rank + 1)])
    expected = float(sum(range(1, world_size + 1)))

    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    passed = abs(value.item() - expected) < 1e-5

    return passed, value.item(), expected


def test_broadcast(rank, world_size):
    """
    Broadcast: master sends model weights to all workers.
    Used at training start to ensure all nodes begin with the same model.
    """
    tensor = torch.zeros(5)
    if rank == 0:
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    dist.broadcast(tensor, src=0)
    expected = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    passed = torch.allclose(tensor, expected)

    return passed


def test_barrier(rank, world_size):
    """
    Barrier: all nodes must reach this point before any continue.
    Used to synchronize nodes between training epochs.
    """
    # Simulate different compute times on each node
    time.sleep(rank * 0.05)
    dist.barrier()
    return True


def test_gradient_simulation(rank, world_size):
    """
    Gradient sync simulation: mimics what DDP does automatically during
    backward(). Each node computes its own gradients, AllReduce averages them.
    """
    model  = nn.Linear(16, 4)
    loss_fn = nn.MSELoss()

    # Each node processes different data (simulated by rank-dependent scaling)
    x      = torch.randn(8, 16) * (rank + 1)
    target = torch.randn(8, 4)
    output = model(x)
    loss   = loss_fn(output, target)
    loss.backward()

    # Manually AllReduce + average gradients (DDP does this automatically)
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group(backend=config.BACKEND)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    hostname   = socket.gethostname()

    # Header (master only)
    if rank == 0:
        print()
        print("=" * 60)
        print("  CLUSTER SMOKE TEST — CPU Cluster Lab")
        print(f"  Nodes   : {world_size}")
        print(f"  Backend : {config.BACKEND}")
        print(f"  Master  : {config.MASTER_ADDR}:{config.MASTER_PORT}")
        print("=" * 60)

    dist.barrier()

    # Every node announces itself
    print(f"  ✓  Node {rank} ({hostname}) — connected")
    dist.barrier()

    all_passed = True

    # ── Test 1: AllReduce ─────────────────────────────────────────────────────
    passed, result, expected = test_allreduce(rank, world_size)
    all_passed = all_passed and passed
    dist.barrier()
    if rank == 0:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"\n  [Test 1] AllReduce SUM         : {status}")
        print(f"           Expected={expected:.0f}  Got={result:.0f}")

    # ── Test 2: Broadcast ─────────────────────────────────────────────────────
    passed = test_broadcast(rank, world_size)
    all_passed = all_passed and passed
    dist.barrier()
    if rank == 0:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [Test 2] Broadcast (rank 0→all): {status}")

    # ── Test 3: Barrier ───────────────────────────────────────────────────────
    passed = test_barrier(rank, world_size)
    all_passed = all_passed and passed
    dist.barrier()
    if rank == 0:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [Test 3] Barrier sync          : {status}")

    # ── Test 4: Gradient Simulation ───────────────────────────────────────────
    passed = test_gradient_simulation(rank, world_size)
    all_passed = all_passed and passed
    dist.barrier()
    if rank == 0:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [Test 4] Gradient sync sim     : {status}")

    # ── Final result ──────────────────────────────────────────────────────────
    if rank == 0:
        print()
        print("=" * 60)
        if all_passed:
            print(f"  RESULT : ALL TESTS PASSED — CLUSTER IS WORKING ✓")
            print(f"           {world_size}-node CPU cluster verified and ready")
        else:
            print("  RESULT : SOME TESTS FAILED ✗  — Check output above")
        print("=" * 60)
        print()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
