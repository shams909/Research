"""
CPU Cluster Lab — Script 02: Distributed CIFAR-10 Training (ResNet-18)
=======================================================================
PURPOSE:
    Quick, working distributed training demo using a well-known benchmark.
    Proves the cluster can actually *train* a model — not just communicate.
    Faster to run than GPT-2, good for showing your professor on Day 1.

MODEL    : ResNet-18 (~11M parameters) — standard CV benchmark model
DATASET  : CIFAR-10 — 60,000 images, 10 classes (auto-downloads ~170 MB)
PARALLEL : PyTorch DistributedDataParallel (DDP) — data parallelism
           Each node trains on a different subset of data.
           Gradients averaged across all nodes after each step.

METRICS LOGGED (saved to results/cifar_<N>nodes_<timestamp>.csv):
    epoch         — current epoch number
    train_loss    — average training loss this epoch
    test_acc      — test accuracy (%) — only computed on rank 0
    epoch_time_s  — seconds for this epoch
    imgs_per_sec  — training throughput (images/second, global)
    world_size    — number of nodes used

RUN:
    Node 01:  bash launch_master.sh 02_train_cifar.py
    Node 02:  bash launch_worker.sh 02_train_cifar.py 1

    For 4 nodes:
    Node 01:  bash launch_master.sh 02_train_cifar.py 4
    Node 02:  bash launch_worker.sh 02_train_cifar.py 1 4
    Node 03:  bash launch_worker.sh 02_train_cifar.py 2 4
    Node 04:  bash launch_worker.sh 02_train_cifar.py 3 4
"""

import os
import sys
import csv
import time
import socket
import datetime
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Data ─────────────────────────────────────────────────────────────────────

def get_dataloaders(rank, world_size, batch_size):
    """
    Prepare CIFAR-10 dataloaders with DistributedSampler.
    DistributedSampler ensures each node gets a unique, non-overlapping
    subset of the dataset — this is data parallelism.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Both nodes ensure dataset is downloaded to their local disk
    torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, download=True
    )
    torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True
    )
    dist.barrier()  # Wait for all nodes to finish downloading

    train_set = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True, transform=transform_train
    )
    test_set = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, transform=transform_test
    )

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=False
    )
    # Test loader: all nodes load full test set for accuracy computation
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, train_sampler


# ── Model ─────────────────────────────────────────────────────────────────────

def get_model():
    """
    ResNet-18 adapted for CIFAR-10 (32×32 images instead of 224×224).
    Replaces the first 7×7 conv with 3×3 and removes the initial maxpool.
    """
    model = resnet18(weights=None, num_classes=10)
    # Adapt for 32x32 CIFAR images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_epoch(model, loader, sampler, optimizer, criterion, rank, world_size, epoch, total_epochs, accum_steps=1):
    model.train()
    sampler.set_epoch(epoch)  # Ensures different shuffling each epoch

    total_loss = 0.0
    total_samples = 0
    start = time.time()
    num_batches = len(loader)

    optimizer.zero_grad()

    for batch_idx, (inputs, targets) in enumerate(loader):
        b_start = time.perf_counter()

        # Only sync gradients across network on the final accumulation step
        is_sync_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == num_batches)

        if not is_sync_step and world_size > 1:
            with model.no_sync():
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accum_steps
                loss.backward()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accum_steps
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        b_time = time.perf_counter() - b_start
        total_loss    += loss.item() * accum_steps * inputs.size(0)
        total_samples += inputs.size(0)

        # Print live streaming progress every 25 batches (or first/last batch) on rank 0
        if rank == 0 and (batch_idx == 0 or (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == num_batches):
            current_loss = total_loss / total_samples
            b_tput = (inputs.size(0) * world_size) / max(b_time, 1e-4)
            pct = 100.0 * (batch_idx + 1) / num_batches
            print(f"  [Epoch {epoch:>2}/{total_epochs} | Batch {batch_idx+1:>3}/{num_batches} ({pct:>4.1f}%)] "
                  f"Loss: {loss.item() * accum_steps:.4f} (avg: {current_loss:.4f}) | "
                  f"{b_tput:>5.1f} imgs/s", flush=True)

    epoch_time  = time.time() - start
    avg_loss    = total_loss / total_samples
    # Global throughput = per-node samples * number of nodes / time
    imgs_per_sec = (total_samples * world_size) / epoch_time

    return avg_loss, epoch_time, imgs_per_sec


def evaluate(model, loader, criterion):
    """Evaluate on the full test set (run on all nodes, identical result)."""
    model.eval()
    correct = total = total_loss = 0

    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            total_loss  += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total   += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distributed CIFAR-10 Training")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE, help="Batch size per node")
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR, help="Learning rate")
    args, unknown = parser.parse_known_args()

    dist.init_process_group(backend=config.BACKEND)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    hostname   = socket.gethostname()

    BATCH_SIZE = args.batch_size
    EPOCHS     = args.epochs
    LR         = args.lr

    if rank == 0:
        print()
        print("=" * 65)
        print("  DISTRIBUTED CIFAR-10 TRAINING — CPU Cluster Lab")
        print(f"  Nodes      : {world_size}")
        print(f"  Model      : ResNet-18 (~11M params)")
        print(f"  Dataset    : CIFAR-10 (60,000 images, 10 classes)")
        print(f"  Batch/node : {BATCH_SIZE}  |  Global batch: {BATCH_SIZE * world_size}")
        print(f"  Epochs     : {EPOCHS}")
        print(f"  Backend    : {config.BACKEND}")
        print("=" * 65)

    # Setup
    train_loader, test_loader, train_sampler = get_dataloaders(
        rank, world_size, BATCH_SIZE
    )

    model     = get_model()
    model     = DDP(model)   # Wraps model for distributed gradient sync
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # TensorBoard writer — only rank 0 logs (master node)
    writer = None
    if rank == 0:
        import datetime
        run_tag = f"cifar_{world_size}nodes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer  = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, run_tag))

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parameters : {param_count:.1f}M")
        print()
        print(f"  {'Epoch':>5}  {'Loss':>8}  {'Acc (%)':>8}  {'Time (s)':>9}  {'Imgs/sec':>9}")
        print("  " + "-" * 46)

    # Training loop
    results = []
    for epoch in range(1, EPOCHS + 1):
        if rank == 0:
            print(f"\n>>> Starting Epoch {epoch}/{EPOCHS} ...", flush=True)

        train_loss, epoch_time, imgs_per_sec = train_epoch(
            model, train_loader, train_sampler,
            optimizer, criterion, rank, world_size, epoch, EPOCHS,
            accum_steps=getattr(config, "GRADIENT_ACCUM_STEPS", 1)
        )
        scheduler.step()

        # Evaluate on rank 0 only
        test_acc = 0.0
        if rank == 0:
            print(f"  [Evaluating Epoch {epoch} Test Accuracy...]", flush=True)
            raw_model = model.module if hasattr(model, "module") else model
            test_acc = evaluate(raw_model, test_loader, criterion)
            print(f"  ================================================================")
            print(f"  ✓ Epoch {epoch}/{EPOCHS} Finished: Loss: {train_loss:.4f} | "
                  f"Test Acc: {test_acc:.2f}% | Time: {epoch_time:.1f}s | Speed: {imgs_per_sec:.1f} imgs/s")
            print(f"  ================================================================\n", flush=True)

            # TensorBoard logging
            if writer:
                writer.add_scalar("Loss/train",        train_loss,  epoch)
                writer.add_scalar("Accuracy/test",     test_acc,    epoch)
                writer.add_scalar("Throughput/imgs_per_sec", imgs_per_sec, epoch)
                writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

            results.append({
                "epoch"       : epoch,
                "train_loss"  : round(train_loss,  4),
                "test_acc_pct": round(test_acc,     2),
                "epoch_time_s": round(epoch_time,   2),
                "imgs_per_sec": round(imgs_per_sec, 1),
                "world_size"  : world_size,
                "accum_steps" : getattr(config, "GRADIENT_ACCUM_STEPS", 1),
            })

        dist.barrier()

    # Save results & checkpoint
    if rank == 0:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR,   exist_ok=True)
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_f = os.path.join(
            config.RESULTS_DIR, f"cifar_{world_size}nodes_{ts}.csv"
        )
        with open(csv_f, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        ckpt_f = os.path.join(
            config.MODEL_DIR, f"resnet18_cifar10_{world_size}nodes_{ts}.pt"
        )
        torch.save(model.module.state_dict(), ckpt_f)

        avg_tput = sum(r["imgs_per_sec"] for r in results) / len(results)
        print()
        print("=" * 65)
        print(f"  TRAINING COMPLETE")
        print(f"  Final accuracy   : {results[-1]['test_acc_pct']:.2f}%")
        print(f"  Avg throughput   : {avg_tput:.1f} images/sec (global)")
        print(f"  Results saved    : {csv_f}")
        print(f"  Checkpoint saved : {ckpt_f}")
        print("=" * 65)
        print()
        # Training complete

    dist.destroy_process_group()


if __name__ == "__main__":
    main()