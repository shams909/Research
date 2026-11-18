"""
CPU Cluster Lab — Script 03: Distributed GPT-2 Training (LLM)
=============================================================
PURPOSE:
    Distributed training of GPT-2 Small (117M parameters) across cluster
    nodes. This directly parallels the base paper "Towards Large-Scale
    Training on Apple Silicon" (2025) but on commodity Intel/CPU hardware.

MODEL    : GPT-2 Small — 117M parameters, 12 layers, 768 hidden dim
DATASET  : WikiText-2 — standard LM benchmark (~2M tokens, auto-downloads)
PARALLEL : PyTorch DistributedDataParallel (DDP) over Gloo backend
           Each node processes different token sequences.
           Gradients synced across nodes via AllReduce after each step.

CONNECTION TO BASE PAPER:
    Base paper: 16× Mac Mini via Thunderbolt 5 (~120 GB/s interconnect)
    This work : N× Intel PCs via Gigabit Ethernet (~0.1 GB/s interconnect)
    Key metric: Tokens/sec — directly comparable to base paper's throughput

METRICS LOGGED (saved to results/gpt2_<N>nodes_<timestamp>.csv):
    step          — training step number
    loss          — cross-entropy language model loss
    perplexity    — exp(loss) — standard LM evaluation metric
    tokens_per_sec— global training throughput (all nodes combined)
    step_time_s   — seconds per step
    world_size    — number of nodes

RUN:
    Node 01:  bash launch_master.sh 03_train_gpt2.py
    Node 02:  bash launch_worker.sh 03_train_gpt2.py 1

NOTE:
    First run downloads ~5 MB (WikiText-2 dataset).
    Training is slow on CPU — expected. That's what you're measuring.
    Use --steps 20 for a quick demo, more for real training.
"""

import os
import sys
import csv
import time
import math
import socket
import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Dataset ───────────────────────────────────────────────────────────────────

class WikiTextDataset(Dataset):
    """
    Token-chunked dataset for distributed LLM training.
    Uses WikiText-2 if accessible, or ultra-fast offline token stream.
    """
    def __init__(self, split="train", seq_len=512, rank=0):
        self.chunks = []
        loaded = False

        try:
            from datasets import load_dataset
            if rank == 0:
                print(f"  Loading WikiText-2 ({split})...")
            raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            all_text = " ".join([x for x in raw["text"] if x.strip()])
            token_ids = tokenizer.encode(all_text)
            for i in range(0, len(token_ids) - seq_len, seq_len):
                self.chunks.append(torch.tensor(token_ids[i : i + seq_len], dtype=torch.long))
            loaded = True
        except Exception:
            pass

        if not loaded:
            if rank == 0:
                print(f"  [Offline Mode] Using synthetic token stream (1000 chunks × {seq_len} tokens)")
            # Deterministic token sequences across ranks for benchmark reproducibility
            g = torch.Generator().manual_seed(42)
            for _ in range(1000):
                self.chunks.append(torch.randint(0, 50257, (seq_len,), generator=g, dtype=torch.long))

        if rank == 0:
            print(f"  Dataset : {len(self.chunks)} chunks × {seq_len} tokens ready ✓")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # Language modelling: input = tokens[:-1], labels = tokens[1:]
        return chunk[:-1], chunk[1:]


# ── Model ─────────────────────────────────────────────────────────────────────

def get_model():
    """GPT-2 Small — created from scratch (no pretrained weights)."""
    cfg = GPT2Config(
        vocab_size   = 50257,
        n_positions  = 512,    # Reduced from 1024 to fit CPU RAM
        n_embd       = 768,    # Hidden dimension
        n_layer      = 12,     # Transformer layers
        n_head       = 12,     # Attention heads
    )
    return GPT2LMHeadModel(cfg)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group(backend=config.BACKEND)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    hostname   = socket.gethostname()

    BATCH_SIZE = config.GPT2_BATCH_SIZE
    SEQ_LEN    = config.GPT2_SEQ_LEN
    NUM_STEPS  = 50   # Increase for real training (e.g. 1000+)
    LR         = 3e-4

    if rank == 0:
        print()
        print("=" * 65)
        print("  DISTRIBUTED GPT-2 TRAINING — CPU Cluster Lab")
        print(f"  Nodes      : {world_size}")
        print(f"  Model      : GPT-2 Small (117M parameters)")
        print(f"  Dataset    : WikiText-2")
        print(f"  Seq length : {SEQ_LEN} tokens")
        print(f"  Batch/node : {BATCH_SIZE}  |  Global: {BATCH_SIZE * world_size}")
        print(f"  Steps      : {NUM_STEPS}")
        print(f"  Backend    : {config.BACKEND}")
        print("=" * 65)

    # Data — download only on rank 0 to avoid conflicts
    dataset = WikiTextDataset(split="train", seq_len=SEQ_LEN + 1, rank=rank)
    dist.barrier()  # Sync after download

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=0, drop_last=True
    )

    # Model
    model     = get_model()
    model     = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parameters : {param_count:.1f}M")
        print()
        print(f"  {'Step':>6}  {'Loss':>8}  {'PPL':>10}  {'Tok/sec':>10}  {'Time(s)':>8}")
        print("  " + "-" * 50)

    # TensorBoard
    writer = None
    if rank == 0:
        run_tag = f"gpt2_{world_size}nodes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer  = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, run_tag))

    results = []
    model.train()
    step = 0

    for input_ids, labels in loader:
        if step >= NUM_STEPS:
            break

        t_start = time.perf_counter()

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss    = outputs.loss
        loss.backward()           # DDP AllReduces gradients here
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        step_time   = time.perf_counter() - t_start
        tokens_step = BATCH_SIZE * SEQ_LEN           # per node
        tokens_sec  = (tokens_step * world_size) / step_time  # global
        ppl         = math.exp(min(loss.item(), 20))  # cap to avoid overflow

        if rank == 0:
            print(f"  {step+1:>6}  {loss.item():>8.4f}  {ppl:>10.2f}  "
                  f"{tokens_sec:>10.1f}  {step_time:>8.3f}")

            if writer:
                writer.add_scalar("Loss/train",           loss.item(), step)
                writer.add_scalar("Perplexity",           ppl,         step)
                writer.add_scalar("Throughput/tokens_per_sec", tokens_sec, step)
                writer.add_scalar("Time/step_seconds",   step_time,   step)

            results.append({
                "step"         : step + 1,
                "loss"         : round(loss.item(), 4),
                "perplexity"   : round(ppl,         2),
                "tokens_per_sec": round(tokens_sec, 1),
                "step_time_s"  : round(step_time,   3),
                "world_size"   : world_size,
            })

        step += 1

    # Save results
    if rank == 0 and results:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR,   exist_ok=True)
        ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_f = os.path.join(
            config.RESULTS_DIR, f"gpt2_{world_size}nodes_{ts}.csv"
        )
        with open(csv_f, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        avg_tput = sum(r["tokens_per_sec"] for r in results) / len(results)
        print()
        print("=" * 65)
        print(f"  TRAINING COMPLETE")
        print(f"  Final loss       : {results[-1]['loss']:.4f}")
        print(f"  Final perplexity : {results[-1]['perplexity']:.2f}")
        print(f"  Avg tokens/sec   : {avg_tput:.1f} (global, {world_size} nodes)")
        print(f"  Results saved    : {csv_f}")
        print("=" * 65)
        print()
        if writer:
            writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
