"""
CPU Cluster Lab — Script 07: Distributed Whisper-Tiny Speech Training (Voice)
=============================================================================
PURPOSE:
    Distributed convergence verification for Voice / Speech-to-Text
    (Whisper-Tiny, ~37.8M parameters) on Google Speech Commands over Gloo.
    Matches Script 02 (ResNet-18/CIFAR-10) and Script 03 (GPT-2/WikiText-2)
    with identical DDP, no_sync() gradient accumulation (K=4), and CSV logging.

RUN:
    Node 01:  bash launch_master.sh 07_train_whisper.py 2 --steps 15
    Single :  bash launch_single.sh 07_train_whisper.py --steps 15
"""

import os
import sys
import csv
import time
import math
import socket
import datetime
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ── Whisper-Tiny Model Builder ────────────────────────────────────────────────

def get_model():
    """Whisper-Tiny (~37.8M params) instantiated with standard OpenAI dimensions."""
    from transformers import WhisperForConditionalGeneration, WhisperConfig

    cfg = WhisperConfig(
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=4,
        encoder_attention_heads=6,
        decoder_layers=4,
        decoder_attention_heads=6,
        encoder_ffn_dim=1536,
        decoder_ffn_dim=1536,
        d_model=384,
        max_source_positions=1500,
        max_target_positions=448,
        decoder_start_token_id=50258,
        pad_token_id=50257,
        bos_token_id=50257,
        eos_token_id=50257,
        suppress_tokens=[],
        begin_suppress_tokens=[],
    )
    return WhisperForConditionalGeneration(cfg)


# ── Speech Commands Dataset (Pure SciPy + TorchAudio MelSpectrogram) ──────────

class SpeechCommandsWhisperDataset(Dataset):
    """
    Loads real spoken-command .wav clips via scipy.io.wavfile (avoids codec
    dependencies) and produces standard (80, 3000) log-Mel inputs + token labels.
    """

    def __init__(self, dec_len=32, max_samples=4096, rank=0):
        import torchaudio

        self.dec_len = dec_len
        self.samples = []
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=400, hop_length=160, n_mels=80
        )

        loaded = False
        try:
            from transformers import WhisperTokenizer

            os.makedirs(config.DATA_DIR, exist_ok=True)
            if rank == 0:
                print("  Loading Speech Commands (training subset)...")
            ds = torchaudio.datasets.SPEECHCOMMANDS(
                root=config.DATA_DIR, subset="training", download=True
            )
            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")
            walker = ds._walker
            step = max(1, len(walker) // max_samples)
            selected = walker[::step][:max_samples]

            for rel_path in selected:
                rel_str = str(rel_path)
                full_path = rel_str if os.path.exists(rel_str) else os.path.join(ds._path, rel_str)
                if not os.path.exists(full_path):
                    raise FileNotFoundError(f"SpeechCommands audio clip missing: {full_path}")
                label_word = os.path.basename(os.path.dirname(full_path)).replace("_", " ")
                tok_ids = tokenizer.encode(label_word, add_special_tokens=True)
                # Pad or truncate decoder label sequence to fixed dec_len
                if len(tok_ids) < dec_len:
                    tok_ids = tok_ids + [-100] * (dec_len - len(tok_ids))
                else:
                    tok_ids = tok_ids[:dec_len]
                self.samples.append((full_path, torch.tensor(tok_ids, dtype=torch.long)))

            loaded = len(self.samples) > 0
            if rank == 0 and loaded:
                print(f"  Dataset : {len(self.samples)} real .wav utterances ready ✓")
        except Exception as exc:
            if rank == 0:
                print(f"  [Warning] Real SpeechCommands load skipped ({exc})")

        self.loaded = loaded
        if not loaded:
            if rank == 0:
                print(f"  [Offline Mode] Using deterministic synthetic audio tensors ({max_samples} clips)")
            self.samples = [(None, torch.randint(0, 50256, (dec_len,), dtype=torch.long)) for _ in range(max_samples)]

    def __len__(self):
        return len(self.samples)

    def _wav_to_log_mel(self, wav_path):
        sr, audio = wavfile.read(wav_path)
        if audio.dtype == np.int16:
            wav = torch.from_numpy(audio.astype(np.float32) / 32768.0)
        else:
            wav = torch.from_numpy(audio.astype(np.float32))
        if wav.ndim > 1:
            wav = wav.mean(dim=-1)

        # Pad / crop into Whisper's standard 30-second (480,000 sample @ 16kHz) window
        target_len = 480000
        if wav.numel() < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.numel()))
        else:
            wav = wav[:target_len]

        mel = self.mel_transform(wav)[:, :3000]
        log_mel = torch.log10(mel.clamp(min=1e-10))
        log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel

    def __getitem__(self, idx):
        wav_path, labels = self.samples[idx]
        if wav_path is None:
            g = torch.Generator().manual_seed(idx)
            return torch.randn((80, 3000), generator=g), labels
        return self._wav_to_log_mel(wav_path), labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distributed Whisper-Tiny Speech Training")
    parser.add_argument("--steps", type=int, default=15, help="Number of optimizer steps to train")
    parser.add_argument("--batch_size", type=int, default=getattr(config, "WHISPER_BATCH_SIZE", 4),
                        help="Batch size per node")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args, unknown = parser.parse_known_args()

    dist.init_process_group(backend=config.BACKEND)
    rank       = dist.get_rank()
    world_size = dist.get_world_size()

    BATCH_SIZE = args.batch_size
    DEC_LEN    = getattr(config, "WHISPER_DEC_LEN", 32)
    NUM_STEPS  = args.steps
    LR         = args.lr

    if rank == 0:
        print()
        print("=" * 65)
        print("  DISTRIBUTED WHISPER-TINY TRAINING (VOICE) — CPU Cluster Lab")
        print(f"  Nodes      : {world_size}")
        print("  Model      : Whisper-Tiny (~37.8M parameters)")
        print("  Dataset    : Speech Commands (16 kHz Audio -> 80-bin Log-Mel)")
        print(f"  Batch/node : {BATCH_SIZE}  |  Global: {BATCH_SIZE * world_size}")
        print(f"  Steps      : {NUM_STEPS}")
        print(f"  Backend    : {config.BACKEND}")
        print("=" * 65)

    dataset = SpeechCommandsWhisperDataset(dec_len=DEC_LEN, rank=rank)
    dist.barrier()

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, drop_last=True)

    model     = DDP(get_model())
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  Parameters : {param_count:.1f}M")
        print()
        print(f"  {'Step':>6}  {'Loss':>8}  {'PPL':>10}  {'Clips/sec':>10}  {'Time(s)':>8}")
        print("  " + "-" * 50)

    writer = None
    if rank == 0:
        run_tag = f"whisper_{world_size}nodes_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer  = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, run_tag))

    accum_steps = getattr(config, "GRADIENT_ACCUM_STEPS", 4)
    results = []
    model.train()
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad()

    epoch = 0
    while step < NUM_STEPS:
        sampler.set_epoch(epoch)
        epoch += 1
        for batch_idx, (input_features, labels) in enumerate(loader):
            if step >= NUM_STEPS:
                break

            if (batch_idx % accum_steps) == 0:
                t_start = time.perf_counter()

            is_sync_step = ((batch_idx + 1) % accum_steps == 0)

            if not is_sync_step and world_size > 1:
                with model.no_sync():
                    outputs = model(input_features=input_features, labels=labels)
                    loss    = outputs.loss / accum_steps
                    loss.backward()
            else:
                outputs = model(input_features=input_features, labels=labels)
                loss    = outputs.loss / accum_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            accum_loss += loss.item() * accum_steps

            if is_sync_step:
                step_time     = time.perf_counter() - t_start
                clips_step    = BATCH_SIZE * accum_steps
                clips_sec     = (clips_step * world_size) / step_time
                avg_step_loss = accum_loss / accum_steps
                ppl           = math.exp(min(avg_step_loss, 20))
                accum_loss    = 0.0

                if rank == 0:
                    print(f"  {step+1:>6}  {avg_step_loss:>8.4f}  {ppl:>10.2f}  "
                          f"{clips_sec:>10.2f}  {step_time:>8.3f}", flush=True)

                    if writer:
                        writer.add_scalar("Loss/train", avg_step_loss, step)
                        writer.add_scalar("Perplexity", ppl, step)
                        writer.add_scalar("Throughput/clips_per_sec", clips_sec, step)
                        writer.add_scalar("Time/step_seconds", step_time, step)

                    results.append({
                        "step"         : step + 1,
                        "loss"         : round(avg_step_loss, 4),
                        "perplexity"   : round(ppl, 2),
                        "clips_per_sec": round(clips_sec, 2),
                        "step_time_s"  : round(step_time, 3),
                        "world_size"   : world_size,
                        "accum_steps"  : accum_steps,
                    })

                step += 1

    if rank == 0 and results:
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_f = os.path.join(config.RESULTS_DIR, f"whisper_{world_size}nodes_{ts}.csv")
        with open(csv_f, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=results[0].keys())
            writer_csv.writeheader()
            writer_csv.writerows(results)

        avg_tput = sum(r["clips_per_sec"] for r in results) / len(results)
        print()
        print("=" * 65)
        print("  VOICE TRAINING COMPLETE")
        print(f"  Final loss       : {results[-1]['loss']:.4f}")
        print(f"  Final perplexity : {results[-1]['perplexity']:.2f}")
        print(f"  Avg clips/sec    : {avg_tput:.2f} (global, {world_size} nodes)")
        print(f"  Results saved    : {csv_f}")
        print("=" * 65)
        print()

    if writer:
        writer.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
