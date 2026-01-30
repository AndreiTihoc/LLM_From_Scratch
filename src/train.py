#!/usr/bin/env python3
"""
Training Script
===============
Complete training loop for the GPT model with:
- Mixed precision training (AMP)
- Gradient accumulation
- Cosine learning rate schedule with warmup
- Gradient clipping
- Periodic evaluation
- Checkpoint save/resume

Usage:
    # Quick test (toy mode)
    python -m src.train --preset toy --max_steps 200

    # Full training
    python -m src.train --preset small --max_steps 10000

    # Resume training
    python -m src.train --resume checkpoints/latest.pt
"""

import os
import sys
import math
import json
import time
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ModelConfig, TrainConfig, save_run_config, get_git_commit
from src.model import GPT
from src.dataset import TokenDataset


# ============================================
# Learning Rate Schedule
# ============================================

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float
) -> float:
    """
    Cosine learning rate schedule with warmup.

    Args:
        step: Current training step
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        max_lr: Peak learning rate
        min_lr: Minimum learning rate

    Returns:
        Learning rate for this step
    """
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Constant min_lr after decay
    if step > max_steps:
        return min_lr

    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================
# Evaluation
# ============================================

@torch.no_grad()
def evaluate(
    model: GPT,
    dataset: TokenDataset,
    eval_steps: int,
    batch_size: int,
    device: str,
    ctx
) -> Tuple[float, float]:
    """
    Evaluate model on dataset.

    Args:
        model: GPT model
        dataset: Evaluation dataset
        eval_steps: Number of evaluation batches
        batch_size: Batch size
        device: Device to use
        ctx: AMP context

    Returns:
        (average_loss, perplexity)
    """
    model.eval()
    losses = []

    for _ in range(eval_steps):
        x, y = dataset.get_random_batch(batch_size, device)
        with ctx:
            _, loss = model(x, targets=y)
        losses.append(loss.item())

    model.train()

    avg_loss = np.mean(losses)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

    return avg_loss, perplexity


# ============================================
# Checkpoint Management
# ============================================

def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    best_val_loss: float,
    path: str
) -> None:
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "git_commit": get_git_commit(),
    }

    # Save to temp file first, then rename (atomic save)
    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.rename(temp_path, path)

    print(f"[Checkpoint] Saved to {path} (step {step})")


def load_checkpoint(
    path: str,
    model: GPT,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[int, float]:
    """
    Load training checkpoint.

    Args:
        path: Path to checkpoint file
        model: GPT model to load weights into
        optimizer: Optimizer to load state into (optional)
        scaler: GradScaler to load state into (optional)

    Returns:
        (step, best_val_loss)
    """
    print(f"[Checkpoint] Loading from {path}...")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model"])
    print(f"[Checkpoint] Loaded model weights")

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[Checkpoint] Loaded optimizer state")

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
        print(f"[Checkpoint] Loaded scaler state")

    step = checkpoint.get("step", 0)
    best_val_loss = checkpoint.get("best_val_loss", float('inf'))

    print(f"[Checkpoint] Resuming from step {step}, best_val_loss={best_val_loss:.4f}")

    return step, best_val_loss


# ============================================
# Main Training Loop
# ============================================

def train(
    model_config: ModelConfig,
    train_config: TrainConfig,
    resume_from: Optional[str] = None
) -> None:
    """
    Main training function.

    Args:
        model_config: Model configuration
        train_config: Training configuration
        resume_from: Path to checkpoint to resume from
    """
    # Set random seed
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"[Train] Using device: {device}")

    # Mixed precision setup
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype) if train_config.use_amp and device_type == "cuda" else nullcontext()
    scaler = torch.amp.GradScaler(device_type, enabled=train_config.use_amp and device_type == "cuda")
    print(f"[Train] Mixed precision: {train_config.use_amp}, dtype: {dtype if train_config.use_amp else 'float32'}")

    # Load datasets
    print(f"\n[Train] Loading datasets from {train_config.data_dir}...")
    train_path = os.path.join(train_config.data_dir, "train.bin")
    val_path = os.path.join(train_config.data_dir, "val.bin")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found: {val_path}")

    train_dataset = TokenDataset(train_path, model_config.block_size)

    # Val dataset might be empty for very small datasets
    val_dataset = None
    if os.path.getsize(val_path) > 0:
        val_dataset = TokenDataset(val_path, model_config.block_size)
        if len(val_dataset) == 0:
            val_dataset = None
            print("[Train] Warning: Val dataset too small, skipping validation")
    else:
        print("[Train] Warning: Val dataset empty, skipping validation")

    # Load vocab size from tokenizer config if available
    tokenizer_config_path = os.path.join(
        os.path.dirname(train_config.data_dir), "tokenizer", "config.json"
    )
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            tok_config = json.load(f)
        model_config.vocab_size = tok_config["vocab_size"]
        print(f"[Train] Loaded vocab_size={model_config.vocab_size} from tokenizer config")

    # Create model
    print(f"\n[Train] Creating model...")
    model = GPT(model_config)
    model = model.to(device)

    # Compile model if requested
    if train_config.compile_model and hasattr(torch, 'compile'):
        print("[Train] Compiling model with torch.compile...")
        model = torch.compile(model)

    # Create optimizer
    optimizer = model.configure_optimizers(
        weight_decay=train_config.weight_decay,
        learning_rate=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        device_type=device_type
    )

    # Resume from checkpoint
    start_step = 0
    best_val_loss = float('inf')

    if resume_from:
        start_step, best_val_loss = load_checkpoint(
            resume_from, model, optimizer, scaler
        )

    # Save run configuration
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    save_run_config(
        model_config, train_config,
        os.path.join(train_config.checkpoint_dir, "run_config.json")
    )

    # Training loop
    print(f"\n[Train] Starting training...")
    print(f"[Train] Batch size: {train_config.batch_size}")
    print(f"[Train] Grad accum steps: {train_config.grad_accum_steps}")
    print(f"[Train] Effective batch size: {train_config.effective_batch_size}")
    print(f"[Train] Max steps: {train_config.max_steps}")
    print(f"[Train] LR: {train_config.lr} -> {train_config.min_lr}")
    print()

    model.train()
    t0 = time.time()

    # Progress bar
    pbar = tqdm(
        range(start_step, train_config.max_steps),
        initial=start_step,
        total=train_config.max_steps,
        desc="Training"
    )

    for step in pbar:
        # Update learning rate
        lr = get_lr(
            step,
            train_config.warmup_steps,
            train_config.max_steps,
            train_config.lr,
            train_config.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation loop
        loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(train_config.grad_accum_steps):
            # Get batch
            x, y = train_dataset.get_random_batch(train_config.batch_size, device)

            # Forward pass with AMP
            with ctx:
                logits, loss = model(x, targets=y)
                # Scale loss for gradient accumulation
                loss = loss / train_config.grad_accum_steps

            loss_accum += loss.item()

            # Backward pass
            scaler.scale(loss).backward()

        # Gradient clipping
        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % train_config.log_interval == 0 or step == train_config.max_steps - 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Tokens per second
            tokens_per_step = train_config.effective_batch_size * model_config.block_size
            tokens_per_sec = tokens_per_step * train_config.log_interval / dt if dt > 0 else 0

            pbar.set_postfix({
                "loss": f"{loss_accum:.4f}",
                "lr": f"{lr:.2e}",
                "tok/s": f"{tokens_per_sec:.0f}"
            })

        # Evaluation
        if step % train_config.eval_interval == 0 or step == train_config.max_steps - 1:
            train_loss, train_ppl = evaluate(
                model, train_dataset, train_config.eval_steps,
                train_config.batch_size, device, ctx
            )

            # Evaluate on val if available, otherwise use train loss
            if val_dataset is not None:
                val_loss, val_ppl = evaluate(
                    model, val_dataset, train_config.eval_steps,
                    train_config.batch_size, device, ctx
                )
                print(f"\n[Eval] Step {step}: train_loss={train_loss:.4f} (ppl={train_ppl:.2f}), "
                      f"val_loss={val_loss:.4f} (ppl={val_ppl:.2f})")
            else:
                val_loss = train_loss
                val_ppl = train_ppl
                print(f"\n[Eval] Step {step}: train_loss={train_loss:.4f} (ppl={train_ppl:.2f})")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scaler,
                    model_config, train_config,
                    step, best_val_loss,
                    os.path.join(train_config.checkpoint_dir, "best.pt")
                )

        # Periodic checkpoint
        if step > 0 and step % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scaler,
                model_config, train_config,
                step, best_val_loss,
                os.path.join(train_config.checkpoint_dir, f"step_{step}.pt")
            )

    # Final checkpoint
    save_checkpoint(
        model, optimizer, scaler,
        model_config, train_config,
        train_config.max_steps, best_val_loss,
        os.path.join(train_config.checkpoint_dir, "latest.pt")
    )

    print(f"\n[Train] Training complete!")
    print(f"[Train] Best validation loss: {best_val_loss:.4f}")


# ============================================
# Main Entry Point
# ============================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ScratchGPT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model config
    parser.add_argument(
        "--preset", type=str, default="toy",
        choices=["toy", "small"],
        help="Model preset"
    )
    parser.add_argument("--vocab_size", type=int, default=None, help="Override vocab size")
    parser.add_argument("--block_size", type=int, default=None, help="Override block size")
    parser.add_argument("--n_layer", type=int, default=None, help="Override number of layers")
    parser.add_argument("--n_head", type=int, default=None, help="Override number of heads")
    parser.add_argument("--d_model", type=int, default=None, help="Override model dimension")
    parser.add_argument("--dropout", type=float, default=None, help="Override dropout")

    # Training config
    parser.add_argument("--batch_size", type=int, default=32, help="Micro batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--eval_interval", type=int, default=250, help="Eval interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Checkpoint interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Paths
    parser.add_argument("--data_dir", type=str, default="data/tokens", help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")

    # Performance
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")

    # Resume
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    args = parser.parse_args()

    # Build model config
    model_config = ModelConfig.from_preset(args.preset)
    if args.vocab_size is not None:
        model_config.vocab_size = args.vocab_size
    if args.block_size is not None:
        model_config.block_size = args.block_size
    if args.n_layer is not None:
        model_config.n_layer = args.n_layer
    if args.n_head is not None:
        model_config.n_head = args.n_head
    if args.d_model is not None:
        model_config.d_model = args.d_model
    if args.dropout is not None:
        model_config.dropout = args.dropout

    # Build training config
    train_config = TrainConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        max_steps=args.max_steps,
        lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
        compile_model=args.compile,
    )

    print("=" * 60)
    print("ScratchGPT Training")
    print("=" * 60)
    print(f"\nModel Config:")
    print(f"  Preset: {args.preset}")
    print(f"  vocab_size: {model_config.vocab_size}")
    print(f"  block_size: {model_config.block_size}")
    print(f"  n_layer: {model_config.n_layer}")
    print(f"  n_head: {model_config.n_head}")
    print(f"  d_model: {model_config.d_model}")
    print(f"  dropout: {model_config.dropout}")
    print(f"  Est. params: {model_config.n_params:,}")

    print(f"\nTraining Config:")
    print(f"  batch_size: {train_config.batch_size}")
    print(f"  grad_accum_steps: {train_config.grad_accum_steps}")
    print(f"  effective_batch_size: {train_config.effective_batch_size}")
    print(f"  max_steps: {train_config.max_steps}")
    print(f"  lr: {train_config.lr}")
    print(f"  use_amp: {train_config.use_amp}")
    print(f"  compile: {train_config.compile_model}")

    print("=" * 60)

    # Train
    train(model_config, train_config, args.resume)


if __name__ == "__main__":
    main()
