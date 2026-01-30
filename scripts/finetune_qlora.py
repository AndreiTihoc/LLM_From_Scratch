#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for CyberExploitDB
============================================

Fine-tunes ScratchGPT using QLoRA (Quantized LoRA) for memory-efficient training.

QLoRA benefits:
- 4-bit quantization reduces base model memory by ~75%
- Only LoRA adapters (~1-5% of params) are trained
- Enables fine-tuning larger models on consumer GPUs
- Fast training with minimal quality loss

Usage:
    # Basic QLoRA fine-tuning (no quantization, just LoRA)
    python scripts/finetune_qlora.py --preset small --max_steps 2000

    # Full QLoRA with 4-bit quantization (requires bitsandbytes)
    python scripts/finetune_qlora.py --preset small --use_4bit --max_steps 2000

    # From a pretrained checkpoint
    python scripts/finetune_qlora.py --base_model checkpoints/best.pt --max_steps 2000

    # Colab-optimized settings
    python scripts/finetune_qlora.py --preset small --batch_size 8 --grad_accum 8 --lora_r 16
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

from src.config import ModelConfig, TrainConfig, save_run_config
from src.model import GPT
from src.dataset import TokenDataset
from src.lora import (
    LoRAConfig,
    apply_lora,
    freeze_base_model,
    save_lora_weights,
    load_lora_weights,
    count_parameters,
    HAS_BNB
)


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float
) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def evaluate(
    model: GPT,
    dataset: TokenDataset,
    eval_steps: int,
    batch_size: int,
    device: str,
    ctx
) -> Tuple[float, float]:
    """Evaluate model on dataset."""
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


@torch.no_grad()
def generate_sample(
    model: GPT,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    device: str = "cuda"
) -> str:
    """Generate a sample from the model."""
    model.eval()

    tokens = tokenizer.encode(prompt)
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        x_cond = x[:, -model.config.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)

        if next_token.item() == tokenizer.eos_id:
            break

    model.train()
    return tokenizer.decode(x[0].tolist())


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    train_config: TrainConfig,
    step: int,
    best_val_loss: float,
    path: str
) -> None:
    """Save QLoRA training checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Save full checkpoint (model + LoRA + optimizer)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "model_config": model_config.to_dict(),
        "lora_config": lora_config.to_dict(),
        "train_config": train_config.to_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "fine_tuned_on": "CyberExploitDB",
        "method": "QLoRA",
    }

    temp_path = path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.rename(temp_path, path)
    print(f"[Checkpoint] Saved to {path} (step {step})")

    # Also save LoRA-only weights
    lora_path = path.replace('.pt', '_lora.pt')
    save_lora_weights(model, lora_config, model_config, lora_path)


def finetune_qlora(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    train_config: TrainConfig,
    data_dir: str,
    base_model_path: Optional[str] = None,
    test_prompts: bool = True
) -> None:
    """
    Main QLoRA fine-tuning function.

    Args:
        model_config: Model configuration
        lora_config: LoRA configuration
        train_config: Training configuration
        data_dir: Directory containing tokenized data
        base_model_path: Path to pretrained base model checkpoint
        test_prompts: Whether to test generation during training
    """
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"[QLoRA] Using device: {device}")

    if device == "cuda":
        print(f"[QLoRA] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[QLoRA] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Mixed precision setup
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type=device_type, dtype=dtype) if train_config.use_amp and device_type == "cuda" else nullcontext()
    scaler = torch.amp.GradScaler(device_type, enabled=train_config.use_amp and device_type == "cuda")

    # Load datasets
    tokens_dir = os.path.join(data_dir, "tokens")
    train_path = os.path.join(tokens_dir, "train.bin")
    val_path = os.path.join(tokens_dir, "val.bin")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    train_dataset = TokenDataset(train_path, model_config.block_size)
    val_dataset = None
    if os.path.exists(val_path) and os.path.getsize(val_path) > 0:
        val_dataset = TokenDataset(val_path, model_config.block_size)
        if len(val_dataset) == 0:
            val_dataset = None

    # Load tokenizer
    tokenizer = None
    if test_prompts:
        try:
            from src.tokenizer import ByteBPETokenizer
            tokenizer = ByteBPETokenizer.load(os.path.join(data_dir, "tokenizer"))
        except Exception as e:
            print(f"[QLoRA] Could not load tokenizer: {e}")
            test_prompts = False

    # Load vocab size from tokenizer
    tokenizer_config_path = os.path.join(data_dir, "tokenizer", "config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r") as f:
            tok_config = json.load(f)
        model_config.vocab_size = tok_config["vocab_size"]

    # Create or load base model
    print(f"\n[QLoRA] Creating base model...")
    model = GPT(model_config)

    if base_model_path:
        print(f"[QLoRA] Loading base weights from {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])

    model = model.to(device)

    # Apply LoRA
    print(f"\n[QLoRA] Applying LoRA adapters...")
    print(f"[QLoRA]   Rank (r): {lora_config.r}")
    print(f"[QLoRA]   Alpha: {lora_config.alpha}")
    print(f"[QLoRA]   Target modules: {lora_config.target_modules}")
    print(f"[QLoRA]   4-bit quantization: {lora_config.use_4bit}")

    model = apply_lora(model, lora_config)

    # Freeze base model, only train LoRA
    freeze_base_model(model)

    # Count parameters
    param_counts = count_parameters(model)
    print(f"\n[QLoRA] Parameter summary:")
    print(f"[QLoRA]   Total: {param_counts['total']:,}")
    print(f"[QLoRA]   Trainable (LoRA): {param_counts['trainable']:,} ({100*param_counts['trainable']/param_counts['total']:.2f}%)")
    print(f"[QLoRA]   Frozen: {param_counts['frozen']:,}")

    # Compile model if requested
    if train_config.compile_model and hasattr(torch, 'compile'):
        print("[QLoRA] Compiling model...")
        model = torch.compile(model)

    # Create optimizer (only for LoRA params)
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=train_config.lr,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )
    print(f"[QLoRA] Optimizer: AdamW with {len(lora_params)} parameter groups")

    # Save config
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    save_run_config(model_config, train_config,
                   os.path.join(train_config.checkpoint_dir, "qlora_config.json"))

    # Training loop
    print(f"\n[QLoRA] Starting QLoRA fine-tuning...")
    print(f"[QLoRA] Batch size: {train_config.batch_size}")
    print(f"[QLoRA] Grad accum: {train_config.grad_accum_steps}")
    print(f"[QLoRA] Effective batch: {train_config.effective_batch_size}")
    print(f"[QLoRA] Max steps: {train_config.max_steps}")
    print()

    test_prompt_list = [
        "<|user|>What is SQL injection?<|assistant|>",
        "<|user|>Explain buffer overflow<|assistant|>",
        "<|user|>What is XSS?<|assistant|>",
    ]

    model.train()
    best_val_loss = float('inf')
    t0 = time.time()

    pbar = tqdm(range(train_config.max_steps), desc="QLoRA Training")

    for step in pbar:
        # Update learning rate
        lr = get_lr(step, train_config.warmup_steps, train_config.max_steps,
                   train_config.lr, train_config.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Gradient accumulation
        loss_accum = 0.0
        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(train_config.grad_accum_steps):
            x, y = train_dataset.get_random_batch(train_config.batch_size, device)

            with ctx:
                logits, loss = model(x, targets=y)
                loss = loss / train_config.grad_accum_steps

            loss_accum += loss.item()
            scaler.scale(loss).backward()

        # Gradient clipping
        if train_config.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, train_config.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % train_config.log_interval == 0 or step == train_config.max_steps - 1:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

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

            if val_dataset is not None:
                val_loss, val_ppl = evaluate(
                    model, val_dataset, train_config.eval_steps,
                    train_config.batch_size, device, ctx
                )
                print(f"\n[Eval] Step {step}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                val_loss = train_loss
                print(f"\n[Eval] Step {step}: train_loss={train_loss:.4f}")

            # Test generation
            if test_prompts and tokenizer and step % (train_config.eval_interval * 2) == 0:
                print("[Generation Test]")
                prompt = test_prompt_list[step // train_config.eval_interval % len(test_prompt_list)]
                try:
                    output = generate_sample(model, tokenizer, prompt, max_tokens=50, device=device)
                    print(f"  {prompt[:40]}...")
                    print(f"  -> {output[len(prompt):100]}...")
                except Exception as e:
                    print(f"  Generation failed: {e}")
                print()

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scaler,
                    model_config, lora_config, train_config,
                    step, best_val_loss,
                    os.path.join(train_config.checkpoint_dir, "best_qlora.pt")
                )

        # Periodic checkpoint
        if step > 0 and step % train_config.save_interval == 0:
            save_checkpoint(
                model, optimizer, scaler,
                model_config, lora_config, train_config,
                step, best_val_loss,
                os.path.join(train_config.checkpoint_dir, f"qlora_step_{step}.pt")
            )

    # Final checkpoint
    save_checkpoint(
        model, optimizer, scaler,
        model_config, lora_config, train_config,
        train_config.max_steps, best_val_loss,
        os.path.join(train_config.checkpoint_dir, "qlora_final.pt")
    )

    print(f"\n[QLoRA] Fine-tuning complete!")
    print(f"[QLoRA] Best validation loss: {best_val_loss:.4f}")
    print(f"\nTo test the model:")
    print(f"  python scripts/test_qlora.py --checkpoint {train_config.checkpoint_dir}/best_qlora.pt")


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for ScratchGPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data
    parser.add_argument("--data_dir", type=str, default="data/cyberexploit")

    # Model
    parser.add_argument("--preset", type=str, default="small", choices=["toy", "small"])
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--base_model", type=str, default=None, help="Pretrained base model")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--target_modules", type=str, nargs="+",
                       default=["c_attn", "c_proj"], help="Modules to apply LoRA")

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-4, help="Higher LR for LoRA")
    parser.add_argument("--min_lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/qlora")

    # Performance
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no_test", action="store_true")

    args = parser.parse_args()

    # Build model config
    model_config = ModelConfig.from_preset(args.preset)
    model_config.block_size = args.block_size

    # Build LoRA config
    lora_config = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=args.target_modules,
        use_4bit=args.use_4bit,
    )

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
        data_dir=os.path.join(args.data_dir, "tokens"),
        checkpoint_dir=args.checkpoint_dir,
        use_amp=not args.no_amp,
        compile_model=args.compile,
    )

    print("=" * 60)
    print("ScratchGPT QLoRA Fine-Tuning")
    print("=" * 60)

    print(f"\nModel: {args.preset}, block_size={model_config.block_size}")
    print(f"\nLoRA Config:")
    print(f"  r={lora_config.r}, alpha={lora_config.alpha}")
    print(f"  dropout={lora_config.dropout}")
    print(f"  target_modules={lora_config.target_modules}")
    print(f"  4-bit quantization: {lora_config.use_4bit}")
    if lora_config.use_4bit and not HAS_BNB:
        print("  WARNING: bitsandbytes not installed, 4-bit disabled")

    print(f"\nTraining Config:")
    print(f"  batch_size={train_config.batch_size}, grad_accum={train_config.grad_accum_steps}")
    print(f"  effective_batch={train_config.effective_batch_size}")
    print(f"  max_steps={train_config.max_steps}, lr={train_config.lr}")

    print("=" * 60)

    finetune_qlora(
        model_config,
        lora_config,
        train_config,
        args.data_dir,
        args.base_model,
        not args.no_test
    )


if __name__ == "__main__":
    main()
