#!/usr/bin/env python3
"""
Text Generation / Sampling Script
==================================
Generate text from a trained TinyGPT model.

Supports various sampling strategies:
- Greedy decoding (temperature=0 or do_sample=False)
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling

Usage:
    # Interactive mode
    python -m src.sample --checkpoint checkpoints/best.pt --interactive

    # Single generation
    python -m src.sample --checkpoint checkpoints/best.pt --prompt "Once upon a time"

    # From file
    python -m src.sample --checkpoint checkpoints/best.pt --prompt_file prompts.txt
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ModelConfig
from src.model import GPT
from src.tokenizer import ByteBPETokenizer


class TextGenerator:
    """
    Text generation wrapper for TinyGPT.

    Handles model loading, tokenization, and generation with
    various sampling parameters.
    """

    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        device: str = None
    ):
        """
        Initialize generator.

        Args:
            checkpoint_path: Path to model checkpoint
            tokenizer_path: Path to tokenizer directory
            device: Device to use (auto-detect if None)
        """
        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[Generator] Using device: {device}")

        # Load tokenizer
        print(f"[Generator] Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = ByteBPETokenizer.load(tokenizer_path)

        # Load checkpoint
        print(f"[Generator] Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Build model config
        model_config_dict = checkpoint.get("model_config", {})
        self.model_config = ModelConfig(**model_config_dict)

        # Ensure vocab size matches tokenizer
        self.model_config.vocab_size = self.tokenizer.vocab_size

        # Create and load model
        self.model = GPT(self.model_config)
        self.model.load_state_dict(checkpoint["model"])
        self.model = self.model.to(device)
        self.model.eval()

        print(f"[Generator] Model loaded successfully!")
        print(f"[Generator] Vocab size: {self.model_config.vocab_size}")
        print(f"[Generator] Block size: {self.model_config.block_size}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        add_bos: bool = True
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for greedy)
            top_k: Top-k sampling (None to disable)
            top_p: Top-p nucleus sampling (None to disable)
            do_sample: Use sampling vs greedy
            add_bos: Add BOS token to prompt

        Returns:
            Generated text (including prompt)
        """
        # Handle greedy mode
        if temperature == 0:
            do_sample = False
            temperature = 1.0  # Avoid division by zero

        # Encode prompt
        tokens = self.tokenizer.encode(prompt, add_bos=add_bos)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Check length
        if len(tokens) >= self.model_config.block_size:
            print(f"[Warning] Prompt length ({len(tokens)}) >= block_size ({self.model_config.block_size})")
            print(f"[Warning] Truncating prompt to last {self.model_config.block_size - 1} tokens")
            input_ids = input_ids[:, -(self.model_config.block_size - 1):]

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample
            )

        # Decode
        generated_tokens = output_ids[0].tolist()
        generated_text = self.tokenizer.decode(generated_tokens)

        return generated_text

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        do_sample: bool = True,
        add_bos: bool = True
    ):
        """
        Generate text token by token (streaming).

        Yields tokens as they are generated.

        Args:
            Same as generate()

        Yields:
            Individual tokens as strings
        """
        if temperature == 0:
            do_sample = False
            temperature = 1.0

        # Encode prompt
        tokens = self.tokenizer.encode(prompt, add_bos=add_bos)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Truncate if needed
        if len(tokens) >= self.model_config.block_size:
            input_ids = input_ids[:, -(self.model_config.block_size - 1):]

        # Yield prompt first
        yield self.tokenizer.decode(input_ids[0].tolist())

        # Generate token by token
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop context
                idx_cond = input_ids if input_ids.size(1) <= self.model_config.block_size \
                    else input_ids[:, -self.model_config.block_size:]

                # Get next token
                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')

                probs = torch.softmax(logits, dim=-1)

                if do_sample:
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)

                # Append and yield
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Decode just the new token
                token_str = self.tokenizer.decode([next_token[0].item()])
                yield token_str

                # Check for EOS
                if next_token[0].item() == self.tokenizer.eos_id:
                    break


def interactive_mode(generator: TextGenerator, args):
    """Run interactive generation loop."""
    print("\n" + "=" * 60)
    print("TinyGPT Interactive Mode")
    print("=" * 60)
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Max tokens: {args.max_tokens}")
    print("\nType your prompt and press Enter.")
    print("Commands: /quit, /temp <val>, /topk <val>, /topp <val>")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ").strip()

            if not prompt:
                continue

            # Handle commands
            if prompt.startswith("/"):
                parts = prompt.split()
                cmd = parts[0].lower()

                if cmd == "/quit" or cmd == "/exit":
                    print("Goodbye!")
                    break
                elif cmd == "/temp" and len(parts) > 1:
                    args.temperature = float(parts[1])
                    print(f"Temperature set to {args.temperature}")
                    continue
                elif cmd == "/topk" and len(parts) > 1:
                    args.top_k = int(parts[1]) if parts[1] != "none" else None
                    print(f"Top-k set to {args.top_k}")
                    continue
                elif cmd == "/topp" and len(parts) > 1:
                    args.top_p = float(parts[1]) if parts[1] != "none" else None
                    print(f"Top-p set to {args.top_p}")
                    continue
                elif cmd == "/help":
                    print("Commands:")
                    print("  /quit       - Exit")
                    print("  /temp <val> - Set temperature")
                    print("  /topk <val> - Set top-k (or 'none')")
                    print("  /topp <val> - Set top-p (or 'none')")
                    continue
                else:
                    print(f"Unknown command: {cmd}")
                    continue

            # Generate
            print()
            if args.stream:
                for token in generator.generate_stream(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=args.temperature > 0
                ):
                    print(token, end="", flush=True)
                print("\n")
            else:
                output = generator.generate(
                    prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=args.temperature > 0
                )
                print(output)
                print()

        except KeyboardInterrupt:
            print("\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with TinyGPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default="data/tokenizer",
        help="Path to tokenizer directory"
    )

    # Input
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default=None,
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="File containing prompts (one per line)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode"
    )

    # Generation params
    parser.add_argument(
        "--max_tokens", "-n",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", "-T",
        type=float,
        default=0.8,
        help="Sampling temperature (0 for greedy)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling (0 to disable)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (1.0 to disable)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream output token by token"
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu, auto-detect if not set)"
    )

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)

    # Process args
    if args.top_k == 0:
        args.top_k = None
    if args.top_p >= 1.0:
        args.top_p = None

    # Load generator
    generator = TextGenerator(
        args.checkpoint,
        args.tokenizer,
        device=args.device
    )

    # Run
    if args.interactive:
        interactive_mode(generator, args)

    elif args.prompt_file:
        # Generate from file
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]

        for i, prompt in enumerate(prompts):
            print(f"\n=== Prompt {i+1}/{len(prompts)} ===")
            print(f"Input: {prompt}")
            print(f"Output:")
            output = generator.generate(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.temperature > 0
            )
            print(output)

    elif args.prompt:
        # Single generation
        if args.stream:
            for token in generator.generate_stream(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.temperature > 0
            ):
                print(token, end="", flush=True)
            print()
        else:
            output = generator.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.temperature > 0
            )
            print(output)

    else:
        print("Error: Provide --prompt, --prompt_file, or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
