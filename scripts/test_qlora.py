#!/usr/bin/env python3
"""
Test QLoRA Fine-Tuned Models
============================

Load and test a QLoRA fine-tuned model with interactive prompts.

Usage:
    # Test with a checkpoint
    python scripts/test_qlora.py --checkpoint checkpoints/qlora/best_qlora.pt

    # Single prompt
    python scripts/test_qlora.py -c checkpoints/qlora/best_qlora.pt -p "What is SQL injection?"

    # Interactive mode
    python scripts/test_qlora.py -c checkpoints/qlora/best_qlora.pt --interactive

    # With custom settings
    python scripts/test_qlora.py -c checkpoints/qlora/best_qlora.pt --temperature 0.5 --max_tokens 200
"""

import argparse
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model import GPT
from src.config import ModelConfig
from src.tokenizer import ByteBPETokenizer
from src.lora import LoRAConfig, apply_lora, load_lora_weights


def load_qlora_model(checkpoint_path: str, device: str = "cuda"):
    """
    Load a QLoRA checkpoint.

    Returns:
        model, tokenizer, model_config, lora_config
    """
    print(f"[Load] Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get configs
    model_config = ModelConfig(**checkpoint['model_config'])
    lora_config = LoRAConfig.from_dict(checkpoint['lora_config'])

    print(f"[Load] Model: {model_config.n_layer}L/{model_config.n_head}H/{model_config.d_model}D")
    print(f"[Load] LoRA: r={lora_config.r}, alpha={lora_config.alpha}")

    # Create base model
    model = GPT(model_config)

    # Apply LoRA with same config
    model = apply_lora(model, lora_config, verbose=False)

    # Load weights
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()

    print(f"[Load] Model loaded on {device}")

    return model, model_config, lora_config


def load_lora_only(
    base_checkpoint_path: str,
    lora_checkpoint_path: str,
    device: str = "cuda"
):
    """
    Load base model + separate LoRA weights.

    Useful when you have a base model and LoRA adapter saved separately.
    """
    print(f"[Load] Base model: {base_checkpoint_path}")
    print(f"[Load] LoRA weights: {lora_checkpoint_path}")

    # Load base model
    base_ckpt = torch.load(base_checkpoint_path, map_location=device, weights_only=False)
    model_config = ModelConfig(**base_ckpt['model_config'])

    model = GPT(model_config)
    model.load_state_dict(base_ckpt['model'])

    # Load and apply LoRA
    lora_ckpt = torch.load(lora_checkpoint_path, map_location=device, weights_only=False)
    lora_config = LoRAConfig.from_dict(lora_ckpt['lora_config'])

    model = apply_lora(model, lora_config, verbose=False)
    load_lora_weights(model, lora_checkpoint_path)

    model.to(device).eval()

    return model, model_config, lora_config


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda"
) -> str:
    """Generate text from prompt."""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=device)

    output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=True
    )

    return tokenizer.decode(output[0].tolist())


def format_qa(question: str, response: str) -> None:
    """Format and print Q&A."""
    # Extract answer after <|assistant|>
    if '<|assistant|>' in response:
        answer = response.split('<|assistant|>')[-1].strip()
    else:
        answer = response

    # Clean up any trailing special tokens
    for token in ['<|endoftext|>', '<|pad|>', '<|user|>']:
        if token in answer:
            answer = answer.split(token)[0].strip()

    print(f"Q: {question}")
    print(f"A: {answer}")
    print("-" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="Test QLoRA model")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                       help="Path to QLoRA checkpoint")
    parser.add_argument("--tokenizer", "-t", type=str, default="data/cyberexploit/tokenizer",
                       help="Path to tokenizer")
    parser.add_argument("--prompt", "-p", type=str, default=None,
                       help="Single prompt to test")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Interactive mode")
    parser.add_argument("--max_tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    print(f"[Load] Tokenizer from {args.tokenizer}")
    tokenizer = ByteBPETokenizer.load(args.tokenizer)

    # Load model
    model, model_config, lora_config = load_qlora_model(args.checkpoint, device)

    print()

    # Generation function
    def ask(question: str):
        prompt = f"<|user|>{question}<|assistant|>"
        response = generate(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        format_qa(question, response)

    if args.interactive:
        # Interactive mode
        print("=" * 60)
        print("QLoRA Model Interactive Mode")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)
        print()

        while True:
            try:
                question = input("You: ").strip()
                if not question:
                    continue
                if question.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                ask(question)
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    elif args.prompt:
        # Single prompt mode
        ask(args.prompt)

    else:
        # Default test prompts
        print("=" * 60)
        print("QLoRA Model Test")
        print("=" * 60)
        print()

        test_prompts = [
            "What is SQL injection?",
            "Explain buffer overflow",
            "What is XSS?",
            "What is remote code execution?",
            "Explain a path traversal vulnerability",
        ]

        for prompt in test_prompts:
            ask(prompt)


if __name__ == "__main__":
    main()
