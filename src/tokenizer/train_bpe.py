#!/usr/bin/env python3
"""
BPE Tokenizer Training Script
=============================
Train a byte-level BPE tokenizer on a text corpus.

Usage:
    python -m src.tokenizer.train_bpe --input data/raw/input.txt --output data/tokenizer --vocab_size 4096

This script:
1. Reads text file(s) in a streaming fashion (low memory)
2. Trains BPE tokenizer to target vocabulary size
3. Saves tokenizer files (vocab.json, merges.txt, config.json)
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizer.byte_bpe import ByteBPETokenizer


def stream_lines(path: str, max_lines: int = None) -> iter:
    """
    Stream lines from a text file.

    Args:
        path: Path to text file
        max_lines: Maximum lines to read (None for all)

    Yields:
        Stripped non-empty lines
    """
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line
                count += 1
                if max_lines and count >= max_lines:
                    break


def main():
    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input text file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/tokenizer",
        help="Output directory for tokenizer files"
    )
    parser.add_argument(
        "--vocab_size", "-v",
        type=int,
        default=4096,
        help="Target vocabulary size"
    )
    parser.add_argument(
        "--min_freq", "-m",
        type=int,
        default=2,
        help="Minimum frequency for BPE merges"
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Maximum lines to process (for testing)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Print configuration
    if not args.quiet:
        print("=" * 50)
        print("BPE Tokenizer Training")
        print("=" * 50)
        print(f"Input file:    {args.input}")
        print(f"Output dir:    {args.output}")
        print(f"Vocab size:    {args.vocab_size}")
        print(f"Min frequency: {args.min_freq}")
        if args.max_lines:
            print(f"Max lines:     {args.max_lines}")
        print("=" * 50)

    # Create tokenizer and train
    tokenizer = ByteBPETokenizer()

    # Stream lines from file
    line_iter = stream_lines(args.input, max_lines=args.max_lines)

    tokenizer.train(
        texts=line_iter,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq,
        verbose=not args.quiet
    )

    # Save tokenizer
    tokenizer.save(args.output)

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 50)
        print("Training Complete!")
        print("=" * 50)
        print(f"Final vocab size: {tokenizer.vocab_size}")
        print(f"Number of merges: {len(tokenizer.merges)}")
        print(f"Vocab hash:       {tokenizer.get_vocab_hash()}")
        print(f"Files saved to:   {args.output}/")
        print("  - config.json")
        print("  - vocab.json")
        print("  - merges.txt")
        print("=" * 50)

    # Quick test
    if not args.quiet:
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"\nQuick test:")
        print(f"  Input:   '{test_text}'")
        print(f"  Tokens:  {tokens}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Match:   {test_text == decoded}")


if __name__ == "__main__":
    main()
