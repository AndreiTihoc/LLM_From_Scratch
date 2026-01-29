#!/usr/bin/env python3
"""
Data Preparation Script
=======================
Complete data preparation pipeline:
1. Clean raw text
2. Train BPE tokenizer
3. Encode text to tokens
4. Write binary memmap files

Usage:
    python scripts/prepare_data.py --input data/raw/input.txt

    # With custom vocab size
    python scripts/prepare_data.py --input data/raw/input.txt --vocab_size 8192

    # Quick test mode
    python scripts/prepare_data.py --input data/raw/input.txt --max_lines 10000
"""

import os
import sys
import re
import json
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tokenizer.byte_bpe import ByteBPETokenizer
from src.dataset import prepare_dataset, TokenDataset


def clean_text(text: str) -> str:
    """
    Clean raw text for training.

    - Normalize whitespace
    - Remove excessive newlines
    - Keep punctuation and structure
    """
    # Normalize various whitespace chars
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove excessive blank lines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text


def clean_file(input_path: str, output_path: str, max_lines: int = None) -> int:
    """
    Clean a text file.

    Args:
        input_path: Path to raw text file
        output_path: Path to cleaned output file
        max_lines: Maximum lines to keep (None for all)

    Returns:
        Number of lines written
    """
    print(f"[Clean] Reading {input_path}...")

    with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Clean
    text = clean_text(text)

    # Limit lines if requested
    if max_lines:
        lines = text.split('\n')[:max_lines]
        text = '\n'.join(lines)

    # Write cleaned file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    line_count = text.count('\n') + 1
    print(f"[Clean] Wrote {line_count:,} lines to {output_path}")

    return line_count


def train_tokenizer(
    input_path: str,
    output_dir: str,
    vocab_size: int,
    min_freq: int = 2
) -> ByteBPETokenizer:
    """
    Train BPE tokenizer on text file.

    Args:
        input_path: Path to text file
        output_dir: Output directory for tokenizer
        vocab_size: Target vocabulary size
        min_freq: Minimum merge frequency

    Returns:
        Trained tokenizer
    """
    print(f"\n[Tokenizer] Training BPE tokenizer...")
    print(f"[Tokenizer] Input: {input_path}")
    print(f"[Tokenizer] Target vocab size: {vocab_size}")

    tokenizer = ByteBPETokenizer()

    # Stream lines from file
    def line_iterator():
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    tokenizer.train(line_iterator(), vocab_size=vocab_size, min_freq=min_freq, verbose=True)

    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(output_dir)

    return tokenizer


def encode_and_save(
    input_path: str,
    output_dir: str,
    tokenizer: ByteBPETokenizer,
    val_split: float = 0.1
) -> dict:
    """
    Encode text file to binary token files.

    Args:
        input_path: Path to text file
        output_dir: Output directory for token files
        tokenizer: Trained tokenizer
        val_split: Validation split fraction

    Returns:
        Metadata dictionary
    """
    print(f"\n[Encode] Encoding to binary tokens...")

    metadata = prepare_dataset(
        input_path=input_path,
        output_dir=output_dir,
        tokenizer=tokenizer,
        val_split=val_split,
        add_eos=True,
        verbose=True
    )

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for TinyGPT training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data/raw/input.txt",
        help="Path to raw text file"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="data",
        help="Base output directory"
    )
    parser.add_argument(
        "--vocab_size", "-v",
        type=int,
        default=4096,
        help="Target vocabulary size for BPE"
    )
    parser.add_argument(
        "--min_freq",
        type=int,
        default=2,
        help="Minimum frequency for BPE merges"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split fraction"
    )
    parser.add_argument(
        "--max_lines",
        type=int,
        default=None,
        help="Maximum lines to process (for testing)"
    )
    parser.add_argument(
        "--skip_clean",
        action="store_true",
        help="Skip text cleaning step"
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        print(f"Run ./scripts/download_data.sh first to download training data.")
        sys.exit(1)

    print("=" * 60)
    print("TinyGPT Data Preparation")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Val split: {args.val_split}")
    if args.max_lines:
        print(f"Max lines: {args.max_lines}")
    print("=" * 60)

    # Paths
    clean_path = os.path.join(args.output_dir, "clean", "input.txt")
    tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
    tokens_dir = os.path.join(args.output_dir, "tokens")

    # Step 1: Clean text
    if not args.skip_clean:
        print("\n" + "=" * 40)
        print("Step 1: Cleaning Text")
        print("=" * 40)
        clean_file(args.input, clean_path, max_lines=args.max_lines)
        input_for_tokenizer = clean_path
    else:
        print("\n[Skip] Skipping text cleaning")
        input_for_tokenizer = args.input

    # Step 2: Train tokenizer
    print("\n" + "=" * 40)
    print("Step 2: Training Tokenizer")
    print("=" * 40)
    tokenizer = train_tokenizer(
        input_for_tokenizer,
        tokenizer_dir,
        vocab_size=args.vocab_size,
        min_freq=args.min_freq
    )

    # Step 3: Encode to tokens
    print("\n" + "=" * 40)
    print("Step 3: Encoding to Binary Tokens")
    print("=" * 40)
    metadata = encode_and_save(
        input_for_tokenizer,
        tokens_dir,
        tokenizer,
        val_split=args.val_split
    )

    # Step 4: Verify
    print("\n" + "=" * 40)
    print("Step 4: Verification")
    print("=" * 40)

    train_path = os.path.join(tokens_dir, "train.bin")
    val_path = os.path.join(tokens_dir, "val.bin")

    if os.path.exists(train_path) and os.path.getsize(train_path) > 0:
        train_ds = TokenDataset(train_path, block_size=128)
        print(f"[Verify] Train dataset: {train_ds.num_tokens:,} tokens, {len(train_ds):,} samples")

    if os.path.exists(val_path) and os.path.getsize(val_path) > 0:
        val_ds = TokenDataset(val_path, block_size=128)
        print(f"[Verify] Val dataset: {val_ds.num_tokens:,} tokens, {len(val_ds):,} samples")
    else:
        print(f"[Verify] Val dataset: (empty or too small)")

    # Quick encode/decode test
    test_text = "Hello, world!"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n[Verify] Quick test:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  '{decoded}'")
    print(f"  Match:    {test_text == decoded}")

    # Summary
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Tokenizer: {tokenizer_dir}/")
    print(f"  Train data: {train_path}")
    print(f"  Val data: {val_path}")
    print(f"\nTokenizer info:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Num merges: {len(tokenizer.merges)}")
    print(f"  Vocab hash: {tokenizer.get_vocab_hash()}")
    print(f"\nNext step: Train the model with:")
    print(f"  python -m src.train --preset toy --max_steps 500")
    print("=" * 60)


if __name__ == "__main__":
    main()
