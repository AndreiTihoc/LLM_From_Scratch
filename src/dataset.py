"""
Dataset Module
==============
Memory-efficient dataset handling using numpy memmap for token storage.

Key features:
- Binary memmap storage: Loads only needed chunks, not entire dataset
- Streaming tokenization: Process large files without loading into RAM
- Simple random sampling: Get random chunks for training batches

Usage:
    # Preparing data
    from src.dataset import TokenDatasetWriter
    writer = TokenDatasetWriter("data/tokens/train.bin")
    writer.add_tokens([1, 2, 3, 4, 5])
    writer.close()

    # Loading data for training
    from src.dataset import TokenDataset
    dataset = TokenDataset("data/tokens/train.bin", block_size=128)
    x, y = dataset[0]  # Get a sample
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional, Iterator
from pathlib import Path


# Use uint16 for token IDs (supports vocab up to 65535)
# Use uint32 if you need larger vocab
TOKEN_DTYPE = np.uint16
TOKEN_DTYPE_MAX = 65535


class TokenDatasetWriter:
    """
    Write tokenized data to binary memmap file.

    The file format is simple: raw array of token IDs (uint16).
    Metadata is stored in a separate .json file.
    """

    def __init__(self, output_path: str, dtype=TOKEN_DTYPE):
        """
        Initialize writer.

        Args:
            output_path: Path to output .bin file
            dtype: NumPy dtype for tokens (default: uint16)
        """
        self.output_path = output_path
        self.dtype = dtype
        self.tokens_written = 0

        # Create output directory
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Open file in binary append mode
        self.file = open(output_path, "wb")

    def add_tokens(self, tokens: List[int]) -> None:
        """
        Add tokens to the dataset.

        Args:
            tokens: List of token IDs
        """
        if not tokens:
            return

        # Convert to numpy array and write
        arr = np.array(tokens, dtype=self.dtype)
        arr.tofile(self.file)
        self.tokens_written += len(tokens)

    def add_tokens_batch(self, token_batches: Iterator[List[int]]) -> None:
        """
        Add multiple token sequences.

        Args:
            token_batches: Iterator of token lists
        """
        for tokens in token_batches:
            self.add_tokens(tokens)

    def close(self) -> dict:
        """
        Close the writer and save metadata.

        Returns:
            Metadata dictionary
        """
        self.file.close()

        # Calculate file size
        file_size = os.path.getsize(self.output_path)
        expected_size = self.tokens_written * np.dtype(self.dtype).itemsize

        # Save metadata
        metadata = {
            "num_tokens": self.tokens_written,
            "dtype": str(self.dtype),
            "file_size_bytes": file_size,
        }

        meta_path = self.output_path.replace(".bin", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Dataset] Wrote {self.tokens_written:,} tokens to {self.output_path}")
        print(f"[Dataset] File size: {file_size / 1024 / 1024:.2f} MB")

        return metadata


class TokenDataset:
    """
    Memory-mapped token dataset for efficient training.

    Uses numpy memmap to load data on-demand without loading entire
    dataset into RAM. Perfect for limited memory environments.
    """

    def __init__(
        self,
        path: str,
        block_size: int,
        dtype=TOKEN_DTYPE
    ):
        """
        Initialize dataset.

        Args:
            path: Path to .bin file containing tokens
            block_size: Sequence length for training
            dtype: NumPy dtype of stored tokens
        """
        self.path = path
        self.block_size = block_size
        self.dtype = dtype

        # Memory-map the file
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.num_tokens = len(self.data)

        # Calculate number of valid starting positions
        # We need block_size + 1 tokens (input + 1 for target shift)
        self.num_samples = max(0, self.num_tokens - block_size)

        print(f"[Dataset] Loaded {self.path}")
        print(f"[Dataset] Total tokens: {self.num_tokens:,}")
        print(f"[Dataset] Block size: {block_size}")
        print(f"[Dataset] Valid samples: {self.num_samples:,}")

    def __len__(self) -> int:
        """Return number of valid samples."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a training sample.

        Args:
            idx: Sample index (starting position in token array)

        Returns:
            (x, y) where x is input tokens and y is target tokens
            Both have shape (block_size,)
        """
        # Get block_size + 1 tokens starting at idx
        chunk = self.data[idx : idx + self.block_size + 1]

        # Input is first block_size tokens
        x = chunk[:-1].astype(np.int64)

        # Target is shifted by 1 (next token prediction)
        y = chunk[1:].astype(np.int64)

        return x, y

    def get_random_batch(
        self,
        batch_size: int,
        device: str = "cpu"
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Get a random batch of samples.

        Args:
            batch_size: Number of samples in batch
            device: PyTorch device to place tensors on

        Returns:
            (x, y) tensors of shape (batch_size, block_size)
        """
        import torch

        # Random starting positions
        indices = np.random.randint(0, self.num_samples, size=batch_size)

        # Gather samples
        x_list = []
        y_list = []
        for idx in indices:
            x, y = self[idx]
            x_list.append(x)
            y_list.append(y)

        # Stack into tensors
        x = torch.tensor(np.stack(x_list), dtype=torch.long, device=device)
        y = torch.tensor(np.stack(y_list), dtype=torch.long, device=device)

        return x, y


def prepare_dataset(
    input_path: str,
    output_dir: str,
    tokenizer,
    val_split: float = 0.1,
    add_eos: bool = True,
    max_lines: int = None,
    verbose: bool = True
) -> dict:
    """
    Prepare dataset: tokenize text and write to binary files.

    Args:
        input_path: Path to input text file
        output_dir: Output directory for train.bin and val.bin
        tokenizer: Tokenizer instance with encode() method
        val_split: Fraction of data for validation
        add_eos: Add EOS token after each line
        max_lines: Maximum lines to process (for testing)
        verbose: Print progress

    Returns:
        Metadata dictionary
    """
    from tqdm import tqdm

    os.makedirs(output_dir, exist_ok=True)

    # Count lines first for progress bar
    if verbose:
        print("[Dataset] Counting lines...")
    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        total_lines = sum(1 for _ in f)
    if max_lines:
        total_lines = min(total_lines, max_lines)

    # Calculate split point
    val_lines = int(total_lines * val_split)
    train_lines = total_lines - val_lines

    if verbose:
        print(f"[Dataset] Total lines: {total_lines:,}")
        print(f"[Dataset] Train lines: {train_lines:,}")
        print(f"[Dataset] Val lines:   {val_lines:,}")

    # Create writers
    train_writer = TokenDatasetWriter(os.path.join(output_dir, "train.bin"))
    val_writer = TokenDatasetWriter(os.path.join(output_dir, "val.bin"))

    # Process file
    line_count = 0
    train_tokens = 0
    val_tokens = 0

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        pbar = tqdm(f, total=total_lines, desc="Tokenizing", disable=not verbose)
        for line in pbar:
            if max_lines and line_count >= max_lines:
                break

            line = line.strip()
            if not line:
                continue

            # Encode line
            tokens = tokenizer.encode(line, add_eos=add_eos)

            # Write to appropriate split
            if line_count < train_lines:
                train_writer.add_tokens(tokens)
                train_tokens += len(tokens)
            else:
                val_writer.add_tokens(tokens)
                val_tokens += len(tokens)

            line_count += 1

            if line_count % 10000 == 0:
                pbar.set_postfix({
                    "train_tok": f"{train_tokens:,}",
                    "val_tok": f"{val_tokens:,}"
                })

    # Close writers
    train_meta = train_writer.close()
    val_meta = val_writer.close()

    # Save combined metadata
    metadata = {
        "input_file": input_path,
        "vocab_size": tokenizer.vocab_size,
        "vocab_hash": tokenizer.get_vocab_hash(),
        "total_lines": line_count,
        "train": train_meta,
        "val": val_meta,
    }

    meta_path = os.path.join(output_dir, "dataset_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"\n[Dataset] Preparation complete!")
        print(f"[Dataset] Train tokens: {train_tokens:,}")
        print(f"[Dataset] Val tokens:   {val_tokens:,}")
        print(f"[Dataset] Saved to:     {output_dir}/")

    return metadata


# ============================================
# PyTorch DataLoader Helper
# ============================================

def create_dataloader(
    data_path: str,
    block_size: int,
    batch_size: int,
    device: str = "cpu"
):
    """
    Create a simple data iterator (not a full DataLoader).

    For simplicity, we use random sampling each iteration rather
    than a proper shuffled epoch-based loader.

    Args:
        data_path: Path to .bin file
        block_size: Sequence length
        batch_size: Batch size
        device: Target device

    Returns:
        Iterator function that yields (x, y) batches
    """
    dataset = TokenDataset(data_path, block_size)

    def get_batch():
        return dataset.get_random_batch(batch_size, device)

    return get_batch, dataset


# ============================================
# Example Usage / Self-Test
# ============================================

if __name__ == "__main__":
    import tempfile

    print("=== Testing Dataset Module ===\n")

    # Create dummy tokenizer for testing
    class DummyTokenizer:
        vocab_size = 1000
        def encode(self, text, add_eos=False):
            tokens = [ord(c) % 256 for c in text]
            if add_eos:
                tokens.append(259)  # EOS token
            return tokens
        def get_vocab_hash(self):
            return "dummy123"

    tokenizer = DummyTokenizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            for i in range(1000):
                f.write(f"This is test line number {i}\n")

        # Prepare dataset
        print("--- Preparing Dataset ---")
        output_dir = os.path.join(tmpdir, "tokens")
        metadata = prepare_dataset(
            test_file,
            output_dir,
            tokenizer,
            val_split=0.1,
            verbose=True
        )

        # Test loading
        print("\n--- Testing Dataset Loading ---")
        train_path = os.path.join(output_dir, "train.bin")
        dataset = TokenDataset(train_path, block_size=32)

        print(f"Dataset length: {len(dataset)}")

        # Get a sample
        x, y = dataset[0]
        print(f"Sample x shape: {x.shape}")
        print(f"Sample y shape: {y.shape}")
        print(f"Sample x[:10]: {x[:10]}")
        print(f"Sample y[:10]: {y[:10]}")

        # Test batch
        print("\n--- Testing Batch Loading ---")
        import torch
        x_batch, y_batch = dataset.get_random_batch(4, "cpu")
        print(f"Batch x shape: {x_batch.shape}")
        print(f"Batch y shape: {y_batch.shape}")

    print("\n[OK] Dataset module working correctly!")
