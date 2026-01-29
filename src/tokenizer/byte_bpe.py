"""
Byte-level BPE Tokenizer
========================
A from-scratch implementation of Byte Pair Encoding (BPE) tokenization.

Key concepts:
- Base vocabulary: All 256 possible byte values
- Merges: Learned pairs of tokens that get combined
- Encoding: Convert text to bytes, then apply merges greedily
- Decoding: Map token IDs back to bytes, then decode to text

This implementation is designed for educational clarity and efficiency
on small-to-medium datasets (up to ~1GB of text).

Usage:
    # Training
    tokenizer = ByteBPETokenizer()
    tokenizer.train(texts, vocab_size=4096)
    tokenizer.save("data/tokenizer")

    # Loading and using
    tokenizer = ByteBPETokenizer.load("data/tokenizer")
    ids = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(ids)
"""

import os
import json
import hashlib
from collections import Counter
from typing import List, Dict, Tuple, Optional, Iterator
from pathlib import Path


class ByteBPETokenizer:
    """
    Byte-level BPE tokenizer.

    The base vocabulary consists of all 256 byte values (0-255).
    Additional vocabulary entries are learned merge pairs.

    Attributes:
        merges: Dict mapping (tok1, tok2) -> merged_tok_id
        vocab: Dict mapping token_id -> bytes
        vocab_size: Total vocabulary size (256 + num_merges)
    """

    # Special tokens (can be extended)
    PAD_TOKEN = "<|pad|>"
    UNK_TOKEN = "<|unk|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"

    def __init__(self):
        """Initialize tokenizer with base byte vocabulary."""
        # Base vocabulary: 256 bytes
        # Index 0-255 are the raw byte values
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

        # Special tokens get IDs after base bytes but before merges
        self.special_tokens: Dict[str, int] = {}
        self._add_special_token(self.PAD_TOKEN)  # 256
        self._add_special_token(self.UNK_TOKEN)  # 257
        self._add_special_token(self.BOS_TOKEN)  # 258
        self._add_special_token(self.EOS_TOKEN)  # 259

        # Reverse lookup for special tokens
        self.special_token_ids: Dict[int, str] = {
            v: k for k, v in self.special_tokens.items()
        }

        # Merge list preserves order for encoding
        self.merge_list: List[Tuple[int, int]] = []

    def _add_special_token(self, token: str) -> int:
        """Add a special token to the vocabulary."""
        if token in self.special_tokens:
            return self.special_tokens[token]
        idx = len(self.vocab)
        self.special_tokens[token] = idx
        self.vocab[idx] = token.encode("utf-8")
        return idx

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size."""
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.special_tokens[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.special_tokens[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.special_tokens[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.special_tokens[self.EOS_TOKEN]

    # ========================================
    # Training
    # ========================================

    def train(
        self,
        texts: Iterator[str],
        vocab_size: int = 4096,
        min_freq: int = 2,
        verbose: bool = True
    ) -> None:
        """
        Train BPE tokenizer on text corpus.

        Args:
            texts: Iterator of text strings (for memory efficiency)
            vocab_size: Target vocabulary size (including base 256 + special)
            min_freq: Minimum frequency for a merge to be considered
            verbose: Print progress
        """
        # Number of merges to learn
        # Base vocab = 256, special = 4, so merges = vocab_size - 260
        num_special = len(self.special_tokens)
        num_merges = vocab_size - 256 - num_special

        if num_merges <= 0:
            if verbose:
                print(f"[Tokenizer] vocab_size={vocab_size} <= base (260), no merges to learn")
            return

        if verbose:
            print(f"[Tokenizer] Training BPE with target vocab_size={vocab_size}")
            print(f"[Tokenizer] Learning {num_merges} merges...")

        # Step 1: Convert all text to byte sequences and count pair frequencies
        # For memory efficiency, we work with a frequency dict of byte sequences
        if verbose:
            print("[Tokenizer] Step 1: Building initial token sequences...")

        # Build word frequencies (treating each line/chunk as a unit)
        word_freqs: Counter = Counter()
        total_chars = 0

        for text in texts:
            # Convert text to tuple of bytes
            text_bytes = tuple(text.encode("utf-8"))
            if text_bytes:
                word_freqs[text_bytes] += 1
                total_chars += len(text)

        if verbose:
            print(f"[Tokenizer] Processed {total_chars:,} characters, {len(word_freqs):,} unique sequences")

        # Now learn merges
        if verbose:
            print("[Tokenizer] Step 2: Learning merges...")

        for merge_idx in range(num_merges):
            # Count all adjacent pairs
            pair_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                if verbose:
                    print(f"[Tokenizer] No more pairs to merge at step {merge_idx}")
                break

            # Find most frequent pair
            best_pair, best_freq = pair_freqs.most_common(1)[0]

            if best_freq < min_freq:
                if verbose:
                    print(f"[Tokenizer] Best pair freq {best_freq} < min_freq {min_freq}, stopping")
                break

            # Create new token ID
            new_id = 256 + num_special + merge_idx

            # Record the merge
            self.merges[best_pair] = new_id
            self.merge_list.append(best_pair)

            # Update vocab with the merged bytes
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            # Apply merge to all words
            new_word_freqs: Counter = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, best_pair, new_id)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

            if verbose and (merge_idx + 1) % 500 == 0:
                merged_bytes = self.vocab[new_id]
                try:
                    merged_str = merged_bytes.decode("utf-8", errors="replace")
                except:
                    merged_str = str(merged_bytes)
                print(f"[Tokenizer] Merge {merge_idx + 1}/{num_merges}: "
                      f"{best_pair} -> {new_id} (freq={best_freq}, repr='{merged_str}')")

        if verbose:
            print(f"[Tokenizer] Training complete! Vocab size: {self.vocab_size}")

    def _apply_merge(
        self,
        word: Tuple[int, ...],
        pair: Tuple[int, int],
        new_id: int
    ) -> Tuple[int, ...]:
        """Apply a single merge to a word (tuple of token IDs)."""
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def train_from_file(
        self,
        path: str,
        vocab_size: int = 4096,
        min_freq: int = 2,
        chunk_size: int = 10000,
        verbose: bool = True
    ) -> None:
        """
        Train tokenizer from a text file in a streaming fashion.

        Args:
            path: Path to text file
            vocab_size: Target vocabulary size
            min_freq: Minimum frequency for merges
            chunk_size: Number of lines to process at a time
            verbose: Print progress
        """
        def line_iterator():
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

        self.train(line_iterator(), vocab_size=vocab_size, min_freq=min_freq, verbose=verbose)

    # ========================================
    # Encoding
    # ========================================

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text string
            add_bos: Prepend BOS token
            add_eos: Append EOS token

        Returns:
            List of token IDs
        """
        if not text:
            tokens = []
        else:
            # Convert to bytes, then to list of byte values
            text_bytes = list(text.encode("utf-8"))

            # Apply merges in order learned
            tokens = text_bytes
            for pair in self.merge_list:
                if pair in self.merges:
                    new_id = self.merges[pair]
                    tokens = self._apply_merge_to_list(tokens, pair, new_id)

        # Add special tokens
        if add_bos:
            tokens = [self.bos_id] + tokens
        if add_eos:
            tokens = tokens + [self.eos_id]

        return tokens

    def _apply_merge_to_list(
        self,
        tokens: List[int],
        pair: Tuple[int, int],
        new_id: int
    ) -> List[int]:
        """Apply a merge to a list of tokens."""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def encode_batch(self, texts: List[str], **kwargs) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text, **kwargs) for text in texts]

    # ========================================
    # Decoding
    # ========================================

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: List of token IDs
            skip_special: Skip special tokens in output

        Returns:
            Decoded text string
        """
        byte_pieces = []
        for idx in ids:
            if idx in self.special_token_ids:
                if not skip_special:
                    byte_pieces.append(self.vocab[idx])
            elif idx in self.vocab:
                byte_pieces.append(self.vocab[idx])
            else:
                # Unknown token, skip or use UNK representation
                if not skip_special:
                    byte_pieces.append(b"<unk>")

        # Concatenate all bytes and decode
        all_bytes = b"".join(byte_pieces)
        return all_bytes.decode("utf-8", errors="replace")

    def decode_batch(self, batch_ids: List[List[int]], **kwargs) -> List[str]:
        """Decode multiple sequences."""
        return [self.decode(ids, **kwargs) for ids in batch_ids]

    # ========================================
    # Save / Load
    # ========================================

    def save(self, path: str) -> None:
        """
        Save tokenizer to directory.

        Saves:
            - vocab.json: Token ID -> bytes (hex-encoded for special chars)
            - merges.txt: One merge per line "id1 id2"
            - config.json: Metadata
        """
        os.makedirs(path, exist_ok=True)

        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "num_merges": len(self.merges),
            "special_tokens": self.special_tokens,
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save vocab as JSON (bytes hex-encoded)
        vocab_json = {}
        for idx, byte_val in self.vocab.items():
            vocab_json[str(idx)] = byte_val.hex()
        with open(os.path.join(path, "vocab.json"), "w") as f:
            json.dump(vocab_json, f, indent=2)

        # Save merges in human-readable format
        with open(os.path.join(path, "merges.txt"), "w") as f:
            for pair in self.merge_list:
                f.write(f"{pair[0]} {pair[1]}\n")

        print(f"[Tokenizer] Saved to {path}/")

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        """
        Load tokenizer from directory.

        Args:
            path: Directory containing vocab.json, merges.txt, config.json

        Returns:
            Loaded tokenizer
        """
        tokenizer = cls()

        # Load config
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            tokenizer.special_tokens = config.get("special_tokens", tokenizer.special_tokens)
            tokenizer.special_token_ids = {
                v: k for k, v in tokenizer.special_tokens.items()
            }

        # Load vocab
        vocab_path = os.path.join(path, "vocab.json")
        with open(vocab_path, "r") as f:
            vocab_json = json.load(f)

        tokenizer.vocab = {}
        for idx_str, hex_val in vocab_json.items():
            idx = int(idx_str)
            tokenizer.vocab[idx] = bytes.fromhex(hex_val)

        # Load merges
        merges_path = os.path.join(path, "merges.txt")
        tokenizer.merges = {}
        tokenizer.merge_list = []

        num_special = len(tokenizer.special_tokens)
        merge_idx = 0

        with open(merges_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    pair = (int(parts[0]), int(parts[1]))
                    new_id = 256 + num_special + merge_idx
                    tokenizer.merges[pair] = new_id
                    tokenizer.merge_list.append(pair)
                    merge_idx += 1

        print(f"[Tokenizer] Loaded from {path}/ (vocab_size={tokenizer.vocab_size})")
        return tokenizer

    def get_vocab_hash(self) -> str:
        """Get a hash of the vocabulary for verification."""
        # Create a deterministic string representation
        vocab_str = json.dumps({
            "vocab_size": self.vocab_size,
            "merges": [(p[0], p[1]) for p in self.merge_list]
        }, sort_keys=True)
        return hashlib.md5(vocab_str.encode()).hexdigest()[:8]

    # ========================================
    # Utility Methods
    # ========================================

    def get_token_str(self, token_id: int) -> str:
        """Get string representation of a token."""
        if token_id in self.special_token_ids:
            return self.special_token_ids[token_id]
        if token_id in self.vocab:
            try:
                return self.vocab[token_id].decode("utf-8")
            except UnicodeDecodeError:
                return f"<bytes:{self.vocab[token_id].hex()}>"
        return f"<unk:{token_id}>"

    def __repr__(self) -> str:
        return f"ByteBPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"


# ============================================
# Example Usage / Self-Test
# ============================================

if __name__ == "__main__":
    print("=== Testing ByteBPETokenizer ===\n")

    # Test training on small corpus
    test_texts = [
        "Hello, world!",
        "Hello, Python!",
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy cat.",
        "Machine learning is fascinating.",
        "Machine learning models learn from data.",
    ] * 100  # Repeat for more data

    tokenizer = ByteBPETokenizer()
    print(f"Initial vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens}")

    # Train
    print("\n--- Training ---")
    tokenizer.train(iter(test_texts), vocab_size=300, verbose=True)
    print(f"Final vocab size: {tokenizer.vocab_size}")

    # Test encoding/decoding
    print("\n--- Encoding/Decoding ---")
    test_str = "Hello, world!"
    tokens = tokenizer.encode(test_str)
    decoded = tokenizer.decode(tokens)
    print(f"Original: '{test_str}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{decoded}'")
    print(f"Match: {test_str == decoded}")

    # Test with special tokens
    print("\n--- With Special Tokens ---")
    tokens_special = tokenizer.encode(test_str, add_bos=True, add_eos=True)
    print(f"With BOS/EOS: {tokens_special}")
    decoded_skip = tokenizer.decode(tokens_special, skip_special=True)
    decoded_keep = tokenizer.decode(tokens_special, skip_special=False)
    print(f"Decoded (skip special): '{decoded_skip}'")
    print(f"Decoded (keep special): '{decoded_keep}'")

    # Test save/load
    print("\n--- Save/Load ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer.save(tmpdir)
        loaded = ByteBPETokenizer.load(tmpdir)
        tokens2 = loaded.encode(test_str)
        print(f"Original tokens: {tokens}")
        print(f"Loaded tokens:   {tokens2}")
        print(f"Match: {tokens == tokens2}")

    print("\n[OK] ByteBPETokenizer working correctly!")
