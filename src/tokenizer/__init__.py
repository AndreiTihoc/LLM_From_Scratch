# TinyGPT Tokenizer Module
"""
Byte-level BPE tokenizer implementation.

Classes:
    ByteBPETokenizer: Main tokenizer class for encoding/decoding text.

Usage:
    from src.tokenizer import ByteBPETokenizer
    tokenizer = ByteBPETokenizer.load("data/tokenizer")
    tokens = tokenizer.encode("Hello, world!")
    text = tokenizer.decode(tokens)
"""

from .byte_bpe import ByteBPETokenizer

__all__ = ["ByteBPETokenizer"]
