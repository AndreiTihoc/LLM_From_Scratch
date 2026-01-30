#!/usr/bin/env python3
"""
Export to Hugging Face-like Format
==================================
Export trained ScratchGPT model to a format compatible with llama.cpp conversion.

This creates:
- config.json: Model configuration in HF-like format
- tokenizer.json: Tokenizer in HF format
- tokenizer_config.json: Tokenizer configuration
- model.safetensors or pytorch_model.bin: Model weights

The exported model can then be converted to GGUF format using:
    python llama.cpp/convert_hf_to_gguf.py exports/my_model/

Usage:
    python -m src.export_hf --checkpoint checkpoints/best.pt --output exports/scratchgpt

Note: This export format is designed to be compatible with llama.cpp's
convert_hf_to_gguf.py script, but full compatibility may require the
custom conversion script provided in scripts/convert_to_gguf.py.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ModelConfig
from src.model import GPT
from src.tokenizer import ByteBPETokenizer


def export_model_weights(
    model: GPT,
    output_dir: str,
    use_safetensors: bool = True
) -> str:
    """
    Export model weights.

    Args:
        model: Trained GPT model
        output_dir: Output directory
        use_safetensors: Use safetensors format (recommended)

    Returns:
        Path to saved weights file
    """
    state_dict = model.state_dict()

    # Rename keys to HF-compatible format
    # Our naming: tok_emb, pos_emb, blocks.N.ln_1, blocks.N.attn, etc.
    # HF naming: model.embed_tokens, model.layers.N.self_attn, etc.

    # For llama.cpp compatibility, we need to match the expected format
    # We'll create a mapping for GPT-2-like architecture

    hf_state_dict = {}

    for key, value in state_dict.items():
        # Map our names to HF-compatible names
        new_key = key

        # Token embedding
        if key == "tok_emb.weight":
            new_key = "model.embed_tokens.weight"
        elif key == "pos_emb.weight":
            new_key = "model.embed_positions.weight"
        elif key == "ln_f.weight":
            new_key = "model.norm.weight"
        elif key == "ln_f.bias":
            new_key = "model.norm.bias"
        elif key == "lm_head.weight":
            new_key = "lm_head.weight"
        elif key.startswith("blocks."):
            # blocks.0.ln_1.weight -> model.layers.0.input_layernorm.weight
            parts = key.split(".")
            layer_idx = parts[1]
            rest = ".".join(parts[2:])

            if rest == "ln_1.weight":
                new_key = f"model.layers.{layer_idx}.input_layernorm.weight"
            elif rest == "ln_1.bias":
                new_key = f"model.layers.{layer_idx}.input_layernorm.bias"
            elif rest == "ln_2.weight":
                new_key = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
            elif rest == "ln_2.bias":
                new_key = f"model.layers.{layer_idx}.post_attention_layernorm.bias"
            elif rest == "attn.c_attn.weight":
                new_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
            elif rest == "attn.c_attn.bias":
                new_key = f"model.layers.{layer_idx}.self_attn.qkv_proj.bias"
            elif rest == "attn.c_proj.weight":
                new_key = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
            elif rest == "attn.c_proj.bias":
                new_key = f"model.layers.{layer_idx}.self_attn.o_proj.bias"
            elif rest == "mlp.c_fc.weight":
                new_key = f"model.layers.{layer_idx}.mlp.up_proj.weight"
            elif rest == "mlp.c_fc.bias":
                new_key = f"model.layers.{layer_idx}.mlp.up_proj.bias"
            elif rest == "mlp.c_proj.weight":
                new_key = f"model.layers.{layer_idx}.mlp.down_proj.weight"
            elif rest == "mlp.c_proj.bias":
                new_key = f"model.layers.{layer_idx}.mlp.down_proj.bias"

        hf_state_dict[new_key] = value

    # Also keep original format for custom conversion
    original_state_dict = {f"original.{k}": v for k, v in state_dict.items()}
    hf_state_dict.update(original_state_dict)

    if use_safetensors:
        try:
            from safetensors.torch import save_file
            output_path = os.path.join(output_dir, "model.safetensors")
            save_file(hf_state_dict, output_path)
            print(f"[Export] Saved weights to {output_path} (safetensors)")
        except ImportError:
            print("[Export] safetensors not available, falling back to PyTorch format")
            use_safetensors = False

    if not use_safetensors:
        output_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(hf_state_dict, output_path)
        print(f"[Export] Saved weights to {output_path} (PyTorch)")

    return output_path


def export_config(
    model_config: ModelConfig,
    output_dir: str
) -> str:
    """
    Export model configuration in HF-like format.

    Args:
        model_config: Model configuration
        output_dir: Output directory

    Returns:
        Path to config file
    """
    # Create HF-compatible config
    hf_config = {
        # Architecture
        "architectures": ["ScratchGPTForCausalLM"],
        "model_type": "scratchgpt",

        # Dimensions
        "vocab_size": model_config.vocab_size,
        "hidden_size": model_config.d_model,
        "intermediate_size": model_config.d_model * 4,
        "num_hidden_layers": model_config.n_layer,
        "num_attention_heads": model_config.n_head,
        "num_key_value_heads": model_config.n_head,  # No GQA
        "max_position_embeddings": model_config.block_size,

        # Architecture details
        "hidden_act": "gelu",
        "rms_norm_eps": 1e-5,
        "use_bias": model_config.bias,
        "tie_word_embeddings": True,

        # Generation
        "bos_token_id": 258,
        "eos_token_id": 259,
        "pad_token_id": 256,

        # ScratchGPT-specific (for custom converter)
        "scratchgpt_config": model_config.to_dict(),
    }

    output_path = os.path.join(output_dir, "config.json")
    with open(output_path, "w") as f:
        json.dump(hf_config, f, indent=2)

    print(f"[Export] Saved config to {output_path}")
    return output_path


def export_tokenizer(
    tokenizer: ByteBPETokenizer,
    output_dir: str
) -> None:
    """
    Export tokenizer in HF-like format.

    Args:
        tokenizer: Trained tokenizer
        output_dir: Output directory
    """
    # Create vocab.json (token -> id)
    vocab = {}
    for idx in range(tokenizer.vocab_size):
        if idx in tokenizer.special_token_ids:
            # Special token
            vocab[tokenizer.special_token_ids[idx]] = idx
        elif idx in tokenizer.vocab:
            # Try to decode as string
            try:
                token_str = tokenizer.vocab[idx].decode("utf-8")
                vocab[token_str] = idx
            except UnicodeDecodeError:
                # Use hex representation
                vocab[f"<0x{tokenizer.vocab[idx].hex().upper()}>"] = idx

    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[Export] Saved vocab to {vocab_path}")

    # Create merges.txt
    merges = []
    for pair in tokenizer.merge_list:
        # Get string representation of each token in the pair
        tok1 = tokenizer.vocab.get(pair[0], b"")
        tok2 = tokenizer.vocab.get(pair[1], b"")
        try:
            tok1_str = tok1.decode("utf-8")
            tok2_str = tok2.decode("utf-8")
            merges.append(f"{tok1_str} {tok2_str}")
        except UnicodeDecodeError:
            # Use hex for non-UTF8 bytes
            merges.append(f"<0x{tok1.hex().upper()}> <0x{tok2.hex().upper()}>")

    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        f.write("\n".join(merges))
    print(f"[Export] Saved merges to {merges_path}")

    # Create tokenizer.json (full tokenizer config)
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": tokenizer.pad_id, "content": "<|pad|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": tokenizer.unk_id, "content": "<|unk|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": tokenizer.bos_id, "content": "<|bos|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": tokenizer.eos_id, "content": "<|eos|>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": "<|unk|>",
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": True,
            "vocab": vocab,
            "merges": merges,
        }
    }

    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False, indent=2)
    print(f"[Export] Saved tokenizer.json to {tokenizer_json_path}")

    # Create tokenizer_config.json
    tokenizer_config = {
        "model_type": "scratchgpt",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "unk_token": "<|unk|>",
        "pad_token": "<|pad|>",
        "clean_up_tokenization_spaces": True,
        "model_max_length": 1024,
        "tokenizer_class": "PreTrainedTokenizerFast",
    }

    tokenizer_config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tokenizer_config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"[Export] Saved tokenizer_config.json to {tokenizer_config_path}")

    # Copy original tokenizer files for custom conversion
    original_dir = os.path.join(output_dir, "original_tokenizer")
    os.makedirs(original_dir, exist_ok=True)
    tokenizer.save(original_dir)
    print(f"[Export] Saved original tokenizer to {original_dir}/")


def export_model(
    checkpoint_path: str,
    tokenizer_path: str,
    output_dir: str,
    use_safetensors: bool = True
) -> None:
    """
    Export complete model for GGUF conversion.

    Args:
        checkpoint_path: Path to training checkpoint
        tokenizer_path: Path to tokenizer directory
        output_dir: Output directory for export
    """
    print("=" * 60)
    print("ScratchGPT Model Export")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    print(f"\n[Export] Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Build model config
    model_config_dict = checkpoint.get("model_config", {})
    model_config = ModelConfig(**model_config_dict)

    # Load tokenizer
    print(f"[Export] Loading tokenizer from {tokenizer_path}...")
    tokenizer = ByteBPETokenizer.load(tokenizer_path)

    # Update vocab size
    model_config.vocab_size = tokenizer.vocab_size

    # Create model and load weights
    print(f"[Export] Creating model...")
    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])

    # Export
    print(f"\n[Export] Exporting to {output_dir}...")

    export_config(model_config, output_dir)
    export_tokenizer(tokenizer, output_dir)
    export_model_weights(model, output_dir, use_safetensors)

    # Create generation config
    generation_config = {
        "max_length": model_config.block_size,
        "do_sample": True,
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "eos_token_id": tokenizer.eos_id,
        "bos_token_id": tokenizer.bos_id,
        "pad_token_id": tokenizer.pad_id,
    }
    gen_config_path = os.path.join(output_dir, "generation_config.json")
    with open(gen_config_path, "w") as f:
        json.dump(generation_config, f, indent=2)
    print(f"[Export] Saved generation_config.json")

    # Create export info
    export_info = {
        "source_checkpoint": checkpoint_path,
        "source_tokenizer": tokenizer_path,
        "model_config": model_config.to_dict(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_hash": tokenizer.get_vocab_hash(),
        "export_format": "safetensors" if use_safetensors else "pytorch",
        "notes": "Use scripts/convert_to_gguf.py for GGUF conversion",
    }
    info_path = os.path.join(output_dir, "export_info.json")
    with open(info_path, "w") as f:
        json.dump(export_info, f, indent=2)

    print("\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nExported files in {output_dir}/:")
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                print(f"  {f} ({size / 1024 / 1024:.2f} MB)")
            else:
                print(f"  {f} ({size / 1024:.2f} KB)")
        else:
            print(f"  {f}/ (directory)")

    print(f"\nNext steps:")
    print(f"  1. Convert to GGUF: ./scripts/gguf_convert.sh {output_dir}")
    print(f"  2. Quantize: ./scripts/gguf_quantize.sh {output_dir}/model.gguf")
    print(f"  3. Run: ./scripts/run_llamacpp.sh {output_dir}/model-q4_k_m.gguf")


def main():
    parser = argparse.ArgumentParser(
        description="Export ScratchGPT model for GGUF conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

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
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="exports/scratchgpt",
        help="Output directory"
    )
    parser.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Use PyTorch format instead of safetensors"
    )

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.tokenizer):
        print(f"Error: Tokenizer not found: {args.tokenizer}")
        sys.exit(1)

    export_model(
        args.checkpoint,
        args.tokenizer,
        args.output,
        use_safetensors=not args.no_safetensors
    )


if __name__ == "__main__":
    main()
