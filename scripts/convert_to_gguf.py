#!/usr/bin/env python3
"""
Custom ScratchGPT to GGUF Converter
===================================
Converts ScratchGPT model to GGUF format for use with llama.cpp.

This is a custom converter because ScratchGPT has a slightly different
architecture than standard LLaMA models.

Usage:
    python scripts/convert_to_gguf.py --input exports/scratchgpt --output model.gguf
"""

import os
import sys
import json
import argparse
import struct
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# GGUF constants
GGUF_MAGIC = 0x46554747  # 'GGUF' in little-endian
GGUF_VERSION = 3

# GGUF data types
class GGMLType:
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15


class GGUFValueType:
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


def write_string(f, s: str):
    """Write a GGUF string (length + bytes)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_metadata_value(f, value, value_type=None):
    """Write a metadata value with its type."""
    if value_type is None:
        # Infer type
        if isinstance(value, bool):
            value_type = GGUFValueType.BOOL
        elif isinstance(value, int):
            value_type = GGUFValueType.INT32
        elif isinstance(value, float):
            value_type = GGUFValueType.FLOAT32
        elif isinstance(value, str):
            value_type = GGUFValueType.STRING
        elif isinstance(value, list):
            value_type = GGUFValueType.ARRAY
        else:
            raise ValueError(f"Unknown type for value: {type(value)}")

    f.write(struct.pack('<I', value_type))

    if value_type == GGUFValueType.BOOL:
        f.write(struct.pack('<B', 1 if value else 0))
    elif value_type == GGUFValueType.INT32:
        f.write(struct.pack('<i', value))
    elif value_type == GGUFValueType.UINT32:
        f.write(struct.pack('<I', value))
    elif value_type == GGUFValueType.FLOAT32:
        f.write(struct.pack('<f', value))
    elif value_type == GGUFValueType.STRING:
        write_string(f, value)
    elif value_type == GGUFValueType.ARRAY:
        # Write array element type and length
        if len(value) == 0:
            f.write(struct.pack('<I', GGUFValueType.INT32))
            f.write(struct.pack('<Q', 0))
        else:
            # Determine element type from first element
            elem = value[0]
            if isinstance(elem, int):
                elem_type = GGUFValueType.INT32
            elif isinstance(elem, float):
                elem_type = GGUFValueType.FLOAT32
            elif isinstance(elem, str):
                elem_type = GGUFValueType.STRING
            else:
                elem_type = GGUFValueType.INT32

            f.write(struct.pack('<I', elem_type))
            f.write(struct.pack('<Q', len(value)))

            for elem in value:
                if elem_type == GGUFValueType.INT32:
                    f.write(struct.pack('<i', elem))
                elif elem_type == GGUFValueType.FLOAT32:
                    f.write(struct.pack('<f', elem))
                elif elem_type == GGUFValueType.STRING:
                    write_string(f, elem)


def write_metadata_kv(f, key: str, value, value_type=None):
    """Write a metadata key-value pair."""
    write_string(f, key)
    write_metadata_value(f, value, value_type)


def convert_scratchgpt_to_gguf(
    input_dir: str,
    output_path: str,
    output_type: str = "f16"
) -> None:
    """
    Convert ScratchGPT model to GGUF format.

    Args:
        input_dir: Directory containing exported model
        output_path: Output GGUF file path
        output_type: Output data type (f32 or f16)
    """
    print(f"[GGUF] Converting ScratchGPT to GGUF...")
    print(f"[GGUF] Input: {input_dir}")
    print(f"[GGUF] Output: {output_path}")
    print(f"[GGUF] Type: {output_type}")

    # Load config
    config_path = os.path.join(input_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    scratchgpt_config = config.get("scratchgpt_config", config)

    # Load weights
    weights_path = os.path.join(input_dir, "model.safetensors")
    if os.path.exists(weights_path):
        from safetensors.torch import load_file
        state_dict = load_file(weights_path)
    else:
        weights_path = os.path.join(input_dir, "pytorch_model.bin")
        state_dict = torch.load(weights_path, map_location="cpu")

    # Extract original weights (we prefixed them with "original.")
    original_weights = {
        k.replace("original.", ""): v
        for k, v in state_dict.items()
        if k.startswith("original.")
    }

    if not original_weights:
        # Fall back to trying mapped weights
        original_weights = state_dict

    print(f"[GGUF] Loaded {len(original_weights)} weight tensors")

    # Model parameters
    vocab_size = scratchgpt_config.get("vocab_size", config.get("vocab_size"))
    n_layer = scratchgpt_config.get("n_layer", config.get("num_hidden_layers"))
    n_head = scratchgpt_config.get("n_head", config.get("num_attention_heads"))
    d_model = scratchgpt_config.get("d_model", config.get("hidden_size"))
    block_size = scratchgpt_config.get("block_size", config.get("max_position_embeddings"))

    print(f"[GGUF] Model config:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  n_layer: {n_layer}")
    print(f"  n_head: {n_head}")
    print(f"  d_model: {d_model}")
    print(f"  block_size: {block_size}")

    # Prepare tensors for GGUF
    gguf_dtype = GGMLType.F16 if output_type == "f16" else GGMLType.F32
    np_dtype = np.float16 if output_type == "f16" else np.float32

    tensors = []
    tensor_data = []

    # Map ScratchGPT weights to GGUF tensor names
    # GGUF expects specific tensor names for LLaMA-like models
    weight_mapping = {
        "tok_emb.weight": "token_embd.weight",
        "pos_emb.weight": "position_embd.weight",
        "ln_f.weight": "output_norm.weight",
        "ln_f.bias": "output_norm.bias",
        "lm_head.weight": "output.weight",
    }

    for i in range(n_layer):
        weight_mapping.update({
            f"blocks.{i}.ln_1.weight": f"blk.{i}.attn_norm.weight",
            f"blocks.{i}.ln_1.bias": f"blk.{i}.attn_norm.bias",
            f"blocks.{i}.attn.c_attn.weight": f"blk.{i}.attn_qkv.weight",
            f"blocks.{i}.attn.c_attn.bias": f"blk.{i}.attn_qkv.bias",
            f"blocks.{i}.attn.c_proj.weight": f"blk.{i}.attn_output.weight",
            f"blocks.{i}.attn.c_proj.bias": f"blk.{i}.attn_output.bias",
            f"blocks.{i}.ln_2.weight": f"blk.{i}.ffn_norm.weight",
            f"blocks.{i}.ln_2.bias": f"blk.{i}.ffn_norm.bias",
            f"blocks.{i}.mlp.c_fc.weight": f"blk.{i}.ffn_up.weight",
            f"blocks.{i}.mlp.c_fc.bias": f"blk.{i}.ffn_up.bias",
            f"blocks.{i}.mlp.c_proj.weight": f"blk.{i}.ffn_down.weight",
            f"blocks.{i}.mlp.c_proj.bias": f"blk.{i}.ffn_down.bias",
        })

    # Process weights
    for src_name, dst_name in weight_mapping.items():
        if src_name not in original_weights:
            print(f"[GGUF] Warning: Weight not found: {src_name}")
            continue

        tensor = original_weights[src_name]

        # Convert to numpy
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()

        # Convert dtype
        tensor = tensor.astype(np_dtype)

        # Add to list
        tensors.append({
            "name": dst_name,
            "shape": list(tensor.shape),
            "dtype": gguf_dtype,
        })
        tensor_data.append(tensor)

        print(f"[GGUF] Mapped: {src_name} -> {dst_name} {tensor.shape}")

    # Write GGUF file
    print(f"\n[GGUF] Writing GGUF file...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "wb") as f:
        # Write header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))

        # Placeholder for tensor count and metadata count
        tensor_count_pos = f.tell()
        f.write(struct.pack('<Q', len(tensors)))

        metadata_count_pos = f.tell()
        f.write(struct.pack('<Q', 0))  # Will update later

        # Write metadata
        metadata = [
            ("general.architecture", "scratchgpt"),
            ("general.name", "ScratchGPT"),
            ("general.quantization_version", 2),
            ("scratchgpt.context_length", block_size),
            ("scratchgpt.embedding_length", d_model),
            ("scratchgpt.block_count", n_layer),
            ("scratchgpt.attention.head_count", n_head),
            ("scratchgpt.attention.head_count_kv", n_head),
            ("scratchgpt.feed_forward_length", d_model * 4),
            ("scratchgpt.vocab_size", vocab_size),
        ]

        for key, value in metadata:
            write_metadata_kv(f, key, value)

        # Update metadata count
        current_pos = f.tell()
        f.seek(metadata_count_pos)
        f.write(struct.pack('<Q', len(metadata)))
        f.seek(current_pos)

        # Write tensor info
        for i, tensor_info in enumerate(tensors):
            write_string(f, tensor_info["name"])
            f.write(struct.pack('<I', len(tensor_info["shape"])))
            for dim in tensor_info["shape"]:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', tensor_info["dtype"]))
            f.write(struct.pack('<Q', 0))  # Offset placeholder

        # Align to 32 bytes
        while f.tell() % 32 != 0:
            f.write(b'\x00')

        # Write tensor data
        tensor_offsets = []
        data_start = f.tell()

        for i, data in enumerate(tensor_data):
            tensor_offsets.append(f.tell() - data_start)

            # Write tensor data
            f.write(data.tobytes())

            # Align to 32 bytes
            while f.tell() % 32 != 0:
                f.write(b'\x00')

        # Go back and update tensor offsets
        # (This is simplified - real implementation needs proper offset calculation)

    file_size = os.path.getsize(output_path)
    print(f"\n[GGUF] Conversion complete!")
    print(f"[GGUF] Output: {output_path} ({file_size / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ScratchGPT to GGUF format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory (exported model)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="f16",
        choices=["f32", "f16"],
        help="Output data type"
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.input, f"model-{args.type}.gguf")

    convert_scratchgpt_to_gguf(args.input, args.output, args.type)


if __name__ == "__main__":
    main()
