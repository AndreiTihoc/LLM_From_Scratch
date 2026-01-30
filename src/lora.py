"""
LoRA (Low-Rank Adaptation) Module for QLoRA Fine-Tuning
=======================================================

Implements LoRA adapters for efficient fine-tuning with optional 4-bit quantization.

QLoRA = Quantized base model (4-bit) + LoRA adapters (trainable)

Key features:
- LoRA adapters for attention layers (Q, K, V, O projections)
- Optional 4-bit quantization via bitsandbytes
- Efficient memory usage for fine-tuning on consumer GPUs
- Easy save/load of LoRA weights only

Usage:
    from src.lora import apply_lora, LoRAConfig, save_lora_weights, load_lora_weights

    # Apply LoRA to model
    config = LoRAConfig(r=16, alpha=32, dropout=0.05)
    model = apply_lora(model, config)

    # Train only LoRA parameters
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

References:
- LoRA: https://arxiv.org/abs/2106.09685
- QLoRA: https://arxiv.org/abs/2305.14314
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import bitsandbytes for 4-bit quantization
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


@dataclass
class LoRAConfig:
    """
    LoRA Configuration.

    Attributes:
        r: Rank of the LoRA matrices (lower = fewer params, higher = more capacity)
        alpha: LoRA scaling factor (scaling = alpha / r)
        dropout: Dropout probability for LoRA layers
        target_modules: Which modules to apply LoRA to
        use_4bit: Whether to quantize base model to 4-bit (requires bitsandbytes)
        bnb_4bit_compute_dtype: Compute dtype for 4-bit operations
        bnb_4bit_quant_type: Quantization type ('nf4' or 'fp4')
    """
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = None  # Default: ['c_attn', 'c_proj']
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"  # or "bfloat16"
    bnb_4bit_quant_type: str = "nf4"  # NormalFloat4, best for QLoRA

    def __post_init__(self):
        if self.target_modules is None:
            # Default: apply LoRA to attention QKV and output projections
            self.target_modules = ['c_attn', 'c_proj']

        if self.use_4bit and not HAS_BNB:
            print("[LoRA] Warning: bitsandbytes not available, disabling 4-bit quantization")
            self.use_4bit = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LoRAConfig":
        return cls(**d)


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adapters.

    Computes: y = W_base @ x + (lora_B @ lora_A) @ x * scaling

    The base weight W_base is frozen, only lora_A and lora_B are trained.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05
    ):
        super().__init__()

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # LoRA matrices
        # A: (in_features, r) - initialized with Kaiming uniform
        # B: (r, out_features) - initialized with zeros
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming uniform, B with zeros
        # This ensures the initial output is just the base layer output
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer forward (frozen)
        base_out = self.base_layer(x)

        # LoRA forward
        # x @ A @ B * scaling
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> None:
        """Merge LoRA weights into base layer (for inference)."""
        with torch.no_grad():
            # W_merged = W_base + (A @ B) * scaling
            delta = (self.lora_A @ self.lora_B * self.scaling).T
            self.base_layer.weight.add_(delta)

    def unmerge_weights(self) -> None:
        """Unmerge LoRA weights from base layer."""
        with torch.no_grad():
            delta = (self.lora_A @ self.lora_B * self.scaling).T
            self.base_layer.weight.sub_(delta)


class LoRALinear4bit(nn.Module):
    """
    4-bit quantized linear layer with LoRA adapters.

    Uses bitsandbytes for 4-bit NormalFloat quantization of the base weights.
    Only LoRA adapters (lora_A, lora_B) are trainable.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        compute_dtype: torch.dtype = torch.float16,
        quant_type: str = "nf4"
    ):
        super().__init__()

        if not HAS_BNB:
            raise ImportError("bitsandbytes is required for 4-bit quantization. "
                            "Install with: pip install bitsandbytes")

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Create 4-bit quantized base layer
        self.base_layer = bnb.nn.Linear4bit(
            in_features,
            out_features,
            bias=base_layer.bias is not None,
            compute_dtype=compute_dtype,
            quant_type=quant_type,
        )

        # Copy weights to 4-bit layer
        self.base_layer.weight = bnb.nn.Params4bit(
            base_layer.weight.data,
            requires_grad=False,
            quant_type=quant_type,
        )
        if base_layer.bias is not None:
            self.base_layer.bias = nn.Parameter(base_layer.bias.data, requires_grad=False)

        # LoRA matrices (full precision)
        self.lora_A = nn.Parameter(torch.zeros(in_features, r, dtype=compute_dtype))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features, dtype=compute_dtype))

        # Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer forward (4-bit quantized)
        base_out = self.base_layer(x)

        # LoRA forward (full precision)
        x_dtype = x.dtype
        lora_out = self.lora_dropout(x.to(self.lora_A.dtype)) @ self.lora_A @ self.lora_B * self.scaling

        return base_out + lora_out.to(x_dtype)


def apply_lora(
    model: nn.Module,
    config: LoRAConfig,
    verbose: bool = True
) -> nn.Module:
    """
    Apply LoRA adapters to a model.

    Args:
        model: The model to modify (GPT)
        config: LoRA configuration
        verbose: Print info about modified layers

    Returns:
        Modified model with LoRA layers
    """
    compute_dtype = torch.float16
    if config.bnb_4bit_compute_dtype == "bfloat16":
        compute_dtype = torch.bfloat16

    modified_count = 0

    # Find and replace target modules
    for name, module in list(model.named_modules()):
        if any(target in name for target in config.target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # Create LoRA layer
                if config.use_4bit:
                    lora_layer = LoRALinear4bit(
                        module,
                        r=config.r,
                        alpha=config.alpha,
                        dropout=config.dropout,
                        compute_dtype=compute_dtype,
                        quant_type=config.bnb_4bit_quant_type
                    )
                else:
                    lora_layer = LoRALinear(
                        module,
                        r=config.r,
                        alpha=config.alpha,
                        dropout=config.dropout
                    )

                # Replace module
                setattr(parent, child_name, lora_layer)
                modified_count += 1

                if verbose:
                    print(f"[LoRA] Applied to: {name}")

    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[LoRA] Modified {modified_count} layers")
        print(f"[LoRA] Total params: {total_params:,}")
        print(f"[LoRA] Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    return model


def freeze_base_model(model: nn.Module) -> None:
    """Freeze all parameters except LoRA adapters."""
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from model."""
    lora_state = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state[name] = param.data.clone()
    return lora_state


def save_lora_weights(
    model: nn.Module,
    lora_config: LoRAConfig,
    model_config: Any,
    path: str
) -> None:
    """
    Save LoRA weights and config to file.

    Args:
        model: Model with LoRA layers
        lora_config: LoRA configuration
        model_config: Base model configuration
        path: Output path
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    checkpoint = {
        'lora_state_dict': get_lora_state_dict(model),
        'lora_config': lora_config.to_dict(),
        'model_config': model_config.to_dict() if hasattr(model_config, 'to_dict') else model_config,
    }

    torch.save(checkpoint, path)
    print(f"[LoRA] Saved weights to {path}")


def load_lora_weights(
    model: nn.Module,
    path: str,
    strict: bool = True
) -> Tuple[LoRAConfig, Dict]:
    """
    Load LoRA weights into model.

    Args:
        model: Model with LoRA layers already applied
        path: Path to LoRA checkpoint
        strict: Whether to require all keys to match

    Returns:
        Tuple of (lora_config, model_config)
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Load LoRA state dict
    lora_state = checkpoint['lora_state_dict']
    model_state = model.state_dict()

    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
        elif strict:
            raise KeyError(f"LoRA parameter not found in model: {name}")

    lora_config = LoRAConfig.from_dict(checkpoint['lora_config'])
    model_config = checkpoint.get('model_config', {})

    print(f"[LoRA] Loaded weights from {path}")
    return lora_config, model_config


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into base model for faster inference.

    Note: This modifies the model in-place and removes LoRA layers.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            module.merge_weights()

            # Replace with base layer
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, module.base_layer)

    print("[LoRA] Merged weights into base model")
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count trainable and total parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for n, p in model.named_parameters() if 'lora_' in n)

    return {
        'total': total,
        'trainable': trainable,
        'lora': lora,
        'frozen': total - trainable,
    }


# ============================================
# Example Usage / Self-Test
# ============================================

if __name__ == "__main__":
    print("=== Testing LoRA Module ===\n")

    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.c_attn = nn.Linear(256, 768)  # QKV projection
            self.c_proj = nn.Linear(256, 256)  # Output projection
            self.mlp = nn.Linear(256, 1024)    # Not targeted by LoRA

        def forward(self, x):
            x = self.c_attn(x)
            x = x[:, :256]  # Simplified
            x = self.c_proj(x)
            return x

    model = SimpleModel()
    print(f"Original model params: {sum(p.numel() for p in model.parameters()):,}")

    # Apply LoRA
    config = LoRAConfig(r=8, alpha=16, dropout=0.0, use_4bit=False)
    model = apply_lora(model, config)

    # Freeze base model
    freeze_base_model(model)

    # Count parameters
    counts = count_parameters(model)
    print(f"\nParameter counts:")
    print(f"  Total: {counts['total']:,}")
    print(f"  Trainable (LoRA): {counts['trainable']:,}")
    print(f"  Frozen: {counts['frozen']:,}")

    # Test forward pass
    x = torch.randn(2, 10, 256)
    y = model(x)
    print(f"\nForward pass: input {x.shape} -> output {y.shape}")

    # Test save/load
    print("\nTesting save/load...")
    save_lora_weights(model, config, {'test': True}, '/tmp/test_lora.pt')

    # Create fresh model and load
    model2 = SimpleModel()
    model2 = apply_lora(model2, config)
    load_lora_weights(model2, '/tmp/test_lora.pt')

    print("\n[OK] LoRA module working correctly!")
