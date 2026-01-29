"""
TinyGPT Configuration Module
============================
Handles model, training, and data configuration.
Supports presets (toy, small) and custom configurations.

Usage:
    from src.config import ModelConfig, TrainConfig
    model_cfg = ModelConfig.from_preset("toy")
    train_cfg = TrainConfig()
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
import os
import subprocess
from pathlib import Path


# ============================================
# Model Configuration
# ============================================

@dataclass
class ModelConfig:
    """
    GPT Model Configuration.

    Attributes:
        vocab_size: Size of the vocabulary (set after tokenizer training)
        block_size: Maximum sequence length (context window)
        n_layer: Number of transformer blocks
        n_head: Number of attention heads
        d_model: Embedding dimension (must be divisible by n_head)
        dropout: Dropout probability (0.0 for inference)
        bias: Whether to use bias in Linear layers and LayerNorm
        use_flash_attn: Use torch.nn.functional.scaled_dot_product_attention
    """
    vocab_size: int = 4096
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1
    bias: bool = True
    use_flash_attn: bool = True  # Use fused attention when available

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_model % self.n_head == 0, \
            f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.block_size > 0, "block_size must be positive"

    @classmethod
    def from_preset(cls, preset: str) -> "ModelConfig":
        """
        Load a preset configuration.

        Presets:
            - "toy": Quick testing (4L/4H/256D, ctx=128)
            - "small": Small model (8L/8H/512D, ctx=256)
        """
        presets = {
            "toy": {
                "n_layer": 4,
                "n_head": 4,
                "d_model": 256,
                "block_size": 128,
                "dropout": 0.1,
            },
            "small": {
                "n_layer": 8,
                "n_head": 8,
                "d_model": 512,
                "block_size": 256,
                "dropout": 0.1,
            },
        }
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
        return cls(**presets[preset])

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.n_head

    @property
    def n_params(self) -> int:
        """Estimate number of parameters (rough)."""
        # Embedding: vocab_size * d_model + block_size * d_model
        embed = self.vocab_size * self.d_model + self.block_size * self.d_model
        # Per layer: attn (4 * d^2) + MLP (8 * d^2) + norms (4 * d)
        per_layer = 4 * self.d_model**2 + 8 * self.d_model**2 + 4 * self.d_model
        # Final norm
        final = self.d_model
        # Note: LM head shares weights with embedding, so not counted twice
        return embed + self.n_layer * per_layer + final


# ============================================
# Training Configuration
# ============================================

@dataclass
class TrainConfig:
    """
    Training Configuration.

    Attributes:
        batch_size: Micro-batch size per forward pass
        grad_accum_steps: Number of gradient accumulation steps
        max_steps: Maximum training steps (iterations)
        eval_interval: Steps between evaluations
        eval_steps: Number of batches for evaluation
        save_interval: Steps between checkpoint saves

        lr: Peak learning rate
        min_lr: Minimum learning rate (for cosine decay)
        warmup_steps: Number of warmup steps
        weight_decay: Weight decay for AdamW
        beta1: AdamW beta1
        beta2: AdamW beta2
        grad_clip: Gradient clipping threshold (0 to disable)

        use_amp: Use automatic mixed precision
        compile_model: Use torch.compile (PyTorch 2.0+)
        seed: Random seed for reproducibility

        data_dir: Directory containing train.bin and val.bin
        checkpoint_dir: Directory for saving checkpoints
        log_interval: Steps between logging
    """
    # Batch settings
    batch_size: int = 32
    grad_accum_steps: int = 4

    # Training duration
    max_steps: int = 5000
    eval_interval: int = 250
    eval_steps: int = 50
    save_interval: int = 1000

    # Learning rate schedule
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100

    # Optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Performance
    use_amp: bool = True
    compile_model: bool = False  # Default off for compatibility

    # Reproducibility
    seed: int = 42

    # Paths
    data_dir: str = "data/tokens"
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 10

    # Resume training
    resume_from: Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> "TrainConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def effective_batch_size(self) -> int:
        """Total batch size across gradient accumulation."""
        return self.batch_size * self.grad_accum_steps


# ============================================
# Data Configuration
# ============================================

@dataclass
class DataConfig:
    """
    Data and Tokenizer Configuration.

    Attributes:
        raw_data_path: Path to raw text file(s)
        output_dir: Directory for processed tokens
        tokenizer_dir: Directory for tokenizer files
        vocab_size: Target vocabulary size for BPE
        min_freq: Minimum frequency for BPE merges
        val_split: Fraction of data for validation
        max_chars: Maximum characters to process (None for all)
    """
    raw_data_path: str = "data/raw/input.txt"
    output_dir: str = "data/tokens"
    tokenizer_dir: str = "data/tokenizer"
    vocab_size: int = 4096  # Small vocab for efficiency
    min_freq: int = 2
    val_split: float = 0.1
    max_chars: Optional[int] = None  # Limit for testing

    @classmethod
    def from_json(cls, path: str) -> "DataConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ============================================
# Utility Functions
# ============================================

def get_git_commit() -> Optional[str]:
    """Get current git commit hash if in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def save_run_config(
    model_config: ModelConfig,
    train_config: TrainConfig,
    output_path: str
) -> None:
    """Save complete run configuration for reproducibility."""
    config = {
        "model": model_config.to_dict(),
        "train": train_config.to_dict(),
        "git_commit": get_git_commit(),
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Config] Saved run config to {output_path}")


def load_run_config(path: str) -> tuple:
    """Load run configuration from file."""
    with open(path, "r") as f:
        config = json.load(f)
    model_cfg = ModelConfig(**config["model"])
    train_cfg = TrainConfig(**config["train"])
    return model_cfg, train_cfg


# ============================================
# Example Usage / Self-Test
# ============================================

if __name__ == "__main__":
    # Test preset loading
    print("=== Testing ModelConfig ===")
    toy = ModelConfig.from_preset("toy")
    print(f"Toy preset: {toy}")
    print(f"  Head dim: {toy.head_dim}")
    print(f"  Est. params: {toy.n_params:,}")

    small = ModelConfig.from_preset("small")
    print(f"\nSmall preset: {small}")
    print(f"  Head dim: {small.head_dim}")
    print(f"  Est. params: {small.n_params:,}")

    print("\n=== Testing TrainConfig ===")
    train_cfg = TrainConfig()
    print(f"Default train config:")
    print(f"  Effective batch size: {train_cfg.effective_batch_size}")
    print(f"  Max steps: {train_cfg.max_steps}")

    print("\n=== Testing Git Commit ===")
    commit = get_git_commit()
    print(f"Current commit: {commit or 'Not in git repo'}")

    print("\n[OK] Config module working correctly!")
