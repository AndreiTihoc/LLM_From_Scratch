"""
GPT Model Implementation
========================
A from-scratch decoder-only transformer language model.

Architecture:
- Token embedding + learned positional embedding
- N transformer blocks, each containing:
  - Multi-head self-attention (causal/masked)
  - MLP with GELU activation and 4x expansion
  - Pre-LayerNorm (LN before attention and MLP)
  - Residual connections
- Final LayerNorm
- LM head (weight-tied with token embedding)

This implementation is designed for clarity and educational purposes.
It includes both a pure attention implementation and an optimized
path using PyTorch's scaled_dot_product_attention.

Usage:
    from src.model import GPT
    from src.config import ModelConfig

    config = ModelConfig.from_preset("toy")
    model = GPT(config)

    # Forward pass
    logits, loss = model(input_ids, targets=target_ids)

    # Generation
    generated = model.generate(prompt_ids, max_new_tokens=50)
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    Implements scaled dot-product attention with a causal mask to prevent
    attending to future tokens. Supports both a pure implementation and
    PyTorch's fused attention kernel for efficiency.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout
        self.use_flash = config.use_flash_attn

        assert config.d_model % config.n_head == 0, \
            "d_model must be divisible by n_head"

        # Key, Query, Value projections (combined for efficiency)
        # Output: [batch, seq, 3 * d_model]
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)

        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular matrix
        # Register as buffer so it moves with model to device
        # Shape: [1, 1, block_size, block_size]
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        mask = mask.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, d_model]

        Returns:
            Output tensor of shape [batch, seq_len, d_model]
        """
        B, T, C = x.size()  # batch, seq_len, d_model

        # Compute Q, K, V projections
        # [B, T, 3*C] -> 3 tensors of [B, T, C]
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Reshape for multi-head attention
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Compute attention
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's efficient fused attention (Flash Attention)
            # This handles the causal mask internally
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention computation (for clarity/debugging)
            y = self._manual_attention(q, k, v, T)

        # Reshape back: [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with dropout
        y = self.resid_dropout(self.c_proj(y))

        return y

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        T: int
    ) -> torch.Tensor:
        """
        Manual scaled dot-product attention with causal masking.

        This is the explicit implementation for educational purposes.
        In practice, use scaled_dot_product_attention for efficiency.

        Args:
            q, k, v: Query, Key, Value tensors [B, n_head, T, head_dim]
            T: Sequence length

        Returns:
            Attention output [B, n_head, T, head_dim]
        """
        # Compute attention scores: Q @ K^T / sqrt(head_dim)
        # [B, n_head, T, head_dim] @ [B, n_head, head_dim, T] -> [B, n_head, T, T]
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Apply causal mask: set future positions to -inf
        # mask is [1, 1, block_size, block_size], we only need [:T, :T]
        causal_mask = self.mask[:, :, :T, :T]
        scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax over last dimension (key positions)
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output: weights @ V
        # [B, n_head, T, T] @ [B, n_head, T, head_dim] -> [B, n_head, T, head_dim]
        y = torch.matmul(attn_weights, v)

        return y


class MLP(nn.Module):
    """
    Feed-forward MLP block with GELU activation.

    Architecture: Linear -> GELU -> Linear -> Dropout
    The hidden dimension is 4x the model dimension (standard GPT).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        # Hidden dimension is 4x model dimension
        hidden_dim = 4 * config.d_model

        # Expand
        self.c_fc = nn.Linear(config.d_model, hidden_dim, bias=config.bias)
        # Contract
        self.c_proj = nn.Linear(hidden_dim, config.d_model, bias=config.bias)
        # GELU activation (use 'tanh' approximation like GPT-2)
        self.gelu = nn.GELU(approximate='tanh')
        # Dropout
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    Architecture (pre-norm, like GPT-2):
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Attention with residual
        x = x + self.attn(self.ln_1(x))
        # MLP with residual
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    GPT Language Model.

    A decoder-only transformer for autoregressive language modeling.
    Predicts the next token given a sequence of tokens.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding: vocab_size -> d_model
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embedding: block_size -> d_model (learned, not sinusoidal)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)

        # Dropout after embedding
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, bias=config.bias)

        # LM head: d_model -> vocab_size
        # Note: We tie weights with token embedding below
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share weights between tok_emb and lm_head
        # This is standard in GPT-style models and reduces parameters
        self.tok_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # Scale by 1/sqrt(2*n_layer) for residual connections
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        n_params = self.get_num_params()
        print(f"[GPT] Model initialized with {n_params:,} parameters")

    def _init_weights(self, module):
        """Initialize weights using GPT-2 style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self, non_embedding: bool = False) -> int:
        """
        Count number of parameters.

        Args:
            non_embedding: If True, exclude position embedding params

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: Input token IDs [batch, seq_len]
            targets: Target token IDs [batch, seq_len] for loss computation

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            loss: Cross-entropy loss if targets provided, else None
        """
        B, T = idx.size()

        # Check sequence length
        assert T <= self.config.block_size, \
            f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Get token embeddings: [B, T, d_model]
        tok_emb = self.tok_emb(idx)

        # Get position embeddings: [T, d_model]
        # We create position indices [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)  # [T, d_model]

        # Combine embeddings: [B, T, d_model]
        x = self.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Compute logits: [B, T, vocab_size]
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy: [B*T, vocab_size] and [B*T]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1  # Optionally ignore padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.

        Args:
            idx: Initial token IDs [batch, seq_len]
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (1.0 = neutral, <1 = sharper, >1 = flatter)
            top_k: Keep only top k tokens for sampling
            top_p: Keep tokens with cumulative prob < top_p (nucleus sampling)
            do_sample: If False, use greedy decoding

        Returns:
            Generated token IDs [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]

            # Get logits for last position
            logits, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]  # [B, vocab_size]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative prob above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift to keep first token above threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                # Scatter back to original indices
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample or greedy
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str
    ) -> torch.optim.Optimizer:
        """
        Configure AdamW optimizer with proper weight decay.

        Weight decay is NOT applied to:
        - Bias parameters
        - LayerNorm parameters
        - 1D parameters (embeddings)

        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam beta parameters
            device_type: Device type for fused optimizer

        Returns:
            Configured optimizer
        """
        # Separate parameters into decay and no-decay groups
        # Note: lm_head.weight is tied to tok_emb.weight, so we work with actual params
        param_dict = {pn: p for pn, p in self.named_parameters()}

        decay = set()
        no_decay = set()

        for pn, p in param_dict.items():
            if pn.endswith('bias'):
                # All biases: no decay
                no_decay.add(pn)
            elif pn.endswith('weight') and ('ln_' in pn or 'ln_f' in pn):
                # LayerNorm weights: no decay
                no_decay.add(pn)
            elif pn.endswith('weight') and ('tok_emb' in pn or 'pos_emb' in pn):
                # Embedding weights: no decay
                no_decay.add(pn)
            elif pn.endswith('weight'):
                # All other weights (Linear): decay
                decay.add(pn)
            else:
                # Catch-all: no decay
                no_decay.add(pn)

        # Validate
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters in both sets: {inter_params}"
        assert len(param_dict.keys() - union_params) == 0, \
            f"Parameters not in any set: {param_dict.keys() - union_params}"

        # Create optimizer groups
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

        # Use fused AdamW if available (faster on CUDA)
        use_fused = device_type == 'cuda' and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )

        print(f"[GPT] Optimizer configured: AdamW, fused={use_fused}")
        print(f"[GPT] Decay params: {len(decay)}, No-decay params: {len(no_decay)}")

        return optimizer


# ============================================
# Utility Functions
# ============================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_memory_mb(config: ModelConfig, batch_size: int) -> float:
    """
    Rough estimate of GPU memory usage in MB.

    This is approximate and doesn't account for all factors.
    """
    # Model parameters (in float32 = 4 bytes)
    n_params = config.n_params
    model_mem = n_params * 4 / 1024 / 1024

    # Activations (rough estimate)
    # Per sample: block_size * d_model * n_layer * 2 (forward + backward)
    act_per_sample = config.block_size * config.d_model * config.n_layer * 2 * 4
    act_mem = batch_size * act_per_sample / 1024 / 1024

    # Optimizer states (AdamW: 2x model size for momentum + variance)
    opt_mem = n_params * 4 * 2 / 1024 / 1024

    total = model_mem + act_mem + opt_mem
    return total


# ============================================
# Example Usage / Self-Test
# ============================================

if __name__ == "__main__":
    print("=== Testing GPT Model ===\n")

    # Create toy config
    config = ModelConfig.from_preset("toy")
    config.vocab_size = 1000  # Small vocab for testing
    print(f"Config: {config}")
    print(f"Estimated params: {config.n_params:,}")
    print(f"Estimated memory: {estimate_memory_mb(config, 32):.1f} MB")

    # Create model
    print("\n--- Creating Model ---")
    model = GPT(config)
    print(f"Actual params: {count_parameters(model):,}")

    # Test forward pass
    print("\n--- Testing Forward Pass ---")
    batch_size = 4
    seq_len = 64

    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(x, targets=y)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    print("\n--- Testing Generation ---")
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    model.eval()
    generated = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=1.0,
        top_k=50,
        do_sample=True
    )
    print(f"Prompt shape: {prompt.shape}")
    print(f"Generated shape: {generated.shape}")
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")

    # Test optimizer configuration
    print("\n--- Testing Optimizer ---")
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=3e-4,
        betas=(0.9, 0.95),
        device_type='cpu'
    )
    print(f"Optimizer groups: {len(optimizer.param_groups)}")

    print("\n[OK] GPT model working correctly!")
