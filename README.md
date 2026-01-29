# TinyGPT: A GPT-Style LLM From Scratch

A complete, educational implementation of a GPT-style language model built from scratch using only PyTorch and standard Python libraries. No HuggingFace Trainer, no nanoGPT copying – just clean, readable code.

## Features

- **Byte-level BPE tokenizer** - Custom implementation with training, encoding, and decoding
- **Decoder-only Transformer** - Multi-head attention, MLP, LayerNorm, residual connections
- **Training loop** - Mixed precision, gradient accumulation, cosine LR schedule, checkpointing
- **Text generation** - Temperature, top-k, top-p (nucleus) sampling
- **GGUF export** - Convert to llama.cpp format for efficient CPU inference

## Project Structure

```
LLM_From_Scratch/
├── src/
│   ├── config.py           # Configuration management
│   ├── model.py            # GPT model implementation
│   ├── train.py            # Training loop
│   ├── sample.py           # Text generation
│   ├── dataset.py          # Memmap dataset handling
│   ├── export_hf.py        # Export for GGUF conversion
│   └── tokenizer/
│       ├── byte_bpe.py     # BPE tokenizer implementation
│       └── train_bpe.py    # Tokenizer training script
├── scripts/
│   ├── download_data.sh    # Download training data
│   ├── prepare_data.sh     # Prepare data (tokenize)
│   ├── prepare_data.py     # Data preparation logic
│   ├── train_colab.sh      # Training script for Colab
│   ├── export.sh           # Export model
│   ├── gguf_convert.sh     # Convert to GGUF
│   ├── convert_to_gguf.py  # Custom GGUF converter
│   ├── gguf_quantize.sh    # Quantize GGUF model
│   └── run_llamacpp.sh     # Run inference with llama.cpp
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick Start (5-Minute Sanity Test)

Run this sequence to validate everything works:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download tiny dataset (~5MB)
./scripts/download_data.sh --tiny

# 3. Prepare tokenizer + tokens
python scripts/prepare_data.py --vocab_size 2048 --max_lines 5000

# 4. Train for 200 steps (CPU, ~2-3 min)
python -m src.train --preset toy --max_steps 200 --batch_size 8 --grad_accum 1

# 5. Generate text
python -m src.sample --checkpoint checkpoints/best.pt --prompt "Once upon a time" --max_tokens 50

# 6. Export for GGUF
./scripts/export.sh

# 7. Convert to GGUF (requires llama.cpp)
./scripts/gguf_convert.sh exports/tinygpt

# 8. Quantize to Q4_K_M
./scripts/gguf_quantize.sh exports/tinygpt/model-f16.gguf

# 9. Run with llama.cpp
./scripts/run_llamacpp.sh exports/tinygpt/model-q4_k_m.gguf "Hello"
```

## Detailed Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required: `torch`, `numpy`, `tqdm`
Optional: `safetensors` (for model export), `matplotlib` (for visualization)

### 2. Download Training Data

```bash
# Tiny dataset (~5MB) - for quick testing
./scripts/download_data.sh --tiny

# Small dataset (~50MB) - for toy model training
./scripts/download_data.sh --small

# Medium dataset (~200MB) - for serious training
./scripts/download_data.sh --medium
```

### 3. Prepare Data

```bash
# Default settings (vocab_size=4096)
python scripts/prepare_data.py

# Custom vocab size
python scripts/prepare_data.py --vocab_size 8192

# Quick test with limited data
python scripts/prepare_data.py --max_lines 10000
```

This will:
1. Clean the raw text
2. Train a BPE tokenizer
3. Encode text to binary tokens (memmap format)

### 4. Train the Model

#### Local CPU Training (Slow)

```bash
# Toy model, 500 steps
python -m src.train --preset toy --max_steps 500 --batch_size 16 --grad_accum 2 --no_amp
```

#### Local GPU Training

```bash
# Toy model with mixed precision
python -m src.train --preset toy --max_steps 2000

# Small model, longer training
python -m src.train --preset small --max_steps 10000 --batch_size 32 --grad_accum 4
```

#### Google Colab Training

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Clone and setup:
```bash
!git clone <your-repo-url> /content/LLM_From_Scratch
%cd /content/LLM_From_Scratch
!pip install -r requirements.txt
```

3. Download and prepare data:
```bash
!./scripts/download_data.sh --small
!python scripts/prepare_data.py
```

4. Train:
```bash
!./scripts/train_colab.sh toy    # Quick test
!./scripts/train_colab.sh small  # Full training
```

Checkpoints are saved to Google Drive automatically.

### 5. Generate Text

```bash
# Single generation
python -m src.sample --checkpoint checkpoints/best.pt --prompt "The quick brown fox"

# Interactive mode
python -m src.sample --checkpoint checkpoints/best.pt --interactive

# Streaming output
python -m src.sample --checkpoint checkpoints/best.pt --prompt "Hello" --stream

# Adjust sampling parameters
python -m src.sample -c checkpoints/best.pt -p "Once upon" \
    --temperature 0.7 --top_k 40 --top_p 0.9 --max_tokens 200
```

### 6. Export and Convert to GGUF

```bash
# Export to HF-like format
./scripts/export.sh

# Convert to GGUF
./scripts/gguf_convert.sh exports/tinygpt

# Quantize to Q4_K_M (recommended for 8GB RAM)
./scripts/gguf_quantize.sh exports/tinygpt/model-f16.gguf

# Run with llama.cpp
./scripts/run_llamacpp.sh exports/tinygpt/model-q4_k_m.gguf "Your prompt here"
```

## Model Presets

| Preset | Layers | Heads | Dim | Context | Est. Params |
|--------|--------|-------|-----|---------|-------------|
| toy    | 4      | 4     | 256 | 128     | ~3M         |
| small  | 8      | 8     | 512 | 256     | ~25M        |

## Training Tips

### Out of Memory (OOM) Solutions

1. **Reduce batch size**: `--batch_size 8`
2. **Increase gradient accumulation**: `--grad_accum 8`
3. **Reduce block size**: `--block_size 64`
4. **Reduce vocab size**: `--vocab_size 2048` when preparing data
5. **Disable mixed precision on CPU**: `--no_amp`

### Improving Results

1. **More data**: Use `--medium` or gather more text
2. **Longer training**: Increase `--max_steps`
3. **Larger model**: Use `--preset small` (needs GPU)
4. **Lower learning rate**: `--lr 1e-4`
5. **More warmup**: `--warmup_steps 200`

## Architecture Details

### Attention Implementation

The attention module includes both:
- **Pure implementation**: Explicit Q,K,V projections, scaled dot-product, masking
- **Fused implementation**: Uses `torch.nn.functional.scaled_dot_product_attention` when available

```python
# Causal mask is registered as buffer for efficiency
mask = torch.tril(torch.ones(block_size, block_size))
self.register_buffer("mask", mask)

# During forward pass
scores = scores.masked_fill(mask == 0, float('-inf'))
```

### Weight Tying

Token embedding and LM head share weights:
```python
self.tok_emb.weight = self.lm_head.weight
```

### AdamW Configuration

Weight decay is NOT applied to:
- Bias parameters
- LayerNorm parameters
- Embedding weights

## Project Constraints

This project is designed to work within these constraints:

- **8GB RAM** - Using memmap datasets, streaming tokenization
- **5GB disk** - Data and checkpoints are gitignored, small default vocab
- **No GPU locally** - Training happens on Colab, local is for testing
- **No HuggingFace dependencies** - Pure PyTorch implementation

### Why Memmap?

Memmap allows accessing large token files without loading into RAM:
```python
# Only loads what's accessed, not the full array
self.data = np.memmap(path, dtype=np.uint16, mode="r")
```

## Files Not in Git

These directories are gitignored (create locally or on Colab):
- `data/` - Raw text and tokenized data
- `checkpoints/` - Model checkpoints
- `exports/` - Exported models
- `llama.cpp/` - llama.cpp clone for GGUF conversion

## Troubleshooting

### "CUDA out of memory"
- Reduce `--batch_size`
- Increase `--grad_accum`
- Use `--preset toy`

### "Tokenizer not found"
- Run `python scripts/prepare_data.py` first

### "llama.cpp not found"
- Run `./scripts/gguf_convert.sh` to clone and build it

### Slow training on CPU
- This is expected. Use Colab for real training.
- For testing, use `--max_steps 100` and small batch size

### Poor generation quality
- Train longer
- Use more data
- Try different sampling parameters (lower temperature)

## License

MIT License - Use freely for learning and experimentation.

## Acknowledgments

Inspired by:
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [minGPT](https://github.com/karpathy/minGPT)

Built from scratch for educational purposes.
