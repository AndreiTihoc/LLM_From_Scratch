#!/bin/bash
# ============================================
# Run llama.cpp Inference Script
# ============================================
# Runs inference using llama.cpp with TinyGPT model.
#
# Optimized settings for 8GB RAM systems.
#
# Usage:
#   ./scripts/run_llamacpp.sh exports/tinygpt/model-q4_k_m.gguf
#   ./scripts/run_llamacpp.sh model.gguf "Once upon a time"
#   ./scripts/run_llamacpp.sh model.gguf --interactive
# ============================================

set -e

# Configuration
MODEL_PATH="${1:-exports/tinygpt/model-q4_k_m.gguf}"
PROMPT="${2:-Once upon a time}"
LLAMA_CPP_DIR="llama.cpp"

# Memory-conservative settings for 8GB RAM
N_CTX=512           # Context size
N_BATCH=256         # Batch size for prompt processing
N_THREADS=4         # CPU threads
N_PREDICT=256       # Tokens to generate
TEMP=0.8            # Temperature
TOP_K=40            # Top-k sampling
TOP_P=0.9           # Top-p (nucleus) sampling
REPEAT_PENALTY=1.1  # Repetition penalty

# Change to project root
cd "$(dirname "$0")/.."

echo "============================================"
echo "TinyGPT Inference with llama.cpp"
echo "============================================"
echo "Model: $MODEL_PATH"
echo "Context: $N_CTX"
echo "Threads: $N_THREADS"
echo "============================================"

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model not found: $MODEL_PATH"
    echo ""
    echo "Available models:"
    find exports -name "*.gguf" 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Check llama.cpp
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Error: llama.cpp not found."
    echo "Run ./scripts/gguf_convert.sh first."
    exit 1
fi

# Find llama-cli or main executable
LLAMA_BIN=""
for bin in "$LLAMA_CPP_DIR/build/bin/llama-cli" \
           "$LLAMA_CPP_DIR/build/bin/main" \
           "$LLAMA_CPP_DIR/llama-cli" \
           "$LLAMA_CPP_DIR/main"; do
    if [ -f "$bin" ]; then
        LLAMA_BIN="$bin"
        break
    fi
done

# Build if not found
if [ -z "$LLAMA_BIN" ] || [ ! -f "$LLAMA_BIN" ]; then
    echo ""
    echo "[Run] Building llama.cpp..."
    cd "$LLAMA_CPP_DIR"

    if command -v cmake &> /dev/null; then
        mkdir -p build && cd build
        cmake .. -DGGML_CUDA=OFF -DGGML_METAL=OFF
        cmake --build . --config Release -j$(nproc)
        cd ../..
        LLAMA_BIN="$LLAMA_CPP_DIR/build/bin/llama-cli"
    else
        make main -j$(nproc)
        cd ..
        LLAMA_BIN="$LLAMA_CPP_DIR/main"
    fi
fi

if [ ! -f "$LLAMA_BIN" ]; then
    echo "Error: Could not find or build llama.cpp executable"
    exit 1
fi

echo ""
echo "[Run] Using: $LLAMA_BIN"
echo ""

# Handle interactive mode
if [ "$PROMPT" = "--interactive" ] || [ "$PROMPT" = "-i" ]; then
    echo "[Run] Starting interactive mode..."
    echo "Type your prompts. Press Ctrl+C to exit."
    echo ""

    "$LLAMA_BIN" \
        -m "$MODEL_PATH" \
        -c "$N_CTX" \
        -b "$N_BATCH" \
        -t "$N_THREADS" \
        --temp "$TEMP" \
        --top-k "$TOP_K" \
        --top-p "$TOP_P" \
        --repeat-penalty "$REPEAT_PENALTY" \
        --interactive \
        --color
else
    # Single generation
    echo "[Run] Generating from prompt: \"$PROMPT\""
    echo ""
    echo "--- Output ---"

    "$LLAMA_BIN" \
        -m "$MODEL_PATH" \
        -c "$N_CTX" \
        -b "$N_BATCH" \
        -t "$N_THREADS" \
        -n "$N_PREDICT" \
        --temp "$TEMP" \
        --top-k "$TOP_K" \
        --top-p "$TOP_P" \
        --repeat-penalty "$REPEAT_PENALTY" \
        -p "$PROMPT" \
        --no-display-prompt

    echo ""
    echo "--- End ---"
fi
