#!/bin/bash
# ============================================
# GGUF Quantization Script
# ============================================
# Quantizes F16 GGUF model to Q4_K_M for efficient inference.
#
# Q4_K_M offers a good balance of quality and size:
# - ~4 bits per weight
# - Good quality retention
# - Fast inference
# - Small file size
#
# Usage:
#   ./scripts/gguf_quantize.sh exports/tinygpt/model-f16.gguf
#   ./scripts/gguf_quantize.sh exports/tinygpt/model-f16.gguf q8_0
# ============================================

set -e

# Configuration
INPUT_GGUF="${1:-exports/tinygpt/model-f16.gguf}"
QUANT_TYPE="${2:-q4_k_m}"
LLAMA_CPP_DIR="llama.cpp"

# Change to project root
cd "$(dirname "$0")/.."

echo "============================================"
echo "GGUF Quantization"
echo "============================================"
echo "Input: $INPUT_GGUF"
echo "Quant type: $QUANT_TYPE"
echo "============================================"

# Check input file
if [ ! -f "$INPUT_GGUF" ]; then
    echo "Error: Input GGUF file not found: $INPUT_GGUF"
    echo "Run ./scripts/gguf_convert.sh first."
    exit 1
fi

# Check llama.cpp
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "Error: llama.cpp not found."
    echo "Run ./scripts/gguf_convert.sh first to clone it."
    exit 1
fi

# Build llama.cpp quantize tool if needed
QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
if [ ! -f "$QUANTIZE_BIN" ]; then
    QUANTIZE_BIN="$LLAMA_CPP_DIR/llama-quantize"
fi

if [ ! -f "$QUANTIZE_BIN" ]; then
    echo ""
    echo "[Quantize] Building llama.cpp..."
    cd "$LLAMA_CPP_DIR"

    # Try cmake build first
    if command -v cmake &> /dev/null; then
        mkdir -p build && cd build
        cmake .. -DGGML_CUDA=OFF -DGGML_METAL=OFF
        cmake --build . --config Release -j$(nproc)
        cd ../..
        QUANTIZE_BIN="$LLAMA_CPP_DIR/build/bin/llama-quantize"
    else
        # Fall back to make
        make quantize -j$(nproc)
        cd ..
        QUANTIZE_BIN="$LLAMA_CPP_DIR/llama-quantize"
    fi
fi

if [ ! -f "$QUANTIZE_BIN" ]; then
    echo "Error: Could not build or find llama-quantize"
    echo "Please build llama.cpp manually:"
    echo "  cd llama.cpp && mkdir build && cd build"
    echo "  cmake .. && cmake --build . --config Release"
    exit 1
fi

# Determine output filename
INPUT_DIR=$(dirname "$INPUT_GGUF")
INPUT_BASE=$(basename "$INPUT_GGUF" .gguf)
OUTPUT_GGUF="$INPUT_DIR/${INPUT_BASE%%-f16}-${QUANT_TYPE}.gguf"

# Alternative: replace f16 with quant type
if [[ "$INPUT_BASE" == *"-f16"* ]]; then
    OUTPUT_GGUF="$INPUT_DIR/${INPUT_BASE/-f16/-$QUANT_TYPE}.gguf"
fi

echo ""
echo "[Quantize] Running quantization..."
echo "[Quantize] Output: $OUTPUT_GGUF"
echo ""

# Run quantization
"$QUANTIZE_BIN" "$INPUT_GGUF" "$OUTPUT_GGUF" "$QUANT_TYPE"

# Check output
if [ -f "$OUTPUT_GGUF" ]; then
    INPUT_SIZE=$(du -h "$INPUT_GGUF" | cut -f1)
    OUTPUT_SIZE=$(du -h "$OUTPUT_GGUF" | cut -f1)
    echo ""
    echo "============================================"
    echo "Quantization Complete!"
    echo "============================================"
    echo "Input:  $INPUT_GGUF ($INPUT_SIZE)"
    echo "Output: $OUTPUT_GGUF ($OUTPUT_SIZE)"
    echo ""
    echo "Next step: Run inference:"
    echo "  ./scripts/run_llamacpp.sh $OUTPUT_GGUF"
else
    echo ""
    echo "Error: Quantization failed."
    echo "Output file not created: $OUTPUT_GGUF"
    exit 1
fi
