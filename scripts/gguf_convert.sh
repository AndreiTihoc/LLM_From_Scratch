#!/bin/bash
# ============================================
# GGUF Conversion Script
# ============================================
# Converts exported model to GGUF format using llama.cpp.
#
# This script:
# 1. Clones llama.cpp if not present
# 2. Builds llama.cpp tools
# 3. Converts model to F16 GGUF
#
# Usage:
#   ./scripts/gguf_convert.sh exports/scratchgpt
# ============================================

set -e

# Configuration
EXPORT_DIR="${1:-exports/scratchgpt}"
LLAMA_CPP_DIR="llama.cpp"

# Change to project root
cd "$(dirname "$0")/.."

echo "============================================"
echo "GGUF Conversion"
echo "============================================"
echo "Export dir: $EXPORT_DIR"
echo "============================================"

# Check export directory
if [ ! -d "$EXPORT_DIR" ]; then
    echo "Error: Export directory not found: $EXPORT_DIR"
    echo "Run ./scripts/export.sh first."
    exit 1
fi

# Clone llama.cpp if not present
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo ""
    echo "[GGUF] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
fi

# Install Python dependencies for conversion
echo ""
echo "[GGUF] Installing conversion dependencies..."
pip install -q gguf sentencepiece

# Try to use llama.cpp's converter first
echo ""
echo "[GGUF] Attempting conversion with llama.cpp converter..."

# Check if we have a custom conversion script
CUSTOM_CONVERTER="scripts/convert_to_gguf.py"

if [ -f "$CUSTOM_CONVERTER" ]; then
    echo "[GGUF] Using custom ScratchGPT converter..."
    python "$CUSTOM_CONVERTER" \
        --input "$EXPORT_DIR" \
        --output "$EXPORT_DIR/model-f16.gguf"
else
    # Try llama.cpp's converter (may not work for custom architectures)
    echo "[GGUF] Trying llama.cpp convert_hf_to_gguf.py..."
    echo "[GGUF] Note: This may fail for custom architectures."
    echo "[GGUF] If it fails, use the custom converter: scripts/convert_to_gguf.py"

    python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
        "$EXPORT_DIR" \
        --outfile "$EXPORT_DIR/model-f16.gguf" \
        --outtype f16 \
        || {
            echo ""
            echo "[GGUF] llama.cpp converter failed (expected for custom architecture)."
            echo "[GGUF] Creating custom converter..."

            # Create the custom converter
            python scripts/convert_to_gguf.py \
                --input "$EXPORT_DIR" \
                --output "$EXPORT_DIR/model-f16.gguf"
        }
fi

# Check output
if [ -f "$EXPORT_DIR/model-f16.gguf" ]; then
    SIZE=$(du -h "$EXPORT_DIR/model-f16.gguf" | cut -f1)
    echo ""
    echo "============================================"
    echo "GGUF Conversion Complete!"
    echo "============================================"
    echo "Output: $EXPORT_DIR/model-f16.gguf ($SIZE)"
    echo ""
    echo "Next step: Quantize to Q4_K_M:"
    echo "  ./scripts/gguf_quantize.sh $EXPORT_DIR/model-f16.gguf"
else
    echo ""
    echo "Error: GGUF file not created."
    echo "Check the output above for errors."
    exit 1
fi
