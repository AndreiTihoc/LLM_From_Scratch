#!/bin/bash
# ============================================
# Model Export Script
# ============================================
# Exports trained model for GGUF conversion.
#
# Usage:
#   ./scripts/export.sh                           # Default paths
#   ./scripts/export.sh --checkpoint path/to/model.pt
# ============================================

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Default paths
CHECKPOINT="${1:-checkpoints/best.pt}"
TOKENIZER="data/tokenizer"
OUTPUT="exports/scratchgpt"

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint|-c)
            CHECKPOINT="$2"
            shift 2
            ;;
        --tokenizer|-t)
            TOKENIZER="$2"
            shift 2
            ;;
        --output|-o)
            OUTPUT="$2"
            shift 2
            ;;
        *)
            # Assume it's the checkpoint path if no flag
            if [ -f "$1" ]; then
                CHECKPOINT="$1"
            fi
            shift
            ;;
    esac
done

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -la checkpoints/*.pt 2>/dev/null || echo "  (none found in checkpoints/)"
    exit 1
fi

# Check tokenizer exists
if [ ! -d "$TOKENIZER" ]; then
    echo "Error: Tokenizer not found: $TOKENIZER"
    exit 1
fi

echo "============================================"
echo "ScratchGPT Model Export"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Tokenizer:  $TOKENIZER"
echo "Output:     $OUTPUT"
echo "============================================"

# Run export
python -m src.export_hf \
    --checkpoint "$CHECKPOINT" \
    --tokenizer "$TOKENIZER" \
    --output "$OUTPUT"

echo ""
echo "Export complete!"
echo ""
echo "Next steps:"
echo "  1. Convert to GGUF: ./scripts/gguf_convert.sh $OUTPUT"
echo "  2. Quantize:        ./scripts/gguf_quantize.sh $OUTPUT/model-f16.gguf"
echo "  3. Run:             ./scripts/run_llamacpp.sh $OUTPUT/model-q4_k_m.gguf"
