#!/bin/bash
# ============================================
# Colab Training Script
# ============================================
# Training configuration optimized for Google Colab GPU.
# Saves checkpoints to Google Drive to preserve progress.
#
# Prerequisites:
#   1. Mount Google Drive in Colab:
#      from google.colab import drive
#      drive.mount('/content/drive')
#
#   2. Clone repo and install deps:
#      !git clone <your-repo> /content/LLM_From_Scratch
#      %cd /content/LLM_From_Scratch
#      !pip install -r requirements.txt
#
#   3. Copy/download data (or run download + prepare scripts)
#
# Usage:
#   ./scripts/train_colab.sh toy      # Quick test
#   ./scripts/train_colab.sh small    # Full training
# ============================================

set -e

# Configuration
PRESET="${1:-toy}"
DRIVE_CHECKPOINT_DIR="/content/drive/MyDrive/scratchgpt_checkpoints"

# Change to project root
cd "$(dirname "$0")/.."

echo "============================================"
echo "ScratchGPT Colab Training"
echo "============================================"
echo "Preset: $PRESET"
echo "Checkpoint dir: $DRIVE_CHECKPOINT_DIR"
echo "============================================"

# Create checkpoint directory on Drive
mkdir -p "$DRIVE_CHECKPOINT_DIR"

# Check if data exists
if [ ! -f "data/tokens/train.bin" ]; then
    echo ""
    echo "Data not found. Downloading and preparing..."
    ./scripts/download_data.sh --tiny
    ./scripts/prepare_data.sh
fi

# Training parameters based on preset
if [ "$PRESET" = "toy" ]; then
    # Toy: Quick validation
    BATCH_SIZE=64
    GRAD_ACCUM=2
    MAX_STEPS=1000
    EVAL_INTERVAL=100
    SAVE_INTERVAL=500
elif [ "$PRESET" = "small" ]; then
    # Small: Serious training
    BATCH_SIZE=64
    GRAD_ACCUM=4
    MAX_STEPS=10000
    EVAL_INTERVAL=500
    SAVE_INTERVAL=2000
else
    echo "Unknown preset: $PRESET"
    echo "Use: toy or small"
    exit 1
fi

echo ""
echo "Training parameters:"
echo "  batch_size: $BATCH_SIZE"
echo "  grad_accum: $GRAD_ACCUM"
echo "  max_steps: $MAX_STEPS"
echo "  eval_interval: $EVAL_INTERVAL"
echo "  save_interval: $SAVE_INTERVAL"
echo ""

# Check for existing checkpoint to resume
RESUME_FLAG=""
if [ -f "$DRIVE_CHECKPOINT_DIR/latest.pt" ]; then
    echo "Found existing checkpoint at $DRIVE_CHECKPOINT_DIR/latest.pt"
    echo "Resuming training..."
    RESUME_FLAG="--resume $DRIVE_CHECKPOINT_DIR/latest.pt"
fi

# Run training
python -m src.train \
    --preset "$PRESET" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --max_steps "$MAX_STEPS" \
    --eval_interval "$EVAL_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --checkpoint_dir "$DRIVE_CHECKPOINT_DIR" \
    $RESUME_FLAG

echo ""
echo "============================================"
echo "Training Complete!"
echo "============================================"
echo "Checkpoints saved to: $DRIVE_CHECKPOINT_DIR"
echo ""
echo "To copy best model locally:"
echo "  cp $DRIVE_CHECKPOINT_DIR/best.pt checkpoints/"
echo ""
echo "Next: Export for GGUF conversion:"
echo "  ./scripts/export.sh"
