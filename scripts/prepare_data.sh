#!/bin/bash
# ============================================
# Data Preparation Wrapper Script
# ============================================
# Prepares data for training: clean, tokenize, encode.
#
# Usage:
#   ./scripts/prepare_data.sh                    # Default settings
#   ./scripts/prepare_data.sh --vocab_size 8192  # Custom vocab size
#   ./scripts/prepare_data.sh --max_lines 5000   # Quick test
# ============================================

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Check if data exists
if [ ! -f "data/raw/input.txt" ]; then
    echo "Error: data/raw/input.txt not found!"
    echo "Run ./scripts/download_data.sh first."
    exit 1
fi

# Run Python preparation script with all arguments
python scripts/prepare_data.py "$@"
