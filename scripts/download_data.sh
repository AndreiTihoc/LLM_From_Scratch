#!/bin/bash
# ============================================
# Download Data Script
# ============================================
# Downloads a small dataset for training TinyGPT.
#
# Options:
#   --tiny     : Download tiny dataset (~5MB) for quick testing
#   --small    : Download small dataset (~50MB) for toy training
#   --medium   : Download medium dataset (~200MB) for serious training
#
# Usage:
#   ./scripts/download_data.sh --tiny
#   ./scripts/download_data.sh --small
# ============================================

set -e

# Default size
SIZE="tiny"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tiny)
            SIZE="tiny"
            shift
            ;;
        --small)
            SIZE="small"
            shift
            ;;
        --medium)
            SIZE="medium"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--tiny|--small|--medium]"
            echo ""
            echo "Options:"
            echo "  --tiny    Download tiny dataset (~5MB) for quick testing"
            echo "  --small   Download small dataset (~50MB) for toy training"
            echo "  --medium  Download medium dataset (~200MB) for full training"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create data directory
DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "TinyGPT Data Download"
echo "============================================"
echo "Size: $SIZE"
echo "Output: $DATA_DIR"
echo "============================================"

# Download based on size
if [ "$SIZE" = "tiny" ]; then
    # Tiny: A single small book from Project Gutenberg
    echo ""
    echo "[Download] Downloading Alice in Wonderland (tiny dataset)..."
    curl -L "https://www.gutenberg.org/files/11/11-0.txt" -o "$DATA_DIR/alice.txt" 2>/dev/null || \
        wget -q "https://www.gutenberg.org/files/11/11-0.txt" -O "$DATA_DIR/alice.txt"

    # Also get a bit more text
    echo "[Download] Downloading The Adventures of Sherlock Holmes..."
    curl -L "https://www.gutenberg.org/files/1661/1661-0.txt" -o "$DATA_DIR/sherlock.txt" 2>/dev/null || \
        wget -q "https://www.gutenberg.org/files/1661/1661-0.txt" -O "$DATA_DIR/sherlock.txt"

    # Combine into single file
    echo "[Download] Combining files..."
    cat "$DATA_DIR/alice.txt" "$DATA_DIR/sherlock.txt" > "$DATA_DIR/input.txt"
    rm "$DATA_DIR/alice.txt" "$DATA_DIR/sherlock.txt"

elif [ "$SIZE" = "small" ]; then
    # Small: Several books from Project Gutenberg
    echo ""
    echo "[Download] Downloading multiple books (small dataset)..."

    BOOKS=(
        "https://www.gutenberg.org/files/11/11-0.txt"      # Alice in Wonderland
        "https://www.gutenberg.org/files/1661/1661-0.txt"  # Sherlock Holmes
        "https://www.gutenberg.org/files/84/84-0.txt"      # Frankenstein
        "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice
        "https://www.gutenberg.org/files/2701/2701-0.txt"  # Moby Dick
        "https://www.gutenberg.org/files/98/98-0.txt"      # A Tale of Two Cities
    )

    > "$DATA_DIR/input.txt"  # Clear/create file

    for url in "${BOOKS[@]}"; do
        echo "  Downloading: $url"
        curl -L "$url" >> "$DATA_DIR/input.txt" 2>/dev/null || \
            wget -q "$url" -O - >> "$DATA_DIR/input.txt"
        echo "" >> "$DATA_DIR/input.txt"  # Add newline between books
    done

elif [ "$SIZE" = "medium" ]; then
    # Medium: TinyStories-style or more Gutenberg
    echo ""
    echo "[Download] Downloading larger dataset (medium)..."

    # Download many books from Gutenberg
    BOOKS=(
        "https://www.gutenberg.org/files/11/11-0.txt"
        "https://www.gutenberg.org/files/1661/1661-0.txt"
        "https://www.gutenberg.org/files/84/84-0.txt"
        "https://www.gutenberg.org/files/1342/1342-0.txt"
        "https://www.gutenberg.org/files/2701/2701-0.txt"
        "https://www.gutenberg.org/files/98/98-0.txt"
        "https://www.gutenberg.org/files/1080/1080-0.txt"  # A Modest Proposal
        "https://www.gutenberg.org/files/174/174-0.txt"    # Picture of Dorian Gray
        "https://www.gutenberg.org/files/345/345-0.txt"    # Dracula
        "https://www.gutenberg.org/files/1232/1232-0.txt"  # The Prince
        "https://www.gutenberg.org/files/16328/16328-0.txt" # Beowulf
        "https://www.gutenberg.org/files/2600/2600-0.txt"  # War and Peace
        "https://www.gutenberg.org/files/1400/1400-0.txt"  # Great Expectations
        "https://www.gutenberg.org/files/1952/1952-0.txt"  # The Yellow Wallpaper
        "https://www.gutenberg.org/files/76/76-0.txt"      # Huckleberry Finn
    )

    > "$DATA_DIR/input.txt"

    for url in "${BOOKS[@]}"; do
        echo "  Downloading: $url"
        curl -L "$url" >> "$DATA_DIR/input.txt" 2>/dev/null || \
            wget -q "$url" -O - >> "$DATA_DIR/input.txt"
        echo "" >> "$DATA_DIR/input.txt"
    done
fi

# Show results
echo ""
echo "============================================"
echo "Download Complete!"
echo "============================================"
FILE_SIZE=$(du -h "$DATA_DIR/input.txt" | cut -f1)
LINE_COUNT=$(wc -l < "$DATA_DIR/input.txt")
CHAR_COUNT=$(wc -c < "$DATA_DIR/input.txt")

echo "File: $DATA_DIR/input.txt"
echo "Size: $FILE_SIZE"
echo "Lines: $LINE_COUNT"
echo "Characters: $CHAR_COUNT"
echo ""
echo "Next step: Run ./scripts/prepare_data.sh to tokenize"
echo "============================================"
