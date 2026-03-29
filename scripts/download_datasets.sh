#!/usr/bin/env bash
# download_datasets.sh — Download Mip-NeRF 360 dataset scenes
#
# Downloads from the original source (jonbarron.info/mipnerf360/)
# and extracts into data/<scene>/ with the expected directory structure.
#
# Usage:
#   ./scripts/download_datasets.sh              # download all scenes
#   ./scripts/download_datasets.sh garden room   # download specific scenes
#   ./scripts/download_datasets.sh --list        # list available scenes
#
# Each scene includes:
#   images/     — full-resolution photos
#   images_2/   — 2x downsampled
#   images_4/   — 4x downsampled
#   images_8/   — 8x downsampled
#   sparse/0/   — COLMAP sparse reconstruction
#   poses_bounds.npy — camera poses (LLFF format)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_DIR/data"

# Source URL for the full Mip-NeRF 360 v2 dataset
DATASET_URL="http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

ALL_SCENES=(bicycle bonsai counter garden kitchen room stump)

usage() {
    echo "Usage: $0 [--list] [scene1 scene2 ...]"
    echo ""
    echo "Download Mip-NeRF 360 dataset scenes for NemoReconstruct."
    echo ""
    echo "Options:"
    echo "  --list     List available scenes and exit"
    echo "  --help     Show this help"
    echo ""
    echo "Available scenes: ${ALL_SCENES[*]}"
    echo ""
    echo "Examples:"
    echo "  $0                    # download all scenes (~12 GB)"
    echo "  $0 garden room        # download only garden and room (~4.2 GB)"
    echo "  $0 --list             # list available scenes with sizes"
}

list_scenes() {
    echo "Available Mip-NeRF 360 scenes:"
    echo ""
    echo "  Indoor scenes:"
    echo "    bonsai    (~1.3 GB)  311 images  — tabletop bonsai tree"
    echo "    counter   (~1.2 GB)  312 images  — kitchen counter"
    echo "    kitchen   (~1.5 GB)  315 images  — full kitchen"
    echo "    room      (~1.3 GB)  311 images  — living room"
    echo ""
    echo "  Outdoor scenes:"
    echo "    bicycle   (~2.3 GB)  291 images  — bicycle on grass"
    echo "    garden    (~2.8 GB)  185 images  — flower garden"
    echo "    stump     (~1.4 GB)  295 images  — tree stump"
    echo ""
    echo "Total: ~12 GB (all scenes)"
    echo ""
    echo "Source: https://jonbarron.info/mipnerf360/"

    # Show which scenes are already downloaded
    echo ""
    echo "Currently downloaded:"
    local found=0
    for scene in "${ALL_SCENES[@]}"; do
        if [[ -d "$DATA_DIR/$scene/images" ]]; then
            local count
            count=$(find "$DATA_DIR/$scene/images" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.png' \) 2>/dev/null | wc -l)
            echo "  ✓ $scene ($count images)"
            found=1
        fi
    done
    if [[ $found -eq 0 ]]; then
        echo "  (none)"
    fi
}

validate_scene() {
    local scene="$1"
    for valid in "${ALL_SCENES[@]}"; do
        if [[ "$scene" == "$valid" ]]; then
            return 0
        fi
    done
    echo "Error: Unknown scene '$scene'"
    echo "Available scenes: ${ALL_SCENES[*]}"
    return 1
}

download_and_extract() {
    local scenes=("$@")
    local zip_path="$DATA_DIR/.360_v2.zip"
    local need_download=false

    # Check which scenes need downloading
    local to_download=()
    for scene in "${scenes[@]}"; do
        if [[ -d "$DATA_DIR/$scene/images" ]]; then
            local count
            count=$(find "$DATA_DIR/$scene/images" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.png' \) 2>/dev/null | wc -l)
            if [[ $count -gt 0 ]]; then
                echo "✓ $scene already downloaded ($count images), skipping"
                continue
            fi
        fi
        to_download+=("$scene")
    done

    if [[ ${#to_download[@]} -eq 0 ]]; then
        echo ""
        echo "All requested scenes are already downloaded."
        return 0
    fi

    echo ""
    echo "Scenes to download: ${to_download[*]}"
    echo ""

    # Download the zip if we don't have it cached
    if [[ -f "$zip_path" ]]; then
        echo "Using cached download: $zip_path"
    else
        echo "Downloading Mip-NeRF 360 dataset (~12 GB)..."
        echo "Source: $DATASET_URL"
        echo ""
        mkdir -p "$DATA_DIR"

        if command -v wget &>/dev/null; then
            wget -O "$zip_path" --show-progress "$DATASET_URL"
        elif command -v curl &>/dev/null; then
            curl -L -o "$zip_path" --progress-bar "$DATASET_URL"
        else
            echo "Error: Neither wget nor curl found. Install one and retry."
            exit 1
        fi

        echo ""
        echo "Download complete."
    fi

    # Extract requested scenes
    for scene in "${to_download[@]}"; do
        echo ""
        echo "Extracting $scene..."
        mkdir -p "$DATA_DIR/$scene"
        unzip -o -q "$zip_path" "${scene}/*" -d "$DATA_DIR"
        local count
        count=$(find "$DATA_DIR/$scene/images" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.png' \) 2>/dev/null | wc -l)
        echo "✓ $scene extracted ($count images)"
    done

    echo ""
    echo "Done. Dataset scenes are in: $DATA_DIR/"
    echo ""

    # Offer to clean up the zip
    if [[ -f "$zip_path" ]]; then
        local zip_size
        zip_size=$(du -h "$zip_path" | cut -f1)
        echo "The cached zip ($zip_size) is at: $zip_path"
        echo "To free disk space: rm $zip_path"
    fi
}

# ── Main ──────────────────────────────────────────────────────────

if [[ $# -eq 0 ]]; then
    # No args: download all scenes
    download_and_extract "${ALL_SCENES[@]}"
    exit 0
fi

case "$1" in
    --list|-l)
        list_scenes
        exit 0
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    -*)
        echo "Unknown option: $1"
        usage
        exit 1
        ;;
esac

# Validate all requested scenes first
for scene in "$@"; do
    validate_scene "$scene"
done

download_and_extract "$@"
