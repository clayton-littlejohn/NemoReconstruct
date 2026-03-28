#!/usr/bin/env bash
# test_iterations.sh — Fast iterative reconstruction test (direct API, no agents)
#
# Uploads the same video multiple times with different parameters to test
# the frontend's ability to display multiple iterations side by side.
#
# Usage:
#   ./nemoclaw/test_iterations.sh [video_path] [base_name] [num_iterations]
#
# Example:
#   ./nemoclaw/test_iterations.sh ~/devl/github/NemoReconstruct/spark.MOV spark 3

set -euo pipefail

API_URL="${API_URL:-http://127.0.0.1:8010}"
VIDEO_PATH="${1:-/home/clayton_littlejohn/devl/github/NemoReconstruct/spark.MOV}"
BASE_NAME="${2:-spark}"
NUM_ITERATIONS="${3:-3}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"

if [[ ! -f "$VIDEO_PATH" ]]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

echo "============================================"
echo " Fast Iteration Test"
echo "============================================"
echo " Video:      $VIDEO_PATH"
echo " Base name:  $BASE_NAME"
echo " Iterations: $NUM_ITERATIONS"
echo " API:        $API_URL"
echo "============================================"
echo ""

# Health check
echo "[test] Checking backend health..."
HEALTH=$(curl -sf "${API_URL}/health" 2>/dev/null || echo "FAILED")
if [[ "$HEALTH" == *"FAILED"* ]]; then
    echo "[test] ERROR: Backend not running at $API_URL"
    exit 1
fi
echo "[test] Backend OK"
echo ""

# Define parameters for each iteration (increasingly better)
declare -a EPOCHS=(5 10 15)
declare -a DOWNSAMPLE=(10 8 6)
declare -a FRAME_RATE=(1.0 1.5 2.0)
declare -a DESCRIPTIONS=(
    "Iteration 1: Fast baseline — 5 epochs, 10x downsample"
    "Iteration 2: More epochs (10), less downsample (8x)"
    "Iteration 3: Further tuning — 15 epochs, 6x downsample, higher frame rate"
)

poll_until_done() {
    local id="$1"
    while true; do
        local response
        response=$(curl -sf "${API_URL}/api/v1/reconstructions/${id}/status" 2>/dev/null)
        local status
        status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
        local step
        step=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('processing_step',''))" 2>/dev/null)
        local pct
        pct=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('processing_pct',0))" 2>/dev/null)

        echo "  [$status] ${step} (${pct}%)" >&2

        if [[ "$status" == "completed" || "$status" == "failed" ]]; then
            echo "$status"
            return
        fi
        sleep "$POLL_INTERVAL"
    done
}

for ((i=0; i<NUM_ITERATIONS && i<3; i++)); do
    ITER=$((i + 1))
    echo "============================================"
    echo " Iteration $ITER / $NUM_ITERATIONS"
    echo "============================================"
    echo " Epochs: ${EPOCHS[$i]}, Downsample: ${DOWNSAMPLE[$i]}x, Frame rate: ${FRAME_RATE[$i]}"
    echo ""

    # Upload
    echo "[iter $ITER] Uploading video..."
    RESPONSE=$(curl -sf -X POST "${API_URL}/api/v1/reconstructions/upload" \
        -F "file=@${VIDEO_PATH}" \
        -F "name=${BASE_NAME} (run ${ITER})" \
        -F "description=${DESCRIPTIONS[$i]}" \
        -F "fvdb_max_epochs=${EPOCHS[$i]}" \
        -F "fvdb_image_downsample_factor=${DOWNSAMPLE[$i]}" \
        -F "frame_rate=${FRAME_RATE[$i]}" \
        -F "splat_only_mode=true")

    ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
    echo "[iter $ITER] Created: $ID"
    echo ""

    # Poll until done
    echo "[iter $ITER] Polling status..."
    FINAL_STATUS=$(poll_until_done "$ID")
    echo ""

    if [[ "$FINAL_STATUS" == "completed" ]]; then
        echo "[iter $ITER] Fetching metrics..."
        curl -sf "${API_URL}/api/v1/reconstructions/${ID}/metrics" | python3 -c "
import sys, json
d = json.load(sys.stdin)
s = d.get('summary', {})
print(f'  Loss:      {s.get(\"reconstruct/loss\", \"n/a\")}')
print(f'  SSIM:      {s.get(\"reconstruct/ssimloss\", \"n/a\")}')
print(f'  Gaussians: {s.get(\"reconstruct/num_gaussians\", \"n/a\")}')
print(f'  L1 Loss:   {s.get(\"reconstruct/l1loss\", \"n/a\")}')
print(f'  GPU Mem:   {s.get(\"reconstruct/mem_allocated\", \"n/a\")} GB')
"
    else
        echo "[iter $ITER] Failed — skipping metrics"
    fi
    echo ""
done

echo "============================================"
echo " All iterations complete"
echo "============================================"
echo ""
echo "View results at: http://localhost:3000"
echo "API docs at:     ${API_URL}/docs"
echo ""
echo "All reconstructions:"
curl -sf "${API_URL}/api/v1/reconstructions" | python3 -c "
import sys, json
for r in json.load(sys.stdin):
    print(f'  {r[\"name\"]:30s}  {r[\"status\"]:20s}  {r[\"id\"]}')
"
