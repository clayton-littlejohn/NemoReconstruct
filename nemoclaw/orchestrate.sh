#!/usr/bin/env bash
# orchestrate.sh — Iterative reconstruction workflow
#
# Flow:
#   1. Start reconstruction (direct API call)
#   2. Wait for completion
#   3. Evaluator agent runs in OpenShell sandbox (inspects metrics, files, API)
#   4. If ITERATE → retry with recommended params → goto 2
#   5. Repeat until ACCEPT or max iterations reached
#
# Usage:
#   ./nemoclaw/orchestrate.sh <video_path> <scene_name> [max_iterations]
#   ./nemoclaw/orchestrate.sh --dataset <dataset_name> <scene_name> [max_iterations]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
API_PORT="${API_PORT:-8010}"
API_URL="${API_URL:-http://127.0.0.1:${API_PORT}}"
WORKFLOW_API_URL="${WORKFLOW_API_URL:-$API_URL}"
WORKFLOW_ID="${WORKFLOW_ID:-}"

# LLM / Agent config
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-nemotron-3-nano}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-300}"
RUN_AGENT_RETRIES="${RUN_AGENT_RETRIES:-10}"

# Parse arguments
DATASET_MODE=false
DATASET_NAME=""
VIDEO_PATH=""

if [[ "${1:-}" == "--dataset" ]]; then
    DATASET_MODE=true
    DATASET_NAME="${2:?Usage: orchestrate.sh --dataset <dataset_name> <scene_name> [max_iterations]}"
    SCENE_NAME="${3:?Usage: orchestrate.sh --dataset <dataset_name> <scene_name> [max_iterations]}"
    MAX_ITERATIONS="${4:-3}"
else
    VIDEO_PATH="${1:?Usage: orchestrate.sh <video_path> <scene_name> [max_iterations]}"
    SCENE_NAME="${2:?Usage: orchestrate.sh <video_path> <scene_name> [max_iterations]}"
    MAX_ITERATIONS="${3:-3}"
fi

ACCEPT_PSNR="${ACCEPT_PSNR_THRESHOLD:-25.0}"
ACCEPT_SSIM="${ACCEPT_SSIM_THRESHOLD:-0.85}"

if [[ "$DATASET_MODE" == "true" ]]; then
    CURRENT_BACKEND="${INITIAL_BACKEND:-fvdb}"
else
    CURRENT_BACKEND="${INITIAL_BACKEND:-3dgrut}"
fi

# --- Logging ---
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SCENE_NAME}_$(date +%Y-%m-%d_%H-%M-%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[orchestrator] Log file: $LOG_FILE"

# Validate inputs
if [[ "$DATASET_MODE" == "false" ]]; then
    if [[ ! -f "$VIDEO_PATH" ]]; then
        echo "[orchestrator] ERROR: Video file not found: $VIDEO_PATH"
        exit 1
    fi
    VIDEO_PATH="$(cd "$(dirname "$VIDEO_PATH")" && pwd)/$(basename "$VIDEO_PATH")"
fi

echo ""
echo "============================================"
echo " NemoReconstruct — Iterative Workflow"
echo "============================================"
if [[ "$DATASET_MODE" == "true" ]]; then
echo " Dataset:        $DATASET_NAME"
else
echo " Video:          $VIDEO_PATH"
fi
echo " Scene:          $SCENE_NAME"
echo " Max iterations: $MAX_ITERATIONS"
echo " Backend:        $CURRENT_BACKEND"
echo " PSNR threshold: $ACCEPT_PSNR"
echo " SSIM threshold: $ACCEPT_SSIM"
echo " API:            $API_URL"
echo " LLM:            $OLLAMA_URL ($OLLAMA_MODEL)"
echo " Agent timeout:   ${AGENT_TIMEOUT}s (retries: $RUN_AGENT_RETRIES)"
echo "============================================"
echo ""

# ── Helpers ─────────────────────────────────────────────

update_workflow() {
    if [[ -z "$WORKFLOW_ID" ]]; then return; fi
    curl -sf -X PATCH "${WORKFLOW_API_URL}/api/v1/workflows/${WORKFLOW_ID}/state" \
        -H "Content-Type: application/json" \
        -d "$1" >/dev/null 2>&1 || true
}

save_notes() {
    local reconstruction_id="$1" notes="$2"
    local json_body
    json_body=$(python3 -c "import json,sys; print(json.dumps({'notes': sys.argv[1]}))" "$notes" 2>/dev/null || echo '{"notes":""}')
    curl -sf -X PATCH "${API_URL}/api/v1/reconstructions/${reconstruction_id}/notes" \
        -H "Content-Type: application/json" \
        -d "$json_body" >/dev/null 2>&1 || true
}

wait_for_completion() {
    local reconstruction_id="$1"
    local poll_interval="${2:-30}"
    local max_wait="${3:-7200}"
    local elapsed=0

    while (( elapsed < max_wait )); do
        local status
        status=$(curl -sf "${API_URL}/api/v1/reconstructions/${reconstruction_id}/status" 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")

        if [[ "$status" == "completed" || "$status" == "failed" ]]; then
            echo "[orchestrator] Pipeline finished: status=$status (${elapsed}s)" >&2
            echo "$status"
            return 0
        fi

        echo "[orchestrator] Pipeline running (status=$status, ${elapsed}s elapsed)..." >&2
        sleep "$poll_interval"
        (( elapsed += poll_interval ))
    done

    echo "[orchestrator] WARNING: Pipeline did not finish within ${max_wait}s" >&2
    echo "timeout"
    return 1
}

# Launch evaluator agent in an OpenShell sandbox.
# The agent has autonomous access to:
#   - Backend API (fetch metrics, details, iterations via curl)
#   - Reconstruction output files (mounted at /sandbox/reconstruction_data/)
#   - Local LLM via OpenClaw agent framework
run_evaluator_agent() {
    local reconstruction_id="$1"
    local backend="$2"
    local iteration="$3"

    # Stage files for sandbox upload
    local stage_dir
    stage_dir=$(mktemp -d)

    # Write OpenClaw config with the configured model
    cat > "$stage_dir/openclaw.json" << CONFEOF
{
  "models": {
    "mode": "merge",
    "providers": {
      "openai": {
        "baseUrl": "https://inference.local/v1",
        "api": "openai-completions",
        "models": [
          {
            "id": "${OLLAMA_MODEL}",
            "name": "${OLLAMA_MODEL}",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 32768,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "openai/${OLLAMA_MODEL}"
      },
      "workspace": "/sandbox/stage"
    }
  },
  "tools": {
    "profile": "coding",
    "deny": ["web_fetch"]
  },
  "commands": {
    "native": "auto",
    "nativeSkills": "auto",
    "restart": true,
    "ownerDisplay": "raw"
  },
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "loopback",
    "auth": {
      "mode": "none"
    }
  }
}
CONFEOF

    # Copy policy and instructions
    cp "$SCRIPT_DIR/sandbox-policy.yaml" "$stage_dir/policy.yaml"
    cp "$SCRIPT_DIR/agent-prompt.md" "$stage_dir/instructions.md"

    # Compute thresholds
    local ssimloss_threshold
    ssimloss_threshold=$(python3 -c "print(round(1.0 - ${ACCEPT_SSIM}, 4))")
    local loss_threshold="${LOSS_THRESHOLD:-0.25}"

    # Write the task-specific message
    cat > "$stage_dir/message.txt" << MSGEOF
Evaluate reconstruction ${reconstruction_id} (backend: ${backend}).

## API Access
Use curl to fetch data from the backend API:
  curl -s http://172.20.0.1:${API_PORT}/api/v1/reconstructions/${reconstruction_id}
  curl -s http://172.20.0.1:${API_PORT}/api/v1/reconstructions/${reconstruction_id}/metrics
  curl -s http://172.20.0.1:${API_PORT}/api/v1/reconstructions/${reconstruction_id}/iterations

## Thresholds
For 3DGRUT: PSNR > ${ACCEPT_PSNR}, SSIM > ${ACCEPT_SSIM}
For fVDB: loss < ${loss_threshold}, ssimloss < ${ssimloss_threshold}

After your investigation, output ONLY a JSON verdict block.
MSGEOF

    # Write the runner script that executes inside the sandbox
    cat > "$stage_dir/run.sh" << 'RUNEOF'
#!/bin/bash
set -e

# Timing breadcrumbs visible in captured output
echo "[sandbox] boot_start=$(date +%s.%N)" >&2

mkdir -p /sandbox/.openclaw
cp /sandbox/stage/openclaw.json /sandbox/.openclaw/openclaw.json
export OPENAI_API_KEY=unused

# Remove OpenClaw template files that distract the agent
rm -f /sandbox/AGENTS.md /sandbox/SOUL.md /sandbox/BOOTSTRAP.md \
      /sandbox/USER.md /sandbox/TOOLS.md /sandbox/IDENTITY.md \
      /sandbox/HEARTBEAT.md /sandbox/MEMORY.md 2>/dev/null || true

cd /sandbox/stage

# Combine instructions + task into the agent message
INSTRUCTIONS=$(cat /sandbox/stage/instructions.md)
TASK=$(cat /sandbox/stage/message.txt)
MESSAGE="${INSTRUCTIONS}

---

${TASK}"

echo "[sandbox] openclaw_start=$(date +%s.%N)" >&2

openclaw agent --local --session-id evaluator \
    --message "$MESSAGE" \
    --json \
    --timeout "${1:-300}"

echo "[sandbox] openclaw_end=$(date +%s.%N)" >&2
RUNEOF
    chmod +x "$stage_dir/run.sh"

    # Build the openshell command
    local sb_cmd="openshell sandbox create"
    sb_cmd+=" --from openclaw"
    sb_cmd+=" --tty"
    sb_cmd+=" --no-keep"
    sb_cmd+=" --no-git-ignore"
    sb_cmd+=" --policy ${stage_dir}/policy.yaml"
    sb_cmd+=" --upload ${stage_dir}:/sandbox/stage"
    sb_cmd+=" -- bash /sandbox/stage/run.sh ${AGENT_TIMEOUT}"

    echo "[orchestrator] Staging evaluator sandbox (timeout=${AGENT_TIMEOUT}s)..." >&2

    # Run agent in sandbox (--tty forces PTY allocation, required by OpenClaw)
    local output=""
    local attempt=0
    local sb_output_file
    sb_output_file=$(mktemp)
    while (( attempt < RUN_AGENT_RETRIES )); do
        (( attempt++ ))
        local t_start t_end t_elapsed
        t_start=$(date +%s)
        echo "[orchestrator] Evaluator agent attempt ${attempt}/${RUN_AGENT_RETRIES} started at $(date +%H:%M:%S)..." >&2

        timeout $((AGENT_TIMEOUT + 120)) \
            $sb_cmd \
            > "$sb_output_file" 2>&1 || true

        # Strip carriage returns from PTY output
        sed -i 's/\r//g' "$sb_output_file" 2>/dev/null || true

        t_end=$(date +%s)
        t_elapsed=$(( t_end - t_start ))

        output=$(<"$sb_output_file")
        local output_len=${#output}

        echo "[orchestrator] Sandbox attempt ${attempt} finished in ${t_elapsed}s (${output_len} bytes)" >&2

        # Extract sandbox timing breadcrumbs if present
        local boot_start openclaw_start openclaw_end
        boot_start=$(grep -oP '\[sandbox\] boot_start=\K[0-9.]+' "$sb_output_file" 2>/dev/null | head -1 || true)
        openclaw_start=$(grep -oP '\[sandbox\] openclaw_start=\K[0-9.]+' "$sb_output_file" 2>/dev/null | head -1 || true)
        openclaw_end=$(grep -oP '\[sandbox\] openclaw_end=\K[0-9.]+' "$sb_output_file" 2>/dev/null | head -1 || true)
        if [[ -n "$boot_start" ]]; then
            echo "[orchestrator]   sandbox timing: boot_start=$boot_start openclaw_start=${openclaw_start:-never} openclaw_end=${openclaw_end:-never}" >&2
            if [[ -n "$boot_start" && -n "$openclaw_start" ]]; then
                local setup_secs
                setup_secs=$(python3 -c "print(f'{float(${openclaw_start}) - float(${boot_start}):.1f}')" 2>/dev/null || echo "?")
                echo "[orchestrator]   sandbox boot→openclaw: ${setup_secs}s" >&2
            fi
            if [[ -n "$openclaw_start" && -n "$openclaw_end" ]]; then
                local agent_secs
                agent_secs=$(python3 -c "print(f'{float(${openclaw_end}) - float(${openclaw_start}):.1f}')" 2>/dev/null || echo "?")
                echo "[orchestrator]   openclaw agent duration: ${agent_secs}s" >&2
            fi
        else
            echo "[orchestrator]   (no sandbox timing breadcrumbs — run.sh may not have started)" >&2
        fi

        # Check if we got meaningful output
        if (( output_len < 100 )); then
            echo "[orchestrator] Agent output too short (${output_len} bytes)" >&2
            if (( output_len > 0 )); then
                echo "[orchestrator]   output preview: $(head -5 "$sb_output_file" | head -c 500)" >&2
            fi
            echo "[orchestrator]   retrying in 60s..." >&2
            sleep 60
            continue
        fi

        # Check for empty payloads (agent ran but LLM never responded)
        if echo "$output" | grep -q '"payloads": \[\]'; then
            echo "[orchestrator] Agent returned empty payloads (LLM did not respond), retrying..." >&2
            sleep 5
            continue
        fi

        echo "[orchestrator] Agent produced ${output_len} bytes of output" >&2
        break
    done
    rm -f "$sb_output_file"

    # Cleanup staging directory
    rm -rf "$stage_dir"

    echo "$output"
}

# Parse LLM verdict from response text
# OpenClaw --json wraps the LLM response in {"payloads":[{"text":"..."}],...}
# Extraction handled by extract_agent_output.py to avoid quoting issues.
EXTRACT_SCRIPT="${SCRIPT_DIR}/extract_agent_output.py"

extract_verdict() {
    python3 "$EXTRACT_SCRIPT" verdict 2>/dev/null || echo "UNKNOWN"
}

extract_reason() {
    python3 "$EXTRACT_SCRIPT" reason 2>/dev/null || echo "No analysis available"
}

extract_params() {
    local current_backend="${1:-}"
    python3 "$EXTRACT_SCRIPT" params "$current_backend" 2>/dev/null || echo "{}"
}

# Default param escalation when LLM returns empty params
default_escalation() {
    local reconstruction_id="$1"
    curl -sf "${API_URL}/api/v1/reconstructions/${reconstruction_id}" 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
pp = d.get('processing_params', {})
backend = pp.get('reconstruction_backend', '${CURRENT_BACKEND}')
result = {}
if backend == 'fvdb':
    epochs = pp.get('fvdb_max_epochs') or 30
    ds = pp.get('fvdb_image_downsample_factor') or 4
    if epochs < 60: result['fvdb_max_epochs'] = min(epochs * 2, 120)
    elif ds > 1: result['fvdb_image_downsample_factor'] = max(ds // 2, 1)
    else: result['fvdb_max_epochs'] = min(epochs + 30, 120)
elif backend == '3dgrut':
    iters = pp.get('grut_n_iterations') or 5000
    ds = pp.get('grut_downsample_factor') or 4
    if iters < 10000: result['grut_n_iterations'] = min(iters * 2, 50000)
    elif ds > 1: result['grut_downsample_factor'] = max(ds // 2, 1)
    else: result['grut_n_iterations'] = min(iters + 5000, 50000)
print(json.dumps(result))
" 2>/dev/null || echo '{}'
}

# ── Health Check ────────────────────────────────────────

echo "[orchestrator] Checking backend health..."
if ! curl -sf "${API_URL}/health" >/dev/null 2>&1; then
    echo "[orchestrator] ERROR: Backend not running at $API_URL"
    exit 1
fi
echo "[orchestrator] Backend healthy."
echo ""

# ── Step 1: Start Reconstruction ───────────────────────

echo "============================================"
echo " Step 1: Start Reconstruction"
echo "============================================"
echo ""

update_workflow '{"status":"running","current_step":"starting reconstruction","iteration":1,"current_agent":"runner"}'

# Build form args into an array for safe quoting
CURL_ARGS=()
if [[ "$DATASET_MODE" == "true" ]]; then
    API_ENDPOINT="${API_URL}/api/v1/reconstructions/from-dataset"
    CURL_ARGS+=(-F "dataset_name=${DATASET_NAME}" -F "name=${SCENE_NAME}")
else
    API_ENDPOINT="${API_URL}/api/v1/reconstructions/upload"
    CURL_ARGS+=(-F "file=@${VIDEO_PATH}" -F "name=${SCENE_NAME}")
    CURL_ARGS+=(-F "frame_rate=${INITIAL_FRAME_RATE:-1.0}")
    CURL_ARGS+=(-F "splat_only_mode=${INITIAL_SPLAT_ONLY:-false}")
fi

CURL_ARGS+=(-F "reconstruction_backend=${CURRENT_BACKEND}")
if [[ "$CURRENT_BACKEND" == "fvdb" ]]; then
    CURL_ARGS+=(-F "fvdb_max_epochs=${INITIAL_FVDB_EPOCHS:-5}")
    CURL_ARGS+=(-F "fvdb_image_downsample_factor=${INITIAL_FVDB_DS:-8}")
else
    CURL_ARGS+=(-F "grut_n_iterations=${INITIAL_GRUT_ITERS:-500}")
    CURL_ARGS+=(-F "grut_downsample_factor=${INITIAL_GRUT_DS:-8}")
fi

echo "[orchestrator] Starting reconstruction..."
START_RESPONSE=$(curl -sf "${CURL_ARGS[@]}" "$API_ENDPOINT" 2>&1) || true

RECONSTRUCTION_ID=$(echo "$START_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null || true)

if [[ -z "$RECONSTRUCTION_ID" ]]; then
    echo "[orchestrator] ERROR: Failed to start reconstruction"
    echo "[orchestrator] Response: $(echo "$START_RESPONSE" | head -c 500)"
    update_workflow '{"status":"failed","error_message":"Failed to start reconstruction"}'
    exit 1
fi

echo "[orchestrator] Reconstruction ID: $RECONSTRUCTION_ID"
update_workflow "{\"reconstruction_id\":\"${RECONSTRUCTION_ID}\",\"current_step\":\"waiting for reconstruction\"}"

# Wait for completion
JOB_STATUS=$(wait_for_completion "$RECONSTRUCTION_ID" 30 7200 | tail -1 || true)
echo "[orchestrator] Status: $JOB_STATUS"

if [[ "$JOB_STATUS" == "failed" ]]; then
    echo "[orchestrator] Initial reconstruction failed."
    update_workflow '{"status":"failed","error_message":"Initial reconstruction failed"}'
    exit 1
fi
if [[ "$JOB_STATUS" == "timeout" ]]; then
    echo "[orchestrator] Initial reconstruction timed out."
    update_workflow '{"status":"failed","error_message":"Reconstruction timed out"}'
    exit 1
fi

# ── Step 2+: Evaluate → Retry Loop ─────────────────────

for ((i=1; i<=MAX_ITERATIONS; i++)); do
    echo ""
    echo "============================================"
    echo " Evaluation $i / $MAX_ITERATIONS"
    echo "============================================"
    echo ""

    update_workflow "{\"current_step\":\"evaluating metrics\",\"iteration\":${i},\"current_agent\":\"evaluator\"}"

    echo "[orchestrator] Launching evaluator agent in sandbox..."
    EVAL_OUTPUT=$(run_evaluator_agent "$RECONSTRUCTION_ID" "$CURRENT_BACKEND" "$i")
    echo "[orchestrator] Evaluator agent output:"
    echo "$EVAL_OUTPUT"
    echo ""

    VERDICT=$(echo "$EVAL_OUTPUT" | extract_verdict)
    REASON=$(echo "$EVAL_OUTPUT" | extract_reason)

    echo "[orchestrator] Verdict: $VERDICT"
    echo "[orchestrator] Reason: $REASON"

    # Save evaluator notes
    save_notes "$RECONSTRUCTION_ID" "[Eval $i] Verdict: ${VERDICT}. ${REASON}"
    update_workflow "{\"last_verdict\":\"${VERDICT}\",\"last_reason\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${REASON}" 2>/dev/null || echo '""')}"

    # Record verdict on the latest iteration
    ITER_COUNT=$(curl -sf "${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}/iterations" 2>/dev/null \
        | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('iterations',[])))" 2>/dev/null || echo "0")
    if [[ "$ITER_COUNT" -gt 0 ]]; then
        VERDICT_JSON=$(python3 -c "import json,sys; print(json.dumps({'verdict':sys.argv[1],'reason':sys.argv[2]}))" "$VERDICT" "$REASON" 2>/dev/null || echo '{}')
        curl -sf -X PATCH "${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}/iterations/${ITER_COUNT}/verdict" \
            -H "Content-Type: application/json" -d "$VERDICT_JSON" >/dev/null 2>&1 || true
    fi

    # ACCEPT → done
    if [[ "$VERDICT" == "ACCEPT" ]]; then
        echo "[orchestrator] Reconstruction accepted!"
        update_workflow '{"status":"completed","current_step":"accepted"}'
        break
    fi

    # Unknown verdict → stop
    if [[ "$VERDICT" != "ITERATE" ]]; then
        echo "[orchestrator] Unknown verdict ($VERDICT). Stopping."
        update_workflow '{"status":"completed","current_step":"unknown verdict - stopped"}'
        break
    fi

    # Max iterations → stop
    if [[ $i -eq $MAX_ITERATIONS ]]; then
        echo "[orchestrator] Max iterations reached. Using last result."
        update_workflow '{"status":"completed","current_step":"max iterations reached"}'
        break
    fi

    # Extract suggested params
    NEW_PARAMS=$(echo "$EVAL_OUTPUT" | extract_params "$CURRENT_BACKEND")
    echo "[orchestrator] LLM suggested params: $NEW_PARAMS"

    # Fallback to default escalation
    if [[ "$NEW_PARAMS" == "{}" || -z "$NEW_PARAMS" ]]; then
        echo "[orchestrator] No params from LLM — using default escalation"
        NEW_PARAMS=$(default_escalation "$RECONSTRUCTION_ID")
        echo "[orchestrator] Default escalation: $NEW_PARAMS"
    fi

    # Track backend switches
    SWITCHED_BACKEND=$(echo "$NEW_PARAMS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('reconstruction_backend',''))" 2>/dev/null || true)
    if [[ -n "$SWITCHED_BACKEND" ]]; then
        echo "[orchestrator] Backend switch: $CURRENT_BACKEND → $SWITCHED_BACKEND"
        CURRENT_BACKEND="$SWITCHED_BACKEND"
    fi

    echo ""
    echo "============================================"
    echo " Retry $((i+1)): New parameters"
    echo "============================================"
    echo "[orchestrator] Retrying with: $NEW_PARAMS"

    update_workflow "{\"current_step\":\"retrying with new params\",\"iteration\":$((i+1)),\"current_agent\":\"runner\"}"

    # Retry via API
    RETRY_RESPONSE=$(curl -sf -X POST "${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}/retry" \
        -H 'Content-Type: application/json' \
        -d "{\"params\": ${NEW_PARAMS}}" 2>&1) || true

    RETRY_STATUS=$(echo "$RETRY_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
    echo "[orchestrator] Retry queued: $RETRY_STATUS"

    if [[ "$RETRY_STATUS" == "unknown" || -z "$RETRY_STATUS" ]]; then
        echo "[orchestrator] ERROR: Retry failed. Response: $(echo "$RETRY_RESPONSE" | head -c 500)"
        update_workflow '{"status":"failed","error_message":"Retry API call failed"}'
        break
    fi

    # Wait for retry to complete
    JOB_STATUS=$(wait_for_completion "$RECONSTRUCTION_ID" 30 7200 | tail -1 || true)
    echo "[orchestrator] Retry result: $JOB_STATUS"

    if [[ "$JOB_STATUS" == "failed" ]]; then
        echo "[orchestrator] Retry failed."
        update_workflow '{"status":"failed","error_message":"Retry reconstruction failed"}'
        break
    fi
    if [[ "$JOB_STATUS" == "timeout" ]]; then
        echo "[orchestrator] Retry timed out."
        update_workflow '{"status":"failed","error_message":"Retry timed out"}'
        break
    fi
done

# ── Final Result ────────────────────────────────────────

echo ""
echo "============================================"
echo " Final Result"
echo "============================================"
echo ""
curl -sf "${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}" 2>/dev/null | python3 -m json.tool 2>/dev/null || true
echo ""
echo "[orchestrator] Reconstruction ID: $RECONSTRUCTION_ID"
echo "[orchestrator] Done."
