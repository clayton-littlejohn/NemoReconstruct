#!/usr/bin/env bash
# orchestrate.sh — Multi-agent iterative reconstruction workflow
#
# Drives two sandboxed agents in a loop:
#   1. Runner agent: executes the reconstruction pipeline
#   2. Evaluator agent: analyzes metrics and suggests parameter improvements
#
# The backend API (on the host) is the shared coordination layer.
# Each agent runs in its own isolated OpenShell sandbox.
#
# Usage:
#   ./nemoclaw/orchestrate.sh <video_path> <scene_name> [max_iterations]
#
# Example:
#   ./nemoclaw/orchestrate.sh ~/videos/my_scene.MOV "kitchen-scan" 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
POLICY="$SCRIPT_DIR/sandbox-policy.yaml"
OPENCLAW_CONFIG="$SCRIPT_DIR/sandbox-openclaw.json"
API_HOST="172.20.0.1"
API_PORT="8010"
API_URL="http://${API_HOST}:${API_PORT}"
WORKFLOW_API_URL="${WORKFLOW_API_URL:-http://127.0.0.1:${API_PORT}}"
WORKFLOW_ID="${WORKFLOW_ID:-}"

# Support two modes:
#   1. Video mode:   orchestrate.sh <video_path> <scene_name> [max_iterations]
#   2. Dataset mode:  orchestrate.sh --dataset <dataset_name> <scene_name> [max_iterations]
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

TIMEOUT="${AGENT_TIMEOUT:-1200}"
ACCEPT_PSNR="${ACCEPT_PSNR_THRESHOLD:-25.0}"
ACCEPT_SSIM="${ACCEPT_SSIM_THRESHOLD:-0.85}"

# Validate inputs
if [[ "$DATASET_MODE" == "false" ]]; then
    if [[ ! -f "$VIDEO_PATH" ]]; then
        echo "Error: Video file not found: $VIDEO_PATH"
        exit 1
    fi
    # Resolve to absolute path
    VIDEO_PATH="$(cd "$(dirname "$VIDEO_PATH")" && pwd)/$(basename "$VIDEO_PATH")"
fi

echo "============================================"
echo " NemoClaw Multi-Agent Reconstruction"
echo "============================================"
if [[ "$DATASET_MODE" == "true" ]]; then
echo " Dataset:        $DATASET_NAME"
else
echo " Video:          $VIDEO_PATH"
fi
echo " Scene:          $SCENE_NAME"
echo " Max iterations: $MAX_ITERATIONS"
echo " PSNR threshold: $ACCEPT_PSNR"
echo " SSIM threshold: $ACCEPT_SSIM"
echo " API:            $API_URL"
echo "============================================"
echo ""

# Health check
echo "[orchestrator] Checking backend health..."
HEALTH=$(curl -sf "${API_URL}/health" 2>/dev/null || echo "FAILED")
if [[ "$HEALTH" == "FAILED" ]]; then
    echo "[orchestrator] ERROR: Backend is not running at $API_URL"
    echo "[orchestrator] Start it with: make backend-dev"
    exit 1
fi
echo "[orchestrator] Backend healthy: $HEALTH"
echo ""

# Helper: run a sandboxed agent with a message
run_agent() {
    local session_id="$1"
    local message="$2"
    local prompt_file="$3"

    # Build a lightweight staging dir with only what the sandbox needs:
    # the nemoclaw config files and the video. This avoids uploading the
    # entire repo (backend/storage alone is multi-GB).
    local stage_dir
    stage_dir=$(mktemp -d)
    # Clean up staging dir on return
    trap "rm -rf '$stage_dir'" RETURN

    cp -a "$SCRIPT_DIR" "$stage_dir/nemoclaw"
    # Only link the video file if we have one (video mode, not dataset mode)
    if [[ -n "$VIDEO_PATH" && -f "$VIDEO_PATH" ]]; then
        ln -s "$VIDEO_PATH" "$stage_dir/$(basename "$VIDEO_PATH")"
    fi

    # Use 'timeout' as a hard kill-switch since openclaw's --timeout is unreliable.
    # Allow TIMEOUT for agent work + 120s overhead for upload/bootstrap.
    local hard_timeout=$(( TIMEOUT + 120 ))

    timeout --signal=KILL "${hard_timeout}s" \
    openshell sandbox create \
        --from openclaw \
        --policy "$POLICY" \
        --no-git-ignore \
        --upload "$stage_dir:/sandbox/NemoReconstruct" \
        -- bash -c "
mkdir -p /sandbox/.openclaw
cp /sandbox/NemoReconstruct/nemoclaw/sandbox-openclaw.json /sandbox/.openclaw/openclaw.json
export OPENAI_API_KEY=unused
cd /sandbox/NemoReconstruct
openclaw agent --local --session-id $session_id \
    --message \"$(echo "$message" | sed 's/"/\\"/g')\" \
    --json --timeout $TIMEOUT
" || true
}

# Wait for reconstruction to reach a terminal state (completed or failed).
# The runner agent may time out before the pipeline finishes training, so
# the orchestrator must poll independently before proceeding to the evaluator.
wait_for_completion() {
    local reconstruction_id="$1"
    local poll_interval="${2:-30}"
    local max_wait="${3:-2400}"  # 40 minutes default
    local elapsed=0

    while (( elapsed < max_wait )); do
        local status
        status=$(curl -sf "${API_URL}/api/v1/reconstructions/${reconstruction_id}/status" 2>/dev/null \
            | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")

        if [[ "$status" == "completed" || "$status" == "failed" ]]; then
            echo "[orchestrator] Pipeline finished with status: $status (waited ${elapsed}s)"
            echo "$status"
            return 0
        fi

        echo "[orchestrator] Pipeline still running (status=$status, ${elapsed}s elapsed)..."
        sleep "$poll_interval"
        (( elapsed += poll_interval ))
    done

    echo "[orchestrator] WARNING: Pipeline did not finish within ${max_wait}s"
    echo "timeout"
    return 1
}

# Extract reconstruction ID from agent output
extract_id() {
    grep -oE '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' | head -1 || true
}

# Extract evaluator verdict from agent output
extract_verdict() {
    python3 -c "
import sys, json, re
text = sys.stdin.read()

def extract_json_objects(s):
    results = []
    i = 0
    while i < len(s):
        if s[i] == '{':
            for end in range(len(s)-1, i, -1):
                if s[end] == '}':
                    try:
                        results.append(json.loads(s[i:end+1]))
                        i = end + 1
                        break
                    except: pass
            else:
                i += 1
                continue
            continue
        i += 1
    return results

def find_all_strings(obj):
    if isinstance(obj, str): return [obj]
    if isinstance(obj, list): return [s for item in obj for s in find_all_strings(item)]
    if isinstance(obj, dict): return [s for v in obj.values() for s in find_all_strings(v)]
    return []

cleaned = re.sub(r'\`\`\`[a-z]*\n?', '', text)
objects = extract_json_objects(cleaned)
for obj in list(objects):
    for s in find_all_strings(obj):
        if '{' in s:
            objects.extend(extract_json_objects(s))

for obj in reversed(objects):
    v = obj.get('verdict', '').upper()
    if v in ('ACCEPT', 'ITERATE'):
        print(v)
        sys.exit(0)

if re.search(r'\bACCEPT\b', text):
    print('ACCEPT')
elif re.search(r'\bITERATE\b', text):
    print('ITERATE')
else:
    print('UNKNOWN')
" 2>/dev/null || echo "UNKNOWN"
}

# Extract evaluator reasoning/notes from agent output
extract_reason() {
    python3 -c "
import sys, json, re
text = sys.stdin.read()
cleaned = re.sub(r'\`\`\`[a-z]*\n?', '', text)

def extract_json_objects(s):
    results = []
    depth = 0
    start = None
    for i, c in enumerate(s):
        if c == '{':
            if depth == 0: start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start is not None:
                try: results.append(json.loads(s[start:i+1]))
                except: pass
                start = None
    return results

def find_all_strings(obj):
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, list):
        r = []
        for item in obj:
            r.extend(find_all_strings(item))
        return r
    if isinstance(obj, dict):
        r = []
        for v in obj.values():
            r.extend(find_all_strings(v))
        return r
    return []

objects = extract_json_objects(cleaned)
all_strings = []
for obj in list(objects):
    for s in find_all_strings(obj):
        all_strings.append(s)
        if '{' in s:
            objects.extend(extract_json_objects(s))

# Strategy 1: find reason key in parsed JSON objects
for obj in reversed(objects):
    for key in ('reason', 'reasoning', 'analysis', 'explanation', 'notes'):
        if key in obj and isinstance(obj[key], str):
            print(obj[key][:500])
            sys.exit(0)

# Strategy 2: regex extract reason from extracted strings (handles truncated JSON)
for s in reversed(all_strings):
    m = re.search(r'\"reason\"\s*:\s*\"((?:[^\"\\\\]|\\\\.)*)\"', s)
    if m:
        val = m.group(1).replace('\\\\n', ' ').replace('\\\\\"', '\"').strip()
        if len(val) > 5:
            print(val[:500])
            sys.exit(0)

# Strategy 3: extract the last substantial line (skip JSON fragments)
lines = [l.strip() for l in text.splitlines() if l.strip() and len(l.strip()) > 3 and not re.match(r'^[\[\]{},\s]*$', l.strip())]
if lines:
    print(lines[-1][:500])
else:
    print('No analysis available')
" 2>/dev/null || echo "No analysis available"
}

# Save notes to a reconstruction via the API
save_notes() {
    local reconstruction_id="$1"
    local notes="$2"
    curl -sf -X PATCH "${API_URL}/api/v1/reconstructions/${reconstruction_id}/notes" \
        -H "Content-Type: application/json" \
        -d "$(python3 -c "import json; print(json.dumps({'notes': $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$notes")}))")" \
        >/dev/null 2>&1 || true
}

# Update workflow state (if WORKFLOW_ID is set)
update_workflow() {
    if [[ -z "$WORKFLOW_ID" ]]; then return; fi
    local json_body="$1"
    curl -sf -X PATCH "${WORKFLOW_API_URL}/api/v1/workflows/${WORKFLOW_ID}/state" \
        -H "Content-Type: application/json" \
        -d "$json_body" \
        >/dev/null 2>&1 || true
}

# Extract suggested params JSON from evaluator output
extract_params() {
    python3 -c "
import sys, json, re

VALID_KEYS = {'frame_rate','sequential_matcher_overlap','colmap_mapper_type',
              'colmap_max_num_features','reconstruction_backend',
              'fvdb_max_epochs','fvdb_sh_degree','fvdb_image_downsample_factor',
              'grut_n_iterations','grut_render_method','grut_strategy',
              'grut_downsample_factor','splat_only_mode'}
text = sys.stdin.read()

# Strip markdown code fences
cleaned = re.sub(r'\`\`\`[a-z]*\n?', '', text)

def extract_json_objects(s):
    results = []
    i = 0
    while i < len(s):
        if s[i] == '{':
            for end in range(len(s)-1, i, -1):
                if s[end] == '}':
                    try:
                        results.append(json.loads(s[i:end+1]))
                        i = end + 1
                        break
                    except: pass
            else:
                i += 1
                continue
            continue
        i += 1
    return results

def find_all_strings(obj):
    if isinstance(obj, str): return [obj]
    if isinstance(obj, list): return [s for item in obj for s in find_all_strings(item)]
    if isinstance(obj, dict): return [s for v in obj.values() for s in find_all_strings(v)]
    return []

objects = extract_json_objects(cleaned)
for obj in list(objects):
    for s in find_all_strings(obj):
        if '{' in s:
            objects.extend(extract_json_objects(s))

# Strategy 1: find a top-level 'params' dict in any JSON object
for obj in reversed(objects):
    if isinstance(obj.get('params'), dict):
        filtered = {k: v for k, v in obj['params'].items() if k in VALID_KEYS}
        if filtered:
            print(json.dumps(filtered))
            sys.exit(0)

# Strategy 2: find 'suggested_parameters' or similar keys
for obj in reversed(objects):
    for key in ('suggested_parameters', 'recommended_params', 'new_params', 'parameters'):
        if isinstance(obj.get(key), dict):
            filtered = {k: v for k, v in obj[key].items() if k in VALID_KEYS}
            if filtered:
                print(json.dumps(filtered))
                sys.exit(0)

# Strategy 3: find any JSON object that has valid param keys directly
for obj in reversed(objects):
    filtered = {k: v for k, v in obj.items() if k in VALID_KEYS}
    if filtered:
        print(json.dumps(filtered))
        sys.exit(0)

# Strategy 4: look for individual param=value patterns in text
params = {}
for key in VALID_KEYS:
    m = re.search(rf'{key}[\"\\s:=]+\\s*([\\d.]+|true|false)', cleaned, re.IGNORECASE)
    if m:
        val = m.group(1)
        if val.lower() in ('true','false'):
            params[key] = val.lower() == 'true'
        elif '.' in val:
            params[key] = float(val)
        else:
            params[key] = int(val)
if params:
    print(json.dumps(params))
else:
    print('{}')
" 2>/dev/null || echo "{}"
}

echo "============================================"
echo " Iteration 1: Initial Run"
echo "============================================"

if [[ "$DATASET_MODE" == "true" ]]; then
    # Dataset mode: runner uses the from-dataset API
    RUNNER_MSG="Check the backend health at ${API_URL}. Then start a reconstruction from the pre-existing dataset '${DATASET_NAME}' using curl to POST to ${API_URL}/api/v1/reconstructions/from-dataset. Use these form fields: dataset_name=${DATASET_NAME}, name=${SCENE_NAME}, reconstruction_backend=${INITIAL_BACKEND:-fvdb}. After posting, extract the reconstruction ID from the response. Then poll the status using a shell loop: run 'while true; do sleep 30; curl -s ${API_URL}/api/v1/reconstructions/<ID>/status; done' and wait for it to show completed or failed. Report the reconstruction ID and final status."
else
    # Video mode: runner uploads the video file
    VIDEO_BASENAME="$(basename "$VIDEO_PATH")"
    VIDEO_EXT="${VIDEO_BASENAME##*.}"
    SANDBOX_VIDEO="/sandbox/NemoReconstruct/${VIDEO_BASENAME}"

    RUNNER_MSG="Check the backend health at ${API_URL}. Then upload the video at ${SANDBOX_VIDEO} with name '${SCENE_NAME}' using curl to the API at ${API_URL}. Use these form fields: file=@${SANDBOX_VIDEO}, name=${SCENE_NAME}, reconstruction_backend=${INITIAL_BACKEND:-3dgrut}, grut_n_iterations=${INITIAL_GRUT_ITERS:-5000}, grut_downsample_factor=${INITIAL_GRUT_DS:-4}, frame_rate=${INITIAL_FRAME_RATE:-1.0}, splat_only_mode=${INITIAL_SPLAT_ONLY:-false}. After uploading, poll the status using a shell loop: run 'while true; do sleep 30; curl -s ${API_URL}/api/v1/reconstructions/<ID>/status; done' and wait for it to show completed or failed. Report the reconstruction ID and final status."
fi

echo "[orchestrator] Starting Runner agent..."
echo ""

update_workflow '{"status":"running","current_agent":"runner","current_step":"uploading and reconstructing","iteration":1}'

RUNNER_OUTPUT=$(run_agent "runner-iter-1" "$RUNNER_MSG" "runner_prompt.md" 2>&1)
echo "$RUNNER_OUTPUT"

RECONSTRUCTION_ID=$(echo "$RUNNER_OUTPUT" | extract_id)

if [[ -z "$RECONSTRUCTION_ID" ]]; then
    echo ""
    echo "[orchestrator] ERROR: Could not extract reconstruction ID from runner output"
    update_workflow '{"status":"failed","error_message":"Could not extract reconstruction ID"}'
    exit 1
fi

echo ""
echo "[orchestrator] Reconstruction ID: $RECONSTRUCTION_ID"
update_workflow "{\"reconstruction_id\":\"${RECONSTRUCTION_ID}\",\"current_step\":\"waiting for reconstruction\"}"

# Wait for the pipeline to actually reach a terminal state.
# The runner agent may time out before training completes.
JOB_STATUS=$(wait_for_completion "$RECONSTRUCTION_ID" 30 7200 | tail -1 || true)
echo "[orchestrator] Job status: $JOB_STATUS"

if [[ "$JOB_STATUS" == "failed" ]]; then
    echo "[orchestrator] Initial run failed. Check the run log for diagnostics."
    update_workflow '{"status":"failed","error_message":"Initial reconstruction failed"}'
    exit 1
fi

if [[ "$JOB_STATUS" == "timeout" ]]; then
    echo "[orchestrator] Initial run timed out waiting for pipeline. Stopping."
    update_workflow '{"status":"failed","error_message":"Pipeline did not complete within timeout"}'
    exit 1
fi

# Iterative evaluation loop
for ((i=1; i<=MAX_ITERATIONS; i++)); do
    echo ""
    echo "============================================"
    echo " Evaluation $i / $MAX_ITERATIONS"
    echo "============================================"

    EVAL_MSG="Evaluate reconstruction ${RECONSTRUCTION_ID}. Use curl to call the API at ${API_URL}. Fetch the reconstruction details from /api/v1/reconstructions/${RECONSTRUCTION_ID} and the training metrics from /api/v1/reconstructions/${RECONSTRUCTION_ID}/metrics. Also fetch the iteration history from /api/v1/reconstructions/${RECONSTRUCTION_ID}/iterations to see what parameters and metrics each previous iteration produced — use this to avoid repeating failed approaches and to identify improvement trends. Analyze the quality. Your final output MUST be exactly one line of JSON like: {\"verdict\": \"ACCEPT\", \"reason\": \"...\"} or {\"verdict\": \"ITERATE\", \"reason\": \"...\", \"params\": {\"fvdb_max_epochs\": 20}}. Use ACCEPT if quality is sufficient (for 3DGRUT: psnr > ${ACCEPT_PSNR} and ssim > ${ACCEPT_SSIM}; for fVDB: loss < 0.25 and ssimloss > ${ACCEPT_SSIM}), otherwise ITERATE with suggested param changes."

    echo "[orchestrator] Starting Evaluator agent..."
    echo ""

    update_workflow "{\"current_agent\":\"evaluator\",\"current_step\":\"analyzing metrics\",\"iteration\":${i}}"

    # Retry evaluator up to 3 times if it returns UNKNOWN (Ollama 500 / empty payloads)
    VERDICT="UNKNOWN"
    REASON=""
    for ((eval_try=1; eval_try<=3; eval_try++)); do
        EVAL_OUTPUT=$(run_agent "eval-iter-${i}-try-${eval_try}" "$EVAL_MSG" "evaluator_prompt.md" 2>&1)
        echo "$EVAL_OUTPUT"

        VERDICT=$(echo "$EVAL_OUTPUT" | extract_verdict || true)
        REASON=$(echo "$EVAL_OUTPUT" | extract_reason || true)
        echo ""
        echo "[orchestrator] Verdict: ${VERDICT:-UNKNOWN} (attempt ${eval_try}/3)"
        echo "[orchestrator] Reason: ${REASON}"

        if [[ "$VERDICT" == "ACCEPT" || "$VERDICT" == "ITERATE" ]]; then
            break
        fi

        if [[ $eval_try -lt 3 ]]; then
            echo "[orchestrator] Evaluator returned UNKNOWN, retrying (attempt $((eval_try+1))/3)..."
            update_workflow "{\"current_step\":\"retrying evaluation (attempt $((eval_try+1)))\",\"iteration\":${i}}"
            sleep 5
        fi
    done

    # Save evaluator notes to the reconstruction
    save_notes "$RECONSTRUCTION_ID" "[Eval $i] Verdict: ${VERDICT}. ${REASON}"
    update_workflow "{\"last_verdict\":\"${VERDICT}\",\"last_reason\":$(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "${REASON}" 2>/dev/null || echo '""')}"

    # Record the verdict on the latest iteration record
    ITER_COUNT=$(curl -sf "${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}/iterations" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('iterations',[])))" 2>/dev/null || echo "0")
    if [[ "$ITER_COUNT" -gt 0 ]]; then
        VERDICT_JSON=$(python3 -c "import json,sys; print(json.dumps({'verdict':sys.argv[1],'reason':sys.argv[2]}))" "$VERDICT" "$REASON" 2>/dev/null || echo '{}')
        curl -sf -X PATCH "${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}/iterations/${ITER_COUNT}/verdict" \
            -H "Content-Type: application/json" -d "$VERDICT_JSON" >/dev/null 2>&1 || true
    fi

    if [[ "$VERDICT" == "ACCEPT" ]]; then
        echo "[orchestrator] Evaluator accepted the reconstruction. Done!"
        update_workflow '{"status":"completed","current_agent":null,"current_step":"accepted"}'
        break
    fi

    if [[ "$VERDICT" != "ITERATE" ]]; then
        echo "[orchestrator] Could not parse evaluator verdict. Stopping."
        update_workflow '{"status":"completed","current_agent":null,"current_step":"unknown verdict - stopped"}'
        break
    fi

    if [[ $i -eq $MAX_ITERATIONS ]]; then
        echo "[orchestrator] Max iterations reached. Using last result."
        update_workflow '{"status":"completed","current_agent":null,"current_step":"max iterations reached"}'
        break
    fi

    # Extract suggested parameters
    NEW_PARAMS=$(echo "$EVAL_OUTPUT" | extract_params || echo "{}")
    echo "[orchestrator] Suggested params: $NEW_PARAMS"

    # Save suggested params as notes
    save_notes "$RECONSTRUCTION_ID" "[Eval $i] Verdict: ${VERDICT}. Suggested params: ${NEW_PARAMS}. ${REASON}"

    echo ""
    echo "============================================"
    echo " Iteration $((i+1)): Retry with new parameters"
    echo "============================================"

    RETRY_MSG="Retry reconstruction ${RECONSTRUCTION_ID} with new parameters. Use curl to POST to ${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}/retry with body: {\"params\": ${NEW_PARAMS}}. After posting, poll the status using a shell loop: run 'while true; do sleep 30; STATUS=\$(curl -s ${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}/status); echo \$STATUS; echo \$STATUS | grep -q '\"completed\"\|\"failed\"' && break; done' and wait for completion. Report the final status."

    echo "[orchestrator] Starting Runner agent (retry)..."
    echo ""

    update_workflow "{\"current_agent\":\"runner\",\"current_step\":\"retrying with new params\",\"iteration\":$((i+1))}"

    RETRY_OUTPUT=$(run_agent "runner-iter-$((i+1))" "$RETRY_MSG" "runner_prompt.md" 2>&1)
    echo "$RETRY_OUTPUT"

    # Wait for the pipeline to finish (agent may time out before training ends)
    JOB_STATUS=$(wait_for_completion "$RECONSTRUCTION_ID" 30 7200 | tail -1 || true)
    echo ""
    echo "[orchestrator] Job status after retry: $JOB_STATUS"

    if [[ "$JOB_STATUS" == "failed" ]]; then
        echo "[orchestrator] Retry failed. Stopping."
        update_workflow '{"status":"failed","error_message":"Retry reconstruction failed"}'
        break
    fi

    if [[ "$JOB_STATUS" == "timeout" ]]; then
        echo "[orchestrator] Retry timed out waiting for pipeline. Stopping."
        update_workflow '{"status":"failed","error_message":"Retry pipeline did not complete within timeout"}'
        break
    fi
done

echo ""
echo "============================================"
echo " Final Result"
echo "============================================"
echo ""
curl -sf "${API_URL}/api/v1/reconstructions/${RECONSTRUCTION_ID}" 2>/dev/null | python3 -m json.tool
echo ""
echo "[orchestrator] Reconstruction ID: $RECONSTRUCTION_ID"
echo "[orchestrator] Done."
