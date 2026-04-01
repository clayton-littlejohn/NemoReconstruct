# NemoClaw — Iterative Reconstruction Orchestrator

NemoClaw drives iterative 3D reconstruction pipelines with an LLM-powered evaluator agent. The flow:

```
Start Reconstruction → Evaluator Agent (in sandbox) → Retry with Recommended Params → Repeat
```

1. **Start** — kicks off a reconstruction via the NemoReconstruct backend API (direct curl call)
2. **Evaluate** — launches an evaluator agent in an OpenShell sandbox. The agent autonomously fetches metrics from the API, inspects reconstruction output files, and uses a local LLM to decide: ACCEPT or ITERATE
3. **Retry** — if the agent says ITERATE, applies the suggested parameter changes and re-runs the reconstruction
4. **Repeat** — loops until quality thresholds are met or max iterations reached

The evaluator agent runs in an isolated OpenShell sandbox with access to:
- The backend API (fetch metrics, details, iterations via curl)
- The reconstruction output directory (training logs, PLY files, USDZ outputs)
- A local LLM via the OpenClaw agent framework

## Files

| File | Purpose |
|------|---------|
| `orchestrate.sh` | Main orchestrator — drives the Reconstruct → Evaluate → Retry loop |
| `agent-prompt.md` | Instructions for the evaluator agent — thresholds, output format, tuning rules |
| `sandbox-policy.yaml` | OpenShell sandbox policy — network and filesystem isolation rules |
| `sandbox-openclaw-evaluator.json` | OpenClaw config — LLM model, tool permissions, workspace settings |
| `logs/` | Timestamped logs for each run |

## Prerequisites

- **Backend running** at `http://127.0.0.1:8010` (`make backend-dev`)
- **Ollama** running at `http://127.0.0.1:11434` with a model pulled (e.g., `ollama pull nemotron-3-nano`)
- **OpenShell** installed with the `openclaw` base image available
- **OpenShell gateway** with an Ollama provider configured

## Quick Start

```bash
# 1. Start the backend
make backend-dev

# 2. Set up OpenShell gateway with Ollama provider (one-time)
openshell provider create --name ollama --type openai \
    --credential OPENAI_API_KEY=unused \
    --config endpoint=http://host.openshell.internal:11434/v1

# 3. Run with a video
./nemoclaw/orchestrate.sh ~/videos/scene.MOV "my-scene" 3

# 4. Or run with a pre-loaded dataset
./nemoclaw/orchestrate.sh --dataset garden "garden-test" 3
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama API base URL |
| `OLLAMA_MODEL` | `nemotron-3-nano` | Model for the evaluator agent |
| `AGENT_TIMEOUT` | `300` | Evaluator agent timeout (seconds) |
| `RUN_AGENT_RETRIES` | `2` | Max retries for sandbox agent launch |
| `ACCEPT_PSNR_THRESHOLD` | `25.0` | PSNR threshold for 3DGRUT |
| `ACCEPT_SSIM_THRESHOLD` | `0.85` | SSIM threshold for both backends |
| `INITIAL_BACKEND` | `3dgrut` (video) / `fvdb` (dataset) | Starting reconstruction backend |
| `INITIAL_GRUT_ITERS` | `5000` | Initial 3DGRUT iterations |
| `INITIAL_FVDB_EPOCHS` | `30` | Initial fVDB epochs |
| `API_URL` | `http://127.0.0.1:8010` | Backend API URL |

## How Evaluation Works

After each reconstruction completes, the orchestrator:
1. Creates a staging directory with the OpenClaw config, sandbox policy, agent instructions, and task message
2. Launches an OpenShell sandbox (`--from openclaw`) with the staging dir and reconstruction output files mounted
3. Inside the sandbox, the OpenClaw agent:
   - Fetches reconstruction details, metrics, and iteration history from the backend API via curl
   - Explores the mounted output files (training logs, PLY files, etc.)
   - Analyzes quality against thresholds using the local LLM
   - Returns a JSON verdict: `ACCEPT` or `ITERATE` (with suggested parameter changes)
4. The orchestrator parses the verdict. If ITERATE with no params, a default escalation strategy kicks in (double epochs/iterations, then reduce downsample factor)

See [NEMOCLAW_SETUP.md](./NEMOCLAW_SETUP.md) for the full setup tutorial.
