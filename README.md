# NemoReconstruct

A fully local Physical AI pipeline — upload a video, get an Omniverse-ready **NuRec USDZ** scene with 3D Gaussians, all orchestrated by an AI agent running inside an isolated sandbox with local inference.

The pipeline uses **NemoClaw** as the agentic orchestrator, **3DGRUT** for neural Gaussian reconstruction, and exports to the **Omniverse NuRec** format for physics simulation and collision visualization in Isaac Sim.

> **This repo is a template.** The [setup guide](docs/NEMOCLAW_SETUP.md) shows how to pair NemoClaw + OpenShell with **any** repo or API — NemoReconstruct is just the example project.

```
Video → ffmpeg → COLMAP → 3DGRUT → NuRec USDZ + PLY
                    ↑ orchestrated by ↑
              NemoClaw Agent (Nemotron)
              inside OpenShell Sandbox
              powered by Ollama (local LLM)
```

---

## Quick Start

> **Prerequisites:** Linux + NVIDIA GPU, Docker, Python 3.10+, Node.js 18+.
> See [the full tutorial](docs/NEMOCLAW_SETUP.md) for step-by-step install of every prerequisite.

```bash
# 1. Clone and install
git clone https://github.com/clayton-littlejohn/NemoReconstruct.git ~/NemoReconstruct
cd ~/NemoReconstruct
make setup                    # creates .venv, installs Python + Node deps

# 2. Download a dataset (optional — or bring your own video)
./scripts/download_datasets.sh garden   # downloads garden scene (~2.8 GB)
make list-datasets                      # see all available scenes

# 3. Start the backend (leave running)
make backend-dev              # starts on 0.0.0.0:8010

# 4. One-time OpenShell + Ollama setup (see full guide for details)
openshell gateway start --gpu
openshell provider create --name ollama --type openai \
  --credential OPENAI_API_KEY=empty \
  --config OPENAI_BASE_URL=http://host.openshell.internal:11434/v1
openshell inference set --provider ollama --model glm-4.7-flash

# 5. Run the agent in a sandbox
openshell sandbox create \
  --from openclaw \
  --policy nemoclaw/sandbox-policy.yaml \
  --upload "$PWD:/sandbox/NemoReconstruct" \
  -- bash -c '
mkdir -p /sandbox/.openclaw
cp /sandbox/NemoReconstruct/nemoclaw/sandbox-openclaw.json /sandbox/.openclaw/openclaw.json
export OPENAI_API_KEY=unused
cd /sandbox/NemoReconstruct
openclaw agent --local --session-id demo \
  --message "List all reconstruction jobs. Use curl to call the API at http://172.20.0.1:8010" \
  --json --timeout 120
'
```

**Full tutorial:** [docs/NEMOCLAW_SETUP.md](docs/NEMOCLAW_SETUP.md) — covers every step from a fresh machine, and shows how to adapt this for your own project.

---

## What the Agent Can Do

- Upload videos and start reconstruction jobs
- Select pre-loaded dataset scenes (Mip-NeRF 360) and run full agent workflows
- Tune parameters (iterations, downsample factor, render method, quality presets)
- Monitor progress in real time
- Download NuRec USDZ and PLY splat outputs
- Iterate on its own outputs — the evaluator agent analyzes metrics and retries with better parameters
- Inspect logs and system state via shell

---

## Datasets

NemoReconstruct supports the **Mip-NeRF 360** dataset for benchmarking and testing reconstruction workflows without needing your own video.

```bash
# Download all 7 scenes (~12 GB)
make download-datasets

# Download specific scenes
./scripts/download_datasets.sh garden room

# List available scenes and what's already downloaded
make list-datasets
```

| Scene | Type | Images | Size |
|-------|------|--------|------|
| bicycle | outdoor | 291 | ~2.3 GB |
| bonsai | indoor | 311 | ~1.3 GB |
| counter | indoor | 312 | ~1.2 GB |
| garden | outdoor | 185 | ~2.8 GB |
| kitchen | indoor | 315 | ~1.5 GB |
| room | indoor | 311 | ~1.3 GB |
| stump | outdoor | 295 | ~1.4 GB |

Scenes are downloaded from the original source at [jonbarron.info/mipnerf360](https://jonbarron.info/mipnerf360/) and extracted into `data/<scene>/`. Each scene includes full-resolution images, COLMAP sparse models, and pre-computed downsampled image sets.

Once downloaded, scenes appear in the frontend dashboard under the "Dataset" tab and can be submitted to the agent workflow.

---

## Project Structure

```
backend/          FastAPI API, SQLite job store, background runner
frontend/         Next.js upload dashboard
scripts/          Dataset download and setup scripts
sdk/python/       Python client for agents and automation
sdk/typescript/   TypeScript client
nemoclaw/         Agent config, system prompt, sandbox policy, example scripts
docs/             OpenAPI schema, setup guide
data/             Downloaded dataset scenes (git-ignored)
```

### NemoClaw Config Files

| File | Purpose |
|------|---------|
| `nemoclaw/sandbox-policy.yaml` | OpenShell sandbox network policy (NemoReconstruct) |
| `nemoclaw/sandbox-policy-template.yaml` | Generic sandbox policy — copy and customize for your project |
| `nemoclaw/sandbox-openclaw.json` | OpenClaw config (NemoReconstruct → `inference.local`) |
| `nemoclaw/sandbox-openclaw-template.json` | Generic OpenClaw config — copy and customize for your project |
| `nemoclaw/nemoclaw_config.yaml` | Agent tools, model, guardrails |
| `nemoclaw/system_prompt.md` | Agent persona and workflow rules |
| `nemoclaw/runner_prompt.md` | Runner agent prompt — executes pipelines |
| `nemoclaw/evaluator_prompt.md` | Evaluator agent prompt — analyzes metrics |
| `nemoclaw/orchestrate.sh` | Multi-agent orchestrator — Runner→Evaluator loop |
| `nemoclaw/example_session.py` | SDK script to test the pipeline without an agent |

---

## Backend API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/pipelines` | List available pipelines |
| POST | `/api/v1/reconstructions/upload` | Upload video + start reconstruction |
| POST | `/api/v1/reconstructions/from-dataset` | Create reconstruction from a downloaded dataset scene |
| GET | `/api/v1/datasets` | List available dataset scenes in `data/` |
| GET | `/api/v1/reconstructions` | List all jobs |
| GET | `/api/v1/reconstructions/{id}` | Job details |
| GET | `/api/v1/reconstructions/{id}/status` | Poll status + progress |
| GET | `/api/v1/reconstructions/{id}/artifacts` | List downloadable artifacts |
| GET | `/api/v1/reconstructions/{id}/download/{artifact}` | Download artifact |
| POST | `/api/v1/reconstructions/{id}/retry` | Retry with new params |
| DELETE | `/api/v1/reconstructions/{id}` | Delete a job |
| GET | `/docs` | Interactive API docs |
| GET | `/openapi.json` | OpenAPI schema |

### Tunable Parameters

Pass these to `POST /api/v1/reconstructions/upload`:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `frame_rate` | 0.25 – 12.0 | 2.0 | Frames/sec extracted by ffmpeg |
| `sequential_matcher_overlap` | 2 – 50 | 12 | COLMAP matcher overlap |
| `colmap_mapper_type` | incremental / global | incremental | COLMAP mapper algorithm ('global' uses GLOMAP) |
| `colmap_max_num_features` | 1000 – 32768 | 8192 | Max SIFT features per image |
| `reconstruction_backend` | 3dgrut / fvdb | 3dgrut | Reconstruction backend |
| `grut_n_iterations` | 1000 – 100000 | 30000 | 3DGRUT training iterations |
| `grut_render_method` | 3dgrt / 3dgut | 3dgrt | 3DGRUT render method |
| `grut_strategy` | gs / mcmc | gs | 3DGRUT densification strategy |
| `grut_downsample_factor` | 1 – 12 | 2 | Image downsampling for 3DGRUT |
| `fvdb_max_epochs` | 5 – 500 | 40 | fVDB training epochs |
| `fvdb_sh_degree` | 0 – 4 | 3 | Spherical harmonics degree |
| `fvdb_image_downsample_factor` | 1 – 12 | 6 | Input image downsampling for fVDB |
| `splat_only_mode` | true/false | false | Skip USDZ, produce PLY only |

---

## Manual Usage (Without Agent)

### Upload a video
```bash
curl -s -X POST http://localhost:8010/api/v1/reconstructions/upload \
  -F "file=@/path/to/video.MOV" \
  -F "name=my-scene" \
  -F "frame_rate=2.0" \
  -F "reconstruction_backend=3dgrut" \
  -F "grut_n_iterations=30000"
```

### Poll status
```bash
curl -s http://localhost:8010/api/v1/reconstructions/<id>/status
```

### Download the PLY
```bash
curl -L -o output.ply \
  http://localhost:8010/api/v1/reconstructions/<id>/download/splat_ply
```

### Python SDK
```python
from nemo_reconstruct_client import NemoReconstructClient

client = NemoReconstructClient("http://localhost:8010")
job = client.upload_video("/tmp/scene.mov", "my-scene",
    params={"reconstruction_backend": "3dgrut", "grut_n_iterations": 30000})
result = client.wait_for_completion(job.id)
print(client.get_artifacts(job.id))
```

---

## Development

```bash
# One-time setup — creates .venv, installs Python + Node dependencies
make setup

# Start the backend   (0.0.0.0:8010, auto-reloads on code change)
make backend-dev

# Start the frontend  (localhost:3000, optional)
cp frontend/.env.example frontend/.env.local   # only needed once
make frontend-dev

# Export OpenAPI schema
make openapi

# Install Python SDK
.venv/bin/pip install -e sdk/python
```

> **Note:** `make backend-dev` and the other targets automatically use `.venv/bin/python` — you do not need to activate the virtualenv.

### System Requirements

Pipeline binaries (must be on PATH or configured via env vars):
- **`ffmpeg`** — frame extraction
- **`colmap`** — feature extraction, matching, sparse reconstruction
- **3DGRUT** (default backend) — neural Gaussian reconstruction + NuRec USDZ export, installed at `/opt/3dgrut`, conda env `3dgrut` (see `NEMO_RECONSTRUCT_GRUT_INSTALL_DIR`)
- **CUDA toolkit** — headers at `/usr/local/cuda` for JIT C++ extension builds
- **fVDB / `frgs`** (alternative backend) — conda env `fvdb` (typically at `~/miniconda3/envs/fvdb/bin/frgs`)

All paths are configurable via environment variables or a `.env` file in the backend directory (prefix: `NEMO_RECONSTRUCT_`).

---

## How It All Fits Together

```
┌─────────────────────────────────────────────────────────────────┐
│  DGX Spark                                                      │
│                                                                  │
│  ┌──────────────────────── OpenShell Sandbox ──────────────────┐ │
│  │                                                              │ │
│  │  OpenClaw Agent ──► inference.local ──► Gateway ──► Ollama  │ │
│  │       │                                           :11434    │ │
│  │       │ curl / OpenAPI tools                                 │ │
│  │       ▼                                                      │ │
│  │  NemoReconstruct API (:8010)                                │ │
│  │       │                                                      │ │
│  │       ▼                                                      │ │
│  │  ffmpeg → COLMAP → 3DGRUT (or fVDB) → PLY + USDZ           │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                    │                             │
│                                    ▼                             │
│                             NVIDIA GPU (Blackwell)               │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Port | Purpose |
|-----------|------|---------|
| Ollama | 11434 | Local LLM (glm-4.7-flash) |
| OpenShell Gateway | 8080 | Sandbox management + inference routing |
| NemoReconstruct Backend | 8010 | FastAPI pipeline server |
| NemoReconstruct Frontend | 3000 | Next.js dashboard (optional) |
| `inference.local` | — | In-sandbox LLM endpoint → Ollama |

---

## Using This as a Template for Your Own Project

The [full tutorial](docs/NEMOCLAW_SETUP.md) has a dedicated section (Part 2) that walks through connecting NemoClaw to **any** project:

1. **Start your service** on `0.0.0.0` (not `127.0.0.1`)
2. **Create a sandbox policy** — allow the sandbox to reach your service's port
3. **Create an OpenClaw config** — point the agent at your workspace
4. **Run the agent** — same `openshell sandbox create` pattern, just swap the paths

The NemoClaw + OpenShell infrastructure (Ollama, gateway, provider, inference routing) is set up once and reused across all projects.

---

## Notes

- Intentionally an MVP: one pipeline, one worker, SQLite, local disk.
- Designed for agent demos and livestream walkthroughs — not HA production.
- The pattern (API + OpenShell sandbox + local inference) generalizes to any tool or SDK.

