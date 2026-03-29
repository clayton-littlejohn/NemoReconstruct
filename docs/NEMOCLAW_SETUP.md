# NemoClaw + OpenShell Tutorial

Set up **NemoClaw** (an AI coding agent) inside an **OpenShell sandbox** with fully local inference — then point it at **any repo or API** you want the agent to control.

This guide walks you through every command from a fresh machine. By the end, you'll have a sandboxed AI agent powered by a local LLM that can read your code, call your APIs, and run shell commands — all inside a kernel-isolated environment that can't touch your host files.

> **NemoReconstruct** is used as the example project throughout. Swap it for your own repo — the steps are the same.

---

## What You're Building

```
┌─────────────────────────────────────────────────────────────────┐
│  Your Machine (DGX Spark, Linux workstation, etc.)              │
│                                                                  │
│  ┌──────────────────────── OpenShell Sandbox ──────────────────┐ │
│  │                                                              │ │
│  │  OpenClaw Agent ──► inference.local ──► Gateway ──► Ollama  │ │
│  │       │                                           :11434    │ │
│  │       │ curl / SDK                                           │ │
│  │       ▼                                                      │ │
│  │  Your API / Service (:PORT)                                 │ │
│  │                                                              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key pieces:**
- **Ollama** — runs the LLM locally on your GPU
- **OpenShell** — provides the sandbox runtime + inference routing
- **OpenClaw** — the coding agent (runs inside the sandbox)
- **Your project** — whatever repo, API, or tool you want the agent to work with

---

## Table of Contents

- [Part 1 — Install NemoClaw + OpenShell](#part-1--install-nemoclaw--openshell) (universal — do this once)
- [Part 2 — Connect Your Project](#part-2--connect-your-project) (works for any repo)
- [Part 3 — Full Example: NemoReconstruct](#part-3--full-example-nemoreconstruct) (concrete walkthrough)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

---

# Part 1 — Install NemoClaw + OpenShell

These steps set up the AI agent infrastructure. Do this once — it works for any project.

---

## Step 1 — Install Prerequisites

You need Docker, Node.js, Python, and Git installed.

```bash
# Check what you have
docker --version          # Docker 24+ required
node --version            # Node.js 18+ required
python3 --version         # Python 3.10+ required
git --version             # Git 2.x
uv --version              # uv (Python package installer)
nvidia-smi                # NVIDIA GPU (optional but recommended)
```

If anything is missing:
```bash
# Docker (Ubuntu)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
# Log out and back in after this

# Node.js 22 (Ubuntu — via NodeSource)
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs

# Python 3 + venv (Ubuntu)
sudo apt-get install -y python3 python3-venv python3-pip

# uv (Python package installer — used to install OpenShell)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Step 2 — Install Ollama and Pull a Model

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Verify it's running
ollama --version
systemctl status ollama

# Pull a model with tool-calling support
ollama pull glm-4.7-flash
```

> **Why glm-4.7-flash?** Reliable tool-calling support, 9B parameters, runs well on consumer GPUs and DGX Spark.
> Other good options: `qwen3.5`, `nemotron-3-super` (needs 48GB+ VRAM).

### Bind Ollama to all interfaces

Sandboxes run in an isolated network namespace — they can't reach `127.0.0.1` on the host. Ollama needs to listen on `0.0.0.0`:

```bash
sudo systemctl edit ollama
```

Add these lines in the editor that opens:
```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

Save, then restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Verify Ollama is serving the model:
```bash
curl -s http://localhost:11434/v1/models | python3 -m json.tool
# Should list glm-4.7-flash in the output
```

---

## Step 3 — Install OpenClaw

OpenClaw is the agent runtime. Install it globally via npm:

```bash
npm install -g openclaw
```

Verify:
```bash
openclaw --version
# Should show 2026.x.x
```

---

## Step 4 — Install OpenShell

OpenShell provides the sandbox runtime and inference routing:

```bash
uv tool install openshell
```

> **Alternative:** `pipx install openshell` also works if you have pipx installed.

Verify:
```bash
openshell --version
# Should show >= 0.0.16
```

> **Tip:** If `which openshell` shows an old version at `/usr/local/bin/openshell`, run `hash -r` to clear the bash cache after installing a newer version.

For more details, see [build.nvidia.com/spark/openshell](https://build.nvidia.com/spark/openshell).

---

## Step 5 — Start the OpenShell Gateway

The gateway manages sandboxes and routes inference requests:

```bash
openshell gateway start --gpu
```

This starts a local k3s cluster inside Docker with GPU passthrough. First run pulls container images and takes a couple of minutes. Subsequent starts are fast.

Verify:
```bash
openshell status
# Should show the gateway running
```

---

## Step 6 — Register the Ollama Provider

Tell the OpenShell gateway where to find your local LLM:

```bash
openshell provider create \
  --name ollama \
  --type openai \
  --credential OPENAI_API_KEY=empty \
  --config OPENAI_BASE_URL=http://host.openshell.internal:11434/v1
```

**Important details:**
- **`host.openshell.internal`** — a special hostname the gateway resolves to your host machine. **Never use `localhost` or `127.0.0.1` here** — those resolve inside the container, not the host.
- **`OPENAI_API_KEY=empty`** — Ollama doesn't need auth, but the field is required.
- **`OPENAI_BASE_URL`** — this must be the exact config key name (not `baseUrl`, not `base_url`).

Verify:
```bash
openshell provider get ollama
# Should show: Type: openai, Config keys: OPENAI_BASE_URL
```

---

## Step 7 — Set Inference Routing

Map the model name to the Ollama provider so sandboxes know where to send requests:

```bash
openshell inference set --provider ollama --model glm-4.7-flash
```

Verify:
```bash
openshell inference get
# Gateway inference:
#   Provider: ollama
#   Model: glm-4.7-flash
```

---

## Step 8 — Test Inference from a Sandbox

This is the moment of truth — does `inference.local` inside a sandbox actually reach Ollama?

```bash
openshell sandbox create -- \
  curl -s https://inference.local/v1/chat/completions \
  --json '{"messages":[{"role":"user","content":"Say hello in one word"}],"max_tokens":10}'
```

**Expected:** A JSON response containing `"model":"glm-4.7-flash"` and `"system_fingerprint":"fp_ollama"`.

> **First run timeout:** The first sandbox after `gateway start` pulls the base container image inside the k3s cluster. This can take 3–5+ minutes and may exceed the default 300s timeout, giving you: `sandbox provisioning timed out after 300s. Last reported status: DependenciesNotReady`. **Just run the command again** — the image pull continues in the background and the next attempt will succeed. Subsequent sandboxes start in seconds.

If you see Cloudflare HTML, `"Incorrect API key"` errors, or connection refused — see [Troubleshooting](#troubleshooting).

**NemoClaw + OpenShell is now fully installed.** Everything below is about connecting it to your project.

---

# Part 2 — Connect Your Project

The setup above works for **any** repo or API. Here's how to connect your project.

You need three things:
1. A **sandbox network policy** — tells OpenShell what the sandbox can talk to
2. An **OpenClaw config** — tells the agent which model and workspace to use
3. Your **service running on the host** — bound to `0.0.0.0` so the sandbox can reach it

---

## Step 9 — Find Your Host IP

Sandboxes reach your host machine through the Docker cluster network. Find the gateway IP:

```bash
docker network inspect openshell-cluster-openshell | grep Gateway
# Typically: "Gateway": "172.20.0.1"
```

Use this IP (not `localhost`) whenever the sandbox needs to reach a service on your host.

---

## Step 10 — Start Your Service

If your project has an API or service, start it and bind to `0.0.0.0`:

```bash
# Example: a FastAPI backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8010

# Example: a Node.js server
HOST=0.0.0.0 PORT=3000 node server.js

# Example: any service
./my-service --bind 0.0.0.0 --port 9000
```

> **Key rule:** Services must listen on `0.0.0.0`, not `127.0.0.1`. The sandbox is in a separate network namespace — `localhost` on the host is unreachable from inside.

Verify from your host:
```bash
curl -s http://localhost:YOUR_PORT/health   # or whatever your health endpoint is
```

---

## Step 11 — Create a Sandbox Network Policy

The sandbox is completely locked down by default. Create a policy file to allow the agent to talk to your service.

Create a file called `sandbox-policy.yaml` in your project:

```yaml
version: 1

filesystem_policy:
  include_workdir: true
  read_only:
    - /usr
    - /lib
    - /proc
    - /dev/urandom
    - /app
    - /etc
    - /var/log
  read_write:
    - /sandbox
    - /tmp
    - /dev/null

landlock:
  compatibility: best_effort

process:
  run_as_user: sandbox
  run_as_group: sandbox

network_policies:
  # Your service on the host
  my_service:
    name: my-service
    endpoints:
      - host: 172.20.0.1        # <-- Host IP from Step 9
        port: 8010               # <-- Your service port
        protocol: tcp
        enforcement: enforce
        access: full
    binaries:
      - path: /usr/bin/curl
      - path: /usr/bin/node
      - path: /bin/bash

  # OpenClaw gateway on the host (required for agent communication)
  openclaw_gateway:
    name: openclaw-gateway
    endpoints:
      - host: 172.20.0.1
        port: 18789
        protocol: tcp
        enforcement: enforce
        access: full
    binaries:
      - path: /usr/bin/node
      - path: /bin/bash

  # (Optional) GitHub read-only access for git operations
  github:
    name: github
    endpoints:
      - host: github.com
        port: 443
        protocol: rest
        tls: terminate
        enforcement: enforce
        rules:
          - allow:
              method: GET
              path: /**/info/refs*
          - allow:
              method: POST
              path: /**/git-upload-pack
    binaries:
      - path: /usr/bin/git

  github_rest_api:
    name: github-rest-api
    endpoints:
      - host: api.github.com
        port: 443
        protocol: rest
        tls: terminate
        enforcement: enforce
        rules:
          - allow:
              method: GET
              path: /**
          - allow:
              method: HEAD
              path: /**
```

**Customize this:**
- Change `port: 8010` to your service's port
- Add more `network_policies` entries if your service depends on other hosts
- Remove the GitHub entries if git access isn't needed

> **Note:** `inference.local` is handled internally by the sandbox runtime — no policy entry needed for LLM access.

---

## Step 12 — Create an OpenClaw Config

Create a file called `sandbox-openclaw.json` in your project:

```json
{
  "models": {
    "mode": "merge",
    "providers": {
      "openai": {
        "baseUrl": "https://inference.local/v1",
        "api": "openai-completions",
        "models": [
          {
            "id": "glm-4.7-flash",
            "name": "glm-4.7-flash",
            "reasoning": false,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 128000,
            "maxTokens": 8192
          }
        ]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "openai/glm-4.7-flash"
      },
      "workspace": "/sandbox/my-project"
    }
  },
  "tools": {
    "profile": "coding"
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
```

**Customize this:**
- Change `"workspace": "/sandbox/my-project"` to match where your repo will be uploaded
- If using a different model, update the `id`, `name`, and `primary` fields
- If using a different model, also update `contextWindow` and `maxTokens` to match

---

## Step 13 — Run the Agent

Upload your project into a sandbox and run the agent:

```bash
cd ~/my-project

openshell sandbox create \
  --from openclaw \
  --policy sandbox-policy.yaml \
  --upload "$PWD:/sandbox/my-project" \
  -- bash -c '
mkdir -p /sandbox/.openclaw
cp /sandbox/my-project/sandbox-openclaw.json /sandbox/.openclaw/openclaw.json
export OPENAI_API_KEY=unused
cd /sandbox/my-project
openclaw agent --local --session-id demo \
  --message "Your prompt here" \
  --json --timeout 120
'
```

**What each flag does:**

| Flag | Purpose |
|------|---------|
| `--from openclaw` | Uses the community OpenClaw sandbox image |
| `--policy sandbox-policy.yaml` | Applies your network policy (from Step 11) |
| `--upload "$PWD:/sandbox/my-project"` | Copies your repo into the sandbox at `/sandbox/my-project` |
| `openclaw agent --local` | Runs the agent locally (no external cloud gateway) |
| `--session-id demo` | Names the agent session |
| `--json` | Returns structured JSON output |
| `--timeout 120` | Seconds before the agent times out |

**What happens inside:**
1. Your OpenClaw config gets copied — it points at `https://inference.local/v1` for LLM calls
2. The agent calls `inference.local` → OpenShell routes it to Ollama on the host
3. The agent reads your code, calls your API, runs shell commands — all inside the sandbox
4. The sandbox can ONLY reach the endpoints in your network policy — nothing else

---

## Step 14 — Interactive Session (Optional)

For a persistent shell where you can run multiple agent turns:

```bash
cd ~/my-project

openshell sandbox create \
  --from openclaw \
  --policy sandbox-policy.yaml \
  --upload "$PWD:/sandbox/my-project"
```

Once inside the sandbox shell:

```bash
# One-time setup inside this session
mkdir -p /sandbox/.openclaw
cp /sandbox/my-project/sandbox-openclaw.json /sandbox/.openclaw/openclaw.json
export OPENAI_API_KEY=unused
cd /sandbox/my-project

# Run agent turns
openclaw agent --local --session-id demo \
  --message "Explore the codebase and summarize what this project does" --json

openclaw agent --local --session-id demo \
  --message "Check the health of the API at http://172.20.0.1:8010" --json

openclaw agent --local --session-id demo \
  --message "List all items from the API and format as a table" --json
```

Type `exit` to leave the sandbox.

---

# Part 3 — Full Example: NemoReconstruct

This section walks through a concrete example using the NemoReconstruct 3D reconstruction API.

---

## Clone and Set Up the Backend

```bash
git clone https://github.com/clayton-littlejohn/NemoReconstruct.git ~/NemoReconstruct
cd ~/NemoReconstruct

# One-command setup — creates .venv, installs Python + Node dependencies
make setup

# (Optional) Install the Python SDK
.venv/bin/pip install -e sdk/python
```

### Start the backend

```bash
cd ~/NemoReconstruct
make backend-dev
# Starts on 0.0.0.0:8010 — leave this terminal open
```

Or run manually:
```bash
cd ~/NemoReconstruct/backend
../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8010
```

Verify:
```bash
curl -s http://localhost:8010/health
# {"status":"ok","service":"NemoReconstruct"}
```

---

## Download Dataset Scenes (Optional)

NemoReconstruct includes a download script for the **Mip-NeRF 360** benchmark dataset. These scenes let you test the full pipeline without needing your own video or images.

```bash
cd ~/NemoReconstruct

# See what's available
./scripts/download_datasets.sh --list

# Download one or more scenes (garden is a good starting point — ~2.8 GB)
./scripts/download_datasets.sh garden

# Or download all 7 scenes (~12 GB total)
make download-datasets
```

Scenes are extracted into `data/<scene>/` with pre-built COLMAP sparse models and downsampled image sets. Once downloaded, they appear in the frontend dashboard under the **Dataset** tab, or you can use them via the API:

```bash
# List available dataset scenes
curl -s http://localhost:8010/api/v1/datasets

# Start a reconstruction from a dataset scene
curl -s -X POST http://localhost:8010/api/v1/reconstructions/from-dataset \
  -H "Content-Type: application/json" \
  -d '{"name": "garden-test", "dataset_scene": "garden"}'
```

> **Tip:** The zip file (~12 GB) is cached at `data/.360_v2.zip` after the first download. To free disk space after extracting, delete it: `rm data/.360_v2.zip`

---

## Run the NemoReconstruct Agent

The repo already includes the sandbox policy and OpenClaw config in `nemoclaw/`:

```bash
cd ~/NemoReconstruct

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

**Expected:** The agent calls the NemoReconstruct API and lists all reconstruction jobs.

> **First run:** The sandbox base image is pulled on first use, which can take several minutes depending on your connection. Subsequent runs start in seconds.

### More example prompts

```bash
# Check system health
--message "Check the health of the backend at http://172.20.0.1:8010"

# Upload and reconstruct
--message "Upload /sandbox/NemoReconstruct/test_video.MOV and start a reconstruction with name 'demo-scene'"

# Monitor a job
--message "Check the status of the most recent reconstruction job"

# Download results
--message "Download the PLY splat output from the completed reconstruction"
```

### Interactive session

```bash
cd ~/NemoReconstruct

openshell sandbox create \
  --from openclaw \
  --policy nemoclaw/sandbox-policy.yaml \
  --upload "$PWD:/sandbox/NemoReconstruct" \
  --tty
```

> **`--tty`** allocates a pseudo-terminal, giving you a proper interactive shell inside the sandbox. Without it, auto-detection may fail in some terminal environments.

Then inside the sandbox:
```bash
mkdir -p /sandbox/.openclaw
cp /sandbox/NemoReconstruct/nemoclaw/sandbox-openclaw.json /sandbox/.openclaw/openclaw.json
export OPENAI_API_KEY=unused
cd /sandbox/NemoReconstruct

openclaw agent --local --session-id demo \
  --message "List all reconstruction jobs. Use curl to call the API at http://172.20.0.1:8010" --json

openclaw agent --local --session-id demo \
  --message "Show me the details and artifacts for the most recent job" --json
```

---

### Multiple sandboxes with different policies

You can create as many sandboxes as you want, each with a different network policy. For example, one sandbox might only access the backend API, while another can also reach GitHub:

```bash
# Sandbox A — backend access only
openshell sandbox create \
  --from openclaw \
  --name agent-backend \
  --policy nemoclaw/sandbox-policy-backend-only.yaml \
  --upload "$PWD:/sandbox/NemoReconstruct" \
  --tty

# Sandbox B — backend + GitHub access
openshell sandbox create \
  --from openclaw \
  --name agent-full \
  --policy nemoclaw/sandbox-policy.yaml \
  --upload "$PWD:/sandbox/NemoReconstruct" \
  --tty
```

Each sandbox is fully isolated — separate filesystem, separate network namespace, separate process tree. Sandboxes **cannot** talk to each other directly. However, they can communicate indirectly through a shared service on the host:

```
┌──────────────┐         ┌──────────────────┐         ┌──────────────┐
│  Sandbox A   │────────►│  Host Service    │◄────────│  Sandbox B   │
│  (policy A)  │  :8010  │  (0.0.0.0:8010)  │  :8010  │  (policy B)  │
└──────────────┘         └──────────────────┘         └──────────────┘
```

Both sandboxes need the host service endpoint in their respective policies. One agent writes data, another reads it — the host API acts as the coordination layer.

---

## NemoReconstruct Config Files

| File | Purpose |
|------|---------|
| `nemoclaw/sandbox-policy.yaml` | Network policy — allows backend (:8010), Ollama (:11434), OpenClaw gateway (:18789), GitHub |
| `nemoclaw/sandbox-openclaw.json` | OpenClaw config — model → `inference.local`, workspace → `/sandbox/NemoReconstruct` |
| `nemoclaw/sandbox-policy-template.yaml` | Generic sandbox policy — copy and customize for your own project (Part 2) |
| `nemoclaw/sandbox-openclaw-template.json` | Generic OpenClaw config — copy and customize for your own project (Part 2) |
| `nemoclaw/nemoclaw_config.yaml` | Agent tools, model, and guardrails |
| `nemoclaw/system_prompt.md` | Agent persona and workflow rules |
| `nemoclaw/runner_prompt.md` | Runner agent prompt — executes pipelines |
| `nemoclaw/evaluator_prompt.md` | Evaluator agent prompt — analyzes metrics, suggests parameter changes |
| `nemoclaw/orchestrate.sh` | Multi-agent orchestrator — drives the Runner→Evaluator loop |
| `nemoclaw/example_session.py` | Python SDK script to test the pipeline without an agent |

---

## Multi-Agent Iterative Workflow

This demonstrates two sandboxed agents collaborating through the shared backend API to iteratively improve a 3D reconstruction.

```
┌─────────────────────────────────────────────────────────────────────┐
│  Host                                                                │
│                                                                      │
│  orchestrate.sh (loop)                                               │
│    │                                                                 │
│    ├──► Sandbox: Runner agent                                        │
│    │      Uploads video, starts pipeline, polls to completion        │
│    │                                                                 │
│    ├──► Sandbox: Evaluator agent                                     │
│    │      Reads metrics, analyzes quality, suggests new params       │
│    │      Verdict: ACCEPT or ITERATE {new params}                    │
│    │                                                                 │
│    ├──► Sandbox: Runner agent (retry with evaluator's params)        │
│    │      ...                                                        │
│    └──► (repeat until ACCEPT or max iterations)                      │
│                                                                      │
│  Backend API (0.0.0.0:8010) ◄── shared coordination layer           │
└─────────────────────────────────────────────────────────────────────┘
```

**Key points:**
- Each agent runs in its **own isolated sandbox** — separate filesystem, network, process tree
- Agents **never talk to each other directly** — the backend API is the shared state
- The host-side orchestrator script drives the loop: Runner → Evaluator → Runner (retry) → ...
- The Evaluator reads training metrics (loss, SSIM, num_gaussians) and reasons about parameter changes
- After a configurable number of iterations, the workflow stops with the best result

### Run the workflow

```bash
cd ~/NemoReconstruct

# Start the backend (if not already running)
make backend-dev

# Run the multi-agent workflow
./nemoclaw/orchestrate.sh ~/videos/my_scene.MOV "kitchen-scan" 3
#                         ─── video path ───   ── name ──   ── max iterations ──
```

### What happens

1. **Iteration 1:** Runner agent uploads the video and starts a reconstruction with default parameters. Polls until complete.
2. **Evaluation 1:** Evaluator agent fetches the reconstruction details and training metrics (`/api/v1/reconstructions/{id}/metrics`). Analyzes loss convergence, SSIM, gaussian count. Outputs a verdict:
   - `ACCEPT` — quality is good enough, stop
   - `ITERATE` — suggests specific parameter changes (e.g., more epochs, higher frame rate)
3. **Iteration 2:** Runner agent retries the reconstruction with the evaluator's suggested parameters. Polls until complete.
4. **Evaluation 2:** Evaluator analyzes the new results. Accepts or suggests further changes.
5. Repeat up to `max_iterations` times.

---

# How It Works

## Inference Routing

Code inside sandboxes calls `https://inference.local/v1` — a special endpoint injected by the sandbox runtime. The OpenShell gateway intercepts these requests and forwards them to your configured provider (Ollama in this tutorial), injecting real credentials. The sandbox code never sees API keys.

## Sandbox Isolation

OpenShell sandboxes use kernel-level isolation (Linux namespaces + Landlock LSM):
- **Filesystem:** Only `/sandbox` and `/tmp` are writable. The host filesystem is completely inaccessible.
- **Network:** The sandbox can ONLY reach endpoints listed in your policy file. All other traffic is blocked.
- **Process:** Runs as a non-root `sandbox` user. Cannot see host processes.
- **Identity:** No host user accounts, SSH keys, tokens, or environment variables leak in.

## Component Map

| Component | Port | Role |
|-----------|------|------|
| Ollama | 11434 | Local LLM serving |
| OpenShell Gateway | 8080 | Sandbox management + inference router |
| OpenClaw Gateway | 18789 | Agent runtime communication |
| Your Service | varies | Whatever the agent controls |
| `inference.local` | — | In-sandbox LLM endpoint → routed to Ollama |
| `host.openshell.internal` | — | Hostname that resolves to your host machine |

## Using a Different LLM Backend

Ollama is one option. The same provider setup works with **vLLM, SGLang, TRT-LLM, or NVIDIA NIM**:

```bash
# Example: NVIDIA NIM
openshell provider create \
  --name nim \
  --type openai \
  --credential OPENAI_API_KEY=your-nim-key \
  --config OPENAI_BASE_URL=http://host.openshell.internal:8000/v1

openshell inference set --provider nim --model meta/llama-3.3-70b-instruct
```

Then update the model in your `sandbox-openclaw.json` to match.

---

# Troubleshooting

## inference.local returns Cloudflare / OpenAI errors

The provider's `OPENAI_BASE_URL` is wrong or was created with the wrong config key.

```bash
openshell provider get ollama
# Check that "Config keys" shows OPENAI_BASE_URL (not baseUrl)

# If wrong, delete and recreate:
openshell provider delete ollama
openshell provider create --name ollama --type openai \
  --credential OPENAI_API_KEY=empty \
  --config OPENAI_BASE_URL=http://host.openshell.internal:11434/v1
```

## Sandbox can't reach your service

1. Your service must listen on `0.0.0.0`, not `127.0.0.1`
2. Your sandbox policy must include the correct host IP and port
3. Find the host IP: `docker network inspect openshell-cluster-openshell | grep Gateway`
4. Test from a sandbox: `openshell sandbox create -- curl -s http://172.20.0.1:YOUR_PORT/health`

## Ollama not reachable from sandbox

- Ollama must have `OLLAMA_HOST=0.0.0.0` set (see [Step 2](#step-2--install-ollama-and-pull-a-model))
- Test directly: `curl -s http://$(hostname -I | awk '{print $1}'):11434/v1/models`

## OpenShell CLI version issues

```bash
hash -r                  # Clear bash command cache
which openshell          # Should be ~/.local/bin/openshell
openshell --version      # Should be >= 0.0.16
```

## Agent times out or loops

- Increase `--timeout` (default 600 seconds)
- Check that your model supports tool calling (`glm-4.7-flash` and `qwen3.5` do)
- Simplify the prompt — shorter, more specific messages get better results from smaller models

---

# Quick Reference

```bash
# ─── One-time setup (do once) ─────────────────────────

# 1. Install Ollama + model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull glm-4.7-flash
sudo systemctl edit ollama                          # Add: Environment="OLLAMA_HOST=0.0.0.0"
sudo systemctl daemon-reload && sudo systemctl restart ollama

# 2. Install OpenClaw
npm install -g openclaw

# 3. Install OpenShell
uv tool install openshell
openshell --version                                 # Verify >= 0.0.16

# 4. Start the gateway
openshell gateway start --gpu

# 5. Register Ollama as a provider
openshell provider create --name ollama \
  --type openai \
  --credential OPENAI_API_KEY=empty \
  --config OPENAI_BASE_URL=http://host.openshell.internal:11434/v1

# 6. Set inference routing
openshell inference set --provider ollama --model glm-4.7-flash

# 7. Test inference from a sandbox
openshell sandbox create -- \
  curl -s https://inference.local/v1/chat/completions \
  --json '{"messages":[{"role":"user","content":"hello"}],"max_tokens":10}'

# ─── Run the agent on your project ────────────────────
cd ~/my-project
openshell sandbox create \
  --from openclaw \
  --policy sandbox-policy.yaml \
  --upload "$PWD:/sandbox/my-project" \
  -- bash -c '
mkdir -p /sandbox/.openclaw
cp /sandbox/my-project/sandbox-openclaw.json /sandbox/.openclaw/openclaw.json
export OPENAI_API_KEY=unused
cd /sandbox/my-project
openclaw agent --local --session-id demo \
  --message "Your prompt here" --json --timeout 120
'

# ─── Verify everything ───────────────────────────────
openshell status                                    # Gateway status
openshell inference get                             # Inference routing
openshell provider get ollama                       # Provider config
ollama list                                         # Available models
```
