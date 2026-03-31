# NemoClaw — Agentic Orchestrator for NemoReconstruct

NemoClaw uses [OpenClaw](https://github.com/nvidia/openclaw) agents running inside isolated [OpenShell](https://github.com/nvidia/openshell) sandboxes to drive 3D reconstruction pipelines. Two agents collaborate through the NemoReconstruct backend API:

- **Agent A (Runner)** — uploads video, starts reconstruction, polls for completion
- **Agent B (Evaluator)** — reads quality metrics (PSNR, SSIM), suggests parameter improvements

The `orchestrate.sh` script drives them in a loop: Runner executes → Evaluator analyzes → Runner retries with new params → repeat until quality thresholds are met or max iterations reached.

## Files

| File | Purpose |
|------|---------|
| `orchestrate.sh` | Multi-agent orchestrator — drives the Runner → Evaluator loop |
| `sandbox-policy.yaml` | OpenShell sandbox policy — controls filesystem, process, and network access |
| `sandbox-openclaw.json` | OpenClaw config — sets model, workspace, gateway, and tool permissions |
| `sandbox-policy-template.yaml` | Starter policy — copy and customize for your own project |
| `sandbox-openclaw-template.json` | Starter OpenClaw config — copy and customize for your own project |

## Sandbox Policy Breakdown

The `sandbox-policy.yaml` defines what an agent can and cannot do inside its sandbox. Here's what each section controls:

### `filesystem_policy`

Controls which host paths are visible inside the sandbox and whether the agent can write to them.

| Field | Value | Meaning |
|-------|-------|---------|
| `include_workdir` | `true` | The working directory is accessible inside the sandbox |
| `read_only` | `/usr`, `/lib`, `/proc`, `/dev/urandom`, `/app`, `/etc`, `/var/log` | Agent can read system binaries and libraries but cannot modify them |
| `read_write` | `/sandbox`, `/tmp`, `/dev/null` | Agent can write to its workspace and scratch space |

Paths **not** listed here are invisible to the agent — it has no access to `/home`, host project directories, or anything outside this allowlist.

### `landlock`

[Landlock](https://docs.kernel.org/security/landlock.html) is a Linux kernel security module that restricts filesystem access at the kernel level — even if a process tries to escape its sandbox.

| Field | Value | Meaning |
|-------|-------|---------|
| `compatibility` | `best_effort` | Use Landlock if the kernel supports it (v5.13+), gracefully fall back if not |

Other options: `strict` (require Landlock or fail) and `disabled`.

### `process`

Controls the user/group identity the agent process runs as inside the sandbox.

| Field | Value | Meaning |
|-------|-------|---------|
| `run_as_user` | `sandbox` | Agent runs as the unprivileged `sandbox` user, not root |
| `run_as_group` | `sandbox` | Same for the group — prevents privilege escalation |

This ensures that even if the agent finds an exploit, it cannot run commands as root inside the container.

### `network_policies`

Defines which network endpoints the agent can reach. **Everything not listed here is blocked** — no internet, no SSH, no arbitrary ports.

Each entry has a name and a list of allowed endpoints plus which binaries can use them:

| Field | Meaning |
|-------|---------|
| `name` | Human-readable label for the policy entry |
| `endpoints[].host` | IP address the agent can connect to (`172.20.0.1` = Docker gateway = host machine) |
| `endpoints[].port` | Allowed port number |
| `endpoints[].protocol` | `tcp` or `udp` |
| `endpoints[].enforcement` | `enforce` = actively block violations; `audit` = log but allow |
| `endpoints[].access` | `full` = complete read/write access to this endpoint |
| `binaries[].path` | Only these executables can use this network rule (e.g., `/usr/bin/curl`) |

**Our policy allows exactly two endpoints:**

| Policy Name | Host:Port | Used For |
|-------------|-----------|----------|
| `nemo_reconstruct` | `172.20.0.1:8010` | NemoReconstruct backend API — upload, poll, retry, read metrics |
| `openclaw_gateway` | `172.20.0.1:18789` | OpenClaw gateway — agent harness communication + LLM inference proxy |

The gateway is how agents reach Ollama. Requests to `https://inference.local` inside the sandbox are proxied through the OpenClaw gateway on port 18789, which forwards them to Ollama on the host (port 11434). The agent never talks to Ollama directly.

## OpenClaw Config Breakdown

The `sandbox-openclaw.json` configures the OpenClaw agent harness running inside the sandbox.

| Section | Key Fields | Purpose |
|---------|-----------|---------|
| `models.providers.openai` | `baseUrl`, `models[]` | Routes LLM calls to `https://inference.local/v1` (proxied via gateway to Ollama) |
| `agents.defaults.model` | `primary` | Default model for agents — `openai/glm-4.7-flash` |
| `agents.defaults.workspace` | — | Working directory inside sandbox — `/sandbox/NemoReconstruct` |
| `tools.profile` | `coding` | Enables the coding tool profile (file read/write, shell commands) |
| `tools.deny` | `["web_fetch"]` | Blocks web browsing — agents should only use the backend API |
| `gateway` | `port`, `mode`, `bind` | Internal gateway on port 18789, local mode, loopback only |

## Quick Start

```bash
# 1. Start the backend
make backend-dev

# 2. Run the orchestrator with a video
./nemoclaw/orchestrate.sh ~/videos/scene.MOV "my-scene" 3

# 3. Or run with a pre-loaded dataset
./nemoclaw/orchestrate.sh --dataset garden "garden-test" 3
```

See [docs/NEMOCLAW_SETUP.md](../docs/NEMOCLAW_SETUP.md) for the full step-by-step tutorial.
