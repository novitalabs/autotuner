# CLAUDE.md

Guidance for Claude Code when working with this repository.

## Project Overview

LLM Autotuner: Automated parameter tuning for LLM inference engines (vLLM, SGLang). Supports three deployment modes: Docker, Local (subprocess), and OME (Kubernetes).

**Key Features:**
- Grid search and Bayesian optimization (80%+ reduction vs grid search)
- SLO-aware scoring with exponential penalties
- Web UI with real-time WebSocket updates
- Agent chat interface with LLM-powered task management
- REST API with ARQ background task queue

## Quick Start

```bash
# Backend + Worker
./scripts/start_dev.sh

# Frontend (separate terminal)
cd frontend && npm run dev

# CLI mode
python src/run_autotuner.py task.yaml --mode docker
```

Access: http://localhost:5173 (frontend), http://localhost:8000/docs (API docs)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (React)  │  REST API (FastAPI)  │  ARQ Worker        │
├─────────────────────────────────────────────────────────────────┤
│  Orchestrator → Controllers (Docker/Local/OME) → genai-bench   │
├─────────────────────────────────────────────────────────────────┤
│  SQLite (tasks, experiments)  │  Redis (job queue)             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `src/orchestrator.py` - Core experiment coordination
- `src/controllers/` - Deployment strategies (docker, local, ome)
- `src/web/workers/autotuner_worker.py` - Background job processor
- `frontend/src/pages/` - React pages (Dashboard, Tasks, Agent)

## Task Configuration (YAML)

```yaml
task_name: llama-tune
model:
  id_or_path: llama-3-2-1b-instruct  # /mnt/data/models/ or HuggingFace ID
base_runtime: sglang  # or vllm
parameters:
  tp-size: [1, 2]
  mem-fraction-static: [0.85, 0.90]
optimization:
  strategy: bayesian  # or grid_search
  objective: maximize_throughput
benchmark:
  num_concurrency: [1, 4, 8]
```

See `examples/docker_task.yaml` for complete example.

## Critical Notes

### Worker Restart Required
After editing `src/orchestrator.py`, `src/controllers/`, or `src/web/workers/`:
```bash
pkill -f arq && ./scripts/start_worker.sh
```

### Data Location
```
~/.local/share/inference-autotuner/
├── autotuner.db       # SQLite database
├── logs/task_*.log    # Task logs
└── datasets/          # Cached remote datasets
```

### Docker Mode
- Model path: `/mnt/data/models/{model_id}` → mounted as `/model`
- Ports: Auto-allocated 8000-8100
- GPU: Uses `device_requests`, NOT `CUDA_VISIBLE_DEVICES`

## Common Issues

| Issue | Solution |
|-------|----------|
| Task stuck in RUNNING | Check `logs/worker.log`, restart worker |
| Worker not processing | Verify Redis: `docker ps \| grep redis` |
| Model not found | Check path mapping in `/mnt/data/models/` |
| genai-bench errors | Ensure `additional_params` uses correct types (float, not string) |

See `docs/troubleshooting.md` for more.

## Meta-Instructions

1. **Restart ARQ worker** after editing backend code
2. **Update dev diary** (`agentlog/yyyy/mmdd.md`) on milestones
3. **Consult `docs/troubleshooting.md`** for issues
4. **Place docs in `./docs/`**
5. **DO NOT** use git commands with writing effects
6. **Follow `CLAUDE.local.md`** if present

## Documentation

- [README.md](README.md) - Overview and quick start
- [docs/getting-started/](docs/getting-started/) - Installation, quickstart
- [docs/user-guide/](docs/user-guide/) - Docker, Kubernetes, presets
- [docs/features/](docs/features/) - Bayesian, SLO, WebSocket, GPU tracking
- [docs/architecture/roadmap.md](docs/architecture/roadmap.md) - Product roadmap
- [agentlog/](agentlog/) - Development diary
