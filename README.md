# LLM Autotuner

<p align="center">
  <img src="frontend/public/favicon.svg" width="256" height="256" alt="Autotuner Logo" />
</p>

Automated parameter tuning for LLM inference engines (SGLang, vLLM).

## Features

- **Multiple Deployment Modes**: Docker, Local (direct GPU), OME (Kubernetes)
- **Web UI**: React frontend with real-time monitoring
- **Agent Assistant**: LLM-powered assistant for task management and troubleshooting
- **Optimization Strategies**: Grid search, Bayesian optimization
- **SLO-Aware Scoring**: Exponential penalties for constraint violations

## Quick Start

**→ [Get started in 5 minutes with Docker](docs/getting-started/quickstart.md)**

```bash
# Install
pip install -r requirements.txt && pip install genai-bench

# Run
python src/run_autotuner.py examples/docker_task.json --mode docker
```

## Web UI

```bash
# Start backend + worker
./scripts/start_dev.sh

# Start frontend (separate terminal)
cd frontend && npm run dev
```

Access at http://localhost:5173

## Documentation

**Full Documentation**: https://novitalabs.github.io/inference-autotuner/

### Project Overview
- [ROADMAP.md](docs/architecture/roadmap.md) - **Product roadmap with completed milestones and future plans**

### Setup & Deployment
- [Installation Guide](docs/getting-started/installation.md) - **Complete installation guide**
- [Quick Start](docs/getting-started/quickstart.md) - Quick start tutorial
- [Docker Mode](docs/user-guide/docker-mode.md) - Docker deployment guide
- [Kubernetes/OME](docs/user-guide/kubernetes.md) - Kubernetes/OME setup

### Features & Configuration
- [SLO Scoring](docs/features/slo-scoring.md) - SLO-aware scoring with exponential penalties
- [Parallel Execution](docs/features/parallel-execution.md) - Parallel experiment execution
- [WebSocket Implementation](docs/features/websocket.md) - Real-time updates via WebSocket
- [Quantization Parameters](docs/UNIFIED_QUANTIZATION_PARAMETERS.md) - Quantization configuration
- [Parameter Presets](docs/user-guide/presets.md) - Parameter preset system
- [Bayesian Optimization](docs/features/bayesian-optimization.md) - Bayesian optimization strategy
- [GPU Tracking](docs/features/gpu-tracking.md) - GPU intelligent scheduling

### Operations & Troubleshooting
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

### Development History
- [agentlog/](agentlog/) - Daily development diary (yyyy/mmdd.md format) written directly to files

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines and project architecture.
