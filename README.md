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

**→ [Get started in 5 minutes with Docker](docs/QUICKSTART.md)**

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

### Project Overview
- [ROADMAP.md](docs/ROADMAP.md) - **Product roadmap with completed milestones and future plans**

### Setup & Deployment
- [INSTALLATION.md](docs/INSTALLATION.md) - **Complete installation guide**
- [QUICKSTART.md](docs/QUICKSTART.md) - Quick start tutorial
- [DOCKER_MODE.md](docs/DOCKER_MODE.md) - Docker deployment guide
- [OME_INSTALLATION.md](docs/OME_INSTALLATION.md) - Kubernetes/OME setup

### Features & Configuration
- [SLO_SCORING.md](docs/SLO_SCORING.md) - SLO-aware scoring with exponential penalties
- [PARALLEL_EXECUTION.md](docs/PARALLEL_EXECUTION.md) - Parallel experiment execution
- [WEBSOCKET_IMPLEMENTATION.md](docs/WEBSOCKET_IMPLEMENTATION.md) - Real-time updates via WebSocket
- [UNIFIED_QUANTIZATION_PARAMETERS.md](docs/UNIFIED_QUANTIZATION_PARAMETERS.md) - Quantization configuration
- [PRESET_QUICK_REFERENCE.md](docs/PRESET_QUICK_REFERENCE.md) - Parameter preset system
- [BAYESIAN_OPTIMIZATION.md](docs/BAYESIAN_OPTIMIZATION.md) - Bayesian optimization strategy
- [GPU_TRACKING.md](docs/GPU_TRACKING.md) - GPU intelligent scheduling

### Operations & Troubleshooting
- [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) - Common issues and solutions

### Development History
- [agentlog/](agentlog/) - Daily development diary (yyyy/mmdd.md format) written directly to files

## Contributing

See [CLAUDE.md](CLAUDE.md) for development guidelines and project architecture.
