# LLM Autotuner (for inference)

<p align="center">
  <img src="frontend/public/favicon.svg" width="256" height="256" alt="Autotuner Logo" />
</p>

Automated parameter tuning for LLM inference engines (SGLang, vLLM) for best performance, while respecting SLOs and hardware constraints.

## Why Autotuner?

LLM inference engines like SGLang and vLLM expose dozens of tunable parameters (`mem-fraction-static`, `max-running-requests`, `chunked-prefill-size`, etc.). Finding the optimal combination manually is:

- **Time-consuming**: Each configuration requires deploying a container, running benchmarks, and analyzing results
- **Error-prone**: Parameter interactions are complex and non-intuitive
- **Hardware-dependent**: Optimal settings vary by GPU model, memory, and workload

### Real-World Example: 4% Throughput Gain in Minutes

We compared an **optimized SGLang configuration** (found via Bayesian optimization) against **baseline defaults** on RTX 4090:

| Configuration | Mean Throughput | P99 Latency |
|--------------|-----------------|-------------|
| **Optimized** | 11,298 tok/s | 350 ms |
| Baseline | 10,862 tok/s | 390 ms |
| **Improvement** | **+4.01%** | **-10.2%** |

The performance gain scales with concurrency:

| Concurrency | Improvement |
|-------------|-------------|
| 1 | +0.2% |
| 4 | +2.1% |
| 8 | **+8.0%** |

**Key optimized parameters:**
```yaml
mem-fraction-static: 0.85      # GPU memory allocation
max-running-requests: 128      # Concurrent request capacity
chunked-prefill-size: 4096     # Prefill batch optimization
enable-mixed-chunk: true       # Overlapped prefill/decode
```

### What Would Take Hours Manually

Without Autotuner, achieving this result requires:

1. **Research** SGLang documentation for tunable parameters
2. **Design** a parameter search space (which combinations to try?)
3. **Script** container deployment, health checks, benchmark execution
4. **Run** experiments sequentially (each takes 2-5 minutes)
5. **Analyze** results, identify best configuration
6. **Repeat** for different models, GPUs, or workloads

### What Autotuner Does in One Command

```bash
python src/run_autotuner.py task.yaml --mode docker
```

Autotuner handles:
- Container lifecycle management (deploy, health check, cleanup)
- Benchmark execution with `genai-bench`
- SLO-aware scoring (penalize latency violations)
- Bayesian optimization (80%+ fewer trials than grid search)
- Result persistence and comparison

### Bottom Line

| Metric | Manual Tuning | With Autotuner |
|--------|---------------|----------------|
| Time to optimal config | Hours/Days | Minutes |
| Parameter combinations tested | ~10 (limited by patience) | 50-100+ |
| Reproducibility | Low | High |
| Cross-hardware portability | Manual rework | Re-run task |

## How to Use

### CLI Mode
<p align="center">
  <img src="docs/assets/cli-flow.svg" width="700" alt="CLI Flow" />
</p>

### Web UI Mode
<p align="center">
  <img src="docs/assets/web-flow.svg" width="700" alt="Web UI Flow" />
</p>

### Agent Mode
<p align="center">
  <img src="docs/assets/agent-flow.svg" width="700" alt="Agent Flow" />
</p>

## Core Concepts
<p align="center">
  <img src="docs/assets/concepts.svg" width="700" alt="Core Concepts" />
</p>

- **Task**: A tuning job containing model config, parameter ranges, SLOs, and optimization strategy
- **Experiment**: Individual trial with specific parameter values; multiple experiments per task
- **ARQ Worker**: Background processor that deploys models, runs benchmarks, and scores results

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
python src/run_autotuner.py examples/docker_task.yaml --mode docker
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

[**Full Documentation**](https://novitalabs.github.io/autotuner/)

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

See [DEVELOPMENT](docs/DEVELOPMENT.md) for development guidelines and project architecture.
