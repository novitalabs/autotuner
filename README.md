# LLM Inference Autotuner

Automated parameters tuning for LLM inference engines.

**Features:**
- **Dual Deployment Modes**: OME (Kubernetes) or Docker (Standalone)
- **Web UI**: React frontend with real-time task monitoring
- **Web API**: FastAPI-based REST API for task management
- **Background Processing**: ARQ task queue with Redis
- **Database**: SQLite for task and experiment tracking
- **SLO-Aware Scoring**: Exponential penalties for Service Level Objective violations

## Deployment Modes

### Mode 1: OME (Kubernetes)

Full-featured Kubernetes deployment using OME operator.

**Use cases:**
- Production deployments
- Multi-node clusters
- Advanced orchestration needs

**Requirements:**
- Kubernetes v1.28+
- OME operator installed
- kubectl configured

**Quick start:**
```bash
./install.sh --install-ome
python src/run_autotuner.py examples/simple_task.json --mode ome
```

### Mode 2: Docker (Standalone)

Lightweight standalone deployment using Docker containers.

**Use cases:**
- Development and testing
- Single-node deployments
- CI/CD pipelines
- Quick prototyping

**Requirements:**
- Docker with GPU support
- Model files downloaded locally
- No Kubernetes needed

**Quick start:**
```bash
pip install docker
python src/run_autotuner.py examples/docker_task.json --mode docker
```

**See [docs/DOCKER_MODE.md](docs/DOCKER_MODE.md) for complete Docker mode documentation.**

## Web API

The autotuner includes a FastAPI-based web service and React frontend for managing tuning tasks:

**Features:**
- React UI with task creation wizard and results visualization
- Create and manage tuning tasks via REST API
- Track experiment progress and results in real-time
- Configure SLO constraints with exponential penalties
- Background job processing with ARQ and Redis
- OpenAPI/Swagger documentation at `/docs`

**Starting the Full Stack:**
```bash
# Terminal 1: Start backend (API + Worker)
./scripts/start_dev.sh

# Terminal 2: Start frontend
cd frontend && npm run dev
```

Then access:
- **Frontend UI**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

**Key Endpoints:**
- `POST /api/tasks/` - Create new tuning task
- `GET /api/tasks/` - List all tasks
- `GET /api/tasks/{id}` - Get task details
- `POST /api/tasks/{id}/start` - Start task execution
- `GET /api/experiments/task/{id}` - Get experiments for task

**Database Storage:**
Task and experiment data is stored in SQLite at:
```
~/.local/share/inference-autotuner/autotuner.db
```

This location follows XDG Base Directory standards and persists independently of the codebase.

### Proxy Configuration

If you're behind a corporate proxy or need to access HuggingFace.co through a proxy:

**IMPORTANT**: The ARQ worker needs proxy settings to download models and tokenizers from HuggingFace. However, `NO_PROXY` must be configured to prevent localhost connections from being proxied (which would break health checks and benchmarking).

To configure proxy for the worker:

1. Edit `scripts/start_worker.sh` or `scripts/start_dev.sh`
2. Uncomment and update the proxy variables:
   ```bash
   export HTTP_PROXY=http://your-proxy-server:port
   export HTTPS_PROXY=http://your-proxy-server:port
   export NO_PROXY=localhost,127.0.0.1
   ```
3. Restart the worker for changes to take effect

**Why NO_PROXY is crucial**: Without `NO_PROXY`, the Python `requests` library will attempt to proxy all HTTP requests including those to `localhost`. This causes:
- Health check failures (can't connect to local inference services)
- Benchmark failures (genai-bench can't reach the local endpoint)
- Service readiness detection timeouts


## Prerequisites

### For OME Mode

**IMPORTANT: OME (Open Model Engine) is a required prerequisite for OME mode.**

1. **OME Operator** (Open Model Engine) - **REQUIRED**
   - Version: v0.1.3 or later
   - Installed in `ome` namespace
   - All CRDs must be present: `inferenceservices`, `benchmarkjobs`, `clusterbasemodels`, `clusterservingruntimes`
   - **Installation Guide**: See [docs/OME_INSTALLATION.md](docs/OME_INSTALLATION.md) for detailed setup instructions

2. **Kubernetes cluster** (v1.28+) with OME installed
   - Tested on Minikube v1.34.0
   - Single-node or multi-node cluster
   - GPU support required for inference workloads

3. **kubectl** configured to access the cluster

4. **Python 3.8+** with pip

5. **Model and Runtime Resources**
   - At least one `ClusterBaseModel` available
   - At least one `ClusterServingRuntime` configured
   - Example: `llama-3-2-1b-instruct` model with `llama-3-2-1b-instruct-rt` runtime
   - Setup instructions in [docs/OME_INSTALLATION.md](docs/OME_INSTALLATION.md)

### For Docker Mode

1. **Docker** with GPU support
   - Docker 20.10+ with NVIDIA Container Toolkit
   - Test: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

2. **Python 3.8+** with dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. **Model files** downloaded locally
   ```bash
   mkdir -p /mnt/data/models
   huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
     --local-dir /mnt/data/models/llama-3-2-1b-instruct
   ```

4. **genai-bench** for benchmarking
   ```bash
   pip install genai-bench
   ```

5. **Redis** (optional, for Web API background jobs)
   ```bash
   docker run -d -p 6379:6379 redis:alpine
   ```

**See [docs/DOCKER_MODE.md](docs/DOCKER_MODE.md) for complete setup guide.**

### Environment Verification

**For OME Mode:**
```bash
# Check Kubernetes connection
kubectl cluster-info

# Check OME installation
kubectl get pods -n ome
kubectl get crd | grep ome.io

# Check available models and runtimes
kubectl get clusterbasemodels
kubectl get clusterservingruntimes

# Verify resources
kubectl describe node | grep -A 5 "Allocated resources"
```

**For Docker Mode:**
```bash
# Check Docker
docker --version
docker ps

# Check GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Python dependencies
python -c "import docker; print('Docker SDK:', docker.__version__)"
python -c "import genai_bench; print('genai-bench installed')"
```

Expected output:
- OME controller pods running
- CRDs: `inferenceservices.ome.io`, `benchmarkjobs.ome.io`, etc.
- At least one model in Ready state
- At least one runtime available

## Installation

### Quick Installation (Recommended)

The installation script automatically installs all dependencies including OME:

```bash
# Clone repository
git clone <repository-url>
cd inference-autotuner

# Run installation with OME
./install.sh --install-ome
```

This will:
- ✅ Install Python virtual environment and dependencies
- ✅ Install genai-bench CLI
- ✅ Install cert-manager (OME dependency)
- ✅ Install OME operator with all CRDs
- ✅ Create Kubernetes namespace and PVC
- ✅ Verify all installations

### Manual Installation

If you prefer to install OME separately or already have it installed:

```bash
# 1. Install OME first (if not already installed)
#    See docs/OME_INSTALLATION.md for detailed instructions

# 2. Run autotuner installation
./install.sh
```

### Installation Options

```bash
./install.sh --help              # Show all options
./install.sh --install-ome       # Install with OME (recommended)
./install.sh --skip-venv         # Skip Python virtual environment
./install.sh --skip-k8s          # Skip Kubernetes resources
```

### Post-Installation

After installation, create model and runtime resources:

```bash
# Apply example resources (requires model access)
kubectl apply -f third_party/ome/config/models/meta/Llama-3.2-1B-Instruct.yaml

# Or create your own ClusterBaseModel and ClusterServingRuntime
# See docs/OME_INSTALLATION.md for examples
```

## Usage

### Command Line Interface

```bash
# Show help
python src/run_autotuner.py --help

# OME mode (default) with K8s BenchmarkJob
python src/run_autotuner.py examples/simple_task.json

# OME mode with direct genai-bench CLI
python src/run_autotuner.py examples/simple_task.json --direct

# Docker mode (standalone)
python src/run_autotuner.py examples/docker_task.json --mode docker

# Docker mode with custom model path
python src/run_autotuner.py examples/docker_task.json --mode docker --model-path /data/models
```

### Mode Selection

| CLI Argument | Description | Default |
|--------------|-------------|---------|
| `--mode ome` | Use Kubernetes + OME | Yes |
| `--mode docker` | Use standalone Docker | No |
| `--direct` | Use direct genai-bench CLI (OME mode only) | No |
| `--kubeconfig PATH` | Path to kubeconfig (OME mode) | Auto-detect |
| `--model-path PATH` | Base path for models (Docker mode) | `/mnt/data/models` |

### Benchmark Execution Modes

The autotuner supports two benchmark execution modes:

1. **Kubernetes BenchmarkJob Mode** (OME mode only):
   - Uses OME's BenchmarkJob CRD
   - Runs genai-bench in Kubernetes pods
   - Requires working genai-bench Docker image
   - More complex but native to OME

2. **Direct CLI Mode** (Recommended):
   - Runs genai-bench directly using local installation
   - Automatic port forwarding to InferenceService
   - Bypasses Docker image issues
   - Faster and more reliable for prototyping

### 1. Direct CLI Mode (Recommended)

Run benchmarks using the local genai-bench installation:

```bash
python src/run_autotuner.py examples/simple_task.json --direct
```

**How it works:**
- Deploys InferenceService via OME
- Automatically sets up `kubectl port-forward` to access the service
- Runs genai-bench CLI directly from `env/bin/genai-bench`
- Cleans up port forward after completion
- No Docker image dependencies

**Requirements:**
- genai-bench installed in Python environment (`pip install genai-bench`)
- `kubectl` configured and accessible
- No additional configuration needed

### 2. Kubernetes BenchmarkJob Mode

Run benchmarks using OME's BenchmarkJob CRD:

```bash
python src/run_autotuner.py examples/simple_task.json
```

**How it works:**
- Creates Kubernetes BenchmarkJob resources
- Uses genai-bench Docker image
- Results stored in PersistentVolumeClaim

**Requirements:**
- PVC created (see installation step 3b)
- Working genai-bench Docker image accessible to cluster

### 1. Create a Tuning Task JSON

See `examples/simple_task.json` for the schema:

```json
{
  "task_name": "simple-tune",
  "description": "Description of the tuning task",
  "model": {
    "name": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang-base-runtime",
  "parameters": {
    "tp_size": {"type": "choice", "values": [1, 2]},
    "mem_frac": {"type": "choice", "values": [0.85, 0.9]}
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency",
    "max_iterations": 4,
    "timeout_per_iteration": 600
  },
  "benchmark": {
    "task": "text-to-text",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [1, 4],
    "max_time_per_iteration": 10,
    "max_requests_per_iteration": 50,
    "additional_params": {"temperature": "0.0"}
  }
}
```

### 2. Run the Autotuner

```bash
# Basic usage (uses default kubeconfig)
python src/run_autotuner.py examples/simple_task.json

# Specify kubeconfig path
python src/run_autotuner.py examples/simple_task.json /path/to/kubeconfig
```

### 3. View Results

Results are saved to `results/<task_name>_results.json`

## How It Works

1. **Load Task**: Read JSON configuration file
2. **Generate Parameter Grid**: Create all parameter combinations (grid search)
3. **For Each Configuration**:
   - Deploy InferenceService with parameters
   - Wait for service to be ready
   - Create and run BenchmarkJob
   - Collect metrics
   - Clean up resources
4. **Find Best**: Compare objective scores and report best configuration

## Workflow Example

```
Task: simple-tune (4 combinations: 2 x 2)

Experiment 1: {tp_size: 1, mem_frac: 0.85}
  → Deploy InferenceService
  → Wait for ready
  → Run benchmark
  → Score: 125.3ms

Experiment 2: {tp_size: 1, mem_frac: 0.9}
  → Deploy InferenceService
  → Wait for ready
  → Run benchmark
  → Score: 118.7ms

... (continue for all combinations)

Best: {tp_size: 2, mem_frac: 0.9} → Score: 89.2ms
```

## Configuration Details

### Parameter Types

Currently supported:
- `choice`: List of discrete values

### Optimization Strategies

Currently supported:
- `grid_search`: Exhaustive search over all combinations

### Objectives

Currently supported:
- `minimize_latency`: Minimize average end-to-end latency
- `maximize_throughput`: Maximize tokens/second
- `minimize_ttft`: Minimize Time to First Token
- `minimize_tpot`: Minimize Time Per Output Token

### SLO-Aware Scoring

The autotuner supports sophisticated Service Level Objective (SLO) constraints with exponential penalties:

**Supported Metrics:**
- **Latency Percentiles**: P50, P90, P99 end-to-end latency
- **TTFT**: Time to First Token (initial response latency)
- **TPOT**: Time Per Output Token (sustained generation speed)

**Key Features:**
- Exponential penalty curves create steep gradients near SLO boundaries
- Tiered enforcement: soft penalties for minor violations, hard failures for severe violations
- Multi-metric cumulative penalties with configurable weights
- Configurable steepness parameter (lower = steeper penalties)

**Example Configuration:**
```json
{
  "slo": {
    "ttft": {
      "threshold": 1.0,
      "weight": 2.0,
      "hard_fail": false
    },
    "tpot": {
      "threshold": 0.05,
      "weight": 2.0,
      "hard_fail": false
    },
    "latency": {
      "p90": {
        "threshold": 5.0,
        "weight": 2.0,
        "hard_fail": true,
        "fail_ratio": 0.2
      }
    },
    "steepness": 0.1
  }
}
```

**See [docs/SLO_SCORING.md](docs/SLO_SCORING.md) for complete documentation.**

## Limitations (Prototype)

- Sequential execution (no parallel experiments)
- Basic error handling
- Simplified metric extraction

## Current Implementation Status

**Completed:**
- ✅ Database persistence (SQLite with SQLAlchemy)
- ✅ REST API (FastAPI with OpenAPI docs)
- ✅ Background job processing (ARQ with Redis)
- ✅ Web frontend (React with Vite, TanStack Query, Tailwind CSS)
- ✅ Dual deployment modes (OME + Docker)
- ✅ User data separation (~/.local/share/)
- ✅ Code reorganization (unified src/ structure)
- ✅ SLO-aware scoring with exponential penalties
- ✅ Real-time task monitoring and log viewing

**TODO:**
- Bayesian optimization (Optuna integration in progress)
- Parallel experiment execution
- Advanced error handling
- WebSocket support for real-time updates
- Enhanced result visualization and comparison

## Troubleshooting

For detailed troubleshooting guidance, common issues, and solutions, see [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

Quick reference:
- InferenceService deployment issues
- GPU resource problems
- Docker and Kubernetes configuration
- Model download and transfer
- Benchmark execution errors
- Monitoring and performance tips

## Next Steps

For production implementation:
1. ~~Add database backend~~ ✅ **Completed** - SQLite with SQLAlchemy ORM
2. ~~Implement REST API~~ ✅ **Completed** - FastAPI with OpenAPI
3. ~~Add web UI~~ ✅ **Completed** - React UI with task management and monitoring
4. Add Bayesian optimization (Optuna integration in progress)
5. Enable parallel experiment execution (multi-threaded/async)
6. Improve error handling and retry logic
7. Add comprehensive logging and monitoring
8. Implement metric aggregation and visualization
9. Add user authentication and multi-tenancy
10. Migrate to PostgreSQL for production scale

## Documentation

### Project Overview
- [ROADMAP.md](docs/ROADMAP.md) - **Product roadmap with completed milestones and future plans**

### Setup & Deployment
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
