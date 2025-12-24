# Installation Guide

Complete installation instructions for the LLM Inference Autotuner.

## System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (for inference)
- **Memory**: 16GB+ RAM recommended

## Automated Install (Recommended)

Use the installation script for automated setup:

```bash
# Clone repository
git clone <repository-url>
cd inference-autotuner

# Run installation script
./install.sh
```

**Script options:**
```bash
./install.sh --help          # Show all options
./install.sh --skip-k8s      # Skip Kubernetes setup (for Docker/Local mode)
./install.sh --skip-venv     # Use system Python instead of venv
./install.sh --install-ome   # Include OME operator installation
```

**What install.sh does:**
- Creates Python virtual environment (`env/`)
- Installs Python dependencies from requirements.txt
- Installs genai-bench CLI
- Creates data directories (`~/.local/share/inference-autotuner/`)
- Verifies installation

## Manual Install

If you prefer manual installation:

```bash
# Clone repository
git clone <repository-url>
cd inference-autotuner

# Create virtual environment
python3 -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install genai-bench

# Install frontend dependencies
cd frontend && npm install && cd ..

# Create data directory
mkdir -p ~/.local/share/inference-autotuner

# Start Redis (for background jobs)
docker run -d -p 6379:6379 redis:alpine
```

## Deployment Mode Setup

### Docker Mode (Recommended for beginners)

**Requirements:**
- Docker 20.10+ with NVIDIA Container Toolkit

**Verify GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Download a model:**
```bash
pip install huggingface_hub
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /mnt/data/models/llama-3-2-1b-instruct
```

### Local Mode (Direct GPU)

For running inference servers directly on local GPU without Docker:

```bash
# Install SGLang
pip install sglang[all]

# Or install vLLM
pip install vllm
```

### OME Mode (Kubernetes)

See [Kubernetes Guide](../user-guide/kubernetes.md) for Kubernetes setup.

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Server ports
SERVER_PORT=8000
FRONTEND_PORT=5173

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Model path (Docker mode)
DOCKER_MODEL_PATH=/mnt/data/models

# Proxy (if needed)
HTTP_PROXY=http://proxy:port
HTTPS_PROXY=http://proxy:port
NO_PROXY=localhost,127.0.0.1

# HuggingFace token (for gated models)
HF_TOKEN=your_token_here
```

### Database

SQLite database is auto-created at:
```
~/.local/share/inference-autotuner/autotuner.db
```

## Starting Services

```bash
# Activate environment
source env/bin/activate

# Start backend + ARQ worker
./scripts/start_dev.sh

# Start frontend (separate terminal)
cd frontend && npm run dev
```

**Default ports:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Verification

```bash
# Check backend health
curl http://localhost:8000/api/system/health

# Expected: {"status":"healthy","database":"ok","redis":"ok"}
```

## Troubleshooting

See [Troubleshooting](../troubleshooting.md) for common issues.

**Common issues:**
- Redis not running → `docker run -d -p 6379:6379 redis:alpine`
- GPU not accessible → Check NVIDIA drivers and Docker runtime
- Port conflicts → Update ports in `.env` file
