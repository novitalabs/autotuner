# Quick Start Guide

Get started with the LLM Autotuner in 5 minutes using Docker mode.

## Prerequisites

1. **Docker with GPU support**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. **Python 3.8+** and **Redis**
   ```bash
   docker run -d -p 6379:6379 redis:alpine
   ```

3. **A model** downloaded locally:
   ```bash
   pip install huggingface_hub
   huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
     --local-dir /mnt/data/models/llama-3-2-1b-instruct
   ```

## Installation

```bash
git clone <repository-url>
cd autotuner

pip install -r requirements.txt
pip install genai-bench
```

## Configuration

Copy the example environment file and customize as needed:

```bash
cp .env.example .env
```

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `SERVER_PORT` | Backend API port | 8000 |
| `FRONTEND_PORT` | Frontend dev server port | 5173 |
| `DOCKER_MODEL_PATH` | Host path to models | /mnt/data/models |
| `HF_TOKEN` | HuggingFace token (for gated models) | - |
| `HTTP_PROXY` | Proxy for HuggingFace downloads | - |

See `.env.example` for all available options including Agent and GitHub integration settings.

## Run Your First Task

**Command Line:**
```bash
python src/run_autotuner.py examples/docker_task.yaml --mode docker
```

**Or use the Web UI:**
```bash
# Terminal 1
./scripts/start_dev.sh

# Terminal 2
cd frontend && npm install && npm run dev
```

Then open http://localhost:5173 to create and monitor tasks.

**Tip:** Drag and drop a YAML file onto the New Task page (`#new-task`) to quickly import a task configuration.

## Example Configuration

```yaml
task_name: my-tune

model:
  id_or_path: llama-3-2-1b-instruct

base_runtime: sglang

parameters:
  mem-fraction-static: [0.8, 0.9]

optimization:
  strategy: grid_search
  objective: maximize_throughput

benchmark:
  num_concurrency: [1, 4]
```

## Results

View results in `results/<task_name>_results.json` or in the Web UI.

## Next Steps

- [Docker Mode](../user-guide/docker-mode.md) - Full Docker documentation
- [SLO Scoring](../features/slo-scoring.md) - Add SLO constraints
- [Bayesian Optimization](../features/bayesian-optimization.md) - Smarter optimization
