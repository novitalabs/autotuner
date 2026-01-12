# Quick Start Guide

Get started with the LLM Autotuner in 5 minutes.

## Option 1: Docker Demo Image (Recommended)

The fastest way to get started is using the pre-built Docker image.

### Prerequisites

1. **Docker with GPU support** - Verify NVIDIA Container Toolkit is working:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   If this fails, install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) first.

2. **A model** downloaded locally:
   ```bash
   pip install huggingface_hub
   huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
     --local-dir /mnt/data/models/llama-3-2-1b-instruct
   ```

### Run the Container

```bash
docker run -d --name autotuner \
  -p 8000:8000 \
  -v /mnt/data/models:/mnt/data/models \
  -v autotuner-data:/data \
  -e HF_TOKEN=your_huggingface_token \
  --gpus all \
  novitalabs/autotuner-demo:v0.2.1
```

**Environment Variables:**

| Variable | Description | Required |
|----------|-------------|----------|
| `HF_TOKEN` | HuggingFace token (for gated models like Llama) | For gated models |
| `HTTP_PROXY` | Proxy for HuggingFace downloads | Optional |
| `HTTPS_PROXY` | Proxy for HTTPS connections | Optional |
| `TZ` | Timezone (e.g., `Asia/Singapore`) | Optional |

**Volume Mounts:**

| Mount | Description |
|-------|-------------|
| `/mnt/data/models` | Directory containing your models |
| `/data` | Persistent storage for database and results |

### Access the Web UI

Open http://localhost:8000 in your browser.

### Create Your First Task

1. Click **Tasks** in the sidebar
2. Click **Create Task**
3. Fill in the configuration:
   - **Task Name:** `my-tune`
   - **Model ID:** `llama-3-2-1b-instruct`
   - **Runtime:** SGLang
   - Add parameter: `mem-fraction-static` with values `0.8, 0.9`
4. Click **Create Task**
5. Click **Start Task** to begin autotuning

Or create via API:

```bash
curl -X POST http://localhost:8000/api/tasks/ \
  -H "Content-Type: application/json" \
  -d '{
    "task_name": "my-tune",
    "model": {"id_or_path": "llama-3-2-1b-instruct"},
    "base_runtime": "sglang",
    "parameters": {"mem-fraction-static": [0.8, 0.9]},
    "optimization": {"strategy": "grid_search", "objective": "maximize_throughput"},
    "benchmark": {"num_concurrency": [1, 4]}
  }'

# Start the task
curl -X POST http://localhost:8000/api/tasks/1/start
```

---

## Option 2: Development Installation

For development or customization, install from source.

### Prerequisites

1. **Docker with GPU support** - Verify NVIDIA Container Toolkit is working:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
   If this fails, install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) first.

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

### Installation

```bash
git clone <repository-url>
cd autotuner

pip install -r requirements.txt
pip install genai-bench
```

### Configuration

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

### Run Your First Task

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

**Tip:** Drag and drop a YAML file onto the New Task page to quickly import a task configuration.

---

## Example Configuration

```yaml
task_name: my-tune

model:
  id_or_path: Qwen/Qwen2.5-0.5B-Instruct

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

View results in the Web UI Dashboard or via API:

```bash
# Get task status
curl http://localhost:8000/api/tasks/1

# Get best experiment results
curl http://localhost:8000/api/experiments/{best_experiment_id}
```

## Next Steps

- [Docker Mode](../user-guide/docker-mode.md) - Full Docker documentation
- [SLO Scoring](../features/slo-scoring.md) - Add SLO constraints
- [Bayesian Optimization](../features/bayesian-optimization.md) - Smarter optimization
