# Docker Deployment Mode

This document describes the standalone Docker deployment mode for the Inference Autotuner.

## Overview

The Docker mode allows you to run autotuning experiments using standalone Docker containers instead of Kubernetes. This is useful for:

- **Development and testing** without full Kubernetes setup
- **Single-node deployments** where Kubernetes overhead is unnecessary
- **Quick prototyping** with direct GPU access
- **CI/CD pipelines** where Docker is available but Kubernetes is not

## Architecture

### Docker Mode
```
Autotuner Orchestrator
    ↓
Docker Controller
    ↓
Docker Containers (with GPUs)
    ↓
Direct Benchmark (genai-bench CLI)
```

### Comparison: OME vs Docker Mode

| Feature | OME Mode | Docker Mode |
|---------|----------|-------------|
| Infrastructure | Kubernetes + OME | Docker only |
| Setup Complexity | High | Low |
| Resource Requirements | K8s cluster | Docker + GPU |
| Model Deployment | InferenceService CRD | Docker containers |
| Benchmark Execution | BenchmarkJob or Direct CLI | Direct CLI only |
| Use Case | Production, multi-node | Development, single-node |

## Prerequisites

### Required
1. **Docker** with GPU support
   ```bash
   docker --version
   # Docker version 20.10+
   ```

2. **NVIDIA Docker Runtime**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

3. **Python 3.8+** with dependencies
   ```bash
   pip install docker requests
   pip install genai-bench  # For benchmarking
   ```

4. **Model files** downloaded locally
   ```bash
   # Example: Download Llama model
   huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
     --local-dir /mnt/data/models/llama-3-2-1b-instruct
   ```

### Optional
- Multiple GPUs for tensor parallelism testing

## Installation

### 1. Install Docker Dependencies

```bash
# Install Docker SDK for Python
pip install docker

# Or use the virtual environment
source env/bin/activate
pip install docker
```

### 2. Verify GPU Access

```bash
# Check GPU availability
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### 3. Download Models

```bash
# Create model directory
mkdir -p /mnt/data/models

# Download model (example)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /mnt/data/models/llama-3-2-1b-instruct
```

## Usage

### Basic Usage

```bash
# Run autotuning in Docker mode
python src/run_autotuner.py examples/docker_task.json --mode docker
```

### Advanced Options

```bash
# Custom model path
python src/run_autotuner.py examples/docker_task.json \
  --mode docker \
  --model-path /data/models

# With verbose logging (future enhancement)
python src/run_autotuner.py examples/docker_task.json \
  --mode docker \
  --verbose
```

### Task Configuration

Create a task JSON file (see `examples/docker_task.json`):

```json
{
  "task_name": "docker-simple-tune",
  "description": "Docker deployment test",
  "deployment_mode": "docker",
  "model": {
    "name": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang",
  "parameters": {
    "tp_size": {"type": "choice", "values": [1, 2]},
    "mem_frac": {"type": "choice", "values": [0.7, 0.8]}
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
    "max_requests_per_iteration": 50
  }
}
```

## Workflow

### For Each Experiment:

1. **Deploy Model Container**
   - Pull Docker image (e.g., `lmsysorg/sglang:v0.5.2-cu126`)
   - Mount model volume
   - Allocate GPUs
   - Start container with tuning parameters

2. **Wait for Service Ready**
   - Poll health endpoint
   - Timeout after 600s (configurable)

3. **Run Benchmark**
   - Execute genai-bench CLI
   - Target: `http://localhost:<port>`
   - Collect metrics

4. **Cleanup**
   - Stop and remove container
   - Release GPU resources

## Supported Runtimes

### SGLang (Default)

```json
{
  "base_runtime": "sglang"
}
```

**Docker Image:** `lmsysorg/sglang:v0.5.2-cu126`

**Parameters:**
- `tp_size`: Tensor parallelism size
- `mem_frac`: GPU memory fraction

### vLLM

```json
{
  "base_runtime": "vllm"
}
```

**Docker Image:** `vllm/vllm-openai:latest`

**Parameters:**
- `tp_size`: Tensor parallel size
- `mem_frac`: GPU memory utilization

## Configuration Details

### Model Path Mapping

Host path → Container path:
```
/mnt/data/models/llama-3-2-1b-instruct → /model
```

The controller automatically:
1. Resolves model name to host path
2. Mounts as read-only volume
3. Sets `MODEL_PATH=/model` environment variable

### GPU Selection

The controller automatically:
1. Queries GPU availability using `nvidia-smi`
2. Selects GPUs with most free memory
3. Sets `CUDA_VISIBLE_DEVICES` environment variable

Example for `tp_size=2`:
```bash
# Selects GPUs 0 and 1 (most free memory)
CUDA_VISIBLE_DEVICES=0,1
```

### Port Management

The controller:
1. Scans ports 8000-8100 for availability
2. Assigns first available port
3. Maps container port 8080 to host port

Example:
```
Container port 8080 → Host port 8001
Service URL: http://localhost:8001
```

## Troubleshooting

### 1. Docker Connection Failed

**Error:**
```
Failed to connect to Docker daemon
```

**Solution:**
```bash
# Check Docker service
sudo systemctl status docker

# Start Docker
sudo systemctl start docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

### 2. GPU Not Available

**Error:**
```
Failed to allocate N GPU(s)
```

**Solution:**
```bash
# Check GPU status
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Install/configure NVIDIA Docker runtime
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### 3. Container Startup Failed

**Error:**
```
Container status: exited
```

**Solution:**
```bash
# Check container logs (preserved for debugging)
docker logs <container-name>

# Common issues:
# - Model path not found
# - Insufficient GPU memory
# - Image pull errors
```

### 4. Model Path Not Found

**Error:**
```
Model path /mnt/data/models/llama-3-2-1b-instruct does not exist
```

**Solution:**
```bash
# Verify model path
ls -l /mnt/data/models/llama-3-2-1b-instruct

# Update model path in CLI
python src/run_autotuner.py task.json \
  --mode docker \
  --model-path /correct/path/to/models
```

### 5. Port Already in Use

**Error:**
```
No available ports in range 8000-8100
```

**Solution:**
```bash
# Check port usage
netstat -tulpn | grep 800

# Kill conflicting processes or wait for cleanup
docker ps | grep autotuner
docker stop <container-id>
```

### 6. Out of Memory

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory
```

**Solution:**
```json
{
  "parameters": {
    "mem_frac": {"type": "choice", "values": [0.6]}
  }
}
```

Adjust `mem_frac` based on GPU memory and model size.

## Performance Tips

### 1. Pre-pull Images

```bash
# Pre-pull Docker images to avoid delays
docker pull lmsysorg/sglang:v0.5.2-cu126
docker pull vllm/vllm-openai:latest
```

### 2. Use Local Models

- Store models on fast SSD
- Avoid network-mounted storage for better performance

### 3. GPU Selection

For multi-GPU systems, the controller automatically selects GPUs with most free memory. You can influence selection by:
- Running cleanup between experiments
- Monitoring GPU usage with `nvidia-smi`

### 4. Reduce Timeout for Testing

```json
{
  "optimization": {
    "timeout_per_iteration": 300
  }
}
```

## Limitations

### Current Limitations

1. **Single-node only** - No distributed deployment
2. **Sequential execution** - One experiment at a time
3. **Basic GPU allocation** - No advanced scheduling
4. **Limited runtime support** - SGLang and vLLM only

### Future Enhancements

- [ ] Parallel experiment execution
- [ ] Advanced GPU scheduling and allocation
- [ ] Support for more runtimes (TensorRT-LLM, etc.)
- [ ] Container resource limits (CPU, memory)
- [ ] Better error recovery and retry logic
- [ ] Docker Compose integration for complex setups

## Comparison with OME Mode

### When to Use Docker Mode

✅ **Use Docker mode when:**
- Developing and testing locally
- Running on a single node
- Quick prototyping without K8s overhead
- CI/CD pipeline with Docker
- Direct GPU access needed

❌ **Don't use Docker mode when:**
- Need multi-node distributed deployment
- Require Kubernetes orchestration features
- Production deployment with HA requirements
- Complex networking or service mesh needed

### Migration Path

**Development → Production:**

1. **Develop with Docker mode**
   ```bash
   python src/run_autotuner.py task.json --mode docker
   ```

2. **Test with OME mode locally**
   ```bash
   minikube start
   ./install.sh --install-ome
   python src/run_autotuner.py task.json --mode ome --direct
   ```

3. **Deploy to production with OME**
   ```bash
   python src/run_autotuner.py task.json --mode ome
   ```

The task configuration remains compatible across modes!

## Examples

### Example 1: Basic Test

```bash
# Download model
mkdir -p /tmp/models
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /tmp/models/llama-3-2-1b-instruct

# Run autotuning
python src/run_autotuner.py examples/docker_task.json \
  --mode docker \
  --model-path /tmp/models
```

### Example 2: Multi-GPU Test

```json
{
  "parameters": {
    "tp_size": {"type": "choice", "values": [1, 2, 4]},
    "mem_frac": {"type": "choice", "values": [0.8]}
  }
}
```

```bash
python src/run_autotuner.py task.json --mode docker
```

### Example 3: Quick Iteration

Minimal config for fast testing:

```json
{
  "parameters": {
    "mem_frac": {"type": "choice", "values": [0.7, 0.8]}
  },
  "optimization": {
    "timeout_per_iteration": 300
  },
  "benchmark": {
    "max_requests_per_iteration": 20
  }
}
```

## See Also

- [Main README](../README.md) - General documentation
- [OME Installation](../docs/OME_INSTALLATION.md) - Kubernetes setup
- [examples/docker_task.json](../examples/docker_task.json) - Example configuration
