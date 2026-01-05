# Qwen2.5 Inference Benchmark Report

Comprehensive benchmark comparing baseline and optimized configurations for Qwen2.5-14B-Instruct and Qwen2.5-32B-Instruct on vLLM.

---

## 1. Experiment Environment

### Hardware

| Component | Specification |
|-----------|---------------|
| **GPUs** | 8x NVIDIA RTX 4090 |
| **VRAM per GPU** | 24GB (23.99GB usable) |
| **Memory Bandwidth** | 1008 GB/s per GPU |
| **Driver** | 565.57.01 |

### Software

| Component | Version |
|-----------|---------|
| **Docker** | 27.4.1 |
| **vLLM** | vllm/vllm-openai:latest (v0.13.0) |
| **Benchmark Tool** | genai-bench 0.0.2 |

### Models

| Model | Source | Size |
|-------|--------|------|
| **Qwen2.5-14B-Instruct** | Qwen/Qwen2.5-14B-Instruct | 28GB |
| **Qwen2.5-32B-Instruct** | Qwen/Qwen2.5-32B-Instruct | 59GB |

---

## 2. Runtime Configurations

### 14B Baseline

```bash
docker run -d \
  --name vllm-qwen14b-baseline \
  --gpus all \
  --ipc=host \
  -p 8010:8000 \
  --entrypoint vllm \
  vllm/vllm-openai:latest \
  serve Qwen/Qwen2.5-14B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2
```

### 14B Optimized

```bash
docker run -d \
  --name vllm-qwen14b-optimized \
  --gpus all \
  --ipc=host \
  -p 8011:8000 \
  -v /mnt/data/models/Qwen2.5-14B-Instruct:/model:ro \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name /model \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization fp8 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.98 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  --max-running-requests 1024 \
  --max-num-batched-tokens 4096
```

| Parameter | Value |
|-----------|-------|
| Quantization | FP8 (dynamic) |
| Chunked Prefill | Enabled (4096 tokens) |
| Tensor Parallel | 2 |
| GPUs | 2x RTX 4090 |
| Memory Utilization | 0.90 |
| Max Context Length | 4096 |

### 32B Baseline

```bash
docker run -d \
  --name vllm-qwen32b-baseline \
  --gpus all \
  --ipc=host \
  -p 8012:8000 \
  --entrypoint vllm \
  vllm/vllm-openai:latest \
  serve Qwen/Qwen2.5-32B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4
```

### 32B Optimized

```bash
docker run -d \
  --name vllm-qwen32b-optimized \
  --gpus all \
  --ipc=host \
  -p 8013:8000 \
  -v /mnt/data/models/Qwen2.5-32B-Instruct:/model:ro \
  vllm/vllm-openai:latest \
  --model /model \
  --served-model-name /model \
  --host 0.0.0.0 \
  --port 8000 \
  --quantization fp8 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096 \
  --enable-chunked-prefill \
  --max-running-requests 1024 \
  --max-num-batched-tokens 4096
```

| Parameter | Value |
|-----------|-------|
| Quantization | FP8 (dynamic) |
| Chunked Prefill | Enabled (4096 tokens) |
| Tensor Parallel | 4 |
| GPUs | 4x RTX 4090 |
| Memory Utilization | 0.85 |
| Max Context Length | 4096 |

---

## 3. Benchmark Methodology

### Dataset

genai-bench default dataset (sonnet.txt) with default traffic scenarios.

### Test Configuration

| Parameter | Value |
|-----------|-------|
| Tool | genai-bench 0.0.2 |
| Dataset | Default (sonnet.txt) |
| Traffic Scenarios | Default |
| Max Requests | 1024 |
| Max Time | 30 minutes |
| API Backend | OpenAI-compatible |

### Metrics Collected

- **Output Throughput**: Generated tokens per second (output only)
- **Total Throughput**: Input + output tokens per second
- **E2E Latency**: End-to-end request latency (P50, P99)
- **TTFT**: Time to first token (P50)

---

## 4. Results

### 4.1 Max Throughput Test

| Config | Concurrency | Output Throughput | Total Throughput | E2E P50 | E2E P99 | TTFT P50 |
|--------|-------------|-------------------|------------------|---------|---------|----------|
| **14B Baseline** | 192 | 1172.50 tok/s | 1292.40 tok/s | 14.95s | 33.22s | 5.33s |
| **14B Optimized** | 512 | 2397.85 tok/s | 2668.99 tok/s | 14.63s | 29.10s | 525.6ms |
| **32B Baseline** | 160 | 1580.47 tok/s | 1707.21 tok/s | 11.21s | 28.82s | 101.8ms |
| **32B Optimized** | 384 | 1854.70 tok/s | 2002.71 tok/s | 20.75s | 43.46s | 180.7ms |

### 4.2 Min Latency Test

| Config | Throughput | E2E Mean | E2E P50 | E2E Min | E2E Max |
|--------|------------|----------|---------|---------|---------|
| **14B Baseline** | 105.28 tok/s | 4853.82 ms | 4853.96 ms | 4828.65 ms | 4887.75 ms |
| **14B Optimized** | 151.26 tok/s | 3384.51 ms | 3380.91 ms | 3377.94 ms | 3401.20 ms |
| **32B Baseline** | 77.52 tok/s | 3795.27 ms | 3744.57 ms | 1758.99 ms | 7531.25 ms |
| **32B Optimized** | 107.29 tok/s | 1980.37 ms | 1933.36 ms | 1019.67 ms | 3889.92 ms |

### 4.3 Improvement Summary

| Model | Metric | Baseline | Optimized | Change |
|-------|--------|----------|-----------|--------|
| **14B** | Max Throughput | 1172.50 tok/s | 2397.85 tok/s | +104.5% |
| **14B** | E2E P50 (c=2) | 4853.96 ms | 3380.91 ms | -30.3% |
| **32B** | Max Throughput | 1580.47 tok/s | 1854.70 tok/s | **+17.4%** |
| **32B** | E2E P50 (c=2) | 3744.57 ms | 1933.36 ms | **-48.4%** |

---

## Appendix

### Model Acquisition

```bash
# 14B
huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
  --local-dir /mnt/data/models/Qwen2.5-14B-Instruct

# 32B
huggingface-cli download Qwen/Qwen2.5-32B-Instruct \
  --local-dir /mnt/data/models/Qwen2.5-32B-Instruct
```

### Benchmark Command

```bash
# Max throughput test
python genai_bench/cli/cli.py benchmark \
  --api-backend openai \
  --api-base http://localhost:8010 \
  --api-key dummy \
  --task text-to-text \
  --api-model-name /model \
  --model-tokenizer /mnt/data/models/Qwen2.5-14B-Instruct \
  --traffic-scenario dataset \
  --num-concurrency 160 \
  --max-requests-per-run 1024 \
  --max-time-per-run 30

# Min latency test
python genai_bench/cli/cli.py benchmark \
  --api-backend openai \
  --api-base http://localhost:8010 \
  --api-key dummy \
  --task text-to-text \
  --api-model-name /model \
  --model-tokenizer /mnt/data/models/Qwen2.5-14B-Instruct \
  --traffic-scenario dataset \
  --num-concurrency 2 \
  --max-requests-per-run 10 \
  --max-time-per-run 5
```
