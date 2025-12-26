# Parallel Parameters Support Across LLM Inference Engines

This document summarizes the parallel parameters supported by vLLM, SGLang, and TensorRT-LLM for distributed inference.

## Overview

Three types of parallelism are commonly used for LLM inference:

1. **Tensor Parallelism (TP)**: Splits model layers across GPUs within a node
2. **Pipeline Parallelism (PP)**: Splits model layers into stages across GPUs/nodes
3. **Data Parallelism (DP)**: Replicates the model across GPUs to handle multiple requests in parallel
4. **Expert Parallelism (EP)**: For MoE models, distributes experts across GPUs

## Comparison Table

| Feature | vLLM | SGLang | TensorRT-LLM | Notes |
|---------|------|--------|--------------|-------|
| **Tensor Parallel** | ✅ | ✅ | ✅ | All engines support TP |
| TP Parameter | `--tensor-parallel-size` / `-tp` | `--tp-size` | `tp_size` (build-time) | |
| **Pipeline Parallel** | ✅ | ✅ | ✅ | All engines support PP |
| PP Parameter | `--pipeline-parallel-size` / `-pp` | `--pp-size` | `pp_size` (build-time) | |
| **Data Parallel** | ✅ | ✅ | ❌ | vLLM & SGLang only |
| DP Parameter | `--data-parallel-size` / `-dp` | `--dp-size` | N/A | |
| **Context Parallel** | ✅ (DCP) | ❌ | ✅ (CP) | For long context |
| CP Parameter | `--decode-context-parallel-size` / `-dcp` | N/A | `cp_size` (build-time) | |
| **Expert Parallel (MoE)** | ✅ | ✅ | ✅ | All support MoE parallelism |
| EP Parameter | `--enable-expert-parallel` | Auto (via TP/DP) | `moe_ep_size`, `moe_tp_size` | Different approaches |

## Detailed Parameter Documentation

### 1. vLLM (Runtime Configuration)

Source: `/root/work/vllm/vllm/config/parallel.py`, `/root/work/vllm/vllm/engine/arg_utils.py`

#### Basic Parallelism

```bash
# Tensor Parallel (TP)
--tensor-parallel-size 4  # or -tp 4
# Number of GPUs to use for tensor parallelism
# Default: 1

# Pipeline Parallel (PP)
--pipeline-parallel-size 2  # or -pp 2
# Number of pipeline stages
# Default: 1

# Data Parallel (DP)
--data-parallel-size 2  # or -dp 2
# Number of data parallel replicas
# Default: 1
```

#### Advanced Parallelism

```bash
# Decode Context Parallel (DCP) - for long context
--decode-context-parallel-size 2  # or -dcp 2
# Only for decode phase parallelism
# Default: 1

# Data Parallel Configuration
--data-parallel-rank 0  # or -dpn 0
# Rank of this DP instance (for external load balancing)

--data-parallel-start-rank 0  # or -dpr 0
# Starting DP rank for secondary nodes

--data-parallel-size-local 1  # or -dpl 1
# Number of DP replicas on this node

--data-parallel-backend mp  # or ray
# Backend for data parallelism (mp=multiprocessing, ray=Ray)

--data-parallel-hybrid-lb
# Enable hybrid DP load balancing mode
```

#### MoE Parallelism

```bash
--enable-expert-parallel
# Enable expert parallelism for MoE models
# MoE layers are sharded across TP * DP GPUs

# Additional MoE Configuration
--eplb-config '{"window_size": 1000, "step_interval": 3000}'
# Expert Parallel Load Balancing configuration
```

#### vLLM Parallelism Formula

```
Total GPUs = TP * PP * DP
MoE sharding = TP * DP (when enable_expert_parallel=True)
```

### 2. SGLang (Runtime Configuration)

Source: `/root/work/sglang/python/sglang/srt/server_args.py`

#### Basic Parallelism

```bash
# Tensor Parallel (TP)
--tp-size 4
# Number of GPUs for tensor parallelism
# Default: 1

# Pipeline Parallel (PP)
--pp-size 2
# Number of pipeline stages
# Default: 1

# Data Parallel (DP)
--dp-size 2
# Number of data parallel replicas
# Default: 1
# Note: DP in SGLang is automatically configured based on available resources
```

#### Pipeline Parallel Configuration

```bash
--pp-max-micro-batch-size 8
# Maximum micro-batch size for pipeline parallelism
# Controls memory vs throughput tradeoff
```

#### MoE Parallelism

```bash
# MoE Dense Tensor Parallel
--moe-dense-tp-size 2
# Tensor parallel size for dense (non-expert) layers in MoE models
# If not set, uses same as --tp-size
# Allows different parallelism for experts vs dense layers
```

**SGLang MoE Strategy:**
- Expert parallelism is automatic based on `tp_size` and `dp_size`
- Experts are distributed across `tp_size * dp_size` GPUs
- Dense layers can use different TP via `moe-dense-tp-size`

#### SGLang Parallelism Formula

```
Total GPUs = TP * PP * DP
MoE expert distribution = TP * DP GPUs
Dense layers TP = moe_dense_tp_size (if set) or TP
```

#### SGLang Special Behaviors

- If `dp_size == 1`, chunked prefill optimization is used
- If `dp_size > 1`, chunked prefill size is divided by DP
- TP must be divisible by DP: `tp_size % dp_size == 0`

### 3. TensorRT-LLM (Build-time Configuration)

Source: `/root/work/TensorRT-LLM/tensorrt_llm/mapping.py`

TensorRT-LLM uses **build-time parallelism configuration**. Parallelism is set when building the engine, not at runtime.

#### Basic Parallelism

```python
# Build-time configuration via Mapping class
from tensorrt_llm import Mapping

mapping = Mapping(
    world_size=8,           # Total number of GPUs
    rank=0,                 # Current GPU rank
    tp_size=4,             # Tensor parallelism
    pp_size=2,             # Pipeline parallelism
    gpus_per_node=8        # GPUs per node
)
```

#### Context Parallelism

```python
mapping = Mapping(
    world_size=8,
    tp_size=4,
    pp_size=1,
    cp_size=2,             # Context parallelism
    cp_config={...}        # CP configuration
)
```

#### MoE Parallelism

```python
mapping = Mapping(
    world_size=8,
    tp_size=4,
    pp_size=1,
    moe_tp_size=2,         # TP size for MoE layers
    moe_ep_size=4,         # Expert parallelism size
    moe_cluster_size=1     # MoE cluster size (advanced)
)
```

#### Advanced MoE Configuration

```python
# TensorRT-LLM MoE provides fine-grained control
mapping = Mapping(
    world_size=16,
    tp_size=4,
    pp_size=2,
    moe_tp_size=2,         # TP within expert groups
    moe_ep_size=4,         # Number of expert groups
    moe_cluster_size=2,    # Expert clustering
    attn_tp_size=4,        # Separate TP for attention
    attn_cp_size=1         # Context parallel for attention
)
```

#### TensorRT-LLM Parallelism Formula

```
Total GPUs (world_size) = TP * PP * CP
MoE distribution = moe_tp_size * moe_ep_size * moe_cluster_size
Attention parallelism = attn_tp_size (or falls back to TP * CP)
```

#### Key Differences

1. **Build-time vs Runtime**: TensorRT-LLM parallelism is fixed at engine build time
2. **No Data Parallelism**: TensorRT-LLM doesn't have built-in DP; use multiple engine instances
3. **Granular MoE Control**: Separate TP for MoE vs attention layers

## Use Case Recommendations

### Single Node (8 GPUs)

**Small to Medium Models (7B-13B):**
```bash
# vLLM / SGLang
--tp-size 1 --pp-size 1 --dp-size 8    # Maximum throughput
--tp-size 2 --pp-size 1 --dp-size 4    # Balanced
--tp-size 4 --pp-size 1 --dp-size 2    # For larger models
```

**Large Models (70B+):**
```bash
# vLLM / SGLang
--tp-size 8 --pp-size 1 --dp-size 1    # Fit in 8 GPUs
--tp-size 4 --pp-size 2 --dp-size 1    # PP if memory constrained

# TensorRT-LLM
tp_size=8, pp_size=1                    # Build with TP=8
```

### Multi-Node (16+ GPUs)

**MoE Models (Mixtral, DeepSeek):**
```bash
# vLLM
--tp-size 2 --pp-size 1 --dp-size 8 --enable-expert-parallel
# 16 GPUs: Experts distributed across 2*8=16 GPUs

# SGLang
--tp-size 2 --pp-size 1 --dp-size 8
# Automatic expert distribution

# TensorRT-LLM
moe_tp_size=2, moe_ep_size=8, pp_size=1
# 16 GPUs: Explicit MoE configuration
```

**Long Context (128K+ tokens):**
```bash
# vLLM
--tp-size 4 --dcp 4 --pp-size 1
# 16 GPUs: 4-way TP, 4-way context parallel

# TensorRT-LLM
tp_size=4, cp_size=4, pp_size=1
# Build with context parallelism
```

## Parameter Constraints

### vLLM Constraints

- `world_size = tp_size * pp_size * dp_size`
- TP must divide total model layers evenly
- PP requires at least 2 stages for benefits
- DCP only applies to decode phase (not prefill)

### SGLang Constraints

- `tp_size % dp_size == 0` (TP must be divisible by DP)
- PP requires `pp_size >= 2`
- `pp_max_micro_batch_size` affects memory usage
- `moe_dense_tp_size` must divide `tp_size` evenly (if set)

### TensorRT-LLM Constraints

- All parallelism configured at build time (cannot change at runtime)
- `world_size = tp_size * pp_size * cp_size`
- `moe_tp_size * moe_ep_size * moe_cluster_size` must divide `tp_size`
- `attn_tp_size` must be compatible with TP and CP configuration

## Examples

### Example 1: vLLM - 4x A100 80GB, Llama-70B

```bash
vllm serve meta-llama/Llama-2-70b-hf \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --max-model-len 4096
```

### Example 2: SGLang - 8x H100, Mixtral-8x7B with DP

```bash
python -m sglang.launch_server \
    --model-path mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --tp-size 2 \
    --dp-size 4 \
    --pp-size 1
```

### Example 3: TensorRT-LLM - 8x H100, DeepSeek MoE

```python
# Build stage
from tensorrt_llm import Mapping

mapping = Mapping(
    world_size=8,
    tp_size=4,
    pp_size=2,
    moe_tp_size=2,
    moe_ep_size=4
)

# Use mapping to build engine...
```

### Example 4: vLLM - Long Context with DCP

```bash
vllm serve meta-llama/Llama-2-13b-hf \
    --tensor-parallel-size 4 \
    --decode-context-parallel-size 4 \
    --max-model-len 131072 \
    --enable-chunked-prefill
```

## Monitoring and Debugging

### Check Parallelism Configuration

**vLLM:**
```python
# Check via API
import requests
response = requests.get("http://localhost:8000/stats")
print(response.json()["num_gpu_blocks"])
```

**SGLang:**
```bash
# Logs show parallelism at startup
grep "tp_size\|pp_size\|dp_size" <log_file>
```

**TensorRT-LLM:**
```python
# Check engine metadata
from tensorrt_llm import Engine
engine = Engine.from_dir("path/to/engine")
print(engine.config.mapping)
```

## Performance Considerations

### Tensor Parallelism (TP)
- **Pros**: Reduces memory per GPU, enables larger models
- **Cons**: All-reduce communication overhead (scales with TP size)
- **Best for**: Large models that don't fit in single GPU memory

### Pipeline Parallelism (PP)
- **Pros**: Reduces memory per GPU with minimal communication
- **Cons**: Bubble overhead, sequential dependency
- **Best for**: Very large models with high batch sizes

### Data Parallelism (DP)
- **Pros**: Near-linear throughput scaling, no model sharding
- **Cons**: Requires full model per replica
- **Best for**: Throughput optimization when model fits in GPU

### Expert Parallelism (EP)
- **Pros**: Distributes experts for MoE models
- **Cons**: Load imbalancing if expert routing is skewed
- **Best for**: MoE models with many experts

## Autotuner Integration

The autotuner supports the following parallel parameters:

### Supported Parameters (All Engines)

```json
{
  "parameters": {
    "tp-size": [1, 2, 4, 8],
    "pp-size": [1, 2],
    "dp-size": [1, 2, 4]
  }
}
```

### Engine-Specific Parameters

**vLLM:**
```json
{
  "parameters": {
    "tensor-parallel-size": [2, 4],
    "pipeline-parallel-size": [1, 2],
    "data-parallel-size": [1, 2, 4],
    "decode-context-parallel-size": [1, 2],
    "enable-expert-parallel": [true, false]
  }
}
```

**SGLang:**
```json
{
  "parameters": {
    "tp-size": [2, 4],
    "pp-size": [1, 2],
    "dp-size": [1, 2, 4],
    "moe-dense-tp-size": [1, 2]
  }
}
```

**TensorRT-LLM:**
Note: TensorRT-LLM requires pre-built engines. Parallelism must be set during engine building, not at autotuner runtime.

## References

- vLLM Documentation: https://docs.vllm.ai/en/latest/
- SGLang Documentation: https://sgl-project.github.io/
- TensorRT-LLM Documentation: https://nvidia.github.io/TensorRT-LLM/
- Source code locations:
  - vLLM: `/root/work/vllm/vllm/config/parallel.py`
  - SGLang: `/root/work/sglang/python/sglang/srt/server_args.py`
  - TensorRT-LLM: `/root/work/TensorRT-LLM/tensorrt_llm/mapping.py`
