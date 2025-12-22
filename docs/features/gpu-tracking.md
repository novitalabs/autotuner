# GPU Resource Tracking

Comprehensive GPU monitoring, allocation, and scheduling system for the LLM Inference Autotuner.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [GPU Monitoring](#gpu-monitoring)
- [GPU Allocation](#gpu-allocation)
- [GPU Scheduling](#gpu-scheduling)
- [Real-Time Monitoring](#real-time-monitoring)
- [Frontend Visualization](#frontend-visualization)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Performance Considerations](#performance-considerations)

## Overview

The GPU tracking system provides end-to-end visibility and intelligent management of GPU resources throughout the autotuning workflow:

- **Monitoring**: Real-time collection of GPU metrics (utilization, memory, temperature, power)
- **Allocation**: Intelligent GPU selection for experiments based on availability scoring
- **Scheduling**: Task-level GPU requirement estimation and availability checking
- **Visualization**: Rich frontend charts and tables for GPU metrics analysis

### Key Features

- Automatic GPU detection via nvidia-smi
- Smart GPU allocation using composite scoring (memory + utilization)
- GPU-aware task scheduling with timeout-based waiting
- Real-time GPU monitoring during benchmark execution
- Frontend visualization with Recharts
- Detailed GPU information in experiment results

### Supported Modes

- **Docker Mode**: Full GPU tracking, allocation, and scheduling support
- **OME/Kubernetes Mode**: GPU monitoring and visualization only (allocation handled by K8s scheduler)

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (React)                        │
│  ┌────────────────┐  ┌──────────────────────────────────┐   │
│  │ GPU Metrics    │  │ Experiments Page                 │   │
│  │ Chart          │  │ - GPU count column               │   │
│  │ - Utilization  │  │ - GPU model info                 │   │
│  │ - Memory       │  │ - Monitoring data indicator      │   │
│  │ - Temperature  │  │                                  │   │
│  │ - Power        │  │                                  │   │
│  └────────────────┘  └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                             ▲
                             │ REST API (JSON)
                             │
┌─────────────────────────────────────────────────────────────┐
│                    Backend (Python)                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ ARQ Worker (autotuner_worker.py)                     │   │
│  │ - GPU requirement estimation                         │   │
│  │ - Availability checking before task start            │   │
│  │ - Wait for GPU availability (timeout)                │   │
│  └──────────────────────────────────────────────────────┘   │
│                             │                                │
│  ┌──────────────────────────┼────────────────────────────┐  │
│  │ Orchestrator             │                            │  │
│  │ - Coordinates experiments│                            │  │
│  │ - Passes GPU indices     │                            │  │
│  └──────────────────────────┼────────────────────────────┘  │
│                             │                                │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │ Controllers                                           │  │
│  │                                                       │  │
│  │ ┌─────────────────────────────────────────────────┐  │  │
│  │ │ DockerController                                │  │  │
│  │ │ - Smart GPU allocation (select_gpus_for_task)   │  │  │
│  │ │ - Device requests with specific GPU IDs         │  │  │
│  │ └─────────────────────────────────────────────────┘  │  │
│  │                                                       │  │
│  │ ┌─────────────────────────────────────────────────┐  │  │
│  │ │ DirectBenchmarkController                       │  │  │
│  │ │ - Real-time GPU monitoring during benchmark     │  │  │
│  │ │ - Aggregates stats (min/max/mean)               │  │  │
│  │ │ - Returns monitoring data with metrics          │  │  │
│  │ └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
│                             │                                │
│  ┌──────────────────────────┴────────────────────────────┐  │
│  │ Utilities                                             │  │
│  │                                                       │  │
│  │ ┌─────────────────────────────────────────────────┐  │  │
│  │ │ gpu_monitor.py                                  │  │  │
│  │ │ - nvidia-smi wrapper                            │  │  │
│  │ │ - GPU availability scoring                      │  │  │
│  │ │ - Continuous monitoring thread                  │  │  │
│  │ └─────────────────────────────────────────────────┘  │  │
│  │                                                       │  │
│  │ ┌─────────────────────────────────────────────────┐  │  │
│  │ │ gpu_scheduler.py                                │  │  │
│  │ │ - GPU requirement estimation                    │  │  │
│  │ │ - Availability checking                         │  │  │
│  │ │ - Wait-for-availability with timeout            │  │  │
│  │ └─────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                             ▲
                             │ nvidia-smi
                             │
                        ┌────┴────┐
                        │   GPUs  │
                        └─────────┘
```

### Data Flow

1. **Task Submission**: User creates task via frontend or API
2. **Task Scheduling**: ARQ worker checks GPU availability before starting
3. **GPU Allocation**: DockerController selects optimal GPUs for experiment
4. **Experiment Execution**: DirectBenchmarkController monitors GPUs during benchmark
5. **Data Collection**: GPU metrics aggregated and stored with experiment results
6. **Visualization**: Frontend displays GPU info and monitoring charts

## GPU Monitoring

### Overview

The `gpu_monitor.py` module provides a singleton-based GPU monitoring system with caching and continuous monitoring capabilities.

### Core Components

#### GPUMonitor Singleton

```python
from src.utils.gpu_monitor import get_gpu_monitor

# Get the global GPU monitor instance
gpu_monitor = get_gpu_monitor()

# Check if nvidia-smi is available
if gpu_monitor.is_available():
    print("GPU monitoring available")
```

#### Query GPU Status

```python
# Get current GPU snapshot (uses cache if recent)
snapshot = gpu_monitor.query_gpus()

# Force fresh query (bypass cache)
snapshot = gpu_monitor.query_gpus(use_cache=False)

# Access GPU data
for gpu in snapshot.gpus:
    print(f"GPU {gpu.index}: {gpu.name}")
    print(f"  Memory: {gpu.memory_used_mb}/{gpu.memory_total_mb} MB")
    print(f"  Utilization: {gpu.utilization_percent}%")
    print(f"  Temperature: {gpu.temperature_c}°C")
    print(f"  Power: {gpu.power_draw_w}W")
```

#### Find Available GPUs

```python
# Get GPUs with <50% utilization and at least 8GB free
available_gpus = gpu_monitor.get_available_gpus(
    min_memory_mb=8000,
    max_utilization=50
)

print(f"Available GPU indices: {available_gpus}")
```

#### GPU Availability Scoring

The system uses a composite scoring algorithm to rank GPUs:

```python
score = 0.6 × memory_score + 0.4 × utilization_score

where:
  memory_score = memory_free_mb / memory_total_mb
  utilization_score = (100 - utilization_percent) / 100
```

This prioritizes GPUs with:
- More free memory (60% weight)
- Lower utilization (40% weight)

### Continuous Monitoring

For real-time monitoring during long-running operations:

```python
# Start monitoring thread (samples every 1 second)
gpu_monitor.start_monitoring(interval_seconds=1.0)

# ... run your workload ...

# Stop monitoring and get aggregated stats
stats = gpu_monitor.stop_monitoring()

# Access aggregated data
print(f"Monitoring duration: {stats['monitoring_duration_seconds']}s")
print(f"Sample count: {stats['sample_count']}")

for gpu_index, gpu_stats in stats['gpu_stats'].items():
    print(f"\nGPU {gpu_index}:")
    print(f"  Utilization: {gpu_stats['utilization']['mean']:.1f}%")
    print(f"    Range: {gpu_stats['utilization']['min']:.0f}% - {gpu_stats['utilization']['max']:.0f}%")
    print(f"  Memory: {gpu_stats['memory_used_mb']['mean']:.0f} MB")
    print(f"  Temperature: {gpu_stats['temperature_c']['mean']:.0f}°C")
    print(f"  Power: {gpu_stats['power_draw_w']['mean']:.1f}W")
```

### Cache Behavior

- Default cache TTL: 5 seconds
- Cache cleared on manual refresh (`use_cache=False`)
- Cache invalidated when monitoring starts/stops

## GPU Allocation

### Overview

The DockerController implements intelligent GPU allocation for experiments using the monitoring system.

### Allocation Strategy

```python
def select_gpus_for_task(self, required_gpus: int, min_memory_mb: int = 8000) -> List[int]:
    """
    Select optimal GPUs for task execution.

    Args:
        required_gpus: Number of GPUs needed
        min_memory_mb: Minimum free memory per GPU (default: 8GB)

    Returns:
        List of GPU indices (e.g., [0, 1])

    Raises:
        RuntimeError: If insufficient GPUs available
    """
```

### Allocation Examples

#### Single GPU Allocation

```python
from src.controllers.docker_controller import DockerController

controller = DockerController(
    docker_model_path="/mnt/data/models",
    verbose=True
)

# Allocate 1 GPU with at least 8GB free
gpu_indices = controller.select_gpus_for_task(
    required_gpus=1,
    min_memory_mb=8000
)
# Result: [2]  # GPU 2 had the highest availability score
```

#### Multi-GPU Allocation

```python
# Allocate 4 GPUs for tensor parallelism
gpu_indices = controller.select_gpus_for_task(
    required_gpus=4,
    min_memory_mb=10000
)
# Result: [1, 2, 3, 5]  # Best 4 GPUs by composite score
```

### Allocation Process

1. **Query GPUs**: Get current status via `gpu_monitor.query_gpus(use_cache=False)`
2. **Filter GPUs**: Remove GPUs with insufficient memory
3. **Score GPUs**: Calculate composite score (memory 60% + utilization 40%)
4. **Sort & Select**: Return top N GPUs by score
5. **Validate**: Raise error if insufficient GPUs available

### Docker Integration

Selected GPUs are passed to Docker via `device_requests`:

```python
device_request = docker.types.DeviceRequest(
    device_ids=[str(idx) for idx in gpu_indices],
    capabilities=[['gpu']]
)

container = client.containers.run(
    image=image_name,
    device_requests=[device_request],
    # ... other params
)
```

**IMPORTANT**: Do NOT set `CUDA_VISIBLE_DEVICES` environment variable when using `device_requests`. Docker handles GPU visibility automatically.

## GPU Scheduling

### Overview

The `gpu_scheduler.py` module provides task-level GPU resource management with intelligent waiting.

### GPU Requirement Estimation

```python
from src.utils.gpu_scheduler import estimate_gpu_requirements

task_config = {
    "model": {"id_or_path": "llama-3-70b"},
    "parameters": {
        "tp-size": [1, 2, 4],      # Tensor parallelism
        "pp-size": [1],             # Pipeline parallelism
        "dp-size": [1]              # Data parallelism
    }
}

required_gpus, estimated_memory_mb = estimate_gpu_requirements(task_config)
# Result: (4, 20000)  # 4 GPUs needed, ~20GB per GPU for 70B model
```

### World Size Calculation

```python
world_size = tp × pp × max(dp, dcp, cp)

where:
  tp = tensor_parallel_size
  pp = pipeline_parallel_size
  dp = data_parallel_size
  cp = context_parallel_size
  dcp = decode_context_parallel_size
```

### Memory Estimation Heuristics

- **70B/65B models**: 20,000 MB per GPU
- **13B/7B models**: 12,000 MB per GPU
- **Unknown/small models**: 8,000 MB per GPU (base)

### Parameter Name Formats

The estimator supports multiple parameter naming conventions:

```python
# All these are equivalent for tensor parallelism:
"tensor-parallel-size": [1, 2, 4]
"tp-size": [1, 2, 4]
"tp_size": [1, 2, 4]
"tp": [1, 2, 4]
```

Supported parameters:
- Tensor Parallel: `tensor-parallel-size`, `tp-size`, `tp_size`, `tp`
- Pipeline Parallel: `pipeline-parallel-size`, `pp-size`, `pp_size`, `pp`
- Data Parallel: `data-parallel-size`, `dp-size`, `dp_size`, `dp`
- Context Parallel: `context-parallel-size`, `cp-size`, `cp_size`, `cp`
- Decode Context Parallel: `decode-context-parallel-size`, `dcp-size`, `dcp_size`, `dcp`

### Availability Checking

```python
from src.utils.gpu_scheduler import check_gpu_availability

# Check if 4 GPUs with 10GB free are available
is_available, message = check_gpu_availability(
    required_gpus=4,
    min_memory_mb=10000
)

if is_available:
    print(f"GPUs available: {message}")
else:
    print(f"GPUs unavailable: {message}")
```

### Wait for Availability

```python
from src.utils.gpu_scheduler import wait_for_gpu_availability

# Wait up to 5 minutes for GPUs to become available
is_available, message = wait_for_gpu_availability(
    required_gpus=4,
    min_memory_mb=10000,
    timeout_seconds=300,    # 5 minutes
    check_interval=30       # Check every 30 seconds
)

if is_available:
    print(f"GPUs became available: {message}")
else:
    print(f"Timeout: {message}")
```

### ARQ Worker Integration

The GPU scheduler is integrated into the task execution workflow:

```python
# In autotuner_worker.py:

if task.deployment_mode == "docker":
    # 1. Estimate GPU requirements
    required_gpus, estimated_memory_mb = estimate_gpu_requirements(task_config)

    # 2. Check immediate availability
    is_available, message = check_gpu_availability(
        required_gpus=required_gpus,
        min_memory_mb=estimated_memory_mb
    )

    # 3. Wait if not immediately available
    if not is_available:
        is_available, message = wait_for_gpu_availability(
            required_gpus=required_gpus,
            min_memory_mb=estimated_memory_mb,
            timeout_seconds=300,  # 5 minutes
            check_interval=30
        )

    # 4. Fail task if still unavailable
    if not is_available:
        task.status = TaskStatus.FAILED
        # ... update database and broadcast event
        return {"status": "failed", "error": message}
```

## Real-Time Monitoring

### Overview

The DirectBenchmarkController monitors GPU metrics during benchmark execution.

### Monitoring Process

1. **Start Monitoring**: Thread begins sampling GPUs every 1 second
2. **Run Benchmark**: genai-bench executes while monitoring collects data
3. **Stop Monitoring**: Thread stops and aggregates statistics
4. **Return Results**: Monitoring data included in experiment metrics

### Implementation

```python
# In direct_benchmark_controller.py:

def run_benchmark_job(self, endpoint_url: str, benchmark_spec: Dict[str, Any],
                      gpu_indices: Optional[List[int]] = None) -> Dict[str, Any]:

    # Start GPU monitoring
    gpu_monitor = get_gpu_monitor()
    if gpu_monitor.is_available():
        gpu_monitor.start_monitoring(interval_seconds=1.0)

    # Run benchmark
    result = self._run_genai_bench(endpoint_url, benchmark_spec)

    # Stop monitoring and get stats
    monitoring_data = None
    if gpu_monitor.is_available():
        monitoring_data = gpu_monitor.stop_monitoring()

    # Include in results
    result["gpu_monitoring"] = monitoring_data
    return result
```

### Monitoring Data Structure

```python
{
    "monitoring_duration_seconds": 45.2,
    "sample_count": 45,
    "gpu_stats": {
        "0": {
            "name": "NVIDIA A100-SXM4-80GB",
            "utilization": {
                "min": 78.0,
                "max": 95.0,
                "mean": 87.3,
                "samples": 45
            },
            "memory_used_mb": {
                "min": 15234.0,
                "max": 15678.0,
                "mean": 15456.2
            },
            "memory_usage_percent": {
                "min": 18.5,
                "max": 19.1,
                "mean": 18.8
            },
            "temperature_c": {
                "min": 56.0,
                "max": 62.0,
                "mean": 59.1
            },
            "power_draw_w": {
                "min": 245.0,
                "max": 280.0,
                "mean": 265.3
            }
        }
    }
}
```

## Frontend Visualization

### Experiments Page

The Experiments page displays GPU information for each experiment:

#### GPU Column

Shows GPU count and model for experiments:

```tsx
<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
  {experiment.gpu_info ? (
    <div className="flex items-center gap-1">
      <svg className="w-4 h-4 text-green-600">...</svg>
      <span className="font-medium">{experiment.gpu_info.count}</span>
      <span className="text-gray-500 text-xs">
        {experiment.gpu_info.model ? `(${experiment.gpu_info.model.split(' ')[0]})` : ''}
      </span>
      {experiment.metrics?.gpu_monitoring && (
        <span className="ml-1 inline-flex items-center text-xs text-blue-600"
              title="GPU monitoring data available">
          📊
        </span>
      )}
    </div>
  ) : (
    <span className="text-gray-400">N/A</span>
  )}
</td>
```

### GPU Metrics Chart Component

The `GPUMetricsChart` component visualizes monitoring data:

```tsx
import GPUMetricsChart from '@/components/GPUMetricsChart';

// In experiment details modal:
{selectedExperiment.metrics?.gpu_monitoring && (
  <div>
    <h3 className="text-sm font-medium text-gray-900 mb-3">
      GPU Monitoring
    </h3>
    <GPUMetricsChart gpuMonitoring={selectedExperiment.metrics.gpu_monitoring} />
  </div>
)}
```

#### Chart Features

1. **Monitoring Summary**
   - Duration and sample count
   - Displayed in blue info box

2. **GPU Stats Table**
   - Per-GPU statistics
   - Columns: GPU ID, Model, Utilization, Memory, Temperature, Power
   - Shows mean values with min-max ranges

3. **Interactive Charts** (Recharts LineChart)
   - GPU Utilization (%)
   - Memory Usage (%)
   - Temperature (°C)
   - Power Draw (W)
   - Responsive design (adapts to container width)

### TypeScript Types

```typescript
// frontend/src/types/api.ts

export interface Experiment {
  // ... other fields
  gpu_info?: {
    model: string;
    count: number;
    device_ids?: string[];
    world_size?: number;
    gpu_info?: {
      count: number;
      indices: number[];
      allocation_method: string;
      details?: Array<{
        index: number;
        name: string;
        memory_free_mb: number;
        utilization_percent: number;
        availability_score: number;
      }>;
    };
  };
  metrics?: {
    // ... other metrics
    gpu_monitoring?: {
      monitoring_duration_seconds: number;
      sample_count: number;
      gpu_stats: {
        [gpu_index: string]: {
          name: string;
          utilization: { min: number; max: number; mean: number; samples: number };
          memory_used_mb: { min: number; max: number; mean: number };
          memory_usage_percent: { min: number; max: number; mean: number };
          temperature_c: { min: number; max: number; mean: number };
          power_draw_w: { min: number; max: number; mean: number };
        };
      };
    };
  };
}
```

## API Reference

### gpu_monitor.py

#### GPUMonitor Class

**`get_gpu_monitor() -> GPUMonitor`**
- Returns the global GPUMonitor singleton instance
- Thread-safe initialization

**`is_available() -> bool`**
- Check if nvidia-smi is available on the system
- Returns False if nvidia-smi not found or execution fails

**`query_gpus(use_cache: bool = True) -> Optional[GPUSnapshot]`**
- Query current GPU status
- Args:
  - `use_cache`: Use cached data if available and recent (default: True)
- Returns: GPUSnapshot with GPU data or None if query fails
- Cache TTL: 5 seconds

**`get_available_gpus(min_memory_mb: Optional[int] = None, max_utilization: int = 50) -> List[int]`**
- Get list of available GPU indices
- Args:
  - `min_memory_mb`: Minimum free memory required (optional)
  - `max_utilization`: Maximum utilization percentage (default: 50)
- Returns: List of GPU indices sorted by availability score (descending)

**`get_gpu_info(gpu_index: int) -> Optional[GPUInfo]`**
- Get information for a specific GPU
- Args:
  - `gpu_index`: GPU index (0-based)
- Returns: GPUInfo object or None if not found

**`start_monitoring(interval_seconds: float = 1.0) -> None`**
- Start continuous GPU monitoring thread
- Args:
  - `interval_seconds`: Sampling interval (default: 1.0)
- Clears any previous monitoring data

**`stop_monitoring() -> Optional[Dict[str, Any]]`**
- Stop monitoring thread and return aggregated statistics
- Returns: Dictionary with monitoring data (see data structure above)
- Returns None if monitoring was not started

#### Data Classes

**`GPUInfo`**
```python
@dataclass
class GPUInfo:
    index: int                    # GPU index (0-based)
    name: str                     # GPU model name
    memory_total_mb: int          # Total memory in MB
    memory_used_mb: int           # Used memory in MB
    memory_free_mb: int           # Free memory in MB
    utilization_percent: int      # GPU utilization (0-100)
    temperature_c: int            # Temperature in Celsius
    power_draw_w: float           # Power draw in Watts
    score: float                  # Availability score (0.0-1.0)
```

**`GPUSnapshot`**
```python
@dataclass
class GPUSnapshot:
    timestamp: datetime           # When snapshot was taken
    gpus: List[GPUInfo]          # List of GPU information
```

### gpu_scheduler.py

**`estimate_gpu_requirements(task_config: Dict[str, Any]) -> Tuple[int, int]`**
- Estimate GPU requirements from task configuration
- Args:
  - `task_config`: Task configuration dictionary
- Returns: Tuple of (min_gpus_required, estimated_memory_mb_per_gpu)
- Calculation: `world_size = tp × pp × max(dp, dcp, cp)`

**`check_gpu_availability(required_gpus: int, min_memory_mb: Optional[int] = None) -> Tuple[bool, str]`**
- Check if sufficient GPUs are available
- Args:
  - `required_gpus`: Number of GPUs required
  - `min_memory_mb`: Minimum memory per GPU (optional)
- Returns: Tuple of (is_available, message)
- Message contains detailed status or error information

**`wait_for_gpu_availability(required_gpus: int, min_memory_mb: Optional[int] = None, timeout_seconds: int = 300, check_interval: int = 30) -> Tuple[bool, str]`**
- Wait for sufficient GPUs to become available
- Args:
  - `required_gpus`: Number of GPUs required
  - `min_memory_mb`: Minimum memory per GPU (optional)
  - `timeout_seconds`: Maximum wait time (default: 300 = 5 minutes)
  - `check_interval`: Polling interval (default: 30 seconds)
- Returns: Tuple of (is_available, message)
- Logs check attempts and status periodically

### docker_controller.py

**`select_gpus_for_task(required_gpus: int, min_memory_mb: int = 8000) -> List[int]`**
- Select optimal GPUs for task execution
- Args:
  - `required_gpus`: Number of GPUs needed
  - `min_memory_mb`: Minimum free memory per GPU (default: 8000)
- Returns: List of GPU indices
- Raises: RuntimeError if insufficient GPUs available

### direct_benchmark_controller.py

**`run_benchmark_job(endpoint_url: str, benchmark_spec: Dict[str, Any], gpu_indices: Optional[List[int]] = None) -> Dict[str, Any]`**
- Run benchmark with GPU monitoring
- Args:
  - `endpoint_url`: Inference service endpoint
  - `benchmark_spec`: Benchmark configuration
  - `gpu_indices`: GPU indices being used (optional, for logging)
- Returns: Dictionary with benchmark results and `gpu_monitoring` field

## Configuration

### Environment Variables

No specific environment variables required. GPU monitoring uses nvidia-smi from PATH.

### Task Configuration

Specify parallel configuration in task JSON:

```json
{
  "parameters": {
    "tp-size": [1, 2, 4],              // Tensor parallelism
    "pp-size": [1],                     // Pipeline parallelism
    "dp-size": [1],                     // Data parallelism
    "cp-size": [1],                     // Context parallelism
    "dcp-size": [1]                     // Decode context parallelism
  }
}
```

### Scheduler Configuration

Modify timeout and interval in `autotuner_worker.py`:

```python
# Wait for GPUs with custom timeout
is_available, message = wait_for_gpu_availability(
    required_gpus=required_gpus,
    min_memory_mb=estimated_memory_mb,
    timeout_seconds=600,     # 10 minutes (default: 300)
    check_interval=60        # Check every minute (default: 30)
)
```

### Monitoring Configuration

Adjust sampling interval for continuous monitoring:

```python
# Sample every 2 seconds instead of 1
gpu_monitor.start_monitoring(interval_seconds=2.0)
```

## Troubleshooting

### nvidia-smi Not Found

**Symptom**: Warnings like "nvidia-smi not available"

**Cause**: nvidia-smi not in PATH or NVIDIA drivers not installed

**Solution**:
- System proceeds without GPU monitoring (graceful degradation)
- Install NVIDIA drivers and CUDA toolkit
- Verify: `nvidia-smi` command works in terminal

### No GPUs Available

**Symptom**: Task fails with "Insufficient GPUs after waiting"

**Cause**: All GPUs are busy or don't meet memory requirements

**Solutions**:
1. Wait for running tasks to complete
2. Reduce `min_memory_mb` requirement
3. Reduce parallel configuration (tp-size, pp-size)
4. Increase timeout: `timeout_seconds=600`

### Incorrect GPU Count Estimation

**Symptom**: Task requests wrong number of GPUs

**Cause**: Parameter names not recognized or misconfigured

**Solutions**:
1. Use standard hyphenated format: `tp-size` not `tp_size`
2. Check parameter values are lists: `[1, 2, 4]` not `1`
3. Verify task config JSON structure
4. Check logs for "Estimated requirements" message

### GPU Allocation Failures

**Symptom**: RuntimeError during GPU selection

**Cause**: Insufficient GPUs with required memory

**Solutions**:
1. Lower memory requirement: `min_memory_mb=6000`
2. Free up GPU memory (stop other processes)
3. Use fewer GPUs (reduce tp-size)

### Monitoring Data Not Appearing

**Symptom**: No GPU charts in experiment details

**Cause**: Monitoring not enabled or failed to collect data

**Solutions**:
1. Verify nvidia-smi works: `nvidia-smi` in terminal
2. Check experiment metrics in database has `gpu_monitoring` field
3. Ensure DirectBenchmarkController is being used (Docker mode)
4. Check worker logs for monitoring errors

### Docker Container Can't Access GPUs

**Symptom**: "No accelerator available" in container logs

**Cause**: Incorrect Docker GPU configuration

**Solutions**:
1. Verify `device_requests` is used, not `CUDA_VISIBLE_DEVICES`
2. Check nvidia-container-toolkit installed: `docker run --gpus all ...`
3. Verify GPU indices are valid: `nvidia-smi -L`
4. Don't mix `device_requests` and `CUDA_VISIBLE_DEVICES`

### Frontend Not Showing GPU Info

**Symptom**: "N/A" in GPU column

**Cause**: Experiment doesn't have gpu_info

**Solutions**:
1. Verify task uses Docker mode (OME mode has limited GPU tracking)
2. Check experiment record in database has `gpu_info` field
3. Ensure DockerController's `select_gpus_for_task` was called
4. Verify frontend TypeScript types are up to date

## Performance Considerations

### Cache Usage

- **Query Cache**: 5-second TTL reduces nvidia-smi overhead
- **Recommendation**: Use default cache for frequent queries
- **Force Refresh**: Use `use_cache=False` for critical decisions (GPU allocation)

### Monitoring Overhead

- **Sampling Rate**: Default 1 second is good for most workloads
- **Overhead**: Minimal (~1% CPU per GPU monitored)
- **Recommendation**: Increase interval to 2-5 seconds for very long benchmarks (>10 minutes)

### Scheduler Polling

- **Default Interval**: 30 seconds balances responsiveness and overhead
- **Recommendation**: Use shorter interval (10-15s) for high-priority tasks
- **Recommendation**: Use longer interval (60s) when many tasks in queue

### GPU Allocation Strategy

- **Scoring Algorithm**: Prioritizes memory (60%) over utilization (40%)
- **Rationale**: Memory is hard constraint, utilization is soft
- **Recommendation**: Adjust if workload is compute-bound rather than memory-bound

### Database Storage

- **GPU Info**: Stored as JSON in experiment record (~1-2 KB per experiment)
- **Monitoring Data**: Can be large for long benchmarks (~50 KB for 1000 samples)
- **Recommendation**: Consider cleanup policy for old experiment monitoring data

## Best Practices

1. **Use Docker Mode**: Full GPU tracking support (OME mode has limited support)

2. **Set Realistic Memory Requirements**: Over-estimation causes unnecessary waits

3. **Configure Timeouts Appropriately**:
   - Short tasks (< 5 min): 300s timeout
   - Long tasks (> 10 min): 600-900s timeout

4. **Monitor System Load**: Use `watch -n 1 nvidia-smi` to understand GPU usage patterns

5. **Tune Scoring Algorithm**: Adjust weights in `_calculate_gpu_score()` for your workload

6. **Archive Monitoring Data**: Consider moving old monitoring data to separate storage

7. **Use Graceful Degradation**: System works without nvidia-smi, but with reduced visibility

8. **Check Logs**: Worker logs contain detailed GPU scheduling information

9. **Test Parallel Configs**: Verify world_size calculation matches your expectation

10. **Frontend Caching**: TanStack Query caches experiment data to reduce API calls

## Future Enhancements

Potential areas for improvement:

- **Multi-Node GPU Scheduling**: Support for distributed GPU allocation across nodes
- **Predictive Scheduling**: ML-based prediction of task duration and GPU requirements
- **Dynamic Reallocation**: Move tasks between GPUs based on load
- **WebSocket Updates**: Real-time GPU metrics streaming to frontend
- **GPU Affinity**: Pin specific experiments to specific GPUs
- **Power Capping**: Enforce power limits for energy efficiency
- **Historical Analytics**: Track GPU utilization trends over time

---

## Intelligent GPU Allocation (OME/Kubernetes)

For Kubernetes deployments, the system includes cluster-wide GPU discovery and intelligent node selection.

### Features

1. **Cluster-wide GPU Discovery**
   - Queries all nodes in Kubernetes cluster
   - Collects GPU capacity, utilization, memory, temperature
   - Node-level GPU availability tracking

2. **Intelligent Node Selection**
   - Determines GPU requirements from task parameters (tp-size)
   - Ranks nodes based on idle GPU availability
   - Idle criteria: <30% utilization AND <50% memory
   - Applies node affinity to InferenceService deployments

3. **Automatic Fallback**
   - Falls back to K8s scheduler if no metrics available
   - Graceful degradation if no idle GPUs found
   - Can disable with `enable_gpu_selection=False`

### Implementation

See `src/controllers/gpu_allocator.py`:
- `get_cluster_gpu_status()`: Cluster-wide discovery
- `select_best_node()`: Node ranking algorithm
- Integrates with OMEController for deployments

### Benefits

- Balanced GPU utilization across cluster
- Avoids overloaded nodes
- Reduces deployment failures from resource contention
- Works with dynamic Kubernetes clusters

