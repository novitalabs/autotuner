# Autotuner: Comprehensive Deployment Architecture Analysis

## Executive Summary

The inference-autotuner is a prototype LLM inference parameter optimization system that uses **Kubernetes-native deployment** through the OME (Open Model Engine) framework. It supports two benchmark execution modes and dynamically deploys InferenceServices with different parameter configurations for automated tuning.

---

## 1. Current Deployment Architecture

### 1.1 Overall Architecture Pattern

```
┌─────────────────────────────────────────────────────────────────┐
│                    Inference Autotuner (Python)                 │
│              src/run_autotuner.py (Orchestrator)                │
└────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴────────────────┐
                │                              │
         ┌──────▼──────┐           ┌──────────▼─────────┐
         │ OMEController │         │ BenchmarkController │
         │ (K8s Native) │         │  (Two Modes)        │
         └──────┬──────┘           └──────────┬──────────┘
                │                             │
    ┌───────────┴──────────┐      ┌──────────┴────────────┐
    │                      │      │                       │
┌───▼────────┐      ┌──────▼───┐ │             ┌──────────▼────┐
│InferenceService  │Model│Runtime │             │ Direct CLI   │
│(Custom CRD)      │(CRD)│ (CRD)  │             │(Port Forward)│
└───────────┘      └─────┴──────┘              └──────────────┘
    │                   │                           │
    └───────────┬───────┴───────────────────────────┘
                │
        ┌───────▼───────┐
        │  Kubernetes   │
        │  (K8s Cluster)│
        └───────────────┘
                │
    ┌───────────┴──────────────┐
    │                          │
┌───▼──────────┐        ┌──────▼───────┐
│SGLang Runtime│        │Genai-Bench   │
│   Pods       │        │ Benchmarking │
└──────────────┘        └──────────────┘
```

### 1.2 Deployment Modes: Two Benchmark Execution Pathways

#### Mode 1: Kubernetes BenchmarkJob (Default)
- Uses OME's BenchmarkJob Custom Resource Definition
- Runs genai-bench inside Kubernetes pods
- Results stored in PersistentVolumeClaim (PVC)
- Command: `python src/run_autotuner.py examples/simple_task.json`

#### Mode 2: Direct CLI Mode (Recommended)
- Uses locally installed genai-bench executable
- Automatic `kubectl port-forward` to InferenceService
- Bypasses Docker image dependencies
- Command: `python src/run_autotuner.py examples/simple_task.json --direct`

---

## 2. Key Deployment Components

### 2.1 OME (Open Model Engine) - The Core Infrastructure

OME is a **required prerequisite** that provides:

**Custom Resource Definitions (CRDs):**
- `InferenceService.ome.io/v1beta1` - Model serving endpoints
- `BenchmarkJob.ome.io/v1beta1` - Performance testing jobs
- `ClusterBaseModel.ome.io/v1beta1` - Model metadata and storage
- `ClusterServingRuntime.ome.io/v1beta1` - Runtime configurations

**Supporting Components:**
- OME Controller Manager (Kubernetes Deployment)
- Model Agent DaemonSet (model distribution)
- RBAC configurations
- Webhooks (validation and mutation)

**Installation:**
- Via Helm: `oci://ghcr.io/moirai-internal/charts/ome-crd` and `charts/ome-resources`
- Dependencies: cert-manager, KEDA
- Namespace: `ome`

### 2.2 InferenceService Deployment

**What It Is:**
- OME custom resource that wraps SGLang inference engines
- Dynamically created per experiment with unique parameter configurations

**Template-Based Generation:**
- File: `/src/templates/inference_service.yaml.j2` (Jinja2 template)
- Contains:
  - Namespace declaration
  - InferenceService metadata with experiment labels
  - Model specification
  - SGLang engine configuration with tunable parameters
  - GPU resource requests

**Template Variables:**
```
{{ namespace }}          - K8s namespace (e.g., "autotuner")
{{ isvc_name }}         - Unique service name (e.g., "simple-tune-exp1")
{{ task_name }}         - Task identifier (e.g., "simple-tune")
{{ experiment_id }}     - Experiment number (1, 2, 3, ...)
{{ model_name }}        - Base model (e.g., "llama-3-2-1b-instruct")
{{ runtime_name }}      - Runtime config (e.g., "llama-3-2-1b-instruct-rt")
{{ tp_size }}           - Tensor parallelism (GPU count)
{{ mem_frac }}          - Memory fraction (0.6-0.95)
{{ max_total_tokens }}  - Optional token limit
{{ schedule_policy }}   - Optional scheduling (lpm, random, fcfs)
```

**SGLang Container Args:**
```yaml
- --host=0.0.0.0
- --port=8080
- --model-path=/mnt/data/models/{model_name}
- --tp-size={tp_size}
- --mem-frac={mem_frac}
```

**GPU Resources:**
```yaml
limits:
  nvidia.com/gpu: {{ tp_size }}
requests:
  nvidia.com/gpu: {{ tp_size }}
```

### 2.3 BenchmarkJob Deployment

**When Used (K8s Mode Only):**
- After InferenceService becomes ready
- For performance evaluation

**Template-Based Generation:**
- File: `/src/templates/benchmark_job.yaml.j2`
- Unique per experiment

**Key Configuration:**
```yaml
podOverride:
  image: "kllambda/genai-bench:v251014"
endpoint:
  url: "http://{isvc_name}.{namespace}.svc.cluster.local"
  apiFormat: "openai"
outputLocation:
  storageUri: "pvc://benchmark-results-pvc/{benchmark_name}"
```

### 2.4 Persistent Storage

**PersistentVolumeClaim:**
- File: `/config/benchmark-pvc.yaml`
- Name: `benchmark-results-pvc`
- Namespace: `autotuner`
- Size: 1Gi
- Access: ReadWriteOnce
- Purpose: Store benchmark results from BenchmarkJobs

---

## 3. Deployment Logic Implementation

### 3.1 Main Orchestrator: `src/run_autotuner.py`

**Class: `AutotunerOrchestrator`**

**Constructor (`__init__`):**
```python
def __init__(self, kubeconfig_path: str = None, use_direct_benchmark: bool = False)
```
- Initializes OMEController for InferenceService management
- Selects benchmark controller based on mode:
  - DirectBenchmarkController (--direct flag)
  - BenchmarkController (default)
- Prints active mode to console

**Main Execution Flow (`run_task`):**
1. Load JSON task configuration
2. Generate parameter grid (Cartesian product of all parameter combinations)
3. For each parameter combination:
   - Call `run_experiment()` with parameters
   - Append results to list
4. Find best result by objective score
5. Save summary to `results/{task_name}_results.json`

**Per-Experiment Flow (`run_experiment`):**
```
Step 1: Deploy InferenceService
  └─> OMEController.deploy_inference_service()
  └─> Creates unique InferenceService with parameters

Step 2: Wait for Ready
  └─> OMEController.wait_for_ready()
  └─> Polls status.conditions for Ready=True
  └─> Timeout: configurable (default 600s)

Step 3: Run Benchmark
  ├─> If --direct mode:
  │   └─> DirectBenchmarkController.run_benchmark()
  │       └─> Sets up kubectl port-forward
  │       └─> Runs genai-bench CLI locally
  │
  └─> Else (K8s mode):
      └─> BenchmarkController.create_benchmark_job()
      └─> BenchmarkController.wait_for_completion()

Step 4: Collect Results
  └─> Extract metrics from benchmark output
  └─> Calculate objective score (minimize_latency / maximize_throughput)

Cleanup: Remove experiment resources
  └─> Delete InferenceService
  └─> Delete BenchmarkJob (if applicable)
```

### 3.2 OME Controller: `src/controllers/ome_controller.py`

**Class: `OMEController`**

**Kubernetes API Initialization:**
```python
from kubernetes import client, config

# Try kubeconfig, then in-cluster config
config.load_kube_config(config_file=kubeconfig_path)
# Or: config.load_incluster_config()

self.custom_api = client.CustomObjectsApi()  # For CRDs
self.core_api = client.CoreV1Api()            # For core resources
```

**Key Methods:**

1. **`deploy_inference_service()`**
   - Renders Jinja2 template with parameters
   - Creates namespace if needed
   - Posts InferenceService CRD to Kubernetes
   - Returns InferenceService name
   - CRD group: `ome.io`, version: `v1beta1`, plural: `inferenceservices`

2. **`wait_for_ready()`**
   - Polls `.status.conditions[].type == "Ready"` field
   - Checks `.status.conditions[].status == "True"`
   - Logs status messages and reasons
   - Returns True on Ready, False on timeout

3. **`delete_inference_service()`**
   - Deletes InferenceService CRD
   - Ignores 404 (already deleted)
   - Returns True on success

4. **`create_namespace()` / `get_service_url()`**
   - Helper methods for namespace management

### 3.3 Benchmark Controllers

#### BenchmarkController (K8s Mode): `src/controllers/benchmark_controller.py`

**Key Methods:**

1. **`create_benchmark_job()`**
   - Renders BenchmarkJob template
   - Posts to Kubernetes
   - CRD group: `ome.io`, version: `v1beta1`, plural: `benchmarkjobs`

2. **`wait_for_completion()`**
   - Polls `.status.state` field
   - Returns True when state == "Complete"
   - Returns False when state == "Failed" or timeout

3. **`get_benchmark_results()`**
   - Reads `.status.results` from BenchmarkJob
   - Returns metrics dictionary

#### DirectBenchmarkController (CLI Mode): `src/controllers/direct_benchmark_controller.py`

**Key Functionality:**

1. **`setup_port_forward()`**
   ```bash
   kubectl port-forward [pod|svc]/[name] 8080:8000 -n [namespace]
   ```
   - Finds pod via label selector: `serving.kserve.io/inferenceservice={service_name}`
   - Falls back to service name if no pods found
   - Subprocess-based port-forward in background
   - Returns `http://localhost:8080` endpoint URL

2. **`run_benchmark()`**
   - Builds genai-bench command with all parameters
   - Executes: `env/bin/genai-bench benchmark --api-backend openai --api-base {endpoint} ...`
   - Captures output and parses JSON results
   - Cleanup port-forward in finally block

3. **`_parse_results()`**
   - Reads JSON result files from benchmark output directory
   - Extracts metrics like latency, throughput, TTFT, TPOT

4. **`cleanup_results()`**
   - Removes local benchmark result directories

---

## 4. Configuration Files

### 4.1 Task Configuration Schema

**File:** `examples/simple_task.json` (User-provided)

```json
{
  "task_name": "simple-tune",
  "description": "Description",
  "model": {
    "name": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "llama-3-2-1b-instruct-rt",
  "parameters": {
    "tp_size": {"type": "choice", "values": [1, 2]},
    "mem_frac": {"type": "choice", "values": [0.8, 0.9]},
    "max_total_tokens": {"type": "choice", "values": [4096, 8192]},
    "schedule_policy": {"type": "choice", "values": ["lpm", "fcfs"]}
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency",
    "max_iterations": 4,
    "timeout_per_iteration": 600
  },
  "benchmark": {
    "task": "text-to-text",
    "model_name": "llama-3-2-1b-instruct",
    "model_tokenizer": "meta-llama/Llama-3.2-1B-Instruct",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [1, 4],
    "max_time_per_iteration": 10,
    "max_requests_per_iteration": 50,
    "additional_params": {"temperature": "0.0"}
  }
}
```

**Key Constraints:**
- `model.name`: Must exist as a ClusterBaseModel
- `base_runtime`: Must exist as a ClusterServingRuntime
- `parameters`: Only "choice" type supported (Cartesian product)
- `optimization.strategy`: Only "grid_search" supported
- `optimization.objective`: "minimize_latency" or "maximize_throughput"

### 4.2 Example Resource Configurations

**ClusterBaseModel Example:**
File: `/config/examples/clusterbasemodel-llama-3.2-1b.yaml`

```yaml
apiVersion: ome.io/v1beta1
kind: ClusterBaseModel
metadata:
  name: llama-3-2-1b-instruct
spec:
  vendor: meta
  version: "3.2"
  modelType: llama
  modelParameterSize: "1B"
  maxTokens: 8192
  modelCapabilities: [text-to-text]
  modelFormat:
    name: safetensors
    version: "1.0.0"
  storage:
    storageUri: hf://meta-llama/Llama-3.2-1B-Instruct
    path: /mnt/data/models/llama-3.2-1b-instruct
```

**ClusterServingRuntime Example:**
File: `/config/examples/clusterservingruntime-sglang.yaml`

```yaml
apiVersion: ome.io/v1beta1
kind: ClusterServingRuntime
metadata:
  name: llama-3-2-1b-instruct-rt
spec:
  engineConfig:
    runner:
      name: ome-container
      image: docker.io/lmsysorg/sglang:v0.5.2-cu126
      command: [python3, -m, sglang.launch_server]
      args:
        - --host=0.0.0.0
        - --port=8080
        - --model-path=/mnt/data/models/llama-3.2-1b-instruct
        - --tp-size=1
        - --enable-metrics
      resources:
        requests:
          nvidia.com/gpu: 1
        limits:
          nvidia.com/gpu: 1
```

**PVC Configuration:**
File: `/config/benchmark-pvc.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: benchmark-results-pvc
  namespace: autotuner
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 1Gi
```

---

## 5. Deployment Entry Points

### 5.1 Primary Entry Point: `run_autotuner.py`

**CLI Interface:**
```bash
python src/run_autotuner.py <task_json_file> [kubeconfig_path] [--direct]

Arguments:
  task_json_file     - Path to task configuration JSON
  kubeconfig_path    - Optional: Path to kubeconfig (defaults to ~/.kube/config)
  --direct           - Optional: Use direct genai-bench CLI instead of K8s BenchmarkJob

Examples:
  python src/run_autotuner.py examples/simple_task.json
  python src/run_autotuner.py examples/simple_task.json /path/to/kubeconfig
  python src/run_autotuner.py examples/simple_task.json --direct
  python src/run_autotuner.py examples/simple_task.json ~/.kube/config --direct
```

**Environment Activation:**
```bash
# Must activate virtual environment first
source env/bin/activate

# Then run
python src/run_autotuner.py examples/simple_task.json --direct
```

### 5.2 Installation and Setup: `install.sh`

**CLI Interface:**
```bash
./install.sh [OPTIONS]

Options:
  --skip-venv         Skip virtual environment creation
  --skip-k8s          Skip Kubernetes resource creation
  --install-ome       Install OME operator automatically
  --venv-path PATH    Custom virtual environment path
  --help              Show help message

Installation Flow:
1. Verify prerequisites (Python, pip, kubectl, git)
2. Initialize git submodules (OME, genai-bench)
3. Create Python virtual environment
4. Install dependencies from requirements.txt
5. Install genai-bench in editable mode
6. Create result/benchmark_results directories
7. Create K8s namespace and PVC
8. Install OME (if --install-ome flag)
9. Verify OME installation and CRDs
```

**Key Dependencies Installed:**
```
kubernetes>=28.1.0
pyyaml>=6.0
jinja2>=3.1.0
genai-bench (from third_party/genai-bench)
```

---

## 6. Deployment Assumptions and Constraints

### 6.1 Hardcoded Assumptions

**Kubernetes Assumptions:**
- Cluster version: v1.28+
- Namespace for autotuner: `autotuner`
- OME namespace: `ome`
- Kubernetes API available via kubeconfig or in-cluster config

**OME Assumptions:**
- OME CRDs installed: InferenceService, BenchmarkJob, ClusterBaseModel, ClusterServingRuntime
- OME controller pod running in `ome` namespace
- KEDA and cert-manager installed as OME dependencies

**Model/Runtime Assumptions:**
- Model stored in K8s cluster accessible path
- Model path: `/mnt/data/models/{model_name}`
- Runtime image: `docker.io/lmsysorg/sglang:v0.5.2-cu126` (SGLang)
- Port: 8080 (InferenceService)
- Port: 8000 (SGLang server before port-forward)

**Hardware Assumptions:**
- GPU available (nvidia.com/gpu resource)
- GPU count specified by `tp_size` parameter
- Model can fit in GPU memory with `mem_frac` setting

**Storage Assumptions:**
- PVC named `benchmark-results-pvc` exists in `autotuner` namespace
- Storage backend supports 1Gi allocation
- AccessMode: ReadWriteOnce

### 6.2 Deployment Mode Selection Logic

```python
# Determined by CLI flag
if "--direct" in sys.argv:
    use_direct_benchmark = True
    benchmark_controller = DirectBenchmarkController()
    print("Using direct genai-bench CLI execution")
else:
    use_direct_benchmark = False
    benchmark_controller = BenchmarkController(kubeconfig_path)
    print("Using Kubernetes BenchmarkJob CRD")
```

**Default Behavior:** K8s BenchmarkJob mode (requires working Docker image and PVC)

### 6.3 Status Polling and Timeouts

**InferenceService Ready Check:**
- Poll interval: 10 seconds
- Default timeout: 600 seconds (10 minutes)
- Condition checked: `.status.conditions[].type == "Ready"`
- Status field: `.status.conditions[].status` (must be "True")

**BenchmarkJob Completion Check:**
- Poll interval: 15 seconds
- Default timeout: 1800 seconds (30 minutes)
- Status field: `.status.state`
- States: "Complete", "Failed", or polling continues

**Configurable Per Task:**
```json
"optimization": {
  "timeout_per_iteration": 600  // seconds
}
```

---

## 7. Results Output and Storage

### 7.1 Results Persistence

**Location:** `results/{task_name}_results.json`

**Structure:**
```json
{
  "task_name": "simple-tune",
  "total_experiments": 4,
  "successful_experiments": 4,
  "elapsed_time": 1245.3,
  "best_result": {
    "experiment_id": 2,
    "parameters": {"tp_size": 1, "mem_frac": 0.9},
    "status": "success",
    "metrics": {...},
    "objective_score": 89.2
  },
  "all_results": [
    {
      "experiment_id": 1,
      "parameters": {"tp_size": 1, "mem_frac": 0.8},
      "status": "success",
      "metrics": {...},
      "objective_score": 125.3
    },
    ...
  ]
}
```

### 7.2 Direct CLI Mode Results

**Storage Location:** `benchmark_results/{task_name}-exp{id}/`

**File Pattern:** Genai-bench creates various output files:
- `*_results.json` - Benchmark metrics
- Various metadata and logging files

---

## 8. Deployment Workflow Diagram

```
User Provides:
  examples/simple_task.json
           │
           ▼
┌──────────────────────────────────────┐
│ run_autotuner.py main()              │
│ - Parse CLI args (kubeconfig, --direct) │
│ - Create AutotunerOrchestrator      │
└──────────────────┬───────────────────┘
                   │
                   ▼
         load_task(task_file)
           Parse JSON config
                   │
                   ▼
      generate_parameter_grid()
      All combinations from "choice" params
                   │
    ┌──────────────┴──────────────┐
    │ For each parameter set:     │
    │                             │
    ▼                             ▼
┌────────────────────┐    ┌────────────────┐
│ run_experiment()   │───▶│ Step 1:        │
│                    │    │ Deploy         │
│                    │    │ InferenceService
└────────────────────┘    └────────┬───────┘
         ▲                         │
         │         ┌───────────────┘
         │         │
         │         ▼
         │    ┌─────────────────┐
         │    │ Step 2:         │
         │    │ Wait for Ready  │
         │    └────────┬────────┘
         │             │
         │             ▼
         │    ┌──────────────────────┐
         │    │ Step 3:              │
         │    │ Run Benchmark        │
         │    │ (Two modes)          │
         │    └─────────┬────────────┘
         │              │
         │    ┌─────────┴─────────┐
         │    │                   │
         │ ┌──▼──────────┐  ┌─────▼──────────┐
         │ │Direct Mode: │  │K8s Mode:       │
         │ │- Port Fwd   │  │- BenchmarkJob  │
         │ │- Local CLI  │  │- Wait complete │
         │ └─────────────┘  │- Read PVC      │
         │                   └────────────────┘
         │                          │
         ▼                          ▼
    ┌────────────────────────────────────┐
    │ Step 4: Collect Results            │
    │ - Parse benchmark output           │
    │ - Calculate objective_score        │
    └────────────────┬───────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────┐
    │ Cleanup: Delete InferenceService   │
    │         Delete BenchmarkJob (if K8s)
    └────────────────┬───────────────────┘
                     │
                     └─────────────────┐
                                       │ Repeat for
                                       │ next parameter
                                       │ combination
                                       │
                                    (Loop continues)
                                       │
                                       ▼
    ┌────────────────────────────────────┐
    │ Find Best Result                   │
    │ - Compare objective_scores         │
    │ - Select minimum (or inverted max) │
    └────────────────┬───────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────┐
    │ Save Results                       │
    │ results/{task_name}_results.json   │
    └────────────────────────────────────┘
```

---

## 9. Key Files and Their Roles

| File | Lines | Purpose |
|------|-------|---------|
| `src/run_autotuner.py` | 305 | Main orchestrator, experiment flow control |
| `src/controllers/ome_controller.py` | 232 | InferenceService CRUD operations |
| `src/controllers/benchmark_controller.py` | 229 | BenchmarkJob CRD management (K8s mode) |
| `src/controllers/direct_benchmark_controller.py` | 335 | Direct genai-bench CLI execution |
| `src/templates/inference_service.yaml.j2` | 36 | InferenceService manifest template |
| `src/templates/benchmark_job.yaml.j2` | 41 | BenchmarkJob manifest template |
| `src/utils/optimizer.py` | 67 | Parameter grid generation, objective scoring |
| `install.sh` | 460 | Full environment setup and validation |
| `examples/simple_task.json` | 37 | Example task configuration |
| `config/benchmark-pvc.yaml` | 12 | PersistentVolumeClaim for K8s mode |
| `requirements.txt` | 4 | Python dependencies |
| `README.md` | 705 | Complete user documentation |

---

## 10. Deployment Scenarios

### Scenario 1: Direct CLI Mode (Recommended)
```
prerequisites:
  ✓ Kubernetes cluster with OME
  ✓ OME InferenceService CRD
  ✓ genai-bench CLI installed locally
  ✓ kubectl port-forward capability

flow:
  1. Deploy InferenceService (OME)
  2. Wait for pod ready
  3. kubectl port-forward to expose service
  4. Run genai-bench CLI → http://localhost:8080
  5. Parse JSON results from local filesystem
  6. Clean up port-forward
  7. Delete InferenceService
```

### Scenario 2: Kubernetes BenchmarkJob Mode
```
prerequisites:
  ✓ Kubernetes cluster with OME
  ✓ OME BenchmarkJob CRD
  ✓ PersistentVolumeClaim for results
  ✓ Working genai-bench Docker image
  ✓ Kubernetes RBAC permissions

flow:
  1. Deploy InferenceService (OME)
  2. Wait for pod ready
  3. Create BenchmarkJob CRD
  4. Wait for BenchmarkJob to complete
  5. Read results from PVC/BenchmarkJob status
  6. Delete BenchmarkJob
  7. Delete InferenceService
```

---

## 11. Summary of Deployment Mechanisms

### Current Deployment Stack:
1. **Orchestration:** Python (run_autotuner.py)
2. **Deployment Engine:** Kubernetes + OME CRDs
3. **Model Serving:** SGLang (via OME InferenceService)
4. **Benchmarking:** genai-bench (CLI or K8s Pod)
5. **Configuration:** JSON task files, YAML templates
6. **Storage:** Kubernetes PersistentVolumeClaim

### Deployment Triggers:
1. **Manual:** `python src/run_autotuner.py <task.json> [--direct]`
2. **Installation:** `./install.sh [--install-ome]`
3. **No automated triggers** (deployment is manual or scheduled externally)

### Infrastructure Code:
- Kubernetes API calls (via kubernetes-client Python library)
- kubectl port-forward (subprocess-based)
- Helm charts for OME installation (shell script)
- No Docker build or Docker Compose

### Resource Lifecycle:
- **Created:** InferenceService per experiment, BenchmarkJob per experiment (K8s mode)
- **Managed:** Namespace `autotuner`, PVC `benchmark-results-pvc`
- **Deleted:** Each resource cleaned up after experiment completes

---

## 12. Hardcoded Configuration Values

| Setting | Value | Location | Modifiable |
|---------|-------|----------|-----------|
| OME CRD Group | `ome.io` | controllers/*.py | Code change |
| OME CRD Version | `v1beta1` | controllers/*.py | Code change |
| Namespace | `autotuner` | install.sh | Yes (shell var) |
| OME Namespace | `ome` | install.sh | Code change |
| PVC Name | `benchmark-results-pvc` | benchmark_job.yaml.j2 | Code change |
| SGLang Port | 8080 | multiple files | Code change |
| Port Forward Remote | 8000 | direct_benchmark_controller.py | Code change |
| Port Forward Local | 8080 | direct_benchmark_controller.py | Code change |
| Poll Interval (ISVC) | 10s | ome_controller.py | Code change |
| Poll Interval (Benchmark) | 15s | benchmark_controller.py | Code change |
| Default Timeout | 600s, 1800s | run_autotuner.py | Task JSON |
| Genai-bench Image | `kllambda/genai-bench:v251014` | benchmark_job.yaml.j2 | Code change |
| SGLang Image | `docker.io/lmsysorg/sglang:v0.5.2-cu126` | config examples | YAML |

