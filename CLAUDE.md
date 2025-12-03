# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The LLM Inference Autotuner is a system for automatically tuning LLM inference engine parameters. It supports dual deployment modes (Kubernetes/OME and standalone Docker) with a web UI for task management and background job processing.

**Key Capabilities:**
- Grid search, random search, and **Bayesian optimization** across parameter combinations
- Bayesian optimization typically finds optimal configs in 20-30 experiments (80-87% reduction vs grid search)
- SLO-aware scoring with exponential penalties for constraint violations
- Benchmarking with genai-bench for performance metrics
- REST API with background task queue (ARQ + Redis)
- SQLite database for persistence
- React frontend with enhanced result visualization and multi-experiment comparison

## Architecture

### System Components

The autotuner follows a **three-tier architecture**:

1. **Presentation Layer**: React frontend (`frontend/`) + FastAPI REST API (`src/web/`)
2. **Business Logic Layer**: Orchestrator (`src/orchestrator.py`) + Controllers (`src/controllers/`)
3. **Deployment Layer**: Kubernetes/OME or Docker containers

### Core Design Patterns

**Strategy Pattern for Deployment Modes:**
- Controllers implement `BaseModelController` abstract interface
- **OMEController** + **BenchmarkController**: Kubernetes InferenceService via CRDs
- **DockerController** + **DirectBenchmarkController**: Standalone Docker containers
- Controllers are selected at runtime via `deployment_mode` parameter

**Async Task Queue Architecture:**
- FastAPI accepts task creation requests synchronously
- Task immediately enqueued to ARQ (Redis-backed queue)
- ARQ worker (`src/web/workers/autotuner_worker.py`) processes tasks in background
- Database tracks task/experiment status in real-time
- Frontend polls for updates (WebSocket support TODO)

**Experiment Lifecycle:**
```
Task Created → Enqueued → Worker Picks Up → For Each Parameter Combination:
  1. Create Experiment Record (status: PENDING)
  2. Deploy Inference Service (status: DEPLOYING)
  3. Wait for Service Ready
  4. Run Benchmark (genai-bench CLI or K8s BenchmarkJob)
  5. Parse Metrics & Calculate Objective Score
  6. Update Experiment (status: SUCCESS/FAILED)
  7. Cleanup Inference Service
→ Mark Best Experiment → Complete Task
```

## Running the System

### Quick Start (Full Stack)

For complete development environment with UI:

```bash
# Terminal 1: Start backend (API + Worker)
./scripts/start_dev.sh

# Terminal 2: Start frontend
cd frontend && npm run dev
```

Then access:
- **Frontend UI**: http://localhost:5173 (all services accessible through this port via proxy)
- **Backend API** (direct): http://localhost:8000
- **API Docs**: http://localhost:8000/docs

**Note**: The frontend proxies all `/api/*` requests to the backend at port 8000, so you can access everything through the frontend port (5173) without CORS issues.

### Development Commands

**Backend (API + Worker):**
```bash
# Start both web server and ARQ worker together
./scripts/start_dev.sh

# Or start them separately:
cd src && python web/server.py              # Web server at http://localhost:8000
./scripts/start_worker.sh                    # ARQ worker
```

**Frontend (React UI):**
```bash
cd frontend
npm install                     # First time only
npm run dev                     # Development server at http://localhost:5173
npm run build                   # Production build
npm run type-check              # TypeScript type checking
npm run format                  # Prettier formatting
npm run format:check            # Check formatting without changes
npm run lint                    # ESLint checking
```

**Frontend Features (Implemented):**
- Task creation wizard with form builder and SLO configuration
- Task list with status tracking and filtering
- Experiment results table with metrics visualization and SLO violation badges
- Real-time log viewer for task execution
- Docker container monitoring (Docker mode)
- Recharts-based performance graphs
- Parameter preset selector for common configurations

**CLI (Direct Execution):**
```bash
# Docker mode (standalone, no K8s required)
python src/run_autotuner.py examples/docker_task.json --mode docker --direct

# OME mode (Kubernetes + OME operator)
python src/run_autotuner.py examples/simple_task.json --mode ome --direct

# With verbose genai-bench output for debugging
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose
```

### Worker Management

**CRITICAL**: After editing any code in `src/orchestrator.py`, `src/controllers/`, or `src/web/workers/`, you **must restart the ARQ worker** for changes to take effect:

```bash
# Find and kill existing worker
ps aux | grep arq
kill <PID>

# Restart worker
./scripts/start_worker.sh
```

The web server (`src/web/server.py`) has hot-reload enabled, but ARQ workers do not.

### Database Location

All user data stored in XDG-compliant location:
```
~/.local/share/inference-autotuner/
├── autotuner.db          # SQLite database (tasks, experiments, metrics)
└── logs/
    └── task_<id>.log     # Per-task execution logs
```

## Task Configuration

Task JSON defines experiments. Critical fields:

```json
{
  "task_name": "unique-identifier",
  "model": {
    "id_or_path": "llama-3-2-1b-instruct",  // Directory in /mnt/data/models/ OR HuggingFace ID
    "namespace": "autotuner"                 // K8s namespace OR Docker label
  },
  "base_runtime": "sglang",                  // Runtime engine: "sglang" or "vllm"
  "runtime_image_tag": "v0.5.2-cu126",       // Docker mode only: image version
  "parameters": {
    "tp-size": [1, 2, 4],                    // Simple format (recommended)
    "mem-fraction-static": [0.7, 0.8, 0.9],
    "schedule-policy": ["lpm", "fcfs"]
  },
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency",         // or "maximize_throughput", "minimize_ttft", "minimize_tpot"
    "max_iterations": 10,
    "timeout_per_iteration": 600
  },
  "slo": {                                   // Optional: SLO-aware scoring
    "ttft": {
      "threshold": 1.0,                      // Time to First Token (seconds)
      "weight": 2.0,
      "hard_fail": false
    },
    "tpot": {
      "threshold": 0.05,                     // Time Per Output Token (seconds)
      "weight": 2.0,
      "hard_fail": false
    },
    "latency": {
      "p90": {
        "threshold": 5.0,                    // P90 latency (seconds)
        "weight": 2.0,
        "hard_fail": true,
        "fail_ratio": 0.2                    // Fail if >20% over threshold
      }
    },
    "steepness": 0.1                         // Exponential curve parameter (lower = steeper)
  },
  "benchmark": {
    "task": "text-to-text",
    "model_name": "Llama-3.2-1B-Instruct",   // Display name (auto-filled in UI)
    "model_tokenizer": "meta-llama/Llama-3.2-1B-Instruct",  // HF tokenizer ID
    "traffic_scenarios": ["D(100,100)"],     // genai-bench traffic pattern
    "num_concurrency": [1, 4, 8],
    "additional_params": {"temperature": 0.0}  // Must be correct types (float not string)
  }
}
```

**Parameter Format Notes:**
- Use exact CLI flag format: `tp-size` not `tp_size` (hyphens not underscores)
- The `--` prefix is added automatically
- Simple format: `"param-name": [val1, val2]` (recommended)
- Legacy format: `"param_name": {"type": "choice", "values": [...]}` (backward compatible)

**SLO Configuration Notes:**
- All SLO fields are optional (can omit metrics, weights, etc.)
- Exponential penalties: `penalty = weight × exp(violation_ratio / steepness)`
- Violation ratio: `(actual - threshold) / threshold`
- Soft penalties (hard_fail=false): Multiply objective score
- Hard failures (hard_fail=true + violation > fail_ratio): Experiment marked as FAILED
- See `docs/SLO_SCORING.md` for detailed documentation

## Critical Implementation Details

### Docker Mode Specifics

**GPU Access** (src/controllers/docker_controller.py):
- Command must be a **list**, not string: `command_str.split()`
- Use `docker.types.DeviceRequest(device_ids=['0'], capabilities=[['gpu']])`
- **DO NOT** set `CUDA_VISIBLE_DEVICES` env var (conflicts with `device_requests`)

**Port Management:**
- Auto-allocates ports 8000-8100 to avoid conflicts
- Each experiment uses unique port

**Container Lifecycle:**
- Containers created with `auto_remove=True` (cleanup on stop)
- Logs retrieved before removal, saved to `~/.local/share/inference-autotuner/logs/task_<id>.log`
- Check running containers: `docker ps`

**Model Path Mapping:**
```
Task JSON "model.id_or_path" → /mnt/data/models/{id_or_path} → mounted as /model in container
```

### OME Mode Specifics

**Kubernetes Templates** (src/templates/):
- Uses Jinja2 for InferenceService/BenchmarkJob YAML generation
- Labels must be **strings**: `"{{ experiment_id }}"` not `{{ experiment_id }}`
- Environment variables use K8s syntax: `$(ENV_VAR)`

**Benchmark Execution Modes:**
1. **BenchmarkJob CRD** (`BenchmarkController`): Native OME job runner (requires genai-bench Docker image)
2. **Direct CLI** (`DirectBenchmarkController`): Local genai-bench + `kubectl port-forward` (recommended)

### Benchmark Execution

**DirectBenchmarkController** (src/controllers/direct_benchmark_controller.py):
- **Docker mode**: Direct URL via `endpoint_url` parameter
- **OME mode**: Automatic `kubectl port-forward` if `endpoint_url=None`
- Use `--verbose` flag for real-time genai-bench output (debugging)

### Web API Architecture

**Database Models** (src/web/db/models.py):
- `Task`: Top-level tuning task with config, status, timing
- `Experiment`: Individual parameter combination trial linked to task
- Status enums: `PENDING → RUNNING → COMPLETED/FAILED`

**Key Routes** (src/web/routes/):
- `POST /api/tasks/`: Create task (synchronous, returns immediately)
- `POST /api/tasks/{id}/start`: Enqueue task to ARQ worker
- `GET /api/tasks/{id}`: Get task status and timing
- `GET /api/experiments/task/{id}`: Get all experiments for task with metrics

**ARQ Worker Configuration** (src/web/workers/autotuner_worker.py):
- `max_jobs = 5`: Maximum concurrent autotuning tasks
- `job_timeout = 7200`: 2 hours per task
- Logs redirected to both file and console via `StreamToLogger` class
- **CRITICAL**: Worker must be restarted after code changes

### Frontend Architecture

**Tech Stack:**
- **React 18** with TypeScript
- **Vite** for build tooling and hot module replacement
- **React Router** for navigation (Dashboard, Tasks, Experiments, Containers)
- **TanStack Query** (React Query) for API state management and caching
- **Axios** for HTTP client
- **Recharts** for metrics visualization
- **Tailwind CSS** for styling
- **React Hot Toast** for notifications

**Key Pages** (frontend/src/pages/):
- `Dashboard.tsx`: Overview and system status
- `Tasks.tsx`: Task list with create/start/monitor capabilities
- `NewTask.tsx`: Task creation wizard with form builder
- `Experiments.tsx`: Experiment results and metrics
- `Containers.tsx`: Docker container monitoring (Docker mode only)

**Components** (frontend/src/components/):
- `Layout.tsx`: Main layout with navigation
- `TaskResults.tsx`: Results visualization with graphs
- `LogViewer.tsx`: Real-time log streaming and viewing

**API Integration Pattern:**
- Uses TanStack Query (React Query) for server state management
- Automatic caching, refetching, and background updates
- API client in `services/` wraps Axios calls
- Type-safe with TypeScript interfaces from `types/`
- Polling-based updates (WebSocket migration planned)

## Common Issues

### Docker Mode
1. **"No accelerator available"**: Command not split into list properly
2. **Model not found**: Path mismatch between task JSON and `/mnt/data/models/`
3. **Port already in use**: Check existing services on ports 8000-8100

### OME Mode
1. **InferenceService not ready**: Check `kubectl describe inferenceservice -n autotuner`
2. **GPU resources unavailable**: Minikube Docker driver doesn't support GPUs (use `--driver=none`)
3. **BenchmarkJob fails**: Use `--direct` mode to bypass Docker image issues

### Both Modes
1. **genai-bench type errors**: `additional_params` must use correct types (float 0.0, not string "0.0")
2. **Missing API key error**: genai-bench requires `--api-key dummy` even for local servers
3. **Network unreachable**: genai-bench fetches tokenizer from HuggingFace - use proxy if needed

### Web API
1. **Task stuck in RUNNING**: Check ARQ worker logs at `logs/worker.log`
2. **Worker not processing tasks**: Verify Redis is running (`docker ps | grep redis`)
3. **Changes not taking effect**: Restart ARQ worker after code modifications

### Frontend
1. **CORS errors**: Ensure backend is running on port 8000 (frontend expects this)
2. **API connection failed**: Check that `./scripts/start_dev.sh` is running
3. **Build errors**: Run `npm run type-check` to verify TypeScript issues
4. **Styling issues**: Ensure Tailwind CSS is configured (`npm run build` to rebuild)
5. **Hot reload not working**: Restart Vite dev server with `npm run dev`

## Development Workflow

### Full Stack Development

1. **Start backend services**: `./scripts/start_dev.sh`
   - Starts both web server (port 8000) and ARQ worker
   - Web server has hot-reload enabled
   - Worker logs to `logs/worker.log`

2. **Start frontend** (separate terminal): `cd frontend && npm run dev`
   - Development server on port 5173 with hot module replacement
   - TypeScript checking: `npm run type-check`
   - Format code: `npm run format`

3. **Make code changes**:
   - **Backend Python** (`src/`): Web server auto-reloads
   - **Worker/Controller changes**: Restart ARQ worker (see Worker Management section)
   - **Frontend** (`frontend/src/`): Vite provides instant HMR

4. **Testing strategy**:
   - Test with **Docker mode first**: Faster iteration than OME mode
   - Use single parameter values to reduce experiment count
   - Monitor logs: `tail -f logs/worker.log`

5. **Debugging**:
   - Backend API: Check `logs/worker.log` and web server console
   - Frontend: Browser DevTools + React Query DevTools
   - Task execution: `~/.local/share/inference-autotuner/logs/task_<id>.log`
   - Database inspection: `sqlite3 ~/.local/share/inference-autotuner/autotuner.db`

### Testing Shortcuts

**Reduce experiment count** for faster testing:
```json
{
  "parameters": {
    "tp-size": [1],                    // Single value instead of [1, 2, 4]
    "mem-fraction-static": [0.85]
  },
  "optimization": {
    "max_iterations": 1                // Limit iterations
  },
  "benchmark": {
    "num_concurrency": [1]             // Single concurrency level
  }
}
```

**Monitor GPU usage**: `watch -n 1 nvidia-smi` (Llama-3.2-1B requires ~3GB GPU memory)

## Project Structure

```
inference-autotuner/
├── src/                           # Backend Python code
│   ├── run_autotuner.py           # CLI entry point
│   ├── orchestrator.py            # Core experiment coordination logic
│   ├── controllers/               # Deployment strategy implementations
│   │   ├── base_controller.py     # Abstract interface
│   │   ├── docker_controller.py   # Docker deployment
│   │   ├── ome_controller.py      # Kubernetes/OME deployment
│   │   ├── direct_benchmark_controller.py  # genai-bench CLI execution
│   │   └── benchmark_controller.py         # K8s BenchmarkJob CRD
│   ├── utils/
│   │   └── optimizer.py           # Parameter grid generation
│   ├── templates/                 # Jinja2 templates for K8s resources
│   └── web/                       # FastAPI web application
│       ├── app.py                 # FastAPI app instance
│       ├── server.py              # Development server (hot reload enabled)
│       ├── config.py              # Settings (env vars, paths)
│       ├── routes/                # API endpoints
│       │   ├── tasks.py           # Task CRUD + start
│       │   ├── experiments.py     # Experiment queries
│       │   ├── docker.py          # Docker-specific endpoints
│       │   └── system.py          # Health checks
│       ├── db/                    # Database layer
│       │   ├── models.py          # SQLAlchemy ORM models
│       │   └── session.py         # Database connection management
│       ├── schemas/               # Pydantic request/response schemas
│       └── workers/               # ARQ background task workers
│           ├── autotuner_worker.py  # Main task execution logic
│           └── client.py          # ARQ client for enqueuing
├── frontend/                      # React UI
│   ├── src/
│   │   ├── components/            # React components
│   │   │   ├── Layout.tsx         # Main layout with navigation
│   │   │   ├── TaskResults.tsx    # Results visualization
│   │   │   └── LogViewer.tsx      # Log streaming viewer
│   │   ├── pages/                 # Page components
│   │   │   ├── Dashboard.tsx      # System overview
│   │   │   ├── Tasks.tsx          # Task management
│   │   │   ├── NewTask.tsx        # Task creation wizard
│   │   │   ├── Experiments.tsx    # Experiment results
│   │   │   └── Containers.tsx     # Docker monitoring (Docker mode)
│   │   ├── services/              # API client services
│   │   ├── types/                 # TypeScript interfaces
│   │   ├── utils/                 # Helper functions
│   │   ├── styles/                # CSS styles
│   │   ├── App.tsx                # Root component
│   │   └── main.tsx               # App entry point
│   ├── package.json               # npm dependencies
│   ├── vite.config.ts             # Vite configuration
│   ├── tailwind.config.js         # Tailwind CSS config
│   └── tsconfig.json              # TypeScript config
├── examples/                      # Task JSON configurations
│   ├── docker_task.json           # Docker mode example
│   └── simple_task.json           # OME mode example
├── config/                        # Kubernetes resources (OME mode)
├── scripts/                       # Helper scripts
│   ├── start_dev.sh               # Start web + worker together
│   └── start_worker.sh            # Start ARQ worker only
├── docs/                          # Documentation
├── requirements.txt               # Python dependencies
└── install.sh                     # Installation script (with optional OME setup)
```

## Environment Variables

Configuration via `.env` file or environment (see `src/web/config.py`):

```bash
# Database
DATABASE_URL=sqlite+aiosqlite:////home/user/.local/share/inference-autotuner/autotuner.db

# Redis (for ARQ)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Docker mode
DOCKER_MODEL_PATH=/mnt/data/models

# Network proxy (for HuggingFace downloads)
HTTP_PROXY=http://172.17.0.1:1081
HTTPS_PROXY=http://172.17.0.1:1081
HF_TOKEN=<your-token>            # Optional: for gated models
```

## Documentation

- `README.md`: Installation and usage overview
- `docs/DOCKER_MODE.md`: Docker deployment guide
- `docs/OME_INSTALLATION.md`: Kubernetes/OME setup
- `docs/TROUBLESHOOTING.md`: Common issues and solutions
- (Benchmark logs info now in TROUBLESHOOTING.md)
- `agentlog/`: Development diary organized by date (yyyy/mmdd.md format)

## Meta-Instructions

**Critical constraints**:
1. **Kubernetes Dashboard on port 8443** - do not use this port
2. **Frontend dev server on port 5173** - Vite default, proxies API to port 8000
3. **Update development diary** when a mini-milestone is accomplished
    - Write directly to today's diary file: `agentlog/yyyy/mmdd.md`
    - Include both user's prompt and agent's response
    - Agent's response should be enclosed in foldable `<details>`
    - Summarize concisely if original text is too redundant
    - Fix typos in prompt text if present
4. **Place new .md docs in `./docs/`**
5. **Consult `docs/TROUBLESHOOTING.md`** when encountering issues, maintain it when resolving new issues
6. **Restart ARQ worker** after editing relevant code files
7. DONOT use git commands with writing effects
8. **Follow `CLAUDE.local.md`** if present for further local instructions

**Development Diary Structure**:
- **Diary Files**: `agentlog/yyyy/mmdd.md` - Daily entries written directly
- **Format**: Date-based section headers + user prompt + agent response in `<details>`
- **Creation**: Auto-create directory and file if today's diary doesn't exist
- **Separators**: Use `---` between multiple entries on same day
- **Legacy Script**: `scripts/split_agentlog.js` preserved for historical log processing


**Current implementation status**:
- ✅ **React frontend is fully implemented** (Dashboard, Tasks, Experiments, Container monitoring)
- ✅ **Enhanced result visualization** with multi-experiment comparison, sensitivity analysis, and Pareto frontier
- ✅ **Bayesian optimization** with Optuna TPE sampler (backend + frontend + docs complete)
- ✅ **SLO-aware scoring** with exponential penalties and tiered enforcement
- ✅ **WebSocket support for real-time updates** (backend + frontend with automatic reconnection)
- ✅ REST API with background processing
- ✅ Docker and OME deployment modes
- ✅ Parameter preset selector for common configurations
- ✅ GPU resource tracking with intelligent allocation and real-time monitoring
