# Parallel Experiment Execution

Design document for parallel experiment execution feature that enables running multiple experiments concurrently.

## Overview

The parallel execution feature enables running multiple experiments simultaneously instead of sequentially, dramatically reducing total task completion time. With intelligent GPU resource management, the system can maximize hardware utilization while preventing resource conflicts.

## Benefits

- **5-10x speedup**: For tasks with many experiments and available GPUs
- **Better GPU utilization**: Run multiple small experiments on different GPUs
- **Configurable concurrency**: Control parallelism based on system capacity
- **Error isolation**: Failed experiments don't block others

## Architecture

### Current Sequential Flow

```
Task Started → Worker Loop:
  Exp 1: suggest → create → deploy → benchmark → record → |
  Exp 2: suggest → create → deploy → benchmark → record → |
  Exp 3: suggest → create → deploy → benchmark → record → |
  ...
  → Task Complete
```

**Total time**: sum(experiment_durations)

### Proposed Parallel Flow

```
Task Started → Worker Loop:
  Batch 1 (parallel):
    ├─ Exp 1: suggest → create → deploy → benchmark → record
    ├─ Exp 2: suggest → create → deploy → benchmark → record
    └─ Exp 3: suggest → create → deploy → benchmark → record
  Batch 2 (parallel):
    ├─ Exp 4: suggest → create → deploy → benchmark → record
    └─ Exp 5: suggest → create → deploy → benchmark → record
  ...
  → Task Complete
```

**Total time**: sum(max(batch_durations))

### Key Components

#### 1. GPU Resource Pool

```python
class GPUResourcePool:
    """Manages GPU allocation for concurrent experiments."""
    
    def __init__(self, max_parallel: int):
        self.max_parallel = max_parallel
        self.available_gpus = asyncio.Queue()
        self.in_use = set()
    
    async def acquire(self, required_gpus: int) -> List[int]:
        """Acquire GPU resources for an experiment."""
        # Wait until required_gpus are available
        # Return list of GPU indices
        
    async def release(self, gpu_indices: List[int]):
        """Release GPU resources back to pool."""
```

#### 2. Async Experiment Executor

```python
async def run_experiment_async(
    orchestrator,
    task_config,
    iteration,
    params,
    gpu_pool: GPUResourcePool,
    db: AsyncSession
):
    """Run single experiment with async GPU allocation."""
    
    # Estimate GPU requirements
    required_gpus = estimate_gpu_requirements(task_config)
    
    # Acquire GPUs from pool (blocks if unavailable)
    gpu_indices = await gpu_pool.acquire(required_gpus)
    
    try:
        # Run experiment with allocated GPUs
        result = await run_experiment_with_timeout(...)
        
        # Update database
        await update_experiment_record(db, iteration, result)
        
    finally:
        # Always release GPUs
        await gpu_pool.release(gpu_indices)
```

#### 3. Parallel Batch Executor

```python
async def run_experiments_parallel(
    strategy,
    orchestrator,
    task_config,
    max_parallel: int,
    db: AsyncSession
):
    """Run experiments in parallel batches."""
    
    gpu_pool = GPUResourcePool(max_parallel)
    tasks = []
    
    while not strategy.should_stop():
        # Suggest parameters for next experiment
        params = strategy.suggest_parameters()
        if params is None:
            break
        
        # Create async task for experiment
        task = asyncio.create_task(
            run_experiment_async(
                orchestrator, task_config, iteration, params,
                gpu_pool, db
            )
        )
        tasks.append(task)
        
        # Limit concurrent tasks
        if len(tasks) >= max_parallel:
            # Wait for at least one to complete
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            tasks = list(pending)
    
    # Wait for all remaining tasks
    await asyncio.gather(*tasks)
```

## Implementation Plan

### Phase 1: Database Preparation ✅

**Goal**: Enable concurrent database writes

**Tasks**:
- Enable SQLite WAL (Write-Ahead Logging) mode
- Test concurrent writes from multiple coroutines
- Add database connection pooling if needed

**Changes**:
```python
# src/web/db/session.py
engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={
        "check_same_thread": False,
        "timeout": 30
    }
)

# Enable WAL mode
async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("PRAGMA journal_mode=WAL"))
```

### Phase 2: GPU Resource Pool ✅

**Goal**: Implement GPU allocation/deallocation system

**Tasks**:
- Create GPUResourcePool class
- Implement acquire/release with asyncio primitives
- Add GPU availability checking
- Integrate with existing gpu_monitor

**File**: `src/utils/gpu_pool.py` (new)

### Phase 3: Async Experiment Execution ✅

**Goal**: Convert experiment execution to async

**Tasks**:
- Wrap orchestrator.run_experiment in async executor
- Update experiment record creation/updates for async
- Add error handling and isolation
- Implement cleanup on failure

**File**: `src/web/workers/autotuner_worker.py` (modified)

### Phase 4: Configuration ✅

**Goal**: Add user-configurable concurrency settings

**Tasks**:
- Add max_parallel_experiments to optimization config
- Update Task model with new field
- Add UI controls in NewTask.tsx
- Add validation and defaults

**Changes**:
```json
{
  "optimization": {
    "strategy": "grid_search",
    "max_iterations": 100,
    "max_parallel_experiments": 3  // NEW
  }
}
```

### Phase 5: Testing & Validation ✅

**Goal**: Verify parallel execution works correctly

**Tasks**:
- Unit tests for GPUResourcePool
- Integration tests with mock experiments
- Load testing with real GPUs
- Verify no database conflicts
- Check GPU resource tracking

## Configuration

### Task JSON

```json
{
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency",
    "max_iterations": 20,
    "max_parallel_experiments": 3,
    "timeout_per_iteration": 600
  }
}
```

### Recommended Settings

| Scenario | max_parallel_experiments | Rationale |
|----------|------------------------|-----------|
| Single GPU system | 1 | No benefit to parallelism |
| 4 GPU system, small models | 4 | One experiment per GPU |
| 8 GPU system, large models (TP=4) | 2 | Two experiments, each using 4 GPUs |
| Limited CPU/memory | 2-3 | Avoid system overload |
| Fast experiments (<2 min) | 1 | Overhead not worth it |
| Slow experiments (>10 min) | 4-8 | Maximize parallelism |

## GPU Resource Management

### Allocation Strategy

```
Available GPUs: [0, 1, 2, 3, 4, 5, 6, 7]

Experiment 1 needs 2 GPUs → Allocate [0, 1] → In use: {0, 1}
Experiment 2 needs 4 GPUs → Allocate [2, 3, 4, 5] → In use: {0, 1, 2, 3, 4, 5}
Experiment 3 needs 2 GPUs → Wait (only 2 GPUs free) → Queued

Experiment 1 completes → Release [0, 1] → In use: {2, 3, 4, 5}
Experiment 3 proceeds → Allocate [0, 1] → In use: {0, 1, 2, 3, 4, 5}
```

### Resource Pool Properties

- **Fair allocation**: FIFO queue for waiting experiments
- **Deadlock prevention**: Acquire all GPUs atomically or wait
- **Automatic cleanup**: GPUs released even if experiment fails
- **Smart selection**: Prefer least-utilized GPUs (from gpu_monitor)

## Error Handling

### Isolation

Each experiment runs in independent async task:
- Exceptions caught and logged
- Failed experiment marked in database
- Other experiments continue unaffected
- GPUs properly released on failure

### Recovery

```python
try:
    result = await run_experiment_async(...)
except asyncio.TimeoutError:
    logger.error(f"Experiment {iteration} timed out")
    await mark_experiment_failed(db, iteration, "Timeout")
except Exception as e:
    logger.error(f"Experiment {iteration} failed: {e}")
    await mark_experiment_failed(db, iteration, str(e))
finally:
    await gpu_pool.release(gpu_indices)
```

## Performance Impact

### Expected Speedup

Assuming 3 experiments run in parallel:

**Sequential**:
```
Exp 1: 10 min
Exp 2: 10 min  → Total: 30 min
Exp 3: 10 min
```

**Parallel** (max_parallel=3):
```
Exp 1, 2, 3 (concurrent): 10 min → Total: 10 min
```

**Speedup**: 3x (linear with concurrency)

### Realistic Scenarios

1. **Small models, many experiments**:
   - 20 experiments × 5 min each = 100 min sequential
   - With max_parallel=4: 25 min (4x speedup)

2. **Large models, few experiments**:
   - 10 experiments × 30 min each = 300 min sequential
   - With max_parallel=2: 150 min (2x speedup)

3. **Mixed workload**:
   - Some experiments fail fast, others run full duration
   - Speedup varies: 2-5x typical

## Limitations

### SQLite Constraints

- WAL mode required for concurrent writes
- Database on NFS may have issues (use local disk)
- Maximum ~1000 concurrent writers (far exceeds our needs)

### GPU Constraints

- Cannot run more experiments than available GPUs
- Multi-GPU experiments (TP>1) reduce effective parallelism
- GPU memory fragmentation may limit concurrency

### System Constraints

- CPU/memory overhead for multiple containers
- Network bandwidth for concurrent downloads (HuggingFace models)
- Disk I/O for logs and database writes

## Best Practices

1. **Start conservative**: Begin with max_parallel=2, increase gradually
2. **Monitor GPU usage**: Use `watch -n 1 nvidia-smi` during task
3. **Check logs**: Ensure no "GPU unavailable" errors
4. **Adjust for model size**: Large models → lower concurrency
5. **Consider experiment duration**: Short experiments → lower concurrency (overhead)

## Troubleshooting

### Problem: No speedup observed

**Symptoms**: Experiments still run sequentially

**Solutions**:
- Check max_parallel_experiments > 1 in task config
- Verify sufficient GPUs available
- Check logs for "Waiting for GPUs" messages
- Ensure WAL mode enabled: `sqlite3 autotuner.db "PRAGMA journal_mode"`

### Problem: GPU allocation errors

**Symptoms**: "No GPUs available" despite free GPUs

**Solutions**:
- Check GPU resource pool initialization
- Verify gpu_monitor is working
- Look for GPU leak (not releasing properly)
- Restart task to reset pool

### Problem: Database lock errors

**Symptoms**: "database is locked" errors in logs

**Solutions**:
- Enable WAL mode: `PRAGMA journal_mode=WAL`
- Increase timeout: `connect_args={"timeout": 30}`
- Reduce max_parallel_experiments
- Check database not on NFS

## Future Enhancements

- **Dynamic concurrency**: Adjust based on GPU availability
- **Priority queuing**: High-priority experiments skip queue
- **Cross-task parallelism**: Multiple tasks share GPU pool
- **Distributed execution**: Run experiments across multiple nodes
- **Smart batching**: Group experiments with similar GPU requirements

## References

- SQLite WAL mode: https://www.sqlite.org/wal.html
- AsyncIO task management: https://docs.python.org/3/library/asyncio-task.html
- GPU resource management: docs/GPU_TRACKING.md

---

## Implementation Status

### Phase 1: Database Preparation ✅ COMPLETE

**Goal**: Enable concurrent database writes using SQLite WAL mode

**Implementation** (`src/web/db/session.py`):
- Added `check_same_thread=False` and `timeout=30` to engine config
- Enabled `PRAGMA journal_mode=WAL` in `init_db()`
- WAL mode allows concurrent readers and single writer

**Benefits**:
- Multiple readers can access database concurrently
- Writer doesn't block readers during commits
- Improved throughput for parallel experiment updates

### Phase 2: GPU Tracking ✅ COMPLETE

**Goal**: Track GPU availability and allocate to experiments

**Implementation** (`src/controllers/gpu_tracker.py`):
- Real-time GPU monitoring via nvidia-smi
- Thread-safe allocation tracking
- Automatic cleanup on experiment completion

**See**: GPU_TRACKING.md for detailed documentation

### Phase 3: Parallel Orchestrator 🚧 IN PROGRESS

**Goal**: Execute multiple experiments concurrently

**Status**: ~60% complete
- Concurrent experiment scheduling
- GPU-aware task distribution
- Progress tracking and error handling

### Expected Performance

With parallel execution:
- **Grid search**: 5-10x speedup with 4+ GPUs
- **Bayesian optimization**: 2-3x speedup (sequential dependencies limit parallelism)
- **Limited by**: GPU count, memory per GPU, parameter space size

