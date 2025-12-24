# Code Refactoring Plan

**Date**: 2025-12-24
**Reviewed by**: GPT-5.2 Agent
**Status**: In Progress

## Executive Summary

This document outlines a comprehensive refactoring plan to address DRY (Don't Repeat Yourself) violations and code quality issues identified in the inference-autotuner codebase. The plan is designed to minimize conflicts with the ongoing distributed worker development on the `develop` branch.

**Estimated Impact**:
- **LOC Reduction**: 350-700 lines
- **Files Affected**: 15-25 files
- **New Helper Modules**: 10-14 files
- **Conflict Risk**: Low (avoids worker infrastructure)

---

## Top 10 Most Impactful Refactorings

Ranked by value/effort ratio and conflict risk:

### 1. Shared GPU Selection + Memory-Balance Logic
- **Value**: High (removes drift across `gpu_monitor`, `gpu_pool`, controllers)
- **Effort**: Low-Medium
- **Conflict Risk**: Low
- **Change**: Add `utils/gpu_selection.py` + use in `gpu_monitor.allocate_gpus()` and `gpu_pool._select_gpus()`
- **LOC Reduction**: ~80-120

### 2. Unify GPU Type Definitions
- **Value**: High (prevents import confusion; easier reuse)
- **Effort**: Low
- **Conflict Risk**: Low
- **Change**: Add `utils/gpu_types.py` with `LocalGPUInfo`/`ClusterGPUInfo`; alias old names
- **LOC Reduction**: ~30-50

### 3. Centralize Parallelism/World-Size Calculation
- **Value**: High (consistent GPU requirement calculation)
- **Effort**: Medium
- **Conflict Risk**: Low
- **Change**: Add `utils/parallelism.py` (`parse_parallel_factors`, `compute_world_size`)
- **LOC Reduction**: ~50-80

### 4. Extract HTTP Readiness Polling Helper
- **Value**: High (cuts repeated readiness loops)
- **Effort**: Low
- **Conflict Risk**: Low
- **Change**: Add `controllers/readiness.py`, use in Docker/Local controllers
- **LOC Reduction**: ~40-60

### 5. Extract CLI-Arg Building Helper
- **Value**: High (reduces repeated bool flag / --key value code)
- **Effort**: Low
- **Conflict Risk**: Low
- **Change**: Add `controllers/cli_args.py`
- **LOC Reduction**: ~30-50

### 6. Shared Name Sanitization Utilities
- **Value**: Medium-High (standardizes naming)
- **Effort**: Low
- **Conflict Risk**: Low
- **Change**: Add `controllers/naming.py` with wrappers
- **LOC Reduction**: ~20-40

### 7. Consolidate Port-Finding + Env-Building
- **Value**: Medium (DRY + correctness)
- **Effort**: Low
- **Conflict Risk**: Low
- **Change**: Add `utils/network.py`, `utils/env.py`
- **LOC Reduction**: ~30-50

### 8. Web: Factor `get_*_or_404` Dependencies
- **Value**: High (removes repeated DB/404 logic)
- **Effort**: Low
- **Conflict Risk**: Low
- **Change**: Add `web/routes/deps.py`, update routes
- **LOC Reduction**: ~40-70

### 9. Web: Extract DB Commit/Refresh Boilerplate
- **Value**: Medium (small but widespread DRY)
- **Effort**: Low
- **Conflict Risk**: Low
- **Change**: Add `web/services/db_utils.py`
- **LOC Reduction**: ~20-40

### 10. Split `optimizer.py` God Module
- **Value**: High (testability, maintainability)
- **Effort**: Medium-High
- **Conflict Risk**: Low-Medium
- **Change**: Split into `utils/slo.py`, `utils/scoring.py`, `utils/search_space.py`
- **LOC Reduction**: ~50-120 (complexity reduction)

---

## Quick Wins (<50 LOC, High Value)

These changes provide immediate value with minimal risk:

### QW1: `web/routes/deps.py` - Get-or-404 Dependencies
**Current Problem**: Repeated select/404 blocks in multiple routes
```python
result = await db.execute(select(Task).where(Task.id == task_id))
task = result.scalar_one_or_none()
if not task:
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
```

**Solution**: FastAPI dependency injection
```python
async def get_task_or_404(task_id: int, db: AsyncSession = Depends(get_db)) -> Task:
    task = (await db.execute(select(Task).where(Task.id == task_id))).scalar_one_or_none()
    if not task:
        raise HTTPException(404, f"Task {task_id} not found")
    return task
```

**Impact**: ~30-45 LOC reduction, standardized error messages

### QW2: `web/services/db_utils.py` - DB Boilerplate
**Current Problem**: Repeated commit/refresh patterns
```python
await db.commit()
await db.refresh(obj)
return obj
```

**Solution**: Generic helpers
```python
async def create_and_refresh(db: AsyncSession, obj: T) -> T:
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj
```

**Impact**: ~15-25 LOC reduction, consistent patterns

### QW3: `utils/param_spec.py` - Parameter Normalization
**Current Problem**: Multiple implementations of parameter value extraction
```python
if isinstance(spec, dict) and "values" in spec:
    return spec["values"]
if isinstance(spec, list):
    return spec
return [spec]
```

**Solution**: Single source of truth
```python
def normalize_choice_values(spec: object) -> list[object]:
    if isinstance(spec, dict) and "values" in spec:
        return list(spec["values"])
    if isinstance(spec, list):
        return spec
    return [spec]
```

**Impact**: ~10-15 LOC reduction, prevents drift

### QW4: Replace `print()` with `logging`
**Current Problem**: Production-unfriendly stdout logging
```python
print(f"[API] Deleted task {task_id} from database")
```

**Solution**: Standard library logging
```python
logger = logging.getLogger(__name__)
logger.info("Deleted task %d from database", task_id)
```

**Impact**: ~10-30 LOC touched, improved production behavior

### QW5: Fix Docker Controller Volume Overwrite Bug
**Current Problem**: HF cache mount gets overwritten
```python
volumes[str(hf_cache_dir)] = {"bind": "/root/.cache/huggingface", "mode": "rw"}
# ... later ...
volumes = {str(model_path): {"bind": "/model", "mode": "ro"}}  # BUG: overwrites!
```

**Solution**: Preserve existing volumes
```python
volumes = {str(hf_cache_dir): {"bind": "/root/.cache/huggingface", "mode": "rw"}}
if use_local_model:
    volumes[str(model_path)] = {"bind": "/model", "mode": "ro"}
```

**Impact**: Correctness fix, minimal code change

---

## Implementation Roadmap

### PR0: Foundation (Safest)
**Goal**: Add helper modules without changing call sites

**New Files**:
- `src/utils/gpu_types.py`
- `src/utils/gpu_selection.py`
- `src/utils/parallelism.py`
- `src/utils/param_spec.py`
- `src/controllers/naming.py`
- `src/controllers/readiness.py`
- `src/controllers/cli_args.py`
- `src/utils/network.py`
- `src/utils/env.py`
- `src/web/routes/deps.py`
- `src/web/services/db_utils.py`

**Modified Files**: None
**Risk**: Minimal

### PR1: Web Layer DRY
**Goal**: Eliminate duplication in routes/services (avoid worker infra)

**Changes**:
- Update `tasks.py` + `experiments.py` to use `get_*_or_404` deps
- Replace `print()` with `logging` across `web/routes/*`
- Do NOT refactor `/start`, `/cancel`, `/restart` (to avoid develop conflicts)

**Modified Files**:
- `src/web/routes/tasks.py`
- `src/web/routes/experiments.py`
- `src/web/services/task_service.py`
- `src/web/services/experiment_service.py`

**Risk**: Low (avoids worker infrastructure)

### PR2: GPU Modules DRY
**Goal**: Unify GPU selection and type definitions

**Changes**:
- Update `gpu_monitor.py` to use memory-balance/selection helper
- Update `gpu_pool.py` to use shared selection logic
- Introduce `GPUInfo` aliasing via `gpu_types.py`

**Modified Files**:
- `src/utils/gpu_monitor.py`
- `src/utils/gpu_pool.py`
- `src/utils/gpu_discovery.py`

**Risk**: Low

### PR3: Parallelism Unification
**Goal**: Single source of truth for world-size calculation

**Changes**:
- Update `gpu_scheduler.py` to use `utils/parallelism.py`
- Update Docker/Local controllers to use same helper
- Use `param_spec.normalize_choice_values()`

**Modified Files**:
- `src/utils/gpu_scheduler.py`
- `src/controllers/docker_controller.py`
- `src/controllers/local_controller.py`

**Risk**: Low

### PR4: Controllers DRY
**Goal**: Share helpers for readiness, CLI args, env, ports, naming

**Changes**:
- Adopt shared helpers in Docker/Local/OME controllers
- Keep controller public method signatures unchanged

**Modified Files**:
- `src/controllers/docker_controller.py`
- `src/controllers/local_controller.py`
- `src/controllers/ome_controller.py`

**Risk**: Low-Medium

### PR5: Optimizer Modularization
**Goal**: Split god module into focused modules

**Changes**:
- Create `utils/slo.py`, `utils/scoring.py`, `utils/search_space.py`
- Keep `optimizer.py` as facade with re-exports
- Add unit tests for SLO penalty/scoring

**Modified Files**:
- `src/utils/optimizer.py` (becomes facade)

**New Files**:
- `src/utils/slo.py`
- `src/utils/scoring.py`
- `src/utils/search_space.py`
- `tests/utils/test_slo.py`
- `tests/utils/test_scoring.py`

**Risk**: Low-Medium (many imports)

### PR6: Schema Cleanup (Optional)
**Goal**: Explicit schema modules

**Changes**:
- Create `schemas/task.py`, `schemas/experiment.py`
- Re-export through `schemas/__init__.py`

**Modified Files**:
- Many (all files importing schemas)

**Risk**: Low (but high churn, best done after develop stabilizes)

---

## Conflict Avoidance Strategy

**Develop Branch Activity** (as of 2025-12-24):
- Redis Pub/Sub implementation
- Distributed worker registry
- Result listener service
- Enhanced worker panel in dashboard

**Areas to Avoid**:
- `src/web/routes/workers.py` (new file)
- `src/web/services/result_listener.py` (new file)
- `src/web/workers/pubsub.py` (new file)
- `src/web/workers/registry.py` (new file)
- Task start/cancel/restart mechanics that touch worker enqueuing

**Safe Areas**:
- Utility modules (`src/utils/*`)
- Controller internal implementations
- Route CRUD operations (except start/cancel/restart)
- Service layer DB operations
- Schema definitions

---

## Detailed Issue Catalog

### Controllers

#### Issue C1: Duplicated Name Sanitization
**Location**: `docker_controller.py`, `local_controller.py`, `ome_controller.py`
**Duplication**: 3 similar implementations
**Impact**: Maintenance burden, potential drift

#### Issue C2: Repeated HTTP Readiness Polling
**Location**: `docker_controller.py`, `local_controller.py`
**Duplication**: Nearly identical polling logic
**Impact**: ~40-60 LOC

#### Issue C3: CLI Parameter Building
**Location**: `docker_controller.py`, `local_controller.py`
**Duplication**: Bool flag handling, prefix addition
**Impact**: ~30-50 LOC

#### Issue C4: GPU World-Size Calculation
**Location**: `docker_controller.py`, `local_controller.py`
**Duplication**: TP/PP/DP/CP parsing
**Impact**: ~50-80 LOC, consistency issues

#### Issue C5: GPU Selection Implementation
**Location**: `docker_controller._select_gpus`, `local_controller._select_gpus`
**Duplication**: Nearly identical
**Impact**: ~40-60 LOC

#### Issue C6: Proxy + HF Token Environment Variables
**Location**: `docker_controller.py`, `local_controller.py`
**Duplication**: Env var setup
**Impact**: ~30-50 LOC

#### Issue C7: Port Allocation
**Location**: `docker_controller._find_available_port`, `local_controller._find_available_port`
**Duplication**: Identical socket-binding loop
**Impact**: ~15-25 LOC

#### Issue C8: Docker Volume Overwrite Bug
**Location**: `docker_controller.py:deploy_model()`
**Type**: Correctness issue
**Impact**: HF cache mount lost when local model used

### Utils

#### Issue U1: GPUInfo Name Collision
**Location**: `gpu_discovery.py`, `gpu_monitor.py`
**Impact**: Import confusion, reuse difficulty
**Severity**: High

#### Issue U2: Memory Balance Check Duplication
**Location**: `gpu_monitor.py`, `gpu_pool.py`, controllers
**Duplication**: 3+ implementations
**Impact**: ~60-100 LOC

#### Issue U3: GPU Allocation Logic Overlap
**Location**: `gpu_monitor.py`, `gpu_pool.py`
**Impact**: Unclear responsibilities
**Severity**: Medium

#### Issue U4: gpu_discovery.py Mixed Responsibilities
**Location**: `gpu_discovery.py`
**Issues**: Uses `print()`, broad exception handling, multiple responsibilities
**Impact**: Maintainability

#### Issue U5: Parameter Spec Normalization Duplication
**Location**: `gpu_scheduler.py`, `optimizer.py`, preset mergers
**Duplication**: Similar normalize-values logic
**Impact**: ~30-50 LOC

#### Issue U6: World-Size Calculation Duplication
**Location**: `gpu_scheduler.py`, controllers, parallel utils
**Impact**: Inconsistent behavior
**Severity**: High

#### Issue U7: optimizer.py God Module
**Location**: `optimizer.py` (1137 LOC)
**Issues**: Mixed concerns, hard to test
**Impact**: Maintainability, testability

#### Issue U8: SLO Penalty Logic Repetition
**Location**: `optimizer.py:calculate_slo_penalty()`
**Pattern**: Repeated for TTFT/TPOT/latency
**Impact**: ~50-80 LOC

#### Issue U9: Runtime Key Normalization Duplication
**Location**: `parallel_mapper.py`, `quantization_mapper.py`, others
**Duplication**: Same canonicalization logic
**Impact**: ~20-30 LOC

#### Issue U10: CLI-Arg Building Duplication
**Location**: Multiple mappers, controllers
**Impact**: ~40-60 LOC

### Web Layer

#### Issue W1: Get-or-404 Duplication
**Location**: `routes/tasks.py`, `routes/experiments.py`
**Duplication**: Repeated in every get/update/delete endpoint
**Impact**: ~40-70 LOC

#### Issue W2: Inconsistent Service Layer Usage
**Location**: `routes/tasks.py`
**Pattern**: Some routes use services, others do raw SQL
**Impact**: Architecture smell, rule drift

#### Issue W3: Name Uniqueness Validation Duplication
**Location**: `routes/tasks.py` (create, update)
**Duplication**: Identical validation logic
**Impact**: ~15-25 LOC

#### Issue W4: print() in API Routes
**Location**: `routes/tasks.py`, others
**Impact**: Production behavior

#### Issue W5: Service DB Boilerplate
**Location**: `services/task_service.py`, `services/experiment_service.py`
**Duplication**: Identical create/update patterns
**Impact**: ~20-40 LOC

#### Issue W6: Status String vs Enum Handling
**Location**: `services/task_service.py`, `services/experiment_service.py`
**Issue**: Accepts string, compares to Enum field
**Impact**: Fragility, type safety

#### Issue W7: Best Experiment Lookup in Wrong Service
**Location**: `services/experiment_service.py`
**Issue**: Task-related logic in experiment service
**Impact**: Architecture smell

#### Issue W8: Missing Transaction Boundaries
**Location**: `routes/tasks.py` (/restart)
**Issue**: Multi-step operations not in transaction
**Impact**: Potential inconsistent state

#### Issue W9: Schema Organization
**Location**: `schemas/__init__.py` (all schemas in one file or unclear structure)
**Impact**: Maintainability

---

## Testing Strategy

### Unit Tests to Add
- `tests/utils/test_gpu_selection.py`
- `tests/utils/test_parallelism.py`
- `tests/utils/test_param_spec.py`
- `tests/controllers/test_naming.py`
- `tests/controllers/test_cli_args.py`
- `tests/web/routes/test_deps.py`
- `tests/web/services/test_db_utils.py`

### Integration Tests to Update
- GPU allocation tests (after unification)
- Controller deployment tests (after shared helpers)
- Web API tests (after dependency injection)

---

## Success Metrics

### Code Quality
- [ ] No `GPUInfo` name collisions
- [ ] Single source of truth for world-size calculation
- [ ] All web routes use service layer consistently
- [ ] No `print()` statements in production code
- [ ] All multi-step operations have transaction boundaries

### DRY Compliance
- [ ] Memory balance check in one place
- [ ] HTTP readiness polling shared
- [ ] CLI arg building shared
- [ ] Parameter normalization shared
- [ ] DB commit/refresh patterns shared

### Architecture
- [ ] `optimizer.py` under 400 LOC (after split)
- [ ] Clear separation of concerns in services
- [ ] Consistent use of dependency injection
- [ ] Proper error handling (no broad exceptions)

### Testing
- [ ] Unit test coverage >80% for new helpers
- [ ] All existing tests pass
- [ ] No regressions in functionality

---

## Timeline

**Week 1**: Quick wins + PR0 (foundation)
**Week 2**: PR1 (web) + PR2 (GPU)
**Week 3**: PR3 (parallelism) + PR4 (controllers)
**Week 4**: PR5 (optimizer) + testing
**Week 5**: PR6 (schemas, optional) + documentation

---

## References

- Code review session with GPT-5.2: 2025-12-24
- Develop branch analysis: commits up to baed532
- Worker infrastructure files to avoid: `workers.py`, `pubsub.py`, `registry.py`, `result_listener.py`

---

## Appendix: Code Examples

### Example: get_task_or_404 Usage

**Before**:
```python
@router.get("/{task_id}")
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse.model_validate(task)
```

**After**:
```python
@router.get("/{task_id}")
async def get_task(task: Task = Depends(get_task_or_404)):
    return TaskResponse.model_validate(task)
```

### Example: Memory Balance Check

**Before** (3 implementations):
```python
# gpu_monitor.py
if len(memory_free) > 1:
    min_mem, max_mem = min(memory_free), max(memory_free)
    if min_mem < max_mem * 0.8:
        logger.warning(f"Memory imbalance: {min_mem} vs {max_mem}")
        # ... continue anyway

# gpu_pool.py (similar but different messaging)
if max_mem > 0 and min_mem / max_mem < 0.8:
    raise Exception("Memory imbalance")

# controllers (inlined, no check)
```

**After**:
```python
# All use:
from utils.gpu_selection import validate_memory_balance

valid, msg = validate_memory_balance(memory_free)
if not valid:
    logger.warning(f"GPU memory imbalance: {msg}")
```

### Example: World-Size Calculation

**Before**:
```python
# docker_controller.py
tp = parameters.get("tp-size", parameters.get("tp_size", 1))
pp = parameters.get("pp-size", parameters.get("pp_size", 1))
dp = parameters.get("dp-size", parameters.get("dp_size", 1))
cp = parameters.get("cp-size", parameters.get("cp_size", 1))
dcp = parameters.get("dcp-size", parameters.get("dcp_size", 1))
num_gpus = tp * pp * max(dp, dcp, cp)

# local_controller.py
tp_size = parameters.get("tp-size", parameters.get("tp_size", 1))
pp_size = parameters.get("pp-size", parameters.get("pp_size", 1))
dp_size = parameters.get("dp-size", parameters.get("dp_size", 1))
num_gpus = tp_size * pp_size * dp_size  # Different!
```

**After**:
```python
# Both use:
from utils.parallelism import compute_world_size

num_gpus = compute_world_size(parameters, runtime=base_runtime)
```

---

## Status Tracking

| PR | Status | Started | Completed | Notes |
|----|--------|---------|-----------|-------|
| PR0 | Not Started | - | - | Foundation helpers |
| QW1-QW5 | In Progress | 2025-12-24 | - | Quick wins |
| PR1 | Not Started | - | - | Web layer |
| PR2 | Not Started | - | - | GPU modules |
| PR3 | Not Started | - | - | Parallelism |
| PR4 | Not Started | - | - | Controllers |
| PR5 | Not Started | - | - | Optimizer split |
| PR6 | Not Started | - | - | Schema cleanup |
