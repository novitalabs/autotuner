# SLO-Aware Objective Scoring

The inference-autotuner now supports sophisticated Service Level Objective (SLO) aware scoring with **exponential penalties** for violations and **tiered enforcement** (soft penalties vs hard failures).

## Overview

The SLO-aware scoring algorithm enhances experiment evaluation by:

1. **Exponential Penalty Curves**: Creates steep score increases near SLO boundaries
2. **Tiered Enforcement**: Distinguishes between minor violations (penalty) and severe violations (hard fail)
3. **Multi-Metric Support**: Monitors P50/P90/P99 latency, TTFT (Time to First Token), and TPOT (Time Per Output Token)
4. **Configurable Per-Task**: Each task defines its own SLO thresholds and weights

## Mathematical Formula

### Base Scoring Formula

```
final_score = base_objective_score × (1 + total_penalty)
```

Where `total_penalty` is the sum of all per-metric penalties.

### Per-Metric Penalty Calculation

For each SLO metric that exceeds its threshold:

```python
violation_ratio = (actual_value - threshold) / threshold  # Normalized percentage
penalty = weight × exp(violation_ratio / steepness)
```

**Key Parameters:**
- `weight`: Penalty multiplier (higher weights = more important metrics)
- `steepness`: Controls curve slope (lower = steeper penalties, default: 0.1)

### Tiered Enforcement

- **Minor Violations** (< fail_ratio): Exponential penalty applied to score
- **Severe Violations** (≥ fail_ratio): Experiment marked as `FAILED` with score = ∞

## Task Configuration

Add an optional `slo` section to your task JSON. **All fields within the SLO configuration are optional** - you can specify only the metrics you care about.

### Full Example (All Options)

```json
{
  "task_name": "my-slo-aware-task",
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency"
  },
  "slo": {
    "latency": {
      "p50": {
        "threshold": 2.0,
        "weight": 1.0,
        "hard_fail": false
      },
      "p90": {
        "threshold": 5.0,
        "weight": 2.0,
        "hard_fail": true,
        "fail_ratio": 0.2
      },
      "p99": {
        "threshold": 10.0,
        "weight": 3.0,
        "hard_fail": true,
        "fail_ratio": 0.5
      }
    },
    "ttft": {
      "threshold": 1.0,
      "weight": 2.0,
      "hard_fail": false
    },
    "tpot": {
      "threshold": 0.05,
      "weight": 2.0,
      "hard_fail": false
    },
    "steepness": 0.1
  }
}
```

### Minimal Example (Only Required Fields)

You can specify just the metrics you want to enforce. Here's a minimal configuration with only P99 latency:

```json
{
  "task_name": "my-minimal-slo-task",
  "optimization": {
    "strategy": "grid_search",
    "objective": "minimize_latency"
  },
  "slo": {
    "latency": {
      "p99": {
        "threshold": 10.0
      }
    }
  }
}
```

### Configuration Parameters

**Important:** All SLO configuration fields are optional. You can:
- Omit entire metric sections (e.g., no P50 if you only care about P99)
- Omit individual metrics (e.g., only configure P90 and P99)
- Omit optional parameters within metrics (weight, hard_fail, fail_ratio)

#### Per-Metric SLO

- **`threshold`** (required if metric specified): Maximum allowed value (in seconds)
- **`weight`** (optional, default: 1.0): Penalty weight for this metric
- **`hard_fail`** (optional, default: false): Enable hard failure enforcement
- **`fail_ratio`** (optional, default: 0.5): Violation threshold for hard fail (e.g., 0.2 = 20% over)

#### Global SLO

- **`steepness`** (optional, default: 0.1): Exponential curve steepness parameter

## Example Scenarios

### Scenario 1: No SLO Violations

**Metrics:** P90 = 4.0s (threshold: 5.0s)

**Result:**
- Penalty multiplier: 1.0
- Final score: base_score × 1.0 (no penalty)

### Scenario 2: Minor Violation (10% over)

**Metrics:** P90 = 5.5s (threshold: 5.0s, weight: 2.0, steepness: 0.1)

**Calculation:**
```
violation_ratio = (5.5 - 5.0) / 5.0 = 0.10 (10%)
penalty = 2.0 × exp(0.10 / 0.1) = 2.0 × exp(1.0) ≈ 5.44
penalty_multiplier = 1 + 5.44 = 6.44
```

**Result:**
- Base score: 3.0s
- Final score: 3.0 × 6.44 = **19.3s** (worse score)
- Status: `SUCCESS` but penalized

### Scenario 3: Severe Violation (Hard Fail)

**Metrics:** P90 = 6.5s (threshold: 5.0s, fail_ratio: 0.2)

**Calculation:**
```
violation_ratio = (6.5 - 5.0) / 5.0 = 0.30 (30%)
30% > 20% fail_ratio → HARD FAILURE
```

**Result:**
- Final score: **∞** (infinity)
- Status: `FAILED`
- Reason: "Hard SLO violation"

### Scenario 4: Multiple Violations (Cumulative Penalties)

**Metrics:**
- P50 = 2.3s (threshold: 2.0s, weight: 1.0) → +4.48 penalty
- P90 = 5.5s (threshold: 5.0s, weight: 2.0) → +5.44 penalty
- P99 = 11.0s (threshold: 10.0s, weight: 3.0) → +8.15 penalty
- TTFT = 1.2s (threshold: 1.0s, weight: 2.0) → +14.78 penalty

**Total Penalty:** 32.85

**Result:**
- Base score: 2.5s
- Final score: 2.5 × 33.85 = **84.6s**
- Score increase: **3285%** 🔥

## Steepness Parameter Impact

The `steepness` parameter controls how aggressively penalties grow:

| Steepness | 20% Violation Penalty | Behavior |
|-----------|----------------------|----------|
| 0.05      | 110.2x              | Very steep (aggressive) |
| **0.1**   | **15.8x**           | **Recommended default** |
| 0.2       | 6.4x                | Gentler curve |

**Lower steepness = Steeper penalties near boundaries** ⚠️

## Frontend Features

### Task Creation UI

Navigate to **Create New Task** → Enable **SLO Configuration** toggle:

- Configure P50/P90/P99 latency thresholds
- Configure TTFT (Time to First Token) thresholds
- Configure TPOT (Time Per Output Token) thresholds
- Set penalty weights per metric
- Enable hard fail enforcement with fail_ratio
- Adjust steepness parameter

### Experiments View

Experiments violating hard SLO constraints display:
- Red "SLO" badge next to status
- `slo_violation: true` flag in experiment data
- Status automatically marked as `FAILED`

## Backend Implementation

### Optimizer Module (`src/utils/optimizer.py`)

**New Functions:**

1. **`calculate_slo_penalty(metrics, slo_config)`**
   - Returns: `(penalty_multiplier, is_hard_failure, violation_details)`
   - Implements exponential penalty formula
   - Checks hard failure conditions

2. **`calculate_objective_score(results, objective, slo_config)`**
   - Enhanced to accept optional `slo_config`
   - Applies SLO penalties to base score
   - Returns `inf` for hard failures

### Orchestrator (`src/orchestrator.py`)

- Passes `task.get("slo")` to scoring function
- Marks experiments as `FAILED` when `score == inf`
- Adds `slo_violation: true` flag to experiment results

## Testing

Run the test suite to verify algorithm behavior:

```bash
python test_slo_algorithm.py
```

**Test Coverage:**
- ✓ No violations (baseline)
- ✓ Minor violations (soft penalties)
- ✓ Severe violations (exponential growth)
- ✓ Hard failure boundary conditions
- ✓ Multiple cumulative violations
- ✓ Steepness parameter effects
- ✓ TPOT SLO enforcement (test_tpot_slo.py)
- ✓ Optional field handling (test_slo_optional_fields.py)

## Example Task

See `examples/docker_task_with_slo.json` for a complete example with SLO configuration.

## Use Cases

### 1. **Production-Like Constraints**

Ensure tuned configurations meet real-world SLOs:
```json
"slo": {
  "latency": {
    "p99": {"threshold": 10.0, "hard_fail": true, "fail_ratio": 0.2}
  }
}
```

### 2. **Multi-Objective Optimization**

Balance latency, TTFT, and TPOT:
```json
"slo": {
  "latency": {
    "p90": {"threshold": 5.0, "weight": 1.0}
  },
  "ttft": {"threshold": 1.0, "weight": 3.0},  // Higher weight = more important
  "tpot": {"threshold": 0.05, "weight": 2.0}
}
```

### 3. **Soft Boundaries for Exploration**

Penalize but don't reject near-boundary configurations:
```json
"slo": {
  "latency": {
    "p90": {"threshold": 5.0, "weight": 2.0, "hard_fail": false}
  },
  "steepness": 0.15  // Gentler curve for exploration
}
```

## Design Rationale

### Why Exponential Penalties?

Linear penalties don't adequately penalize configurations near SLO boundaries:

| Violation | Linear (2x weight) | Exponential (weight=2, s=0.1) |
|-----------|-------------------|-------------------------------|
| 5% over   | 1.10x            | 2.30x                         |
| 10% over  | 1.20x            | 3.72x                         |
| 20% over  | 1.40x            | 15.78x                        |
| 50% over  | 2.00x            | 297.4x                        |

Exponential curves create **steep gradients** that guide optimization away from SLO boundaries.

### Why Tiered Enforcement?

- **Soft Penalties**: Allow exploration of configurations slightly over SLO
- **Hard Failures**: Reject configurations that egregiously violate critical SLOs

This mirrors real-world SLO design where some violations are tolerable (warn) and others are not (page).

## Backward Compatibility

Tasks without `slo` configuration continue to work unchanged. SLO scoring is fully optional and backward compatible.

## Future Enhancements

- Support for throughput SLOs (minimum thresholds)
- Custom penalty functions (polynomial, piecewise)
- SLO violation budgets (allow N% of experiments to violate)
- SLO-aware Bayesian optimization (constrained BO)

## References

- **Exponential Penalty Functions**: Common in constrained optimization
- **SLO Design**: Google SRE Book - Chapter 4 (Service Level Objectives)
- **Tiered Enforcement**: Inspired by alerting thresholds (warn/critical)

---

## Graded Failure Penalties for Bayesian Optimization

### Problem

When all experiments fail with infinite scores (`-inf` or `+inf`), Bayesian optimization cannot distinguish between parameter configurations and degrades to random search.

### Solution: Time-Based Failure Penalties

Failed experiments receive **graded penalties** based on failure timing - earlier failures get worse penalties.

### Penalty Calculation

Located in `src/web/workers/autotuner_worker.py`:

```python
def calculate_failure_penalty(started_at, failed_at, timeout_seconds, 
                              experiment_status, error_message, objective_name):
    elapsed = (failed_at - started_at).total_seconds()
    completion_pct = min(elapsed / timeout_seconds, 1.0)

    # Base penalty by completion percentage
    if completion_pct < 0.20:
        base_penalty = -1000  # Very early (deployment, immediate crash)
    elif completion_pct < 0.60:
        base_penalty = -500   # Mid-stage (benchmark started but failed)
    elif completion_pct < 0.95:
        base_penalty = -200   # Late-stage (benchmark mostly done)
    else:
        base_penalty = -100   # Timeout (full duration)

    # Modifiers based on error type
    if "oom" or "memory" in error: base_penalty *= 1.5  # Resource failures
    if "deploy" in error: base_penalty *= 1.2            # Deployment failures
    if "connection" in error: base_penalty *= 0.8        # Transient issues

    # Invert for minimize objectives
    return -base_penalty if "minimize" in objective_name else base_penalty
```

### Benefits

1. **Provides gradient**: Bayesian optimizer can distinguish parameter quality even when all fail
2. **Prioritizes stability**: Configs that run longer are preferred
3. **Contextual penalties**: Error types affect severity
4. **Enables learning**: Optimizer learns to avoid problematic parameter regions

### Example Scenarios

| Failure Timing | Completion % | Base Penalty | Scenario |
|---------------|-------------|--------------|----------|
| 10 seconds | 2% | -1000 | Deployment failure, config clearly broken |
| 200 seconds | 50% | -500 | Benchmark started but OOM |
| 450 seconds | 90% | -200 | Almost complete, near-miss |
| 500 seconds | 100% | -100 | Timeout, config might work with more time |

This allows the optimizer to progressively learn which parameters cause early vs late failures.

