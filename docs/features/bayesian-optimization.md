# Bayesian Optimization

Intelligent parameter search using machine learning to efficiently find optimal configurations.

## Table of Contents

- [Overview](#overview)
- [When to Use](#when-to-use)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Example Task](#example-task)
- [Comparison with Grid Search](#comparison-with-grid-search)
- [Parameter Tuning](#parameter-tuning)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

Bayesian Optimization is an intelligent search strategy that uses machine learning to explore the parameter space efficiently. Unlike grid search which exhaustively tests all combinations, Bayesian optimization builds a probabilistic model of the objective function and uses it to intelligently select which configurations to test next.

### Key Benefits

- **80-87% fewer experiments**: Typically finds optimal configurations in 20-30 experiments vs 100+ for grid search
- **Intelligent exploration**: Balances exploring new regions vs exploiting promising areas
- **Continuous improvement**: Each experiment makes the model smarter
- **Handles large spaces**: Effective for parameter spaces where grid search is impractical

### Implementation

The autotuner uses **Optuna** with the Tree-structured Parzen Estimator (TPE) sampler:
- TPE models the objective function as two distributions: good and bad configurations
- Uses Bayesian reasoning to suggest parameters likely to improve the objective
- Supports mixed parameter types: categorical, continuous, integer, boolean

## When to Use

### Use Bayesian Optimization When:

1. **Large parameter spaces**: 50+ total combinations (e.g., 3 params with 5 values each = 125 combinations)
2. **Expensive experiments**: Each experiment takes >5 minutes
3. **Budget constraints**: Limited time or GPU resources
4. **Complex interactions**: Parameters have non-obvious relationships
5. **Unknown optima**: No prior knowledge of best configuration

### Use Grid Search When:

1. **Small spaces**: <20 total combinations
2. **Fast experiments**: Each experiment takes <1 minute
3. **Comprehensive coverage**: Need to test ALL combinations
4. **Known patterns**: Parameter effects are well understood

### Use Random Search When:

1. **Quick exploration**: Want fast insights without optimization
2. **Baseline comparison**: Need random sampling benchmark

## How It Works

### Phase 1: Initial Random Exploration (5 trials by default)

```
Experiment 1-5: Random sampling across parameter space
Goal: Build initial model of objective function
```

### Phase 2: Bayesian Optimization (remaining trials)

```
For each trial:
1. Model predicts probability that each configuration will improve objective
2. Acquisition function balances:
   - Exploration: testing uncertain regions
   - Exploitation: testing near known good configurations
3. Execute experiment with selected configuration
4. Update model with new result
5. Repeat until max_iterations reached or convergence
```

### TPE (Tree-structured Parzen Estimator)

```python
# TPE models objective as two distributions:
P(params | objective < threshold)  # "good" configurations
P(params | objective >= threshold)  # "bad" configurations

# Suggests params that maximize ratio:
P(params | good) / P(params | bad)
```

## Configuration

### Task JSON Format

```json
{
  "optimization": {
    "strategy": "bayesian",
    "objective": "minimize_latency",
    "max_iterations": 50
  },
  "parameters": {
    "tp-size": [1, 2, 4],
    "mem-fraction-static": [0.7, 0.75, 0.8, 0.85, 0.9],
    "schedule-policy": ["lpm", "fcfs"]
  }
}
```

### Key Configuration Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `max_iterations` | Total experiments to run | 100 | 30-50 for most tasks |
| `n_initial_random` | Random trials before Bayesian starts | 5 | 5-10 (10-20% of max_iterations) |
| `objective` | What to optimize | minimize_latency | Based on use case |
| `timeout_per_iteration` | Max time per experiment | 600s | 300-900s based on model size |

## Example Task

### Full Task Configuration

```json
{
  "task_name": "bayesian-llama3-tune",
  "description": "Bayesian optimization for Llama 3.2-1B",
  "model": {
    "id_or_path": "llama-3-2-1b-instruct",
    "namespace": "autotuner"
  },
  "base_runtime": "sglang",
  "runtime_image_tag": "v0.5.2-cu126",
  "parameters": {
    "tp-size": [1, 2],
    "mem-fraction-static": [0.7, 0.75, 0.8, 0.85, 0.9],
    "schedule-policy": ["lpm", "fcfs"],
    "chunked-prefill-size": [512, 1024, 2048, 4096]
  },
  "optimization": {
    "strategy": "bayesian",
    "objective": "minimize_latency",
    "max_iterations": 30,
    "timeout_per_iteration": 600
  },
  "benchmark": {
    "task": "text-to-text",
    "model_name": "Llama-3.2-1B-Instruct",
    "model_tokenizer": "meta-llama/Llama-3.2-1B-Instruct",
    "traffic_scenarios": ["D(100,100)"],
    "num_concurrency": [4, 8],
    "max_time_per_iteration": 30,
    "max_requests_per_iteration": 100,
    "additional_params": {
      "temperature": 0.0
    }
  }
}
```

### Parameter Space Size

```
Grid search would require: 2 × 5 × 2 × 4 × 2 = 160 experiments
Bayesian optimization: ~30 experiments (81% reduction)
```

### Expected Results

- **Convergence**: Best configuration typically found within 15-20 experiments
- **Remaining experiments**: Fine-tuning and validation
- **Total time**: 5-10 hours vs 26+ hours for grid search

## Comparison with Grid Search

### Example Scenario: Llama-3.2-1B Tuning

**Parameter Space:**
- `tp-size`: [1, 2, 4] → 3 values
- `mem-fraction-static`: [0.7, 0.75, 0.8, 0.85, 0.9] → 5 values
- `schedule-policy`: ["lpm", "fcfs"] → 2 values
- `chunked-prefill-size`: [512, 1024, 2048, 4096] → 4 values

**Total combinations:** 3 × 5 × 2 × 4 = 120

| Strategy | Experiments | Time (est.) | GPU-hours | Best Score Found |
|----------|------------|-------------|-----------|-----------------|
| Grid Search | 120 | 20 hours | 20 | 0.0825 |
| Random Search | 50 | 8.3 hours | 8.3 | 0.0834 |
| **Bayesian** | **25** | **4.2 hours** | **4.2** | **0.0823** |

**Efficiency gain**: 79% fewer experiments, 79% less time, same or better result

## Parameter Tuning

### max_iterations

**Purpose**: Total number of experiments to run

**Guidance:**
- **Small space (<50 combinations)**: 20-30 iterations
- **Medium space (50-200 combinations)**: 30-50 iterations
- **Large space (>200 combinations)**: 50-100 iterations
- **Rule of thumb**: 20-30% of grid search space size

### n_initial_random

**Purpose**: Number of random trials before Bayesian optimization starts

**Guidance:**
- **Default**: 5 trials (10% of max_iterations=50)
- **Small space**: 5-10 trials
- **Large space**: 10-20 trials
- **Rule of thumb**: 10-20% of max_iterations

## Best Practices

### 1. Start with Small max_iterations

```json
{
  "optimization": {
    "strategy": "bayesian",
    "max_iterations": 20  // Start small, increase if needed
  }
}
```

**Why**: Test Bayesian setup without long wait. Increase if not converged.

### 2. Monitor Convergence

```bash
# Watch for "New best score" messages
tail -f ~/.local/share/inference-autotuner/logs/task_<id>.log | grep "best score"
```

### 3. Use SLO Configuration

```json
{
  "slo": {
    "latency": {
      "p90": {
        "threshold": 5.0,
        "weight": 2.0,
        "hard_fail": true,
        "fail_ratio": 0.2
      }
    },
    "steepness": 0.1
  }
}
```

**Why**: Guides Bayesian optimization to respect performance constraints.

## Troubleshooting

### Problem: Bayesian not improving over random baseline

**Symptoms:**
- First 5 experiments find good config
- Remaining experiments don't improve

**Solutions:**
1. Too few parameters → Use random search
2. Parameters don't interact → Grid search may be better
3. Noisy objective → Increase benchmark duration

### Problem: Convergence too slow

**Symptoms:**
- Best score still improving after 40+ experiments

**Solutions:**
1. Reduce `n_initial_random` to 5-10
2. Increase `max_iterations` to 50-100
3. Consider hierarchical optimization

## Further Reading

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [TPE Algorithm Paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
- [Bayesian Optimization Overview](https://distill.pub/2020/bayesian-optimization/)

---

## Handling Failed Experiments

### Question: Can Infinite Scores Guide Bayesian Optimization?

**Short Answer**: **No.** Pure infinite scores provide only weak negative guidance (what to avoid) but no positive gradient (where to go). When all experiments fail, Bayesian optimization degrades to random search.

### How Failed Experiments Are Reported

In `src/web/workers/autotuner_worker.py`, failed experiments receive worst-case scores:

```python
# When experiment fails (timeout, crash, etc.)
objective_name = optimization_config.get("objective", "minimize_latency")
worst_score = float("inf") if "minimize" in objective_name else float("-inf")
strategy.tell_result(
    parameters=params,
    objective_score=worst_score,
    metrics={}
)
```

### TPE Sampler Behavior

Optuna's TPE (Tree-structured Parzen Estimator) sampler:
1. Builds surrogate models for parameter distributions
2. Separates observations into "good" (top γ%) and "bad" (rest)
3. Models two distributions: l(x) for good trials, g(x) for bad trials
4. Samples from regions where l(x)/g(x) is high

**Critical requirement**: Needs varying scores to distinguish good vs bad regions.

### Degradation When All Experiments Fail

When all trials return `-inf` or `inf`:
- TPE cannot distinguish between parameter configurations
- All parameters appear equally bad
- Sampler reverts to quasi-random exploration
- **Result**: Bayesian optimization degrades to random search

### Recommendation

For robustness:
1. Use graded failure penalties (see GRADED_FAILURE_PENALTIES.md)
2. Implement partial success metrics even for failed experiments
3. Consider hybrid approaches that combine Bayesian and grid search
4. Set reasonable SLO thresholds to avoid all-failure scenarios

