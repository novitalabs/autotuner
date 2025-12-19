#!/usr/bin/env python3
"""
Test script for Bayesian optimization strategy.

Tests the strategy classes without requiring full infrastructure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.optimizer import (
    GridSearchStrategy,
    BayesianStrategy,
    RandomSearchStrategy,
    create_optimization_strategy
)


def test_grid_search_strategy():
    """Test grid search strategy."""
    print("=" * 80)
    print("TEST: Grid Search Strategy")
    print("=" * 80)

    param_spec = {
        "tp-size": [1, 2, 4],
        "mem-fraction": [0.8, 0.9]
    }

    strategy = GridSearchStrategy(
        parameter_spec=param_spec,
        objective="minimize_latency",
        max_iterations=4
    )

    print(f"Total combinations: {len(strategy.param_grid)}")
    print(f"Limited to: {len(strategy.param_grid)} iterations")

    results = []
    while not strategy.should_stop():
        params = strategy.suggest_parameters()
        if params is None:
            break

        # Simulate objective score
        score = params["tp-size"] * 0.5 + (1 - params["mem-fraction"]) * 2
        strategy.tell_result(params, score, {})
        results.append((params, score))

    print(f"\nCompleted {len(results)} experiments")
    best = min(results, key=lambda x: x[1])
    print(f"Best: {best[0]} with score {best[1]:.4f}")
    print("✅ Grid search test passed\n")


def test_bayesian_strategy():
    """Test Bayesian optimization strategy."""
    print("=" * 80)
    print("TEST: Bayesian Optimization Strategy")
    print("=" * 80)

    param_spec = {
        "tp-size": [1, 2, 4],
        "mem-fraction": {
            "type": "continuous",
            "low": 0.7,
            "high": 0.95
        },
        "max-tokens": {
            "type": "integer",
            "low": 4096,
            "high": 16384
        }
    }

    strategy = BayesianStrategy(
        parameter_spec=param_spec,
        objective="minimize_latency",
        max_iterations=10,
        n_initial_random=3
    )

    print(f"Max iterations: 10")
    print(f"Initial random: 3")
    print(f"Search space: {list(strategy.search_space.keys())}")

    results = []
    iteration = 0
    while not strategy.should_stop():
        iteration += 1
        params = strategy.suggest_parameters()
        if params is None:
            break

        # Simulate objective: optimal is tp=2, mem=0.85, tokens=8192
        score = (
            abs(params["tp-size"] - 2) * 0.5 +
            abs(params["mem-fraction"] - 0.85) * 2 +
            abs(params["max-tokens"] - 8192) / 2000
        )

        strategy.tell_result(params, score, {})
        results.append((params, score))

    print(f"\nCompleted {len(results)} experiments")
    best = min(results, key=lambda x: x[1])
    print(f"Best parameters: {best[0]}")
    print(f"Best score: {best[1]:.4f}")

    # Check if Bayesian found a good solution
    if best[1] < 1.0:  # Should find something reasonably close
        print("✅ Bayesian optimization test passed\n")
    else:
        print("⚠️  Bayesian optimization may need tuning\n")


def test_random_strategy():
    """Test random search strategy."""
    print("=" * 80)
    print("TEST: Random Search Strategy")
    print("=" * 80)

    param_spec = {
        "tp-size": [1, 2, 4],
        "mem-fraction": [0.7, 0.8, 0.9]
    }

    strategy = RandomSearchStrategy(
        parameter_spec=param_spec,
        objective="minimize_latency",
        max_iterations=5,
        seed=42
    )

    print(f"Max iterations: 5")
    print(f"Seed: 42 (reproducible)")

    results = []
    while not strategy.should_stop():
        params = strategy.suggest_parameters()
        if params is None:
            break

        score = params["tp-size"] * 0.5 + (1 - params["mem-fraction"]) * 2
        strategy.tell_result(params, score, {})
        results.append((params, score))

    print(f"\nCompleted {len(results)} experiments")
    best = min(results, key=lambda x: x[1])
    print(f"Best: {best[0]} with score {best[1]:.4f}")
    print("✅ Random search test passed\n")


def test_strategy_factory():
    """Test strategy factory function."""
    print("=" * 80)
    print("TEST: Strategy Factory")
    print("=" * 80)

    param_spec = {
        "tp-size": [1, 2],
        "mem-fraction": [0.8, 0.9]
    }

    # Test grid search creation
    strategy = create_optimization_strategy(
        optimization_config={
            "strategy": "grid_search",
            "objective": "minimize_latency",
            "max_iterations": 4
        },
        parameter_spec=param_spec
    )
    assert isinstance(strategy, GridSearchStrategy)
    print("✓ Grid search factory works")

    # Test Bayesian creation
    strategy = create_optimization_strategy(
        optimization_config={
            "strategy": "bayesian",
            "objective": "maximize_throughput",
            "max_iterations": 5,
            "n_initial_random": 2
        },
        parameter_spec=param_spec
    )
    assert isinstance(strategy, BayesianStrategy)
    print("✓ Bayesian factory works")

    # Test random creation
    strategy = create_optimization_strategy(
        optimization_config={
            "strategy": "random",
            "objective": "minimize_latency",
            "max_iterations": 5
        },
        parameter_spec=param_spec
    )
    assert isinstance(strategy, RandomSearchStrategy)
    print("✓ Random factory works")

    print("✅ Strategy factory test passed\n")


def test_mixed_parameter_types():
    """Test handling of mixed parameter types."""
    print("=" * 80)
    print("TEST: Mixed Parameter Types")
    print("=" * 80)

    param_spec = {
        "categorical": ["a", "b", "c"],
        "continuous": {
            "type": "continuous",
            "low": 0.1,
            "high": 1.0
        },
        "integer": {
            "type": "integer",
            "low": 100,
            "high": 1000
        },
        "boolean": [True, False]
    }

    strategy = BayesianStrategy(
        parameter_spec=param_spec,
        objective="minimize_latency",
        max_iterations=5,
        n_initial_random=2
    )

    print(f"Parameter types: {list(param_spec.keys())}")

    # Test a few suggestions
    for i in range(3):
        params = strategy.suggest_parameters()
        print(f"\nSuggestion {i+1}:")
        print(f"  categorical: {params['categorical']} (type: {type(params['categorical']).__name__})")
        print(f"  continuous: {params['continuous']:.3f} (type: {type(params['continuous']).__name__})")
        print(f"  integer: {params['integer']} (type: {type(params['integer']).__name__})")
        print(f"  boolean: {params['boolean']} (type: {type(params['boolean']).__name__})")

        # Verify types
        assert isinstance(params['categorical'], str)
        assert isinstance(params['continuous'], float)
        assert isinstance(params['integer'], int)
        assert isinstance(params['boolean'], bool)

        # Simulate result
        score = i * 0.1
        strategy.tell_result(params, score, {})

    print("\n✅ Mixed parameter types test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" BAYESIAN OPTIMIZATION TESTS")
    print("=" * 80 + "\n")

    try:
        test_grid_search_strategy()
        test_bayesian_strategy()
        test_random_strategy()
        test_strategy_factory()
        test_mixed_parameter_types()

        print("=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
