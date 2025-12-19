#!/usr/bin/env python3
"""
Test script for SLO-aware objective scoring algorithm
Demonstrates exponential penalty behavior and tiered enforcement
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.optimizer import calculate_slo_penalty, calculate_objective_score


def test_no_slo_violations():
    """Test case: All metrics within SLO bounds"""
    print("\n" + "="*80)
    print("TEST 1: No SLO Violations (All metrics within bounds)")
    print("="*80)

    metrics = {
        "p50_e2e_latency": 1.5,  # Below 2.0 threshold
        "p90_e2e_latency": 4.0,  # Below 5.0 threshold
        "p99_e2e_latency": 8.0,  # Below 10.0 threshold
        "mean_ttft": 0.8,        # Below 1.0 threshold
        "mean_e2e_latency": 1.5
    }

    slo_config = {
        "latency": {
            "p50": {"threshold": 2.0, "weight": 1.0, "hard_fail": False},
            "p90": {"threshold": 5.0, "weight": 2.0, "hard_fail": True, "fail_ratio": 0.2},
            "p99": {"threshold": 10.0, "weight": 3.0, "hard_fail": True, "fail_ratio": 0.5}
        },
        "ttft": {"threshold": 1.0, "weight": 2.0, "hard_fail": False},
        "steepness": 0.1
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Penalty Multiplier: {penalty_multiplier:.4f}")
    print(f"Hard Failure: {is_hard_failure}")
    print(f"Violations: {violations}")

    score = calculate_objective_score(metrics, "minimize_latency", slo_config)
    print(f"Final Score: {score:.4f}")

    assert penalty_multiplier == 1.0, "Should have no penalty"
    assert not is_hard_failure, "Should not be hard failure"
    assert len(violations) == 0, "Should have no violations"
    print("✓ PASSED")


def test_minor_violation_soft_penalty():
    """Test case: Minor P50 violation (10% over) - soft penalty only"""
    print("\n" + "="*80)
    print("TEST 2: Minor Violation (P50 10% over threshold)")
    print("="*80)

    metrics = {
        "p50_e2e_latency": 2.2,  # 10% over 2.0 threshold
        "p90_e2e_latency": 4.0,
        "p99_e2e_latency": 8.0,
        "mean_ttft": 0.8,
        "mean_e2e_latency": 2.2
    }

    slo_config = {
        "latency": {
            "p50": {"threshold": 2.0, "weight": 1.0, "hard_fail": False}
        },
        "steepness": 0.1
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Metrics: P50 = {metrics['p50_e2e_latency']}s (threshold: 2.0s)")
    print(f"Violation Ratio: {(2.2-2.0)/2.0:.2%}")
    print(f"Penalty Multiplier: {penalty_multiplier:.4f}x")
    print(f"Hard Failure: {is_hard_failure}")
    print(f"Violation Details: {violations}")

    base_score = metrics['mean_e2e_latency']
    final_score = base_score * penalty_multiplier
    print(f"Base Score: {base_score:.4f}")
    print(f"Final Score: {final_score:.4f}")
    print(f"Score Increase: {((final_score/base_score - 1) * 100):.2f}%")

    assert penalty_multiplier > 1.0, "Should have penalty"
    assert not is_hard_failure, "Should not be hard failure"
    assert violations["p50"]["severity"] == "MINOR", "Should be MINOR severity"
    print("✓ PASSED")


def test_severe_violation_exponential_penalty():
    """Test case: P90 violation (25% over) - steep exponential penalty"""
    print("\n" + "="*80)
    print("TEST 3: Severe Violation (P90 25% over threshold)")
    print("="*80)

    metrics = {
        "p50_e2e_latency": 2.0,
        "p90_e2e_latency": 6.25,  # 25% over 5.0 threshold
        "p99_e2e_latency": 8.0,
        "mean_ttft": 0.8,
        "mean_e2e_latency": 2.5
    }

    slo_config = {
        "latency": {
            "p90": {"threshold": 5.0, "weight": 2.0, "hard_fail": True, "fail_ratio": 0.3}
        },
        "steepness": 0.1
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Metrics: P90 = {metrics['p90_e2e_latency']}s (threshold: 5.0s)")
    print(f"Violation Ratio: {(6.25-5.0)/5.0:.2%}")
    print(f"Penalty Multiplier: {penalty_multiplier:.4f}x")
    print(f"Hard Failure: {is_hard_failure}")
    print(f"Violation Details: {violations}")

    # With fail_ratio=0.3, 25% violation should NOT be hard fail (below 30%)
    assert not is_hard_failure, "Should not be hard failure (below fail_ratio)"
    assert penalty_multiplier > 5.0, "Should have steep exponential penalty"
    assert violations["p90"]["severity"] == "SEVERE", "Should be SEVERE severity"
    print("✓ PASSED - Steep exponential penalty applied")


def test_hard_failure_boundary():
    """Test case: Hard failure when violation exceeds fail_ratio"""
    print("\n" + "="*80)
    print("TEST 4: Hard Failure (P90 violation exceeds fail_ratio)")
    print("="*80)

    metrics = {
        "p50_e2e_latency": 2.0,
        "p90_e2e_latency": 6.5,  # 30% over 5.0 threshold
        "p99_e2e_latency": 8.0,
        "mean_ttft": 0.8,
        "mean_e2e_latency": 2.5
    }

    slo_config = {
        "latency": {
            "p90": {"threshold": 5.0, "weight": 2.0, "hard_fail": True, "fail_ratio": 0.2}
        },
        "steepness": 0.1
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Metrics: P90 = {metrics['p90_e2e_latency']}s (threshold: 5.0s)")
    print(f"Violation Ratio: {(6.5-5.0)/5.0:.2%} (fail_ratio: 20%)")
    print(f"Hard Failure: {is_hard_failure}")
    print(f"Violation Details: {violations}")

    score = calculate_objective_score(metrics, "minimize_latency", slo_config)
    print(f"Final Score: {score} (should be inf)")

    assert is_hard_failure, "Should be hard failure"
    assert violations["p90"]["severity"] == "HARD_FAIL", "Should be HARD_FAIL severity"
    assert score == float("inf"), "Score should be infinity"
    print("✓ PASSED - Experiment marked as failed")


def test_multiple_violations_cumulative_penalty():
    """Test case: Multiple SLO violations - cumulative exponential penalty"""
    print("\n" + "="*80)
    print("TEST 5: Multiple Violations (Cumulative exponential penalties)")
    print("="*80)

    metrics = {
        "p50_e2e_latency": 2.3,  # 15% over
        "p90_e2e_latency": 5.5,  # 10% over
        "p99_e2e_latency": 11.0, # 10% over
        "mean_ttft": 1.2,        # 20% over
        "mean_e2e_latency": 2.5
    }

    slo_config = {
        "latency": {
            "p50": {"threshold": 2.0, "weight": 1.0, "hard_fail": False},
            "p90": {"threshold": 5.0, "weight": 2.0, "hard_fail": False},
            "p99": {"threshold": 10.0, "weight": 3.0, "hard_fail": False}
        },
        "ttft": {"threshold": 1.0, "weight": 2.0, "hard_fail": False},
        "steepness": 0.1
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print("Violations:")
    for metric, details in violations.items():
        print(f"  {metric}: {details['actual']:.2f}s > {details['threshold']:.2f}s "
              f"(+{details['violation_ratio']*100:.1f}%, penalty: +{details['penalty']:.2f})")

    print(f"\nTotal Penalty Multiplier: {penalty_multiplier:.4f}x")
    print(f"Hard Failure: {is_hard_failure}")

    base_score = metrics['mean_e2e_latency']
    final_score = base_score * penalty_multiplier
    print(f"Base Score: {base_score:.4f}")
    print(f"Final Score: {final_score:.4f}")
    print(f"Score Increase: {((penalty_multiplier - 1) * 100):.2f}%")

    assert not is_hard_failure, "Should not be hard failure"
    assert penalty_multiplier > 10.0, "Should have steep cumulative penalty"
    assert len(violations) == 4, "Should have 4 violations"
    print("✓ PASSED - Cumulative penalties applied")


def test_steepness_parameter_effect():
    """Test case: Compare different steepness values"""
    print("\n" + "="*80)
    print("TEST 6: Steepness Parameter Effect (0.05 vs 0.1 vs 0.2)")
    print("="*80)

    metrics = {
        "p90_e2e_latency": 6.0,  # 20% over 5.0 threshold
        "mean_e2e_latency": 3.0
    }

    for steepness in [0.05, 0.1, 0.2]:
        slo_config = {
            "latency": {
                "p90": {"threshold": 5.0, "weight": 2.0, "hard_fail": False}
            },
            "steepness": steepness
        }

        penalty_multiplier, _, _ = calculate_slo_penalty(metrics, slo_config)
        print(f"  Steepness {steepness}: penalty_multiplier = {penalty_multiplier:.4f}x")

    print("\nObservation: Lower steepness = steeper penalty curve (higher penalties)")
    print("✓ PASSED")


def main():
    print("\n" + "#"*80)
    print("# SLO-Aware Objective Scoring Algorithm Test Suite")
    print("#"*80)

    tests = [
        test_no_slo_violations,
        test_minor_violation_soft_penalty,
        test_severe_violation_exponential_penalty,
        test_hard_failure_boundary,
        test_multiple_violations_cumulative_penalty,
        test_steepness_parameter_effect
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1

    print("\n" + "#"*80)
    print(f"# Test Summary: {passed} passed, {failed} failed")
    print("#"*80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
