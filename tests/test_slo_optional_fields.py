#!/usr/bin/env python3
"""
Test script for optional SLO fields
Validates that all SLO configuration fields are truly optional
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.optimizer import calculate_slo_penalty, calculate_objective_score


def test_minimal_slo_single_metric():
    """Test: Only threshold specified for single metric"""
    print("\n" + "="*80)
    print("TEST 1: Minimal SLO - Only P99 threshold specified")
    print("="*80)

    metrics = {
        "p99_e2e_latency": 12.0,  # Above threshold
        "mean_e2e_latency": 5.0
    }

    # Minimal config: only threshold
    slo_config = {
        "latency": {
            "p99": {
                "threshold": 10.0
            }
        }
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Config: Only P99 threshold=10.0 (no weight, no hard_fail, no steepness)")
    print(f"Metrics: P99 = 12.0s")
    print(f"Penalty Multiplier: {penalty_multiplier:.4f}x")
    print(f"Hard Failure: {is_hard_failure}")
    print(f"Violation Details: {violations}")

    # Should use defaults: weight=1.0, hard_fail=False, steepness=0.1
    assert penalty_multiplier > 1.0, "Should have penalty with default weight"
    assert not is_hard_failure, "Should not fail (hard_fail defaults to False)"
    assert violations["p99"]["severity"] in ["MINOR", "SEVERE"], "Should classify severity"
    print("✓ PASSED - Uses defaults for omitted fields")


def test_omit_entire_metrics():
    """Test: Only P90 specified, P50 and P99 omitted"""
    print("\n" + "="*80)
    print("TEST 2: Omit Entire Metrics - Only P90 specified")
    print("="*80)

    metrics = {
        "p50_e2e_latency": 10.0,  # No SLO for P50, should be ignored
        "p90_e2e_latency": 6.0,   # Has SLO, will be penalized
        "p99_e2e_latency": 20.0,  # No SLO for P99, should be ignored
        "mean_e2e_latency": 5.0
    }

    slo_config = {
        "latency": {
            "p90": {
                "threshold": 5.0,
                "weight": 2.0
            }
        }
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Config: Only P90 (P50 and P99 omitted)")
    print(f"Metrics: P50=10.0s (ignored), P90=6.0s (penalized), P99=20.0s (ignored)")
    print(f"Violations: {list(violations.keys())}")
    print(f"Penalty Multiplier: {penalty_multiplier:.4f}x")

    assert len(violations) == 1, "Should only have P90 violation"
    assert "p90" in violations, "Should track P90 violation"
    assert "p50" not in violations, "P50 should be ignored (no SLO)"
    assert "p99" not in violations, "P99 should be ignored (no SLO)"
    print("✓ PASSED - Omitted metrics ignored")


def test_omit_ttft():
    """Test: Latency SLOs specified, TTFT omitted"""
    print("\n" + "="*80)
    print("TEST 3: TTFT Omitted - Only latency SLOs specified")
    print("="*80)

    metrics = {
        "p90_e2e_latency": 6.0,
        "mean_ttft": 2.0,  # No SLO, should be ignored
        "mean_e2e_latency": 5.0
    }

    slo_config = {
        "latency": {
            "p90": {"threshold": 5.0}
        }
        # TTFT section omitted entirely
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Config: P90 latency only (no TTFT)")
    print(f"Metrics: P90=6.0s (penalized), TTFT=2.0s (ignored)")
    print(f"Violations: {list(violations.keys())}")

    assert "ttft" not in violations, "TTFT should be ignored when omitted"
    assert "p90" in violations, "P90 should be tracked"
    print("✓ PASSED - TTFT ignored when omitted")


def test_omit_steepness():
    """Test: Steepness omitted, should use default 0.1"""
    print("\n" + "="*80)
    print("TEST 4: Steepness Omitted - Uses default 0.1")
    print("="*80)

    metrics = {
        "p90_e2e_latency": 6.0,  # 20% over 5.0
        "mean_e2e_latency": 5.0
    }

    slo_config_no_steepness = {
        "latency": {
            "p90": {"threshold": 5.0, "weight": 2.0}
        }
        # steepness omitted
    }

    penalty_no_steepness, _, _ = calculate_slo_penalty(metrics, slo_config_no_steepness)

    slo_config_explicit_steepness = {
        "latency": {
            "p90": {"threshold": 5.0, "weight": 2.0}
        },
        "steepness": 0.1
    }

    penalty_explicit_steepness, _, _ = calculate_slo_penalty(metrics, slo_config_explicit_steepness)

    print(f"Penalty without steepness: {penalty_no_steepness:.4f}x")
    print(f"Penalty with steepness=0.1: {penalty_explicit_steepness:.4f}x")

    assert abs(penalty_no_steepness - penalty_explicit_steepness) < 0.01, \
        "Omitted steepness should equal explicit 0.1"
    print("✓ PASSED - Default steepness 0.1 applied")


def test_omit_weight():
    """Test: Weight omitted, should use default 1.0"""
    print("\n" + "="*80)
    print("TEST 5: Weight Omitted - Uses default 1.0")
    print("="*80)

    metrics = {
        "p90_e2e_latency": 5.5,  # 10% over
        "mean_e2e_latency": 5.0
    }

    slo_no_weight = {
        "latency": {
            "p90": {"threshold": 5.0}  # weight omitted
        }
    }

    penalty_no_weight, _, violations_no_weight = calculate_slo_penalty(metrics, slo_no_weight)

    slo_weight_1 = {
        "latency": {
            "p90": {"threshold": 5.0, "weight": 1.0}
        }
    }

    penalty_weight_1, _, violations_weight_1 = calculate_slo_penalty(metrics, slo_weight_1)

    print(f"Penalty without weight: {violations_no_weight['p90']['penalty']:.4f}")
    print(f"Penalty with weight=1.0: {violations_weight_1['p90']['penalty']:.4f}")

    assert abs(violations_no_weight['p90']['penalty'] - violations_weight_1['p90']['penalty']) < 0.01, \
        "Omitted weight should equal explicit 1.0"
    print("✓ PASSED - Default weight 1.0 applied")


def test_omit_fail_ratio():
    """Test: fail_ratio omitted when hard_fail=true, should use default 0.5"""
    print("\n" + "="*80)
    print("TEST 6: fail_ratio Omitted - Uses default 0.5")
    print("="*80)

    metrics = {
        "p90_e2e_latency": 7.0,  # 40% over 5.0
        "mean_e2e_latency": 5.0
    }

    slo_no_fail_ratio = {
        "latency": {
            "p90": {
                "threshold": 5.0,
                "hard_fail": True
                # fail_ratio omitted, should default to 0.5
            }
        }
    }

    _, is_hard_failure_no_ratio, violations_no_ratio = calculate_slo_penalty(metrics, slo_no_fail_ratio)

    print(f"Violation: 40% over threshold")
    print(f"Config: hard_fail=True, fail_ratio omitted (defaults to 0.5)")
    print(f"Hard Failure: {is_hard_failure_no_ratio}")

    # 40% < 50% default fail_ratio, so should NOT be hard failure
    assert not is_hard_failure_no_ratio, "Should not fail (40% < 50% default)"
    assert violations_no_ratio['p90']['severity'] == "SEVERE", "Should be SEVERE but not HARD_FAIL"
    print("✓ PASSED - Default fail_ratio 0.5 applied")


def test_completely_empty_slo():
    """Test: Empty SLO config should return no penalty"""
    print("\n" + "="*80)
    print("TEST 7: Completely Empty SLO - No penalty")
    print("="*80)

    metrics = {
        "p90_e2e_latency": 10.0,
        "mean_e2e_latency": 5.0
    }

    slo_config = {}

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Config: Empty SLO dict")
    print(f"Penalty Multiplier: {penalty_multiplier:.4f}")
    print(f"Violations: {violations}")

    assert penalty_multiplier == 1.0, "Empty SLO should have no penalty"
    assert not is_hard_failure, "Empty SLO should not fail"
    assert len(violations) == 0, "Empty SLO should have no violations"
    print("✓ PASSED - Empty SLO ignored gracefully")


def main():
    print("\n" + "#"*80)
    print("# SLO Optional Fields Test Suite")
    print("#"*80)

    tests = [
        test_minimal_slo_single_metric,
        test_omit_entire_metrics,
        test_omit_ttft,
        test_omit_steepness,
        test_omit_weight,
        test_omit_fail_ratio,
        test_completely_empty_slo
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
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "#"*80)
    print(f"# Test Summary: {passed} passed, {failed} failed")
    print("#"*80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
