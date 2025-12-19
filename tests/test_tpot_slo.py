#!/usr/bin/env python3
"""
Test TPOT SLO functionality
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.optimizer import calculate_slo_penalty, calculate_objective_score


def test_tpot_slo():
    """Test TPOT SLO with violation"""
    print("\n" + "="*80)
    print("TEST: TPOT SLO - Time Per Output Token violation")
    print("="*80)

    metrics = {
        "mean_tpot": 0.06,  # Above 0.05 threshold
        "mean_e2e_latency": 3.0
    }

    slo_config = {
        "tpot": {
            "threshold": 0.05,
            "weight": 2.0
        }
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Metrics: TPOT = {metrics['mean_tpot']}s (threshold: 0.05s)")
    print(f"Violation Ratio: {((0.06-0.05)/0.05):.2%}")
    print(f"Penalty Multiplier: {penalty_multiplier:.4f}x")
    print(f"Hard Failure: {is_hard_failure}")
    print(f"Violations: {violations}")

    assert penalty_multiplier > 1.0, "Should have penalty"
    assert not is_hard_failure, "Should not be hard failure"
    assert "tpot" in violations, "Should track TPOT violation"
    print("✓ PASSED - TPOT violation detected and penalized")


def test_tpot_with_ttft():
    """Test TPOT and TTFT together"""
    print("\n" + "="*80)
    print("TEST: TPOT + TTFT - Both metrics enforced")
    print("="*80)

    metrics = {
        "mean_ttft": 1.2,  # Above 1.0 threshold
        "mean_tpot": 0.055,  # Above 0.05 threshold
        "mean_e2e_latency": 3.0
    }

    slo_config = {
        "ttft": {
            "threshold": 1.0,
            "weight": 2.0
        },
        "tpot": {
            "threshold": 0.05,
            "weight": 1.5
        },
        "steepness": 0.1
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Metrics:")
    print(f"  TTFT = {metrics['mean_ttft']}s (threshold: 1.0s)")
    print(f"  TPOT = {metrics['mean_tpot']}s (threshold: 0.05s)")
    print(f"Penalty Multiplier: {penalty_multiplier:.4f}x")
    print(f"Violations: {list(violations.keys())}")

    assert "ttft" in violations, "Should have TTFT violation"
    assert "tpot" in violations, "Should have TPOT violation"
    assert len(violations) == 2, "Should have both violations"
    print("✓ PASSED - Both TTFT and TPOT violations tracked")


def test_tpot_no_violation():
    """Test TPOT within bounds"""
    print("\n" + "="*80)
    print("TEST: TPOT Within Bounds - No violation")
    print("="*80)

    metrics = {
        "mean_tpot": 0.04,  # Below 0.05 threshold
        "mean_e2e_latency": 3.0
    }

    slo_config = {
        "tpot": {
            "threshold": 0.05,
            "weight": 2.0
        }
    }

    penalty_multiplier, is_hard_failure, violations = calculate_slo_penalty(metrics, slo_config)

    print(f"Metrics: TPOT = {metrics['mean_tpot']}s (threshold: 0.05s)")
    print(f"Penalty Multiplier: {penalty_multiplier:.4f}")
    print(f"Violations: {violations}")

    assert penalty_multiplier == 1.0, "Should have no penalty"
    assert "tpot" not in violations, "Should not have TPOT violation"
    print("✓ PASSED - No violation when within bounds")


def main():
    print("\n" + "#"*80)
    print("# TPOT SLO Test Suite")
    print("#"*80)

    tests = [
        test_tpot_slo,
        test_tpot_with_ttft,
        test_tpot_no_violation
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
