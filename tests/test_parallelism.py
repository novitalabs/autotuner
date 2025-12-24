"""
Tests for parallelism calculation utilities.
"""

import pytest
from src.utils.parallelism import (
    ParallelFactors,
    parse_parallel_factors,
    compute_world_size,
    validate_parallel_factors,
    format_parallel_config,
    _extract_max_int
)


class TestParallelFactors:
    """Test ParallelFactors dataclass."""

    def test_default_values(self):
        """Test default parallel factors."""
        factors = ParallelFactors()
        assert factors.tensor_parallel == 1
        assert factors.pipeline_parallel == 1
        assert factors.data_parallel == 1
        assert factors.context_parallel == 1
        assert factors.disaggregation_parallel == 1
        assert factors.world_size == 1

    def test_world_size_tp_only(self):
        """Test world size with only tensor parallelism."""
        factors = ParallelFactors(tensor_parallel=4)
        assert factors.world_size == 4

    def test_world_size_tp_pp(self):
        """Test world size with tensor and pipeline parallelism."""
        factors = ParallelFactors(tensor_parallel=2, pipeline_parallel=2)
        assert factors.world_size == 4

    def test_world_size_tp_pp_dp(self):
        """Test world size with TP, PP, and DP."""
        factors = ParallelFactors(
            tensor_parallel=2,
            pipeline_parallel=2,
            data_parallel=2
        )
        assert factors.world_size == 8  # 2 * 2 * 2

    def test_world_size_exclusive_max(self):
        """Test that world size uses max of exclusive parallel dimensions."""
        # DP=4, CP=2 -> should use max(4, 2) = 4
        factors = ParallelFactors(
            tensor_parallel=2,
            data_parallel=4,
            context_parallel=2
        )
        assert factors.world_size == 8  # 2 * 1 * 4

    def test_world_size_dcp(self):
        """Test world size with disaggregation parallelism."""
        factors = ParallelFactors(
            tensor_parallel=2,
            disaggregation_parallel=2
        )
        assert factors.world_size == 4  # 2 * 1 * 2

    def test_repr(self):
        """Test string representation."""
        factors = ParallelFactors(tensor_parallel=2, pipeline_parallel=2)
        repr_str = repr(factors)
        assert "tp=2" in repr_str
        assert "pp=2" in repr_str
        assert "world_size=4" in repr_str


class TestParseParallelFactors:
    """Test parse_parallel_factors function."""

    def test_empty_parameters(self):
        """Test parsing empty parameter dict."""
        factors = parse_parallel_factors({})
        assert factors.tensor_parallel == 1
        assert factors.pipeline_parallel == 1
        assert factors.world_size == 1

    def test_parse_tp_hyphen_format(self):
        """Test parsing tp-size parameter."""
        params = {"tp-size": 4}
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 4
        assert factors.world_size == 4

    def test_parse_tp_underscore_format(self):
        """Test parsing tp_size parameter."""
        params = {"tp_size": 2}
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 2

    def test_parse_tp_list(self):
        """Test parsing tp-size as list (takes max)."""
        params = {"tp-size": [1, 2, 4]}
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 4

    def test_parse_tp_dict_values(self):
        """Test parsing tp-size as dict with values key."""
        params = {"tp-size": {"values": [1, 2, 4]}}
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 4

    def test_parse_all_factors(self):
        """Test parsing all parallel factors."""
        params = {
            "tp-size": 2,
            "pp-size": 2,
            "dp-size": 2,
            "cp-size": 1,
            "dcp-size": 1
        }
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 2
        assert factors.pipeline_parallel == 2
        assert factors.data_parallel == 2
        assert factors.world_size == 8  # 2 * 2 * 2

    def test_parse_mixed_formats(self):
        """Test parsing with mixed naming conventions."""
        params = {
            "tp_size": 2,  # underscore
            "pp-size": 2,  # hyphen
            "dp": 1  # short form
        }
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 2
        assert factors.pipeline_parallel == 2
        assert factors.data_parallel == 1

    def test_custom_defaults(self):
        """Test parsing with custom default values."""
        factors = parse_parallel_factors(
            {},
            default_tp=2,
            default_pp=2
        )
        assert factors.tensor_parallel == 2
        assert factors.pipeline_parallel == 2
        assert factors.world_size == 4


class TestComputeWorldSize:
    """Test compute_world_size convenience function."""

    def test_compute_simple(self):
        """Test world size computation."""
        params = {"tp-size": 2, "pp-size": 2}
        world_size = compute_world_size(params)
        assert world_size == 4

    def test_compute_with_dp(self):
        """Test world size with data parallelism."""
        params = {"tp-size": 2, "dp-size": 4}
        world_size = compute_world_size(params)
        assert world_size == 8  # 2 * 1 * 4

    def test_compute_default(self):
        """Test world size with defaults."""
        world_size = compute_world_size({})
        assert world_size == 1


class TestExtractMaxInt:
    """Test _extract_max_int helper function."""

    def test_extract_int(self):
        """Test extracting from int."""
        assert _extract_max_int(4, 1) == 4

    def test_extract_list(self):
        """Test extracting max from list."""
        assert _extract_max_int([1, 2, 4], 1) == 4

    def test_extract_dict_values(self):
        """Test extracting from dict with values key."""
        assert _extract_max_int({"values": [1, 2, 4]}, 1) == 4

    def test_extract_str(self):
        """Test extracting from string."""
        assert _extract_max_int("4", 1) == 4

    def test_extract_none(self):
        """Test extracting from None returns default."""
        assert _extract_max_int(None, 99) == 99

    def test_extract_empty_list(self):
        """Test extracting from empty list returns default."""
        assert _extract_max_int([], 99) == 99

    def test_extract_invalid_dict(self):
        """Test extracting from dict without values key returns default."""
        assert _extract_max_int({"foo": "bar"}, 99) == 99

    def test_extract_invalid_type(self):
        """Test extracting from invalid type returns default."""
        assert _extract_max_int({"complex": "object"}, 99) == 99


class TestValidateParallelFactors:
    """Test validate_parallel_factors function."""

    def test_validate_default(self):
        """Test validation of default factors."""
        factors = ParallelFactors()
        valid, msg = validate_parallel_factors(factors)
        assert valid
        assert msg == ""

    def test_validate_valid_tp_pp(self):
        """Test validation of valid TP+PP configuration."""
        factors = ParallelFactors(tensor_parallel=2, pipeline_parallel=2)
        valid, msg = validate_parallel_factors(factors)
        assert valid
        assert msg == ""

    def test_validate_valid_tp_dp(self):
        """Test validation of valid TP+DP configuration."""
        factors = ParallelFactors(tensor_parallel=2, data_parallel=4)
        valid, msg = validate_parallel_factors(factors)
        assert valid

    def test_validate_invalid_tp_zero(self):
        """Test validation fails for TP=0."""
        factors = ParallelFactors(tensor_parallel=0)
        valid, msg = validate_parallel_factors(factors)
        assert not valid
        assert "tensor_parallel" in msg

    def test_validate_invalid_pp_negative(self):
        """Test validation fails for negative PP."""
        factors = ParallelFactors(pipeline_parallel=-1)
        valid, msg = validate_parallel_factors(factors)
        assert not valid
        assert "pipeline_parallel" in msg

    def test_validate_conflicting_dp_cp(self):
        """Test validation fails for DP and CP both > 1."""
        factors = ParallelFactors(
            data_parallel=2,
            context_parallel=2
        )
        valid, msg = validate_parallel_factors(factors)
        assert not valid
        assert "exclusive" in msg.lower()
        assert "data_parallel" in msg
        assert "context_parallel" in msg

    def test_validate_conflicting_dp_dcp(self):
        """Test validation fails for DP and DCP both > 1."""
        factors = ParallelFactors(
            data_parallel=2,
            disaggregation_parallel=2
        )
        valid, msg = validate_parallel_factors(factors)
        assert not valid
        assert "exclusive" in msg.lower()

    def test_validate_conflicting_all_three(self):
        """Test validation fails for DP, CP, DCP all > 1."""
        factors = ParallelFactors(
            data_parallel=2,
            context_parallel=2,
            disaggregation_parallel=2
        )
        valid, msg = validate_parallel_factors(factors)
        assert not valid
        assert "exclusive" in msg.lower()

    def test_validate_dp_cp_one_is_ok(self):
        """Test validation passes when only one exclusive mode is > 1."""
        # DP=2, CP=1, DCP=1 -> OK
        factors = ParallelFactors(
            data_parallel=2,
            context_parallel=1,
            disaggregation_parallel=1
        )
        valid, msg = validate_parallel_factors(factors)
        assert valid


class TestFormatParallelConfig:
    """Test format_parallel_config function."""

    def test_format_default(self):
        """Test formatting default configuration."""
        factors = ParallelFactors()
        formatted = format_parallel_config(factors)
        assert "TP=1" in formatted
        assert "world_size=1" in formatted

    def test_format_tp_only(self):
        """Test formatting TP-only configuration."""
        factors = ParallelFactors(tensor_parallel=4)
        formatted = format_parallel_config(factors)
        assert "TP=4" in formatted
        assert "PP" not in formatted  # PP=1 not shown
        assert "world_size=4" in formatted

    def test_format_tp_pp(self):
        """Test formatting TP+PP configuration."""
        factors = ParallelFactors(tensor_parallel=2, pipeline_parallel=2)
        formatted = format_parallel_config(factors)
        assert "TP=2" in formatted
        assert "PP=2" in formatted
        assert "world_size=4" in formatted

    def test_format_tp_pp_dp(self):
        """Test formatting TP+PP+DP configuration."""
        factors = ParallelFactors(
            tensor_parallel=2,
            pipeline_parallel=2,
            data_parallel=2
        )
        formatted = format_parallel_config(factors)
        assert "TP=2" in formatted
        assert "PP=2" in formatted
        assert "DP=2" in formatted
        assert "world_size=8" in formatted

    def test_format_with_cp(self):
        """Test formatting with context parallelism."""
        factors = ParallelFactors(tensor_parallel=2, context_parallel=2)
        formatted = format_parallel_config(factors)
        assert "TP=2" in formatted
        assert "CP=2" in formatted
        assert "world_size=4" in formatted

    def test_format_with_dcp(self):
        """Test formatting with disaggregation parallelism."""
        factors = ParallelFactors(tensor_parallel=2, disaggregation_parallel=2)
        formatted = format_parallel_config(factors)
        assert "TP=2" in formatted
        assert "DCP=2" in formatted
        assert "world_size=4" in formatted


class TestIntegration:
    """Integration tests for real-world scenarios."""

    def test_vllm_default_config(self):
        """Test parsing typical vLLM configuration."""
        params = {
            "tp-size": 2,
            "max-model-len": 4096,
            "gpu-memory-utilization": 0.9
        }
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 2
        assert factors.world_size == 2

    def test_sglang_multi_gpu(self):
        """Test parsing SGLang multi-GPU configuration."""
        params = {
            "tp": 4,
            "mem-fraction-static": 0.85
        }
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 4
        assert factors.world_size == 4

    def test_grid_search_parameters(self):
        """Test parsing grid search with multiple values."""
        params = {
            "tp-size": [1, 2, 4],  # Grid search over TP
            "pp-size": [1, 2],  # Grid search over PP
            "dp-size": 1  # Fixed DP
        }
        # Should extract max values for GPU requirement
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 4  # Max of [1, 2, 4]
        assert factors.pipeline_parallel == 2  # Max of [1, 2]
        assert factors.world_size == 8  # 4 * 2 * 1

    def test_bayesian_optimization_dict_format(self):
        """Test parsing Bayesian optimization dict format."""
        params = {
            "tp-size": {"values": [1, 2, 4]},
            "max-model-len": {"min": 2048, "max": 8192}
        }
        factors = parse_parallel_factors(params)
        assert factors.tensor_parallel == 4
        assert factors.world_size == 4
