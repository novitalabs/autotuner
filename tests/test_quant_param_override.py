"""
Tests for quantization parameter overrides from task parameters.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.quantization_integration import (
	merge_parameters_with_quant_config,
	prepare_runtime_parameters
)
from utils.optimizer import generate_parameter_grid


def test_task_params_promote_to_quant_grid():
	"""Ensure kv-cache-dtype in task parameters becomes __quant__kvcache_dtype."""
	base_params = {
		"kv-cache-dtype": ["auto", "fp8_e5m2"],
		"tp-size": [2],
	}

	merged = merge_parameters_with_quant_config(base_params, quant_config=None)
	assert "__quant__kvcache_dtype" in merged
	assert merged["__quant__kvcache_dtype"] == ["auto", "fp8_e5m2"]
	assert merged["tp-size"] == [2]


def test_task_params_override_quant_config():
	"""Task parameters should override quant_config for quantization fields."""
	base_params = {
		"kv-cache-dtype": ["fp8_e4m3"],
		"tp-size": [2],
	}
	quant_config = {
		"kvcache_dtype": ["auto"],
	}

	merged = merge_parameters_with_quant_config(base_params, quant_config=quant_config)
	assert merged["__quant__kvcache_dtype"] == ["fp8_e4m3"]


def test_prepare_runtime_parameters_respects_override():
	"""Runtime args should reflect the user-specified kv-cache-dtype."""
	merged = merge_parameters_with_quant_config(
		{"kv-cache-dtype": ["fp8_e5m2"], "tp-size": [2]},
		quant_config=None
	)
	grid = generate_parameter_grid(merged)
	assert grid, "Expected non-empty parameter grid"

	params = grid[0]
	runtime_params = prepare_runtime_parameters(
		base_runtime="sglang",
		params=params,
		model_path="meta-llama/Llama-3.2-1B-Instruct",
		model_config=None
	)

	assert runtime_params["kv-cache-dtype"] == "fp8_e5m2"
	assert runtime_params["tp-size"] == 2
