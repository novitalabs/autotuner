"""
Integration helper for parallel configuration in orchestrator.

This module provides utilities to integrate parallel configuration
with the autotuner orchestrator and parameter grid expansion.
Mirrors the quantization_integration.py pattern.
"""

import logging
from typing import Dict, Any, Optional, List

from utils.parallel_mapper import get_runtime_parallel_args, resolve_parallel_preset, validate_parallel_config

logger = logging.getLogger(__name__)


def expand_parallel_presets(parallel_config: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""
	Expand parallel configuration if it contains multiple presets.

	Args:
	    parallel_config: Parallel config (can contain "presets" list)

	Returns:
	    List of expanded parallel configs (one per preset)

	Example:
	    >>> expand_parallel_presets({"presets": ["single-gpu", "high-throughput"]})
	    [
	        {"preset": "single-gpu"},
	        {"preset": "high-throughput"}
	    ]
	"""
	if not parallel_config:
		return [None]

	# Check if multiple presets are specified
	if "presets" in parallel_config:
		presets = parallel_config["presets"]
		if not isinstance(presets, list):
			presets = [presets]

		expanded = []
		for preset_name in presets:
			preset_config = {"preset": preset_name}
			# Allow overriding preset values with explicit fields
			preset_config.update({k: v for k, v in parallel_config.items() if k != "presets"})
			expanded.append(preset_config)

		return expanded

	# Single configuration
	return [parallel_config]


def validate_task_parallel_config(base_runtime: str, parallel_config: Optional[Dict[str, Any]]) -> tuple[bool, str]:
	"""
	Validate parallel configuration from task.

	Args:
	    base_runtime: Runtime name
	    parallel_config: Parallel configuration to validate

	Returns:
	    (is_valid, error_message) tuple
	"""
	if not parallel_config:
		return True, "Valid (no parallel configuration)"

	return validate_parallel_config(base_runtime, parallel_config)


def get_parallel_config_summary(parallel_config: Optional[Dict[str, Any]]) -> str:
	"""
	Get a human-readable summary of parallel configuration.

	Args:
	    parallel_config: Parallel configuration

	Returns:
	    Summary string

	Example:
	    >>> get_parallel_config_summary({"tp": 4, "pp": 2, "dp": 1})
	    "TP: 4, PP: 2, DP: 1 (Total GPUs: 8)"
	"""
	if not parallel_config:
		return "No parallelism"

	# Handle preset references (array mode only)
	if "presets" in parallel_config:
		presets = parallel_config["presets"]
		if isinstance(presets, list):
			return f"Presets: {', '.join(presets)}"
		return f"Preset: {presets}"

	parts = []

	tp = parallel_config.get("tp", 1)
	pp = parallel_config.get("pp", 1)
	dp = parallel_config.get("dp", 1)
	cp = parallel_config.get("cp") or parallel_config.get("dcp")

	if tp != 1:
		parts.append(f"TP: {tp}")
	if pp != 1:
		parts.append(f"PP: {pp}")
	if dp != 1:
		parts.append(f"DP: {dp}")
	if cp and cp != 1:
		parts.append(f"CP: {cp}")

	if parallel_config.get("enable_expert_parallel"):
		parts.append("EP: enabled")

	if "moe_tp" in parallel_config or "moe_ep" in parallel_config:
		moe_tp = parallel_config.get("moe_tp", "?")
		moe_ep = parallel_config.get("moe_ep", "?")
		parts.append(f"MoE: TP={moe_tp}, EP={moe_ep}")

	if not parts:
		return "Single GPU"

	summary = ", ".join(parts)

	# Calculate total GPUs
	total_gpus = tp * pp * dp
	if cp and cp != 1:
		total_gpus = tp * pp * cp

	summary += f" (Total GPUs: {total_gpus})"

	return summary


def expand_parallel_config_to_parameter_spec(parallel_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Convert parallel_config with arrays into parameter spec format for generate_parameter_grid.

	This function handles parallel_config that contains arrays of values and converts them
	into the format expected by generate_parameter_grid.

	IMPORTANT: This function keeps the field names (tp, pp, dp, etc.) as-is.
	They will be converted to runtime-specific CLI args later in prepare_runtime_parameters.

	Args:
	    parallel_config: Parallel config with possible arrays or presets
	                     Example: {"tp": [1, 2, 4], "dp": [1, 2]}
	                     Or: {"presets": ["single-gpu", "high-throughput"]}

	Returns:
	    Parameter spec dict in generate_parameter_grid format with original field names
	    Example: {"__parallel__tp": [1, 2, 4], "__parallel__dp": [1, 2]}
	    Or for presets: {"__parallel__preset": ["single-gpu", "high-throughput"]}

	Note:
	    - Single values are wrapped in lists
	    - Keys are prefixed with __parallel__ to distinguish from regular parameters
	    - All values are kept for grid expansion
	    - Presets are stored as __parallel__preset for later resolution
	"""
	if not parallel_config:
		return {}

	# Handle preset mode
	if "presets" in parallel_config:
		presets = parallel_config["presets"]
		if not isinstance(presets, list):
			presets = [presets]
		return {"__parallel__preset": presets}

	# Fields that should be expanded into parameter grid
	parallel_fields = [
		"tp",
		"pp",
		"dp",
		"cp",
		"dcp",
		"enable_expert_parallel",
		"moe_tp",
		"moe_ep",
		"moe_cluster",
		"moe_dense_tp",
	]

	param_spec = {}

	for field in parallel_fields:
		if field not in parallel_config:
			continue

		value = parallel_config[field]

		# Convert to list if single value
		if not isinstance(value, list):
			value = [value]

		# Skip if empty list
		if not value:
			continue

		# Add with __parallel__ prefix to distinguish from regular CLI parameters
		param_spec[f"__parallel__{field}"] = value

	return param_spec


def merge_parameters_with_parallel_config(
	base_parameters: Dict[str, Any], parallel_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
	"""
	Merge base parameters with parallel_config parameter spec.

	This combines user-defined parameters with parallel parameters,
	preparing a unified parameter specification for grid expansion.

	Args:
	    base_parameters: User-defined parameters (e.g., {"mem-fraction-static": [0.7, 0.9]})
	    parallel_config: Parallel configuration with arrays

	Returns:
	    Merged parameter specification

	Example:
	    >>> merge_parameters_with_parallel_config(
	    ...     {"mem-fraction-static": [0.7, 0.9]},
	    ...     {"tp": [1, 2], "dp": [1, 2]}
	    ... )
	    {"mem-fraction-static": [0.7, 0.9], "__parallel__tp": [1, 2], "__parallel__dp": [1, 2]}
	"""
	# Start with base parameters
	merged = base_parameters.copy() if base_parameters else {}

	# Add parallel_config parameters
	parallel_params = expand_parallel_config_to_parameter_spec(parallel_config)

	# Merge (parallel_params can override base_parameters if same key exists)
	merged.update(parallel_params)

	return merged


def extract_parallel_config_from_params(params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
	"""
	Extract parallel config from experiment parameters and separate regular parameters.

	Experiment parameters may contain __parallel__ prefixed fields from grid expansion.
	This function separates them and reconstructs the parallel_config.

	Args:
	    params: Experiment parameters dict, may contain __parallel__ prefixed keys
	            Example: {"mem-fraction-static": 0.7, "__parallel__tp": 2, "__parallel__dp": 1}
	            Or: {"__parallel__preset": "high-throughput"}

	Returns:
	    Tuple of (regular_params, parallel_config)
	    Example: ({"mem-fraction-static": 0.7}, {"tp": 2, "dp": 1})
	    Or: ({}, {"preset": "high-throughput"})
	"""
	regular_params = {}
	parallel_config = {}

	for key, value in params.items():
		if key.startswith("__parallel__"):
			# Extract field name
			field_name = key.replace("__parallel__", "")
			parallel_config[field_name] = value
		else:
			regular_params[key] = value

	return regular_params, parallel_config if parallel_config else None


def prepare_runtime_parallel_parameters(base_runtime: str, params: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Prepare runtime-specific CLI parameters from experiment parameters with parallel config.

	This function:
	1. Extracts parallel_config from __parallel__ prefixed parameters
	2. Converts parallel_config to runtime-specific CLI arguments
	3. Merges with regular parameters

	Args:
	    base_runtime: Runtime engine (\"vllm\", \"sglang\", \"tensorrt_llm\")
	    params: Experiment parameters (may contain __parallel__ prefixed fields)

	Returns:
	    Runtime-ready parameter dict with correct CLI argument names

	Example:
	    >>> prepare_runtime_parallel_parameters(
	    ...     "vllm",
	    ...     {"mem-fraction-static": 0.7, "__parallel__tp": 4, "__parallel__dp": 2}
	    ... )
	    {"mem-fraction-static": 0.7, "--tensor-parallel-size": "4", "--data-parallel-size": "2"}
	"""
	# Separate regular params and parallel config
	regular_params, parallel_config = extract_parallel_config_from_params(params)

	if not parallel_config:
		# No parallel configuration, return as-is
		return regular_params

	# Use get_runtime_parallel_args to convert parallel_config to runtime args
	try:
		parallel_args = get_runtime_parallel_args(base_runtime=base_runtime, parallel_config=parallel_config)
	except ValueError as e:
		logger.error(f"Failed to process parallel config: {e}")
		# Fall back to regular parameters only
		return regular_params

	# Merge parallel args with regular params
	# Remove CLI prefix (--) for cleaner parameter dict
	# Controllers will add it back when building commands
	cleaned_parallel_args = {}
	for key, value in parallel_args.items():
		clean_key = key.lstrip("-")
		cleaned_parallel_args[clean_key] = value

	# Merge (parallel args have lower priority than explicit user params)
	merged = cleaned_parallel_args.copy()
	merged.update(regular_params)

	return merged


def prepare_experiment_parameters(
	base_runtime: str, parallel_config: Optional[Dict[str, Any]], param_combination: Dict[str, Any]
) -> Dict[str, Any]:
	"""
	Prepare final experiment parameters by merging parallel_config and user parameters.

	Args:
	    base_runtime: Runtime engine (\"vllm\", \"sglang\", \"tensorrt_llm\")
	    parallel_config: Parallel configuration from task
	    param_combination: Single parameter combination from grid (user parameters)

	Returns:
	    Merged parameter dictionary with parallel_config-derived args + user overrides

	Example:
	    >>> prepare_experiment_parameters(
	    ...     "vllm",
	    ...     {"tp": 4, "dp": 2},
	    ...     {"mem-fraction-static": 0.7}
	    ... )
	    {"tensor-parallel-size": 4, "data-parallel-size": 2, "mem-fraction-static": 0.7}
	"""
	# Get runtime arguments from parallel_config
	try:
		runtime_args = get_runtime_parallel_args(
			base_runtime=base_runtime, parallel_config=parallel_config, user_parameters=param_combination
		)
	except ValueError as e:
		logger.error(f"Failed to process parallel config: {e}")
		# Fall back to user parameters only
		runtime_args = param_combination.copy()

	# Remove CLI prefix (--) for cleaner parameter dict
	# Controllers will add it back when building commands
	cleaned_params = {}
	for key, value in runtime_args.items():
		clean_key = key.lstrip("-")
		cleaned_params[clean_key] = value

	return cleaned_params


# Export all functions
__all__ = [
	"expand_parallel_presets",
	"validate_task_parallel_config",
	"get_parallel_config_summary",
	"expand_parallel_config_to_parameter_spec",
	"merge_parameters_with_parallel_config",
	"extract_parallel_config_from_params",
	"prepare_runtime_parallel_parameters",
	"prepare_experiment_parameters",
]
