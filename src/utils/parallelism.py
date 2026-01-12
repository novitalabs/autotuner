"""
Parallelism calculation utilities.

This module provides utilities for parsing parallel factors (TP, PP, DP, CP, DCP)
from various parameter formats and calculating the total GPU requirement (world size).

Eliminates duplication between:
- src/utils/gpu_scheduler.py
- src/controllers/docker_controller.py
- src/controllers/local_controller.py
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParallelFactors:
	"""Parallel execution factors for distributed inference."""

	tensor_parallel: int = 1  # TP
	pipeline_parallel: int = 1  # PP
	data_parallel: int = 1  # DP
	context_parallel: int = 1  # CP
	disaggregation_parallel: int = 1  # DCP (for disaggregated prefill/decode)

	@property
	def world_size(self) -> int:
		"""
		Calculate total GPU requirement (world size).

		Formula: tp * pp * max(dp, dcp, cp)

		Rationale:
		- TP and PP are always multiplicative (model sharded across tp*pp GPUs)
		- DP, CP, and DCP are mutually exclusive parallelism strategies
		- Use the maximum of these for total GPU count
		"""
		# Find max of exclusive parallelism dimensions
		exclusive_parallel = max(self.data_parallel, self.disaggregation_parallel, self.context_parallel)

		return self.tensor_parallel * self.pipeline_parallel * exclusive_parallel

	def __repr__(self) -> str:
		"""Human-readable representation."""
		return (
			f"ParallelFactors(tp={self.tensor_parallel}, pp={self.pipeline_parallel}, "
			f"dp={self.data_parallel}, cp={self.context_parallel}, "
			f"dcp={self.disaggregation_parallel}, world_size={self.world_size})"
		)


# Parameter name variations across vLLM, SGLang, and task configs
TP_NAMES = {"tp", "tp-size", "tp_size", "tensor-parallel-size", "tensor_parallel_size"}
PP_NAMES = {"pp", "pp-size", "pp_size", "pipeline-parallel-size", "pipeline_parallel_size"}
DP_NAMES = {"dp", "dp-size", "dp_size", "data-parallel-size", "data_parallel_size"}
CP_NAMES = {"cp", "cp-size", "cp_size", "context-parallel-size", "context_parallel_size"}
DCP_NAMES = {"dcp", "dcp-size", "dcp_size", "disagg-parallel-size", "disagg_parallel_size"}


def parse_parallel_factors(
	parameters: Dict[str, Any],
	*,
	default_tp: int = 1,
	default_pp: int = 1,
	default_dp: int = 1,
	default_cp: int = 1,
	default_dcp: int = 1,
) -> ParallelFactors:
	"""
	Parse parallel factors from parameter dictionary.

	Handles multiple naming conventions (tp, tp-size, tp_size, etc.) and
	various value formats (int, list, dict with "values" key).

	Args:
	    parameters: Parameter dictionary from task config
	    default_tp: Default tensor parallel size if not found
	    default_pp: Default pipeline parallel size if not found
	    default_dp: Default data parallel size if not found
	    default_cp: Default context parallel size if not found
	    default_dcp: Default disaggregation parallel size if not found

	Returns:
	    ParallelFactors object with parsed values

	Example:
	    >>> params = {"tp-size": [1, 2, 4], "pp-size": 1}
	    >>> factors = parse_parallel_factors(params)
	    >>> factors.tensor_parallel
	    4  # Max value from list
	    >>> factors.world_size
	    4  # tp=4, pp=1, dp=1
	"""
	factors = ParallelFactors()

	# Parse TP
	tp_value = _find_parameter(parameters, TP_NAMES, default_tp)
	factors.tensor_parallel = _extract_max_int(tp_value, default_tp)

	# Parse PP
	pp_value = _find_parameter(parameters, PP_NAMES, default_pp)
	factors.pipeline_parallel = _extract_max_int(pp_value, default_pp)

	# Parse DP
	dp_value = _find_parameter(parameters, DP_NAMES, default_dp)
	factors.data_parallel = _extract_max_int(dp_value, default_dp)

	# Parse CP
	cp_value = _find_parameter(parameters, CP_NAMES, default_cp)
	factors.context_parallel = _extract_max_int(cp_value, default_cp)

	# Parse DCP
	dcp_value = _find_parameter(parameters, DCP_NAMES, default_dcp)
	factors.disaggregation_parallel = _extract_max_int(dcp_value, default_dcp)

	return factors


def compute_world_size(
	parameters: Dict[str, Any],
	*,
	default_tp: int = 1,
	default_pp: int = 1,
	default_dp: int = 1,
	default_cp: int = 1,
	default_dcp: int = 1,
) -> int:
	"""
	Compute world size (total GPU requirement) from parameters.

	Convenience function that combines parse_parallel_factors and world_size.

	Args:
	    parameters: Parameter dictionary from task config
	    default_*: Default values for each parallel dimension

	Returns:
	    Total number of GPUs required

	Example:
	    >>> params = {"tp-size": 2, "pp-size": 2, "dp-size": 1}
	    >>> compute_world_size(params)
	    4  # 2 * 2 * 1
	"""
	factors = parse_parallel_factors(
		parameters,
		default_tp=default_tp,
		default_pp=default_pp,
		default_dp=default_dp,
		default_cp=default_cp,
		default_dcp=default_dcp,
	)
	return factors.world_size


def _find_parameter(parameters: Dict[str, Any], names: set, default: Any) -> Any:
	"""
	Find parameter value using multiple possible names.

	Args:
	    parameters: Parameter dictionary
	    names: Set of possible parameter names
	    default: Default value if not found

	Returns:
	    Parameter value or default
	"""
	for name in names:
		if name in parameters:
			return parameters[name]
	return default


def _extract_max_int(value: Any, default: int) -> int:
	"""
	Extract maximum integer from various value formats.

	Handles:
	- int: 4 -> 4
	- list: [1, 2, 4] -> 4
	- dict with "values": {"values": [1, 2, 4]} -> 4
	- str: "4" -> 4
	- None: -> default

	Args:
	    value: Value to extract from
	    default: Default value if extraction fails

	Returns:
	    Maximum integer value
	"""
	if value is None:
		return default

	# Handle dict with "values" key
	if isinstance(value, dict):
		if "values" in value:
			value = value["values"]
		else:
			# No "values" key, use default
			return default

	# Handle list
	if isinstance(value, list):
		if not value:
			return default
		# Filter out non-numeric values and take max
		try:
			int_values = [int(v) for v in value]
			return max(int_values)
		except (ValueError, TypeError):
			return default

	# Handle scalar (int, str, etc.)
	try:
		return int(value)
	except (ValueError, TypeError):
		return default


def validate_parallel_factors(factors: ParallelFactors) -> Tuple[bool, str]:
	"""
	Validate parallel factors for common configuration errors.

	Args:
	    factors: ParallelFactors object to validate

	Returns:
	    Tuple of (is_valid, error_message)
	    - is_valid: True if configuration is valid
	    - error_message: Empty string if valid, error description otherwise

	Example:
	    >>> factors = ParallelFactors(tensor_parallel=2, data_parallel=2, context_parallel=2)
	    >>> valid, msg = validate_parallel_factors(factors)
	    >>> print(valid)
	    False  # Multiple exclusive parallelism modes enabled
	"""
	# All factors must be positive
	if factors.tensor_parallel < 1:
		return False, f"tensor_parallel must be >= 1, got {factors.tensor_parallel}"
	if factors.pipeline_parallel < 1:
		return False, f"pipeline_parallel must be >= 1, got {factors.pipeline_parallel}"
	if factors.data_parallel < 1:
		return False, f"data_parallel must be >= 1, got {factors.data_parallel}"
	if factors.context_parallel < 1:
		return False, f"context_parallel must be >= 1, got {factors.context_parallel}"
	if factors.disaggregation_parallel < 1:
		return False, f"disaggregation_parallel must be >= 1, got {factors.disaggregation_parallel}"

	# Check for conflicting parallelism modes
	exclusive_modes = [
		("data_parallel", factors.data_parallel),
		("context_parallel", factors.context_parallel),
		("disaggregation_parallel", factors.disaggregation_parallel),
	]

	active_modes = [(name, val) for name, val in exclusive_modes if val > 1]

	if len(active_modes) > 1:
		mode_names = ", ".join(f"{name}={val}" for name, val in active_modes)
		return False, (
			f"Multiple exclusive parallelism modes enabled: {mode_names}. " f"Only one of DP, CP, or DCP can be > 1."
		)

	return True, ""


def format_parallel_config(factors: ParallelFactors) -> str:
	"""
	Format parallel configuration as human-readable string.

	Args:
	    factors: ParallelFactors object

	Returns:
	    Formatted string like "TP=2, PP=1, DP=1 (world_size=2)"

	Example:
	    >>> factors = ParallelFactors(tensor_parallel=2, pipeline_parallel=2)
	    >>> format_parallel_config(factors)
	    'TP=2, PP=2, DP=1 (world_size=4)'
	"""
	parts = [f"TP={factors.tensor_parallel}"]

	if factors.pipeline_parallel > 1:
		parts.append(f"PP={factors.pipeline_parallel}")

	if factors.data_parallel > 1:
		parts.append(f"DP={factors.data_parallel}")

	if factors.context_parallel > 1:
		parts.append(f"CP={factors.context_parallel}")

	if factors.disaggregation_parallel > 1:
		parts.append(f"DCP={factors.disaggregation_parallel}")

	config_str = ", ".join(parts)
	return f"{config_str} (world_size={factors.world_size})"
