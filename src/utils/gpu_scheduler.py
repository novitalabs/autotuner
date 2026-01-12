"""
GPU-Aware Task Scheduling Utilities

Provides utilities for estimating GPU requirements and checking GPU availability
before task execution to prevent resource conflicts and optimize scheduling.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from .gpu_monitor import get_gpu_monitor

logger = logging.getLogger(__name__)


def estimate_gpu_requirements(task_config: Dict[str, Any]) -> Tuple[int, int]:
	"""Estimate GPU requirements from task configuration.

	Args:
	    task_config: Task configuration dictionary

	Returns:
	    Tuple of (min_gpus_required, estimated_memory_mb_per_gpu)
	"""
	parameters = task_config.get("parameters", {})

	# Extract parallel configuration parameters
	# Support both hyphenated and underscore formats
	tp = _extract_max_param_value(parameters, ["tensor-parallel-size", "tp-size", "tp_size", "tp"], default=1)
	pp = _extract_max_param_value(parameters, ["pipeline-parallel-size", "pp-size", "pp_size", "pp"], default=1)
	dp = _extract_max_param_value(parameters, ["data-parallel-size", "dp-size", "dp_size", "dp"], default=1)
	cp = _extract_max_param_value(parameters, ["context-parallel-size", "cp-size", "cp_size", "cp"], default=1)
	dcp = _extract_max_param_value(
		parameters, ["decode-context-parallel-size", "dcp-size", "dcp_size", "dcp"], default=1
	)

	# Calculate world_size = tp × pp × max(dp, dcp, cp)
	world_size = tp * pp * max(dp, dcp, cp)

	# Estimate memory requirements per GPU (rough heuristics)
	# Base requirement: 8GB per GPU
	# Add more for larger models
	base_memory_mb = 8000

	# Try to get model size hint from model name/path
	model_id = task_config.get("model", {}).get("id_or_path", "")
	if any(size in model_id.lower() for size in ["70b", "65b"]):
		# Large models need more memory
		estimated_memory_mb = 20000
	elif any(size in model_id.lower() for size in ["13b", "7b"]):
		# Medium models
		estimated_memory_mb = 12000
	else:
		# Small models or unknown
		estimated_memory_mb = base_memory_mb

	logger.info(f"[GPU Scheduler] Estimated requirements: {world_size} GPUs, ~{estimated_memory_mb}MB per GPU")
	logger.info(f"[GPU Scheduler]   Parallel config: TP={tp}, PP={pp}, DP={dp}, CP={cp}, DCP={dcp}")

	return world_size, estimated_memory_mb


def _extract_max_param_value(parameters: Dict[str, Any], param_names: list, default: int = 1) -> int:
	"""Extract maximum value from parameter list (handles lists and single values).

	Args:
	    parameters: Parameter dictionary
	    param_names: List of possible parameter names to check
	    default: Default value if not found

	Returns:
	    Maximum value found or default
	"""
	for name in param_names:
		if name in parameters:
			value = parameters[name]

			# Handle dict format: {"type": "choice", "values": [1, 2, 4]}
			if isinstance(value, dict) and "values" in value:
				value = value["values"]

			# Handle list format: [1, 2, 4]
			if isinstance(value, list):
				return max(int(v) for v in value)

			# Handle single value format: 4
			return int(value)

	return default


def check_gpu_availability(required_gpus: int, min_memory_mb: Optional[int] = None) -> Tuple[bool, str]:
	"""Check if sufficient GPUs are available for task execution.

	Args:
	    required_gpus: Number of GPUs required
	    min_memory_mb: Minimum memory required per GPU (optional)

	Returns:
	    Tuple of (is_available, message)
	"""
	gpu_monitor = get_gpu_monitor()

	# Check if nvidia-smi is available
	if not gpu_monitor.is_available():
		logger.warning("[GPU Scheduler] nvidia-smi not available - assuming GPUs are available")
		return True, "GPU monitoring unavailable, proceeding without checks"

	# Get current GPU status
	snapshot = gpu_monitor.query_gpus(use_cache=False)
	if not snapshot:
		logger.warning("[GPU Scheduler] Failed to query GPU status - assuming GPUs are available")
		return True, "GPU query failed, proceeding without checks"

	# Check total number of GPUs
	if len(snapshot.gpus) < required_gpus:
		message = f"Insufficient GPUs: need {required_gpus}, system has {len(snapshot.gpus)}"
		logger.error(f"[GPU Scheduler] {message}")
		return False, message

	# Check if enough GPUs meet the requirements
	available_gpus = gpu_monitor.get_available_gpus(
		min_memory_mb=min_memory_mb, max_utilization=50  # Consider GPUs with <50% utilization
	)

	if len(available_gpus) < required_gpus:
		# Try with relaxed memory constraint
		if min_memory_mb:
			available_gpus_relaxed = gpu_monitor.get_available_gpus(min_memory_mb=None, max_utilization=50)

			if len(available_gpus_relaxed) >= required_gpus:
				message = (
					f"GPUs available but with limited memory: "
					f"{len(available_gpus_relaxed)} GPUs with <50% util, "
					f"but only {len(available_gpus)} with >{min_memory_mb}MB free"
				)
				logger.warning(f"[GPU Scheduler] {message}")
				return True, message  # Allow with warning

		# Build detailed status message
		gpu_status = []
		for gpu in snapshot.gpus:
			gpu_status.append(f"GPU {gpu.index}: {gpu.utilization_gpu}% util, " f"{gpu.memory_free_mb}MB free")

		message = (
			f"Insufficient available GPUs: need {required_gpus}, "
			f"{len(available_gpus)} available with requirements. "
			f"Status: {'; '.join(gpu_status)}"
		)
		logger.warning(f"[GPU Scheduler] {message}")
		return False, message

	# Success - enough GPUs available
	message = f"{len(available_gpus)} GPUs available (need {required_gpus})"
	logger.info(f"[GPU Scheduler] ✓ {message}")

	# Log details of available GPUs
	for gpu_idx in available_gpus[:required_gpus]:
		gpu_info = gpu_monitor.get_gpu_info(gpu_idx)
		if gpu_info:
			logger.info(
				f"[GPU Scheduler]   GPU {gpu_idx}: {gpu_info.utilization_gpu}% util, "
				f"{gpu_info.memory_free_mb}/{gpu_info.memory_total_mb}MB free, "
				f"score={gpu_info.score:.2f}"
			)

	return True, message


def wait_for_gpu_availability(
	required_gpus: int, min_memory_mb: Optional[int] = None, timeout_seconds: int = 300, check_interval: int = 30
) -> Tuple[bool, str]:
	"""Wait for sufficient GPUs to become available.

	Args:
	    required_gpus: Number of GPUs required
	    min_memory_mb: Minimum memory required per GPU
	    timeout_seconds: Maximum time to wait (default: 5 minutes)
	    check_interval: How often to check (default: 30 seconds)

	Returns:
	    Tuple of (is_available, message)
	"""
	import time

	start_time = time.time()
	check_count = 0

	logger.info(
		f"[GPU Scheduler] Waiting for {required_gpus} GPUs " f"(timeout={timeout_seconds}s, interval={check_interval}s)"
	)

	while (time.time() - start_time) < timeout_seconds:
		check_count += 1
		is_available, message = check_gpu_availability(required_gpus, min_memory_mb)

		if is_available:
			elapsed = time.time() - start_time
			logger.info(f"[GPU Scheduler] GPUs became available after {elapsed:.1f}s " f"({check_count} checks)")
			return True, message

		# Log status periodically
		if check_count == 1 or check_count % 5 == 0:
			logger.info(f"[GPU Scheduler] Check {check_count}: {message}")

		# Wait before next check
		time.sleep(check_interval)

	# Timeout reached
	elapsed = time.time() - start_time
	timeout_message = (
		f"Timeout waiting for GPUs after {elapsed:.1f}s " f"({check_count} checks). Last status: {message}"
	)
	logger.error(f"[GPU Scheduler] {timeout_message}")
	return False, timeout_message
