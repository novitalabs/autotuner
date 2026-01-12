"""
Utility functions and classes for parameter optimization.
"""

import itertools
import math
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import optuna


def generate_parameter_grid(parameter_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
	"""Generate all parameter combinations for grid search.

	Supports two formats:
	1. Simple format: {"param_name": [value1, value2]}
	   - Direct list of values for each parameter
	2. Structured format: {"param_name": {"type": "choice", "values": [value1, value2]}}
	   - Legacy format with explicit type specification

	Args:
	    parameter_spec: Dict mapping parameter names to their values or specifications
	                   Simple: {"tp-size": [1, 2], "mem-fraction-static": [0.7, 0.8]}
	                   Structured: {"tp_size": {"type": "choice", "values": [1, 2]}}

	Returns:
	    List of parameter dictionaries, one for each combination
	"""
	param_names = []
	param_values = []

	for param_name, spec in parameter_spec.items():
		# Check if it's the simple format (direct list) or structured format (dict with type/values)
		if isinstance(spec, list):
			# Simple format: direct list of values
			# Skip empty lists (no values to search)
			if len(spec) == 0:
				continue
			param_names.append(param_name)
			param_values.append(spec)
		elif isinstance(spec, dict) and "type" in spec:
			# Structured format: legacy support
			if spec["type"] == "choice":
				# Skip empty value lists
				values = spec["values"]
				if len(values) == 0:
					continue
				param_names.append(param_name)
				param_values.append(values)
			else:
				raise ValueError(f"Unsupported parameter type: {spec['type']}")
		else:
			raise ValueError(
				f"Invalid parameter specification for '{param_name}'. "
				f"Expected list of values or dict with 'type' and 'values' keys."
			)

	# Generate Cartesian product
	combinations = list(itertools.product(*param_values))

	# Convert to list of dicts
	grid = []
	for combo in combinations:
		param_dict = dict(zip(param_names, combo))
		grid.append(param_dict)

	return grid


def calculate_slo_penalty(
	metrics: Dict[str, Any], slo_config: Optional[Dict[str, Any]] = None
) -> Tuple[float, bool, Dict[str, Any]]:
	"""Calculate SLO penalty with exponential curve near boundaries.

	Implements tiered enforcement:
	- Minor violations: exponential penalty only
	- Severe violations: mark as hard failure

	Args:
	    metrics: Benchmark metrics dictionary
	    slo_config: SLO configuration from task JSON
	                Format: {
	                  "latency": {
	                    "p50": {"threshold": 2.0, "weight": 1.0, "hard_fail": false},
	                    "p90": {"threshold": 5.0, "weight": 2.0, "hard_fail": true, "fail_ratio": 0.2}
	                  },
	                  "ttft": {"threshold": 1.0, "weight": 2.0, "hard_fail": false},
	                  "steepness": 0.1  # Controls exponential slope (lower = steeper)
	                }

	Returns:
	    Tuple of (total_penalty_multiplier, is_hard_failure, violation_details)
	    - penalty_multiplier: Value to multiply base score by (1.0 = no penalty)
	    - is_hard_failure: True if experiment should be marked as failed
	    - violation_details: Dict with per-metric violation info
	"""
	# No SLO configuration means no penalties
	if not slo_config or not metrics:
		return 1.0, False, {}

	# Default steepness parameter (lower = steeper penalty curve)
	steepness = slo_config.get("steepness", 0.1)

	total_penalty = 0.0
	is_hard_failure = False
	violation_details = {}

	# Process latency SLOs (P50, P90, P99)
	latency_slo = slo_config.get("latency", {})
	for percentile in ["p50", "p90", "p99"]:
		if percentile not in latency_slo:
			continue

		slo_spec = latency_slo[percentile]
		threshold = slo_spec.get("threshold")
		weight = slo_spec.get("weight", 1.0)
		hard_fail = slo_spec.get("hard_fail", False)
		fail_ratio = slo_spec.get("fail_ratio", 0.5)  # Default: fail if >50% over

		if threshold is None:
			continue

		# Get actual metric value
		metric_key = f"{percentile}_e2e_latency"
		actual_value = metrics.get(metric_key)

		if actual_value is None:
			continue

		# Calculate violation ratio (normalized)
		if actual_value > threshold:
			violation_ratio = (actual_value - threshold) / threshold

			# Check for hard failure condition
			if hard_fail and violation_ratio > fail_ratio:
				is_hard_failure = True
				violation_details[percentile] = {
					"threshold": threshold,
					"actual": actual_value,
					"violation_ratio": violation_ratio,
					"severity": "HARD_FAIL",
				}
			else:
				# Calculate exponential penalty
				# penalty = weight × exp(violation_ratio / steepness)
				# As violation_ratio increases, penalty grows exponentially
				penalty = weight * math.exp(violation_ratio / steepness)
				total_penalty += penalty

				severity = "SEVERE" if violation_ratio > 0.2 else "MINOR"
				violation_details[percentile] = {
					"threshold": threshold,
					"actual": actual_value,
					"violation_ratio": violation_ratio,
					"penalty": penalty,
					"severity": severity,
				}

	# Process TTFT SLO
	ttft_slo = slo_config.get("ttft", {})
	if ttft_slo:
		threshold = ttft_slo.get("threshold")
		weight = ttft_slo.get("weight", 1.0)
		hard_fail = ttft_slo.get("hard_fail", False)
		fail_ratio = ttft_slo.get("fail_ratio", 0.5)

		if threshold is not None:
			actual_value = metrics.get("mean_ttft")

			if actual_value is not None and actual_value > threshold:
				violation_ratio = (actual_value - threshold) / threshold

				if hard_fail and violation_ratio > fail_ratio:
					is_hard_failure = True
					violation_details["ttft"] = {
						"threshold": threshold,
						"actual": actual_value,
						"violation_ratio": violation_ratio,
						"severity": "HARD_FAIL",
					}
				else:
					penalty = weight * math.exp(violation_ratio / steepness)
					total_penalty += penalty

					severity = "SEVERE" if violation_ratio > 0.2 else "MINOR"
					violation_details["ttft"] = {
						"threshold": threshold,
						"actual": actual_value,
						"violation_ratio": violation_ratio,
						"penalty": penalty,
						"severity": severity,
					}

	# Process TPOT SLO
	tpot_slo = slo_config.get("tpot", {})
	if tpot_slo:
		threshold = tpot_slo.get("threshold")
		weight = tpot_slo.get("weight", 1.0)
		hard_fail = tpot_slo.get("hard_fail", False)
		fail_ratio = tpot_slo.get("fail_ratio", 0.5)

		if threshold is not None:
			actual_value = metrics.get("mean_tpot")

			if actual_value is not None and actual_value > threshold:
				violation_ratio = (actual_value - threshold) / threshold

				if hard_fail and violation_ratio > fail_ratio:
					is_hard_failure = True
					violation_details["tpot"] = {
						"threshold": threshold,
						"actual": actual_value,
						"violation_ratio": violation_ratio,
						"severity": "HARD_FAIL",
					}
				else:
					penalty = weight * math.exp(violation_ratio / steepness)
					total_penalty += penalty

					severity = "SEVERE" if violation_ratio > 0.2 else "MINOR"
					violation_details["tpot"] = {
						"threshold": threshold,
						"actual": actual_value,
						"violation_ratio": violation_ratio,
						"penalty": penalty,
						"severity": severity,
					}

	# Calculate final penalty multiplier
	# penalty_multiplier > 1.0 means the score gets worse
	penalty_multiplier = 1.0 + total_penalty

	return penalty_multiplier, is_hard_failure, violation_details


def check_batch_slo_compliance(
	batch_metrics: Dict[str, Any], slo_config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Any]]:
	"""Check if a single batch (concurrency level) meets SLO requirements.

	This is used to filter out batches that violate SLO constraints before aggregation.

	Args:
	    batch_metrics: Single batch metrics from genai-bench (one concurrency level)
	    slo_config: SLO configuration from task JSON

	Returns:
	    Tuple of (is_compliant, violation_details)
	    - is_compliant: True if batch meets all SLO requirements
	    - violation_details: Dict with violation information for logging
	"""
	# No SLO means all batches are compliant
	if not slo_config or not batch_metrics:
		return True, {}

	violation_details = {}
	is_compliant = True

	# Extract metrics from batch (genai-bench structure: {"stats": {...}})
	stats = batch_metrics.get("stats", {})
	if not stats:
		return True, {}  # No stats means we can't check, assume compliant

	# Check latency SLOs (P50, P90, P99)
	latency_slo = slo_config.get("latency", {})
	for percentile in ["p50", "p90", "p99"]:
		if percentile not in latency_slo:
			continue

		slo_spec = latency_slo[percentile]
		threshold = slo_spec.get("threshold")
		hard_fail = slo_spec.get("hard_fail", False)
		fail_ratio = slo_spec.get("fail_ratio", 0.5)

		if threshold is None:
			continue

		# Get actual value from batch stats
		e2e_latency_stats = stats.get("e2e_latency", {})
		actual_value = e2e_latency_stats.get(percentile)

		if actual_value is None:
			continue

		if actual_value > threshold:
			violation_ratio = (actual_value - threshold) / threshold

			# For SLO compliance check, we only care about hard_fail violations
			# Soft violations are OK - they just get penalties later
			if hard_fail and violation_ratio > fail_ratio:
				is_compliant = False
				violation_details[percentile] = {
					"threshold": threshold,
					"actual": actual_value,
					"violation_ratio": violation_ratio,
					"type": "HARD_FAIL",
				}
			else:
				# Soft violation - still compliant, but log it
				violation_details[percentile] = {
					"threshold": threshold,
					"actual": actual_value,
					"violation_ratio": violation_ratio,
					"type": "SOFT_VIOLATION",
				}

	# Check TTFT SLO
	ttft_slo = slo_config.get("ttft", {})
	if ttft_slo:
		threshold = ttft_slo.get("threshold")
		hard_fail = ttft_slo.get("hard_fail", False)
		fail_ratio = ttft_slo.get("fail_ratio", 0.5)

		if threshold is not None:
			ttft_stats = stats.get("ttft", {})
			actual_value = ttft_stats.get("mean")

			if actual_value is not None and actual_value > threshold:
				violation_ratio = (actual_value - threshold) / threshold

				if hard_fail and violation_ratio > fail_ratio:
					is_compliant = False
					violation_details["ttft"] = {
						"threshold": threshold,
						"actual": actual_value,
						"violation_ratio": violation_ratio,
						"type": "HARD_FAIL",
					}
				else:
					violation_details["ttft"] = {
						"threshold": threshold,
						"actual": actual_value,
						"violation_ratio": violation_ratio,
						"type": "SOFT_VIOLATION",
					}

	# Check TPOT SLO
	tpot_slo = slo_config.get("tpot", {})
	if tpot_slo:
		threshold = tpot_slo.get("threshold")
		hard_fail = tpot_slo.get("hard_fail", False)
		fail_ratio = tpot_slo.get("fail_ratio", 0.5)

		if threshold is not None:
			tpot_stats = stats.get("tpot", {})
			actual_value = tpot_stats.get("mean")

			if actual_value is not None and actual_value > threshold:
				violation_ratio = (actual_value - threshold) / threshold

				if hard_fail and violation_ratio > fail_ratio:
					is_compliant = False
					violation_details["tpot"] = {
						"threshold": threshold,
						"actual": actual_value,
						"violation_ratio": violation_ratio,
						"type": "HARD_FAIL",
					}
				else:
					violation_details["tpot"] = {
						"threshold": threshold,
						"actual": actual_value,
						"violation_ratio": violation_ratio,
						"type": "SOFT_VIOLATION",
					}

	return is_compliant, violation_details


def calculate_objective_score(
	results: Dict[str, Any], objective: str = "minimize_latency", slo_config: Optional[Dict[str, Any]] = None
) -> float:
	"""Calculate objective score from benchmark results with optional SLO penalties.

	Args:
	    results: Benchmark results dictionary from DirectBenchmarkController._parse_results()
	    objective: Optimization objective - 'minimize_latency' or 'maximize_throughput'
	    slo_config: Optional SLO configuration for penalty calculation

	Returns:
	    Objective score with SLO penalties applied (lower is better for minimization, higher for maximization)
	    Note: For hard SLO violations, returns worst possible score (inf or -inf)
	"""
	if not results:
		print("[Optimizer] No results provided, returning worst score")
		return float("inf") if "minimize" in objective else float("-inf")

	# Calculate base score based on objective
	try:
		if objective == "minimize_latency":
			# Use mean E2E latency as primary metric (in seconds)
			# Fallback to P50 if mean not available
			base_score = results.get("mean_e2e_latency", results.get("p50_e2e_latency"))
			if base_score is None:
				print(f"[Optimizer] Warning: No latency metrics found in results")
				print(f"[Optimizer] Available keys: {list(results.keys())}")
				return float("inf")

		elif objective == "maximize_throughput":
			# Use mean total throughput per GPU (tokens/s/GPU) as primary metric
			# This allows fair comparison across different GPU counts
			# Fallback to total throughput if per-GPU metrics not available
			throughput = results.get("mean_total_throughput_per_gpu")

			if throughput is None:
				# Fallback to absolute throughput (for backward compatibility)
				throughput = results.get(
					"mean_total_throughput", results.get("mean_output_throughput", results.get("max_total_throughput"))
				)
				if throughput is not None:
					print(f"[Optimizer] Warning: Using absolute throughput (per-GPU metrics not available)")

			if throughput is None or throughput == 0:
				print(f"[Optimizer] Warning: No throughput metrics found in results")
				print(f"[Optimizer] Available keys: {list(results.keys())}")
				return float("-inf")

			# Negate for minimization (optimizer looks for minimum score)
			base_score = -throughput

		elif objective == "minimize_ttft":
			# Time to First Token optimization
			base_score = results.get("mean_ttft")
			if base_score is None:
				print(f"[Optimizer] Warning: No TTFT metrics found in results")
				return float("inf")

		elif objective == "minimize_tpot":
			# Time Per Output Token optimization
			base_score = results.get("mean_tpot")
			if base_score is None:
				print(f"[Optimizer] Warning: No TPOT metrics found in results")
				return float("inf")

		else:
			raise ValueError(f"Unsupported objective: {objective}")

		# Apply SLO penalties if configured
		if slo_config:
			penalty_multiplier, is_hard_failure, violation_details = calculate_slo_penalty(results, slo_config)

			# Hard failure: return worst possible score
			if is_hard_failure:
				print(f"[Optimizer] HARD SLO FAILURE detected:")
				for metric, details in violation_details.items():
					if details.get("severity") == "HARD_FAIL":
						print(
							f"  {metric}: {details['actual']:.4f} >> {details['threshold']:.4f} (violation: {details['violation_ratio']*100:.1f}%)"
						)
				return float("inf") if "minimize" in objective else float("-inf")

			# NOTE: Batch-level SLO filtering already handles hard failures and filtering.
			# No additional penalty should be applied here for SUCCESS experiments.
			# This section is kept for backward compatibility but should not trigger
			# since batch-level filtering removes non-compliant data.

			# Soft penalties: ADD to base score (not multiply, to handle negative scores correctly)
			if penalty_multiplier > 1.0:
				# Convert penalty_multiplier (1.0 + penalty) to addition value
				penalty_value = (penalty_multiplier - 1.0) * abs(base_score)

				# For minimize objectives (positive base_score), add penalty makes it worse (larger)
				# For maximize objectives (negative base_score), add penalty makes it worse (less negative)
				if "maximize" in objective:
					# base_score is negative (e.g., -5000), adding positive penalty makes it less negative (worse)
					final_score = base_score + penalty_value
				else:
					# base_score is positive, adding penalty makes it larger (worse)
					final_score = base_score + penalty_value

				print(
					f"[Optimizer] Base score: {base_score:.4f}, SLO penalty: +{penalty_value:.2f}, Final: {final_score:.4f}"
				)
				if violation_details:
					print(f"[Optimizer] SLO violations detected:")
					for metric, details in violation_details.items():
						print(
							f"  {metric}: {details['actual']:.4f} > {details['threshold']:.4f} "
							f"(+{details['violation_ratio']*100:.1f}%, penalty: +{details['penalty']:.2f}, severity: {details['severity']})"
						)
				return final_score
			else:
				print(f"[Optimizer] Score: {base_score:.4f} (no SLO violations)")
				return base_score
		else:
			# No SLO config, return base score
			print(f"[Optimizer] Score: {base_score:.4f} (lower is better)")
			return base_score

	except Exception as e:
		print(f"[Optimizer] Error calculating objective score: {e}")
		return float("inf") if "minimize" in objective else float("-inf")


# ============================================================================
# Optimization Strategy Abstraction
# ============================================================================


class OptimizationStrategy(ABC):
	"""Abstract base class for optimization strategies."""

	def __init__(self, parameter_spec: Dict[str, Any], objective: str = "minimize_latency"):
		"""Initialize optimization strategy.

		Args:
		    parameter_spec: Parameter specification dictionary
		    objective: Optimization objective (minimize_latency, maximize_throughput, etc.)
		"""
		self.parameter_spec = parameter_spec
		self.objective = objective
		self.history: List[Dict[str, Any]] = []

	@abstractmethod
	def suggest_parameters(self) -> Optional[Dict[str, Any]]:
		"""Suggest next parameter configuration to try.

		Returns:
		    Dictionary of parameter values, or None if strategy is done
		"""
		pass

	@abstractmethod
	def tell_result(self, parameters: Dict[str, Any], objective_score: float, metrics: Dict[str, Any]):
		"""Update strategy with experiment result.

		Args:
		    parameters: Parameter configuration that was tested
		    objective_score: Objective score from calculate_objective_score()
		    metrics: Full metrics dictionary from benchmark
		"""
		pass

	def should_stop(self) -> bool:
		"""Check if optimization should stop early.

		Returns:
		    True if strategy has converged or no more suggestions
		"""
		return False

	def get_state(self) -> Dict[str, Any]:
		"""Serialize strategy state for checkpoint.

		Returns:
		    Dictionary containing strategy state
		"""
		return {
			"parameter_spec": self.parameter_spec,
			"objective": self.objective,
			"history": self.history,
		}

	@classmethod
	def from_state(cls, state: Dict[str, Any]) -> "OptimizationStrategy":
		"""Restore strategy from serialized state.

		Args:
		    state: Dictionary containing strategy state

		Returns:
		    Restored strategy instance
		"""
		# Base implementation - subclasses should override
		raise NotImplementedError("Subclass must implement from_state()")


class GridSearchStrategy(OptimizationStrategy):
	"""Grid search optimization - exhaustive evaluation of all combinations."""

	def __init__(
		self, parameter_spec: Dict[str, Any], objective: str = "minimize_latency", max_iterations: Optional[int] = None
	):
		"""Initialize grid search strategy.

		Args:
		    parameter_spec: Parameter specification dictionary
		    objective: Optimization objective
		    max_iterations: Maximum number of iterations (limits grid size)
		"""
		super().__init__(parameter_spec, objective)
		self.param_grid = generate_parameter_grid(parameter_spec)

		# Limit grid size if max_iterations specified
		if max_iterations is not None and max_iterations < len(self.param_grid):
			print(f"[GridSearch] Limiting grid from {len(self.param_grid)} to {max_iterations} combinations")
			self.param_grid = self.param_grid[:max_iterations]

		self.current_index = 0
		print(f"[GridSearch] Initialized with {len(self.param_grid)} parameter combinations")

	def suggest_parameters(self) -> Optional[Dict[str, Any]]:
		"""Get next parameter combination from grid."""
		if self.current_index >= len(self.param_grid):
			return None

		params = self.param_grid[self.current_index]
		self.current_index += 1
		print(f"[GridSearch] Suggesting combination {self.current_index}/{len(self.param_grid)}: {params}")
		return params

	def tell_result(self, parameters: Dict[str, Any], objective_score: float, metrics: Dict[str, Any]):
		"""Record result (grid search doesn't adapt)."""
		self.history.append({"parameters": parameters, "objective_score": objective_score, "metrics": metrics})
		print(f"[GridSearch] Recorded result: score={objective_score:.4f}")

	def should_stop(self) -> bool:
		"""Stop when all combinations evaluated."""
		return self.current_index >= len(self.param_grid)

	def get_state(self) -> Dict[str, Any]:
		"""Serialize GridSearch state for checkpoint."""
		base_state = super().get_state()
		base_state.update(
			{
				"strategy_class": "GridSearchStrategy",
				"current_index": self.current_index,
				"param_grid": self.param_grid,
			}
		)
		return base_state

	@classmethod
	def from_state(cls, state: Dict[str, Any]) -> "GridSearchStrategy":
		"""Restore GridSearch from serialized state."""
		# Create instance with basic parameters
		strategy = cls(
			parameter_spec=state["parameter_spec"],
			objective=state["objective"],
			max_iterations=None,  # Already limited in param_grid
		)
		# Restore state
		strategy.current_index = state["current_index"]
		strategy.param_grid = state["param_grid"]
		strategy.history = state.get("history", [])
		return strategy


class BayesianStrategy(OptimizationStrategy):
	"""Bayesian optimization using Optuna's TPE sampler."""

	def __init__(
		self,
		parameter_spec: Dict[str, Any],
		objective: str = "minimize_latency",
		max_iterations: int = 100,
		n_initial_random: int = 5,
		study_name: Optional[str] = None,
		storage: Optional[str] = None,
	):
		"""Initialize Bayesian optimization strategy.

		Args:
		    parameter_spec: Parameter specification dictionary
		    objective: Optimization objective
		    max_iterations: Maximum number of trials
		    n_initial_random: Number of random trials before Bayesian optimization
		    study_name: Optional name for Optuna study
		    storage: Optional Optuna storage URL (e.g., sqlite:///optuna.db)
		"""
		super().__init__(parameter_spec, objective)
		self.max_iterations = max_iterations
		self.n_initial_random = n_initial_random
		self.trial_count = 0

		# Parse parameter specification into Optuna search space
		self.search_space = self._parse_search_space(parameter_spec)

		# Track tried parameter combinations to avoid duplicates
		self.tried_params = set()

		# Create Optuna study
		direction = "minimize"  # All objectives use minimize (throughput is negated)
		sampler = optuna.samplers.TPESampler(n_startup_trials=n_initial_random)

		self.study = optuna.create_study(
			direction=direction,
			sampler=sampler,
			study_name=study_name or f"autotuner_{objective}",
			storage=storage,
			load_if_exists=True,  # Resume if study exists
		)

		print(f"[Bayesian] Initialized with {len(self.search_space)} parameters")
		print(f"[Bayesian] Max iterations: {max_iterations}, Initial random: {n_initial_random}")
		print(f"[Bayesian] Search space: {list(self.search_space.keys())}")

	def _parse_search_space(self, parameter_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
		"""Parse parameter specification into Optuna search space.

		Supports:
		- Simple format: {"param": [val1, val2]} → categorical
		- Explicit format: {"param": {"type": "categorical", "values": [...]}}
		- Continuous: {"param": {"type": "continuous", "low": 0.5, "high": 1.0}}
		- Integer: {"param": {"type": "integer", "low": 1, "high": 10}}

		Returns:
		    Dict mapping parameter names to Optuna space definitions
		"""
		search_space = {}

		for param_name, spec in parameter_spec.items():
			if isinstance(spec, list):
				# Simple format: treat as categorical
				# Skip empty lists (no values to search)
				if len(spec) == 0:
					continue
				search_space[param_name] = {"type": "categorical", "choices": spec}
			elif isinstance(spec, dict):
				param_type = spec.get("type", "categorical")

				if param_type in ["choice", "categorical"]:
					# Categorical parameter
					values = spec.get("values", spec.get("choices", []))
					# Skip empty value lists
					if len(values) == 0:
						continue
					search_space[param_name] = {"type": "categorical", "choices": values}
				elif param_type == "continuous":
					# Continuous (float) parameter
					search_space[param_name] = {
						"type": "continuous",
						"low": spec["low"],
						"high": spec["high"],
						"log": spec.get("log", False),  # Log scale for parameters like learning rate
					}
				elif param_type == "integer":
					# Integer parameter
					search_space[param_name] = {
						"type": "integer",
						"low": spec["low"],
						"high": spec["high"],
						"log": spec.get("log", False),
					}
				else:
					raise ValueError(f"Unsupported parameter type: {param_type}")
			else:
				raise ValueError(f"Invalid parameter specification for '{param_name}'")

		return search_space

	def suggest_parameters(self) -> Optional[Dict[str, Any]]:
		"""Suggest next parameter configuration using Optuna.

		Ensures no duplicate parameter combinations are tried by adding
		random perturbation if sampler suggests a duplicate.
		"""
		if self.trial_count >= self.max_iterations:
			print(f"[Bayesian] Reached max iterations ({self.max_iterations})")
			return None

		# Create Optuna trial - only called ONCE
		trial = self.study.ask()
		self.current_trial = trial

		# Extract parameter suggestions from the trial
		params = {}
		for param_name, space_def in self.search_space.items():
			if space_def["type"] == "categorical":
				params[param_name] = trial.suggest_categorical(param_name, space_def["choices"])
			elif space_def["type"] == "continuous":
				params[param_name] = trial.suggest_float(
					param_name, space_def["low"], space_def["high"], log=space_def.get("log", False)
				)
			elif space_def["type"] == "integer":
				params[param_name] = trial.suggest_int(
					param_name, space_def["low"], space_def["high"], log=space_def.get("log", False)
				)

		# Create hashable representation for duplicate checking
		params_tuple = tuple(sorted(params.items()))

		# Check if these parameters have been tried before
		if params_tuple not in self.tried_params:
			# New parameter combination - use it
			self.tried_params.add(params_tuple)
			self.trial_count += 1
			print(f"[Bayesian] Trial {self.trial_count}/{self.max_iterations}: {params}")
			return params

		# Duplicate detected - try perturbations
		print(f"[Bayesian] Duplicate detected: {params}")

		# Try up to 10 perturbations to find non-duplicate
		max_attempts = 10
		for attempt in range(max_attempts):
			# Apply perturbation based on parameter type
			perturbed_params = self._perturb_parameters(params)
			perturbed_tuple = tuple(sorted(perturbed_params.items()))

			if perturbed_tuple not in self.tried_params:
				# Use perturbed parameters
				self.tried_params.add(perturbed_tuple)
				self.trial_count += 1
				print(f"[Bayesian] Using perturbed params (attempt {attempt + 1}): {perturbed_params}")
				return perturbed_params

		# After max_attempts, use the original suggestion even if duplicate
		# Better to run a duplicate than to stop optimization early
		print(f"[Bayesian] Could not find non-duplicate after {max_attempts} perturbation attempts")
		print(f"[Bayesian] Using sampler's suggestion anyway: {params}")
		params_tuple = tuple(sorted(params.items()))
		self.tried_params.add(params_tuple)
		self.trial_count += 1
		return params

	def _perturb_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
		"""Add random perturbation to parameters to avoid exact duplicates.

		Args:
		    params: Original parameter dictionary

		Returns:
		    New parameter dictionary with random perturbation
		"""
		perturbed = params.copy()

		# Choose a random parameter to perturb
		param_names = list(self.search_space.keys())
		if not param_names:
			return perturbed

		param_to_perturb = random.choice(param_names)
		space_def = self.search_space[param_to_perturb]

		if space_def["type"] == "categorical":
			# For categorical, choose a different value if possible
			choices = space_def["choices"]
			if len(choices) > 1:
				current_value = perturbed[param_to_perturb]
				other_choices = [c for c in choices if c != current_value]
				if other_choices:
					perturbed[param_to_perturb] = random.choice(other_choices)

		elif space_def["type"] == "continuous":
			# For continuous, add small random noise (1-5% of range)
			low, high = space_def["low"], space_def["high"]
			range_size = high - low
			noise = random.uniform(0.01, 0.05) * range_size * random.choice([-1, 1])
			new_value = perturbed[param_to_perturb] + noise
			# Clamp to valid range
			perturbed[param_to_perturb] = max(low, min(high, new_value))

		elif space_def["type"] == "integer":
			# For integer, add ±1 or ±2
			low, high = space_def["low"], space_def["high"]
			current_value = perturbed[param_to_perturb]
			delta = random.choice([-2, -1, 1, 2])
			new_value = current_value + delta
			# Clamp to valid range
			perturbed[param_to_perturb] = max(low, min(high, new_value))

		return perturbed

	def tell_result(self, parameters: Dict[str, Any], objective_score: float, metrics: Dict[str, Any]):
		"""Update Optuna study with experiment result."""
		# Tell Optuna the result
		self.study.tell(self.current_trial, objective_score)

		# Record in history
		self.history.append({"parameters": parameters, "objective_score": objective_score, "metrics": metrics})

		# Print progress
		best_score = self.study.best_value
		best_params = self.study.best_params
		print(f"[Bayesian] Trial complete: score={objective_score:.4f}")
		print(f"[Bayesian] Best so far: score={best_score:.4f}, params={best_params}")

	def should_stop(self) -> bool:
		"""Check if Bayesian optimization should stop."""
		# Stop if max iterations reached
		if self.trial_count >= self.max_iterations:
			return True

		# Optional: Add convergence detection
		# e.g., no improvement in last N trials
		# For now, just use max_iterations

		return False

	def get_best_params(self) -> Dict[str, Any]:
		"""Get best parameters found so far."""
		return self.study.best_params

	def get_best_score(self) -> float:
		"""Get best objective score found so far."""
		return self.study.best_value

	def get_state(self) -> Dict[str, Any]:
		"""Serialize Bayesian state for checkpoint."""
		base_state = super().get_state()
		base_state.update(
			{
				"strategy_class": "BayesianStrategy",
				"trial_count": self.trial_count,
				"max_iterations": self.max_iterations,
				"n_initial_random": self.n_initial_random,
				"tried_params": [list(p) for p in self.tried_params],  # Convert tuples to lists for JSON serialization
			}
		)
		return base_state

	@classmethod
	def from_state(cls, state: Dict[str, Any]) -> "BayesianStrategy":
		"""Restore Bayesian from serialized state."""
		# Create instance (Optuna study will be restored from history)
		strategy = cls(
			parameter_spec=state["parameter_spec"],
			objective=state["objective"],
			max_iterations=state["max_iterations"],
			n_initial_random=state.get("n_initial_random", 5),
		)
		# Restore trial count
		strategy.trial_count = state["trial_count"]
		strategy.history = state.get("history", [])

		# Restore tried_params set
		tried_params_list = state.get("tried_params", [])
		strategy.tried_params = {tuple(p) for p in tried_params_list}

		# Re-populate Optuna study with history
		for entry in strategy.history:
			if entry.get("objective_score") is not None:
				# Create a completed trial from history
				trial = strategy.study.ask()
				# Set parameters from history
				for param_name, param_value in entry["parameters"].items():
					if param_name in strategy.search_space:
						space_def = strategy.search_space[param_name]
						if space_def["type"] == "categorical":
							trial.suggest_categorical(param_name, space_def["choices"])
						elif space_def["type"] == "continuous":
							trial.suggest_float(param_name, space_def["low"], space_def["high"])
						elif space_def["type"] == "integer":
							trial.suggest_int(param_name, space_def["low"], space_def["high"])
				strategy.study.tell(trial, entry["objective_score"])

		return strategy


class RandomSearchStrategy(OptimizationStrategy):
	"""Random search - random sampling from parameter space."""

	def __init__(
		self,
		parameter_spec: Dict[str, Any],
		objective: str = "minimize_latency",
		max_iterations: int = 100,
		seed: Optional[int] = None,
	):
		"""Initialize random search strategy.

		Args:
		    parameter_spec: Parameter specification dictionary
		    objective: Optimization objective
		    max_iterations: Maximum number of random samples
		    seed: Random seed for reproducibility
		"""
		super().__init__(parameter_spec, objective)
		self.max_iterations = max_iterations
		self.trial_count = 0

		# Use Optuna's RandomSampler for convenience
		import random

		if seed is not None:
			random.seed(seed)

		sampler = optuna.samplers.RandomSampler(seed=seed)
		self.study = optuna.create_study(
			direction="minimize", sampler=sampler, study_name=f"autotuner_random_{objective}"
		)

		# Parse search space (reuse BayesianStrategy's parser)
		bayesian_helper = BayesianStrategy(parameter_spec, objective, max_iterations=1)
		self.search_space = bayesian_helper.search_space

		print(f"[Random] Initialized with {max_iterations} random samples")

	def suggest_parameters(self) -> Optional[Dict[str, Any]]:
		"""Suggest random parameter configuration."""
		if self.trial_count >= self.max_iterations:
			return None

		# Create Optuna trial with random sampler
		trial = self.study.ask()
		self.current_trial = trial
		self.trial_count += 1

		# Extract parameters
		params = {}
		for param_name, space_def in self.search_space.items():
			if space_def["type"] == "categorical":
				params[param_name] = trial.suggest_categorical(param_name, space_def["choices"])
			elif space_def["type"] == "continuous":
				params[param_name] = trial.suggest_float(
					param_name, space_def["low"], space_def["high"], log=space_def.get("log", False)
				)
			elif space_def["type"] == "integer":
				params[param_name] = trial.suggest_int(
					param_name, space_def["low"], space_def["high"], log=space_def.get("log", False)
				)

		print(f"[Random] Sample {self.trial_count}/{self.max_iterations}: {params}")
		return params

	def tell_result(self, parameters: Dict[str, Any], objective_score: float, metrics: Dict[str, Any]):
		"""Record result."""
		self.study.tell(self.current_trial, objective_score)
		self.history.append({"parameters": parameters, "objective_score": objective_score, "metrics": metrics})

	def should_stop(self) -> bool:
		"""Stop after max iterations."""
		return self.trial_count >= self.max_iterations

	def get_state(self) -> Dict[str, Any]:
		"""Serialize Random state for checkpoint."""
		base_state = super().get_state()
		base_state.update(
			{
				"strategy_class": "RandomSearchStrategy",
				"trial_count": self.trial_count,
				"max_iterations": self.max_iterations,
			}
		)
		return base_state

	@classmethod
	def from_state(cls, state: Dict[str, Any]) -> "RandomSearchStrategy":
		"""Restore Random from serialized state."""
		strategy = cls(
			parameter_spec=state["parameter_spec"],
			objective=state["objective"],
			max_iterations=state["max_iterations"],
		)
		strategy.trial_count = state["trial_count"]
		strategy.history = state.get("history", [])
		return strategy


# ============================================================================
# Strategy Factory
# ============================================================================


def create_optimization_strategy(
	optimization_config: Dict[str, Any], parameter_spec: Dict[str, Any]
) -> OptimizationStrategy:
	"""Factory function to create optimization strategy.

	Args:
	    optimization_config: Optimization configuration from task
	                        {"strategy": "grid_search", "objective": "minimize_latency", ...}
	    parameter_spec: Parameter specification dictionary

	Returns:
	    OptimizationStrategy instance
	"""
	strategy_name = optimization_config.get("strategy", "grid_search")
	objective = optimization_config.get("objective", "minimize_latency")
	max_iterations = optimization_config.get("max_iterations", 100)

	if strategy_name == "grid_search":
		return GridSearchStrategy(parameter_spec=parameter_spec, objective=objective, max_iterations=max_iterations)

	elif strategy_name == "bayesian":
		n_initial_random = optimization_config.get("n_initial_random", 5)
		study_name = optimization_config.get("study_name")
		storage = optimization_config.get("storage")  # e.g., "sqlite:///optuna.db"

		return BayesianStrategy(
			parameter_spec=parameter_spec,
			objective=objective,
			max_iterations=max_iterations,
			n_initial_random=n_initial_random,
			study_name=study_name,
			storage=storage,
		)

	elif strategy_name == "random":
		seed = optimization_config.get("seed")

		return RandomSearchStrategy(
			parameter_spec=parameter_spec, objective=objective, max_iterations=max_iterations, seed=seed
		)

	else:
		raise ValueError(
			f"Unsupported optimization strategy: {strategy_name}. " f"Supported: grid_search, bayesian, random"
		)


def restore_optimization_strategy(state: Dict[str, Any]) -> OptimizationStrategy:
	"""Restore optimization strategy from serialized state.

	Args:
	    state: Serialized strategy state (from strategy.get_state())

	Returns:
	    Restored OptimizationStrategy instance
	"""
	strategy_class = state.get("strategy_class")

	if strategy_class == "GridSearchStrategy":
		return GridSearchStrategy.from_state(state)
	elif strategy_class == "BayesianStrategy":
		return BayesianStrategy.from_state(state)
	elif strategy_class == "RandomSearchStrategy":
		return RandomSearchStrategy.from_state(state)
	else:
		raise ValueError(f"Unknown strategy class: {strategy_class}")
