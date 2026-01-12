"""
Utility for merging multiple parameter presets with different strategies.
"""

from typing import Dict, List, Any, Tuple
from enum import Enum


class MergeStrategy(str, Enum):
	"""Merge strategy for combining multiple parameter presets."""

	UNION = "union"
	INTERSECTION = "intersection"
	LAST_WINS = "last_wins"


class PresetMerger:
	"""Handles merging of multiple parameter presets."""

	@staticmethod
	def merge_parameters(
		presets: List[Dict[str, Any]], strategy: MergeStrategy = MergeStrategy.UNION
	) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
		"""
		Merge multiple parameter presets.

		Args:
			presets: List of preset dictionaries with 'parameters' field
			strategy: Merge strategy to use

		Returns:
			Tuple of (merged_parameters, conflicts)
		"""
		if not presets:
			return {}, []

		if len(presets) == 1:
			return presets[0].get("parameters", {}), []

		if strategy == MergeStrategy.UNION:
			return PresetMerger._merge_union(presets)
		elif strategy == MergeStrategy.INTERSECTION:
			return PresetMerger._merge_intersection(presets)
		elif strategy == MergeStrategy.LAST_WINS:
			return PresetMerger._merge_last_wins(presets)
		else:
			raise ValueError(f"Unknown merge strategy: {strategy}")

	@staticmethod
	def _merge_union(presets: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
		"""
		Union merge: combine all parameter values from all presets.

		Example:
			Preset A: {"tp-size": [1, 2], "mem-fraction": [0.8]}
			Preset B: {"tp-size": [3], "schedule-policy": ["lpm"]}
			Result: {"tp-size": [1, 2, 3], "mem-fraction": [0.8], "schedule-policy": ["lpm"]}
		"""
		merged = {}
		conflicts = []

		for preset in presets:
			parameters = preset.get("parameters", {})
			for param_name, values in parameters.items():
				if param_name not in merged:
					merged[param_name] = []

				# Convert single value to list
				if not isinstance(values, list):
					values = [values]

				# Add new values (deduplicate)
				for value in values:
					if value not in merged[param_name]:
						merged[param_name].append(value)

		return merged, conflicts

	@staticmethod
	def _merge_intersection(presets: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
		"""
		Intersection merge: only keep values present in all presets.

		Example:
			Preset A: {"tp-size": [1, 2], "mem-fraction": [0.8]}
			Preset B: {"tp-size": [2, 3], "schedule-policy": ["lpm"]}
			Result: {"tp-size": [2]}  # Only common value
		"""
		if not presets:
			return {}, []

		# Get all parameter names from first preset
		first_params = presets[0].get("parameters", {})
		merged = {}
		conflicts = []

		for param_name, first_values in first_params.items():
			if not isinstance(first_values, list):
				first_values = [first_values]

			# Find intersection of values across all presets
			common_values = set(first_values)

			for preset in presets[1:]:
				preset_params = preset.get("parameters", {})
				if param_name in preset_params:
					preset_values = preset_params[param_name]
					if not isinstance(preset_values, list):
						preset_values = [preset_values]
					common_values = common_values.intersection(preset_values)
				else:
					# Parameter not in this preset, intersection is empty
					common_values = set()
					break

			if common_values:
				merged[param_name] = sorted(list(common_values))
			else:
				# Track conflict: parameter exists but no common values
				conflicts.append({"parameter": param_name, "reason": "No common values across all presets"})

		return merged, conflicts

	@staticmethod
	def _merge_last_wins(presets: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
		"""
		Last wins merge: later presets override earlier ones.

		Example:
			Preset A: {"tp-size": [1, 2], "mem-fraction": [0.8]}
			Preset B: {"tp-size": [4], "schedule-policy": ["lpm"]}
			Result: {"tp-size": [4], "mem-fraction": [0.8], "schedule-policy": ["lpm"]}
		"""
		merged = {}
		conflicts = []

		for preset in presets:
			parameters = preset.get("parameters", {})
			for param_name, values in parameters.items():
				if param_name in merged:
					# Track conflict
					conflicts.append(
						{
							"parameter": param_name,
							"overridden_by": preset.get("name", "unknown"),
							"previous_values": merged[param_name],
							"new_values": values if isinstance(values, list) else [values],
						}
					)

				merged[param_name] = values if isinstance(values, list) else [values]

		return merged, conflicts

	@staticmethod
	def validate_parameters(parameters: Dict[str, Any]) -> List[str]:
		"""
		Validate merged parameters.

		Args:
			parameters: Merged parameter dictionary

		Returns:
			List of validation errors (empty if valid)
		"""
		errors = []

		if not parameters:
			errors.append("Parameters cannot be empty")
			return errors

		# Check that all parameter values are lists
		for param_name, values in parameters.items():
			if not isinstance(values, list):
				errors.append(f"Parameter '{param_name}' must be a list")
			elif len(values) == 0:
				errors.append(f"Parameter '{param_name}' cannot be empty")

		return errors
