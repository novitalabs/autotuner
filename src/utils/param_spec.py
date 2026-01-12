"""
Parameter specification normalization utilities.

This module provides functions to normalize different parameter specification
formats into a consistent representation, eliminating duplication across
gpu_scheduler, optimizer, and preset merging code.
"""

from typing import Any


def normalize_choice_values(spec: Any) -> list[Any]:
	"""
	Normalize parameter specification to list of values.

	Handles multiple input formats:
	- dict with "values" key: {"values": [1, 2, 3]} -> [1, 2, 3]
	- list: [1, 2, 3] -> [1, 2, 3]
	- scalar: 5 -> [5]

	Args:
	    spec: Parameter specification in any supported format

	Returns:
	    List of parameter values

	Examples:
	    >>> normalize_choice_values({"values": [1, 2, 3]})
	    [1, 2, 3]

	    >>> normalize_choice_values([1, 2, 3])
	    [1, 2, 3]

	    >>> normalize_choice_values(5)
	    [5]

	    >>> normalize_choice_values("auto")
	    ["auto"]
	"""
	if isinstance(spec, dict) and "values" in spec:
		return list(spec["values"])
	if isinstance(spec, list):
		return spec
	return [spec]


def extract_max_value(spec: Any) -> Any:
	"""
	Extract the maximum value from a parameter specification.

	Useful for GPU requirement estimation where we need the worst-case
	(maximum) value for parameters like tp-size, pp-size, etc.

	Args:
	    spec: Parameter specification

	Returns:
	    Maximum value from the specification

	Examples:
	    >>> extract_max_value({"values": [1, 2, 4]})
	    4

	    >>> extract_max_value([1, 2, 4])
	    4

	    >>> extract_max_value(8)
	    8
	"""
	values = normalize_choice_values(spec)
	if not values:
		return None

	# Handle numeric values
	try:
		return max(values)
	except TypeError:
		# Non-comparable types, return first value
		return values[0]


def extract_min_value(spec: Any) -> Any:
	"""
	Extract the minimum value from a parameter specification.

	Args:
	    spec: Parameter specification

	Returns:
	    Minimum value from the specification

	Examples:
	    >>> extract_min_value({"values": [1, 2, 4]})
	    1

	    >>> extract_min_value([1, 2, 4])
	    1

	    >>> extract_min_value(8)
	    8
	"""
	values = normalize_choice_values(spec)
	if not values:
		return None

	try:
		return min(values)
	except TypeError:
		return values[0]
