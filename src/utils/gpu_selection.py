"""
Shared GPU selection and validation utilities.

This module consolidates GPU selection logic that was previously
duplicated across gpu_monitor.py, gpu_pool.py, and controllers.
"""

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def validate_memory_balance(memory_free: List[int], min_ratio: float = 0.8) -> Tuple[bool, str]:
	"""
	Validate that GPU memory is balanced across multiple GPUs.

	For multi-GPU setups, ensures no GPU is significantly less loaded
	than others, which could indicate resource allocation issues.

	Args:
	    memory_free: List of free memory in MB for each GPU
	    min_ratio: Minimum ratio of (min_memory / max_memory) to accept

	Returns:
	    (is_valid, message) tuple:
	    - is_valid: True if memory is balanced or single GPU
	    - message: Description of the check result

	Examples:
	    >>> validate_memory_balance([8000, 8000, 8000])
	    (True, "balanced: min=8000 max=8000 ratio=1.00")

	    >>> validate_memory_balance([8000, 8000, 2000])
	    (False, "unbalanced: min=2000 max=8000 ratio=0.25")

	    >>> validate_memory_balance([8000])
	    (True, "single GPU")
	"""
	if not memory_free:
		return False, "no memory info"
	if len(memory_free) == 1:
		return True, "single GPU"

	min_mem = min(memory_free)
	max_mem = max(memory_free)

	if max_mem == 0:
		return False, "max memory is zero"

	ratio = min_mem / max_mem

	if ratio < min_ratio:
		return False, f"unbalanced: min={min_mem} max={max_mem} ratio={ratio:.2f}"

	return True, f"balanced: min={min_mem} max={max_mem} ratio={ratio:.2f}"


def filter_gpus_by_memory(
	gpus: List, min_memory_mb: int, require_balance: bool = True, min_balance_ratio: float = 0.8
) -> Tuple[List, Optional[str]]:
	"""
	Filter GPUs by memory requirements and optional balance check.

	Args:
	    gpus: List of GPU info objects (must have memory_free_mb attribute)
	    min_memory_mb: Minimum free memory required per GPU
	    require_balance: Whether to check memory balance
	    min_balance_ratio: Minimum memory balance ratio (if checking)

	Returns:
	    (filtered_gpus, warning_message) tuple:
	    - filtered_gpus: GPUs meeting memory requirement
	    - warning_message: None if OK, warning string if balance issue

	Example:
	    >>> from gpu_types import LocalGPUInfo
	    >>> gpus = [
	    ...     LocalGPUInfo(0, "uuid1", "A100", 40000, 35000, 5000, 10, 20),
	    ...     LocalGPUInfo(1, "uuid2", "A100", 40000, 10000, 30000, 50, 60)
	    ... ]
	    >>> filtered, warning = filter_gpus_by_memory(gpus, 8000)
	    >>> len(filtered)
	    2
	    >>> warning
	    'Memory imbalance: min=10000 max=35000 ratio=0.29'
	"""
	# Filter by minimum memory
	suitable = [g for g in gpus if g.memory_free_mb >= min_memory_mb]

	if not suitable:
		return [], None

	# Check balance if required and multiple GPUs
	warning = None
	if require_balance and len(suitable) > 1:
		memory_values = [g.memory_free_mb for g in suitable]
		is_balanced, msg = validate_memory_balance(memory_values, min_balance_ratio)

		if not is_balanced:
			warning = f"Memory imbalance: {msg}"
			logger.warning(warning)

	return suitable, warning


def sort_gpus_by_score(gpus: List) -> List:
	"""
	Sort GPUs by selection score (highest first).

	Args:
	    gpus: List of GPU info objects (must have score property)

	Returns:
	    Sorted list of GPUs (best candidates first)
	"""
	return sorted(gpus, key=lambda g: g.score, reverse=True)


def select_best_gpus(
	gpus: List, count: int, min_memory_mb: int = 0, require_balance: bool = True, exclude_indices: Optional[set] = None
) -> Tuple[List, Optional[str]]:
	"""
	Select the best N GPUs from available options.

	Combines filtering, scoring, and selection logic that was previously
	duplicated across multiple modules.

	Args:
	    gpus: List of GPU info objects
	    count: Number of GPUs to select
	    min_memory_mb: Minimum free memory required per GPU
	    require_balance: Whether to check memory balance
	    exclude_indices: Set of GPU indices to exclude from selection

	Returns:
	    (selected_gpus, warning_message) tuple:
	    - selected_gpus: List of selected GPU objects (may be empty)
	    - warning_message: None if OK, warning string if issues

	Example:
	    >>> gpus = get_all_gpus()
	    >>> selected, warning = select_best_gpus(gpus, 2, min_memory_mb=8000)
	    >>> if warning:
	    ...     print(f"Warning: {warning}")
	    >>> gpu_indices = [g.index for g in selected]
	"""
	# Apply exclusion filter
	if exclude_indices:
		available = [g for g in gpus if g.index not in exclude_indices]
	else:
		available = gpus

	if not available:
		return [], "No available GPUs after exclusion"

	# Filter by memory
	suitable, balance_warning = filter_gpus_by_memory(available, min_memory_mb, require_balance=require_balance)

	if not suitable:
		return [], f"No GPUs with >= {min_memory_mb}MB free memory"

	if len(suitable) < count:
		warning = f"Only {len(suitable)} GPUs available, need {count}"
		logger.warning(warning)
		return suitable, warning

	# Sort by score and take top N
	sorted_gpus = sort_gpus_by_score(suitable)
	selected = sorted_gpus[:count]

	return selected, balance_warning


def format_gpu_allocation(gpus: List) -> str:
	"""
	Format GPU allocation for logging/display.

	Args:
	    gpus: List of allocated GPU info objects

	Returns:
	    Human-readable string describing allocation

	Example:
	    >>> gpus = [gpu0, gpu1]
	    >>> print(format_gpu_allocation(gpus))
	    "GPUs [0, 1]: NVIDIA A100 (35GB free), NVIDIA A100 (34GB free)"
	"""
	if not gpus:
		return "No GPUs allocated"

	indices = [g.index for g in gpus]
	details = [f"{g.name} ({g.memory_free_mb//1024}GB free)" for g in gpus]

	return f"GPUs {indices}: {', '.join(details)}"
