"""
Checkpoint management for long-running autotuning tasks.

Enables saving and restoring task progress to handle timeouts and interruptions.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone


class TaskCheckpoint:
	"""Manages task progress checkpoints."""

	@staticmethod
	def save_checkpoint(
		task_metadata: Optional[Dict[str, Any]],
		iteration: int,
		best_score: float,
		best_experiment_id: Optional[int],
		strategy_state: Dict[str, Any],
	) -> Dict[str, Any]:
		"""Save task progress checkpoint to metadata.

		Args:
		    task_metadata: Current task metadata dict (can be None)
		    iteration: Current iteration number
		    best_score: Best objective score so far
		    best_experiment_id: ID of best experiment
		    strategy_state: Optimization strategy state

		Returns:
		    Updated metadata dict with checkpoint
		"""
		if task_metadata is None:
			task_metadata = {}

		task_metadata["checkpoint"] = {
			"iteration": iteration,
			"best_score": best_score,
			"best_experiment_id": best_experiment_id,
			"strategy_state": strategy_state,
			"timestamp": datetime.now(timezone.utc).isoformat(),
		}

		return task_metadata

	@staticmethod
	def load_checkpoint(task_metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
		"""Load task progress checkpoint from metadata.

		Args:
		    task_metadata: Task metadata dict

		Returns:
		    Checkpoint dict if exists, None otherwise
		"""
		if not task_metadata:
			return None

		checkpoint = task_metadata.get("checkpoint")
		if not checkpoint:
			return None

		# Validate checkpoint structure
		required_fields = ["iteration", "best_score", "strategy_state"]
		if not all(field in checkpoint for field in required_fields):
			return None

		return checkpoint

	@staticmethod
	def clear_checkpoint(task_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
		"""Clear checkpoint from metadata after task completion.

		Args:
		    task_metadata: Current task metadata dict

		Returns:
		    Updated metadata dict without checkpoint
		"""
		if not task_metadata:
			return {}

		if "checkpoint" in task_metadata:
			del task_metadata["checkpoint"]

		return task_metadata

	@staticmethod
	def should_resume(task_status: str, task_metadata: Optional[Dict[str, Any]]) -> bool:
		"""Check if task should be resumed from checkpoint.

		Args:
		    task_status: Current task status
		    task_metadata: Task metadata dict

		Returns:
		    True if task should resume, False otherwise
		"""
		# Only resume if task is in RUNNING or PENDING state with checkpoint
		if task_status not in ["RUNNING", "PENDING"]:
			return False

		checkpoint = TaskCheckpoint.load_checkpoint(task_metadata)
		return checkpoint is not None
