"""
Experiment service - Business logic for experiment operations.

Shared between REST API routes and agent tools.
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from web.db.models import Experiment, ExperimentStatus


class ExperimentService:
	"""Service for experiment-related business logic."""

	@staticmethod
	async def list_experiments(
		db: AsyncSession,
		task_id: Optional[int] = None,
		status: Optional[str] = None,
		skip: int = 0,
		limit: int = 100
	) -> List[Experiment]:
		"""
		List experiments with optional filtering.

		Args:
			db: Database session
			task_id: Optional task ID filter
			status: Optional status filter
			skip: Number of records to skip
			limit: Maximum number of records to return

		Returns:
			List of Experiment objects
		"""
		query = select(Experiment).order_by(Experiment.created_at.desc())

		if task_id is not None:
			query = query.where(Experiment.task_id == task_id)

		if status:
			query = query.where(Experiment.status == status.upper())

		query = query.offset(skip).limit(limit)
		result = await db.execute(query)
		return result.scalars().all()

	@staticmethod
	async def get_experiment_by_id(db: AsyncSession, experiment_id: int) -> Optional[Experiment]:
		"""
		Get experiment by ID.

		Args:
			db: Database session
			experiment_id: Experiment ID

		Returns:
			Experiment object or None if not found
		"""
		result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
		return result.scalar_one_or_none()

	@staticmethod
	async def get_best_experiment(db: AsyncSession, task_id: int) -> Optional[Experiment]:
		"""
		Get best experiment for a task.

		Args:
			db: Database session
			task_id: Task ID

		Returns:
			Best experiment or None
		"""
		from web.db.models import Task
		
		# Get task to find best_experiment_id
		result = await db.execute(select(Task).where(Task.id == task_id))
		task = result.scalar_one_or_none()
		
		if not task or not task.best_experiment_id:
			return None
		
		return await ExperimentService.get_experiment_by_id(db, task.best_experiment_id)

	@staticmethod
	async def create_experiment(db: AsyncSession, experiment: Experiment) -> Experiment:
		"""
		Create a new experiment.

		Args:
			db: Database session
			experiment: Experiment object to create

		Returns:
			Created experiment with ID assigned
		"""
		db.add(experiment)
		await db.commit()
		await db.refresh(experiment)
		return experiment

	@staticmethod
	async def update_experiment(db: AsyncSession, experiment: Experiment) -> Experiment:
		"""
		Update an existing experiment.

		Args:
			db: Database session
			experiment: Experiment object with updates

		Returns:
			Updated experiment
		"""
		await db.commit()
		await db.refresh(experiment)
		return experiment

	@staticmethod
	async def delete_experiment(db: AsyncSession, experiment_id: int) -> bool:
		"""
		Delete an experiment.

		Args:
			db: Database session
			experiment_id: Experiment ID to delete

		Returns:
			True if deleted, False if not found
		"""
		experiment = await ExperimentService.get_experiment_by_id(db, experiment_id)
		if not experiment:
			return False

		await db.delete(experiment)
		await db.commit()
		return True
