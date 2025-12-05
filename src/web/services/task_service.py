"""
Task service - Business logic for task operations.

Shared between REST API routes and agent tools.
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from web.db.models import Task, TaskStatus


class TaskService:
	"""Service for task-related business logic."""

	@staticmethod
	async def list_tasks(
		db: AsyncSession,
		status: Optional[str] = None,
		skip: int = 0,
		limit: int = 100
	) -> List[Task]:
		"""
		List tasks with optional filtering.

		Args:
			db: Database session
			status: Optional status filter (pending, running, completed, failed, cancelled)
			skip: Number of records to skip
			limit: Maximum number of records to return

		Returns:
			List of Task objects
		"""
		query = select(Task).order_by(Task.created_at.desc())

		if status:
			query = query.where(Task.status == status.upper())

		query = query.offset(skip).limit(limit)
		result = await db.execute(query)
		return result.scalars().all()

	@staticmethod
	async def get_task_by_id(db: AsyncSession, task_id: int) -> Optional[Task]:
		"""
		Get task by ID.

		Args:
			db: Database session
			task_id: Task ID

		Returns:
			Task object or None if not found
		"""
		result = await db.execute(select(Task).where(Task.id == task_id))
		return result.scalar_one_or_none()

	@staticmethod
	async def get_task_by_name(db: AsyncSession, task_name: str) -> Optional[Task]:
		"""
		Get task by name.

		Args:
			db: Database session
			task_name: Task name

		Returns:
			Task object or None if not found
		"""
		result = await db.execute(select(Task).where(Task.task_name == task_name))
		return result.scalar_one_or_none()

	@staticmethod
	async def create_task(db: AsyncSession, task: Task) -> Task:
		"""
		Create a new task.

		Args:
			db: Database session
			task: Task object to create

		Returns:
			Created task with ID assigned
		"""
		db.add(task)
		await db.commit()
		await db.refresh(task)
		return task

	@staticmethod
	async def update_task(db: AsyncSession, task: Task) -> Task:
		"""
		Update an existing task.

		Args:
			db: Database session
			task: Task object with updates

		Returns:
			Updated task
		"""
		await db.commit()
		await db.refresh(task)
		return task

	@staticmethod
	async def delete_task(db: AsyncSession, task_id: int) -> bool:
		"""
		Delete a task.

		Args:
			db: Database session
			task_id: Task ID to delete

		Returns:
			True if deleted, False if not found
		"""
		task = await TaskService.get_task_by_id(db, task_id)
		if not task:
			return False

		await db.delete(task)
		await db.commit()
		return True

	@staticmethod
	async def task_exists(db: AsyncSession, task_name: str) -> bool:
		"""
		Check if task with given name exists.

		Args:
			db: Database session
			task_name: Task name to check

		Returns:
			True if exists, False otherwise
		"""
		task = await TaskService.get_task_by_name(db, task_name)
		return task is not None
