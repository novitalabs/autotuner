"""
FastAPI dependency injection helpers for common route operations.

This module provides reusable dependencies that eliminate repeated patterns
in route handlers, particularly get-by-id-or-404 operations.
"""

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from web.db.session import get_db
from web.db.models import Task, Experiment


async def get_task_or_404(
    task_id: int,
    db: AsyncSession = Depends(get_db)
) -> Task:
    """
    Get a task by ID or raise 404.

    Args:
        task_id: Task ID to fetch
        db: Database session (injected)

    Returns:
        Task model instance

    Raises:
        HTTPException: 404 if task not found

    Example:
        @router.get("/{task_id}")
        async def get_task(task: Task = Depends(get_task_or_404)):
            return TaskResponse.model_validate(task)
    """
    result = await db.execute(select(Task).where(Task.id == task_id))
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )

    return task


async def get_experiment_or_404(
    experiment_id: int,
    db: AsyncSession = Depends(get_db)
) -> Experiment:
    """
    Get an experiment by ID or raise 404.

    Args:
        experiment_id: Experiment ID to fetch
        db: Database session (injected)

    Returns:
        Experiment model instance

    Raises:
        HTTPException: 404 if experiment not found

    Example:
        @router.get("/{experiment_id}")
        async def get_exp(exp: Experiment = Depends(get_experiment_or_404)):
            return ExperimentResponse.model_validate(exp)
    """
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if not experiment:
        raise HTTPException(
            status_code=404,
            detail=f"Experiment {experiment_id} not found"
        )

    return experiment
