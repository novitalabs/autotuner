"""
Database utility functions for common operations.

This module provides reusable helpers that eliminate repeated patterns
in service layer code, particularly for create/commit/refresh operations.
"""

from typing import TypeVar
from sqlalchemy.ext.asyncio import AsyncSession

T = TypeVar("T")


async def create_and_refresh(db: AsyncSession, obj: T) -> T:
    """
    Add object to session, commit, and refresh.

    Args:
        db: Database session
        obj: SQLAlchemy model instance to create

    Returns:
        The created object with refreshed state from database

    Example:
        task = Task(task_name="new-task", ...)
        task = await create_and_refresh(db, task)
    """
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def commit_and_refresh(db: AsyncSession, obj: T) -> T:
    """
    Commit changes and refresh object state.

    Use this after modifying an already-attached object.

    Args:
        db: Database session
        obj: SQLAlchemy model instance to update

    Returns:
        The object with refreshed state from database

    Example:
        task.status = TaskStatus.COMPLETED
        task = await commit_and_refresh(db, task)
    """
    await db.commit()
    await db.refresh(obj)
    return obj


async def create_many_and_commit(db: AsyncSession, *objs: T) -> list[T]:
    """
    Add multiple objects and commit in one transaction.

    Args:
        db: Database session
        *objs: Variable number of SQLAlchemy model instances

    Returns:
        List of created objects (not refreshed)

    Example:
        exp1 = Experiment(...)
        exp2 = Experiment(...)
        experiments = await create_many_and_commit(db, exp1, exp2)
    """
    for obj in objs:
        db.add(obj)
    await db.commit()
    return list(objs)
