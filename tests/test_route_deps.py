"""
Tests for route dependency injection helpers.
"""

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.web.routes.deps import get_task_or_404, get_experiment_or_404
from src.web.db.models import Task, TaskStatus, Experiment


@pytest.mark.asyncio
async def test_get_task_or_404_success(db_session: AsyncSession):
    """Test get_task_or_404 returns task when it exists."""
    # Create a task
    task = Task(
        task_name="test-task",
        description="Test task",
        model_config={"id_or_path": "test-model"},
        base_runtime="vllm",
        parameters={},
        optimization_config={},
        benchmark_config={},
        status=TaskStatus.PENDING
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # Test get_task_or_404
    result = await get_task_or_404(task.id, db_session)
    assert result.id == task.id
    assert result.task_name == "test-task"


@pytest.mark.asyncio
async def test_get_task_or_404_not_found(db_session: AsyncSession):
    """Test get_task_or_404 raises 404 when task doesn't exist."""
    with pytest.raises(HTTPException) as exc_info:
        await get_task_or_404(99999, db_session)

    assert exc_info.value.status_code == 404
    assert "Task 99999 not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_get_experiment_or_404_success(db_session: AsyncSession):
    """Test get_experiment_or_404 returns experiment when it exists."""
    # Create a task first
    task = Task(
        task_name="test-task",
        description="Test task",
        model_config={"id_or_path": "test-model"},
        base_runtime="vllm",
        parameters={},
        optimization_config={},
        benchmark_config={},
        status=TaskStatus.PENDING
    )
    db_session.add(task)
    await db_session.commit()
    await db_session.refresh(task)

    # Create an experiment
    experiment = Experiment(
        task_id=task.id,
        experiment_id=1,
        parameters={"tp-size": 1},
        status="pending"
    )
    db_session.add(experiment)
    await db_session.commit()
    await db_session.refresh(experiment)

    # Test get_experiment_or_404
    result = await get_experiment_or_404(experiment.id, db_session)
    assert result.id == experiment.id
    assert result.task_id == task.id
    assert result.experiment_id == 1


@pytest.mark.asyncio
async def test_get_experiment_or_404_not_found(db_session: AsyncSession):
    """Test get_experiment_or_404 raises 404 when experiment doesn't exist."""
    with pytest.raises(HTTPException) as exc_info:
        await get_experiment_or_404(99999, db_session)

    assert exc_info.value.status_code == 404
    assert "Experiment 99999 not found" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_multiple_tasks(db_session: AsyncSession):
    """Test get_task_or_404 with multiple tasks."""
    # Create multiple tasks
    task1 = Task(
        task_name="task-1",
        description="First task",
        model_config={"id_or_path": "model-1"},
        base_runtime="vllm",
        parameters={},
        optimization_config={},
        benchmark_config={},
        status=TaskStatus.PENDING
    )
    task2 = Task(
        task_name="task-2",
        description="Second task",
        model_config={"id_or_path": "model-2"},
        base_runtime="sglang",
        parameters={},
        optimization_config={},
        benchmark_config={},
        status=TaskStatus.RUNNING
    )
    db_session.add(task1)
    db_session.add(task2)
    await db_session.commit()
    await db_session.refresh(task1)
    await db_session.refresh(task2)

    # Fetch each task
    result1 = await get_task_or_404(task1.id, db_session)
    result2 = await get_task_or_404(task2.id, db_session)

    assert result1.task_name == "task-1"
    assert result1.base_runtime == "vllm"
    assert result2.task_name == "task-2"
    assert result2.base_runtime == "sglang"
