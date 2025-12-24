"""
Tests for database utility functions.
"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.web.services.db_utils import (
    create_and_refresh,
    commit_and_refresh,
    create_many_and_commit,
)
from src.web.db.models import Task, Experiment, TaskStatus, ExperimentStatus, Base


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def async_db_session():
    """Create an async database session for testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_maker = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as session:
        yield session

    await engine.dispose()


@pytest.mark.asyncio
async def test_create_and_refresh(async_db_session):
    """Test create_and_refresh creates and returns refreshed object."""
    task = Task(
        task_name="test-task",
        description="Test task",
        model_id="test-model",
        base_runtime="vllm",
        status=TaskStatus.PENDING,
    )

    result = await create_and_refresh(async_db_session, task)

    assert result.id is not None
    assert result.task_name == "test-task"
    assert result.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_commit_and_refresh(async_db_session):
    """Test commit_and_refresh updates and refreshes object."""
    task = Task(
        task_name="test-task",
        description="Test task",
        model_id="test-model",
        base_runtime="vllm",
        status=TaskStatus.PENDING,
    )
    await create_and_refresh(async_db_session, task)

    task.status = TaskStatus.RUNNING
    task.description = "Updated description"

    result = await commit_and_refresh(async_db_session, task)

    assert result.status == TaskStatus.RUNNING
    assert result.description == "Updated description"


@pytest.mark.asyncio
async def test_create_many_and_commit(async_db_session):
    """Test create_many_and_commit creates multiple objects."""
    task = Task(
        task_name="test-task",
        description="Test task",
        model_id="test-model",
        base_runtime="vllm",
        status=TaskStatus.PENDING,
    )
    await create_and_refresh(async_db_session, task)

    exp1 = Experiment(
        task_id=task.id,
        experiment_number=1,
        parameters={"param1": "value1"},
        status=ExperimentStatus.PENDING,
    )
    exp2 = Experiment(
        task_id=task.id,
        experiment_number=2,
        parameters={"param2": "value2"},
        status=ExperimentStatus.PENDING,
    )

    results = await create_many_and_commit(async_db_session, exp1, exp2)

    assert len(results) == 2
    assert all(isinstance(r, Experiment) for r in results)
