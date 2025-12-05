"""
Database query tools for agent.

These tools allow the agent to query autotuning tasks, experiments, and parameter presets.
All tools are SAFE (no authorization required) as they only perform read-only operations.

NOTE: These tools use shared Service layer (web.services) to avoid code duplication
with REST API routes.
"""

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from web.services import TaskService, ExperimentService, PresetService
from web.agent.tools.base import register_tool, ToolCategory
import json
from typing import Optional


@tool
@register_tool(ToolCategory.DATABASE)
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 10,
    db: AsyncSession = None
) -> str:
    """
    List autotuning tasks with optional status filter.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled). Leave empty for all.
        limit: Maximum number of tasks to return (default 10, max 50)

    Returns:
        JSON string with task list containing id, name, status, created_at, total_experiments
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    limit = min(limit, 50)  # Cap at 50

    # Use TaskService for business logic
    tasks = await TaskService.list_tasks(db, status=status, skip=0, limit=limit)

    # Use model's to_dict() for consistent serialization
    return json.dumps([t.to_dict() for t in tasks], indent=2)


@tool
@register_tool(ToolCategory.DATABASE)
async def get_task_details(task_id: int, db: AsyncSession = None) -> str:
    """
    Get detailed information about a specific task.

    Args:
        task_id: Task ID to query

    Returns:
        JSON string with complete task configuration and results
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Use TaskService for business logic
    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Use model's to_dict() with full config
    return json.dumps(task.to_dict(include_full_config=True), indent=2)


@tool
@register_tool(ToolCategory.DATABASE)
async def get_task_experiments(
    task_id: int,
    status: Optional[str] = None,
    limit: int = 20,
    db: AsyncSession = None
) -> str:
    """
    Get experiments for a task with optional status filter.

    Args:
        task_id: Task ID to query
        status: Filter by experiment status (pending, deploying, benchmarking, success, failed). Leave empty for all.
        limit: Maximum number of experiments to return (default 20, max 100)

    Returns:
        JSON string with experiment list including parameters and metrics
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    limit = min(limit, 100)  # Cap at 100

    # Use ExperimentService for business logic
    experiments = await ExperimentService.list_experiments(
        db, task_id=task_id, status=status, skip=0, limit=limit
    )

    # Use model's to_dict() for consistent serialization
    return json.dumps([exp.to_dict() for exp in experiments], indent=2)


@tool
@register_tool(ToolCategory.DATABASE)
async def get_experiment_metrics(experiment_id: int, db: AsyncSession = None) -> str:
    """
    Get detailed metrics for a specific experiment.

    Args:
        experiment_id: Experiment ID to query

    Returns:
        JSON string with experiment metrics including latency, throughput, TTFT, TPOT, etc.
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Use ExperimentService for business logic
    exp = await ExperimentService.get_experiment_by_id(db, experiment_id)
    if not exp:
        return json.dumps({"error": f"Experiment {experiment_id} not found"})

    # Use model's to_dict() with logs (truncated)
    result = exp.to_dict(include_logs=True)
    # Truncate logs to first 500 chars
    if result.get("benchmark_logs"):
        result["benchmark_logs"] = result["benchmark_logs"][:500]

    return json.dumps(result, indent=2)


@tool
@register_tool(ToolCategory.DATABASE)
async def get_best_experiment(task_id: int, db: AsyncSession = None) -> str:
    """
    Get the best performing experiment for a task.

    Args:
        task_id: Task ID to query

    Returns:
        JSON string with best experiment details including parameters and metrics
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Use ExperimentService for business logic
    best_exp = await ExperimentService.get_best_experiment(db, task_id)
    if not best_exp:
        return json.dumps({"error": "No best experiment found for this task"})

    # Use model's to_dict() for consistent serialization
    return json.dumps(best_exp.to_dict(), indent=2)


@tool
@register_tool(ToolCategory.DATABASE)
async def search_presets(
    category: Optional[str] = None,
    runtime: Optional[str] = None,
    limit: int = 10,
    db: AsyncSession = None
) -> str:
    """
    Search parameter presets by category and runtime.

    Args:
        category: Filter by category (performance, memory, latency, throughput). Leave empty for all.
        runtime: Filter by runtime (sglang, vllm). Leave empty for all.
        limit: Maximum number of presets to return (default 10, max 50)

    Returns:
        JSON string with matching presets including parameters
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    limit = min(limit, 50)  # Cap at 50

    # Use PresetService for business logic
    presets = await PresetService.list_presets(
        db, category=category, runtime=runtime, skip=0, limit=limit
    )

    # Use model's to_dict() for consistent serialization
    return json.dumps([p.to_dict() for p in presets], indent=2)
