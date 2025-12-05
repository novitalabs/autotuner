"""
High-level task management tools for agent.

These tools provide business-friendly operations for managing autotuning tasks.
They wrap REST API logic and do NOT require authorization (safe business operations).

All tools use shared Service layer (web.services) for consistency with REST API.
"""

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from web.services import TaskService
from web.agent.tools.base import register_tool, ToolCategory
from web.db.models import Task, TaskStatus, Experiment
from web.schemas import TaskCreate, TaskUpdate
from datetime import datetime
import json
from typing import Optional


@tool
@register_tool(ToolCategory.TASK)
async def create_task(
    task_name: str,
    description: str,
    model_id: str,
    model_namespace: str,
    base_runtime: str,
    parameters: dict,
    optimization_strategy: str = "grid_search",
    optimization_objective: str = "minimize_latency",
    benchmark_task: str = "text-to-text",
    traffic_scenarios: list = None,
    db: AsyncSession = None
) -> str:
    """
    Create a new autotuning task.

    Args:
        task_name: Unique task name
        description: Task description
        model_id: Model ID or path (e.g., "llama-3-2-1b-instruct")
        model_namespace: Namespace for deployment (K8s namespace or Docker label)
        base_runtime: Runtime engine ("sglang" or "vllm")
        parameters: Parameter grid as dict, e.g., {"tp-size": [1, 2], "mem-fraction-static": [0.8, 0.9]}
        optimization_strategy: Optimization strategy ("grid_search", "random_search", "bayesian")
        optimization_objective: Objective to optimize ("minimize_latency", "maximize_throughput", etc.)
        benchmark_task: Benchmark task type (default: "text-to-text")
        traffic_scenarios: Traffic patterns (default: ["D(100,100)"])

    Returns:
        JSON string with created task details including ID
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    if traffic_scenarios is None:
        traffic_scenarios = ["D(100,100)"]

    # Check if task name already exists
    result = await db.execute(select(Task).where(Task.task_name == task_name))
    existing_task = result.scalar_one_or_none()

    if existing_task:
        return json.dumps({"error": f"Task '{task_name}' already exists"})

    # Create task
    db_task = Task(
        task_name=task_name,
        description=description,
        model_config={"id_or_path": model_id, "namespace": model_namespace},
        base_runtime=base_runtime,
        parameters=parameters,
        optimization_config={
            "strategy": optimization_strategy,
            "objective": optimization_objective
        },
        benchmark_config={
            "task": benchmark_task,
            "model_name": model_id,
            "model_tokenizer": model_id,
            "traffic_scenarios": traffic_scenarios
        },
        status=TaskStatus.PENDING,
        deployment_mode="docker"  # Default to docker mode
    )

    db.add(db_task)
    await db.commit()
    await db.refresh(db_task)

    return json.dumps({
        "success": True,
        "task": db_task.to_dict(include_full_config=True)
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def start_task(task_id: int, db: AsyncSession = None) -> str:
    """
    Start a pending autotuning task.

    Args:
        task_id: Task ID to start

    Returns:
        JSON string with updated task status
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    if task.status != TaskStatus.PENDING:
        return json.dumps({"error": f"Task must be in PENDING status to start. Current status: {task.status}"})

    # Update status to RUNNING
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.utcnow()

    await db.commit()
    await db.refresh(task)

    # Enqueue ARQ job
    from web.workers import enqueue_autotuning_task
    job_id = await enqueue_autotuning_task(task.id)

    return json.dumps({
        "success": True,
        "message": f"Task {task_id} started successfully",
        "job_id": job_id,
        "task": task.to_dict(include_full_config=True)
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def cancel_task(task_id: int, db: AsyncSession = None) -> str:
    """
    Cancel a running task.

    Args:
        task_id: Task ID to cancel

    Returns:
        JSON string with updated task status
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    if task.status != TaskStatus.RUNNING:
        return json.dumps({"error": f"Task is not running. Current status: {task.status}"})

    task.status = TaskStatus.CANCELLED
    await db.commit()
    await db.refresh(task)

    return json.dumps({
        "success": True,
        "message": f"Task {task_id} cancelled successfully",
        "task": task.to_dict(include_full_config=True)
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def restart_task(task_id: int, db: AsyncSession = None) -> str:
    """
    Restart a completed, failed, or cancelled task.

    This will delete old experiments and start the task fresh.

    Args:
        task_id: Task ID to restart

    Returns:
        JSON string with updated task status and job ID
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Only allow restart for completed, failed, or cancelled tasks
    if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        return json.dumps({
            "error": f"Task must be completed, failed, or cancelled to restart. Current status: {task.status}"
        })

    # Delete old experiments from previous runs
    await db.execute(delete(Experiment).where(Experiment.task_id == task_id))

    # Reset task fields
    task.completed_at = None
    task.elapsed_time = None
    task.total_experiments = 0
    task.successful_experiments = 0
    task.best_experiment_id = None
    task.task_metadata = None

    # Set status to RUNNING and start immediately
    task.status = TaskStatus.RUNNING
    task.started_at = datetime.utcnow()

    await db.commit()
    await db.refresh(task)

    # Enqueue ARQ job
    from web.workers import enqueue_autotuning_task
    job_id = await enqueue_autotuning_task(task.id)

    return json.dumps({
        "success": True,
        "message": f"Task {task_id} restarted successfully",
        "job_id": job_id,
        "task": task.to_dict(include_full_config=True)
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def delete_task(task_id: int, db: AsyncSession = None) -> str:
    """
    Delete a task from database.

    Note: This does not clean up log files. Use clear_task_data() to clean logs.
    Cannot delete running tasks - cancel them first.

    Args:
        task_id: Task ID to delete

    Returns:
        JSON string with success message
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Don't allow deletion of running tasks
    if task.status == TaskStatus.RUNNING:
        return json.dumps({
            "error": "Cannot delete running task. Cancel it first."
        })

    # Delete task from database (cascades to experiments)
    await db.delete(task)
    await db.commit()

    return json.dumps({
        "success": True,
        "message": f"Task {task_id} deleted successfully"
    })


@tool
@register_tool(ToolCategory.TASK)
async def clear_task_data(task_id: int, db: AsyncSession = None) -> str:
    """
    Clear task experiments and logs without deleting the task configuration.

    This resets the task to PENDING status and removes all experimental data,
    allowing you to re-run the task with the same configuration.

    Args:
        task_id: Task ID to clear

    Returns:
        JSON string with success message and number of experiments deleted
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Don't allow clearing running tasks
    if task.status == TaskStatus.RUNNING:
        return json.dumps({
            "error": "Cannot clear running task. Cancel it first."
        })

    # Delete all experiments for this task
    delete_result = await db.execute(delete(Experiment).where(Experiment.task_id == task_id))
    experiments_deleted = delete_result.rowcount

    # Reset task fields
    task.completed_at = None
    task.elapsed_time = None
    task.started_at = None
    task.total_experiments = 0
    task.successful_experiments = 0
    task.best_experiment_id = None
    task.task_metadata = None
    task.status = TaskStatus.PENDING

    await db.commit()
    await db.refresh(task)

    # Clear log file (delete it entirely)
    from pathlib import Path
    log_dir = Path.home() / ".local/share/inference-autotuner/logs"
    log_file = log_dir / f"task_{task_id}.log"
    if log_file.exists():
        try:
            log_file.unlink()
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to delete log file: {str(e)}",
                "experiments_deleted": experiments_deleted
            })

    return json.dumps({
        "success": True,
        "message": f"Task {task_id} data cleared successfully",
        "experiments_deleted": experiments_deleted,
        "task": task.to_dict(include_full_config=True)
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def update_task_description(
    task_id: int,
    description: str,
    db: AsyncSession = None
) -> str:
    """
    Update task description.

    Args:
        task_id: Task ID to update
        description: New description

    Returns:
        JSON string with updated task
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    task.description = description
    await db.commit()
    await db.refresh(task)

    return json.dumps({
        "success": True,
        "message": f"Task {task_id} description updated",
        "task": task.to_dict(include_full_config=True)
    }, indent=2)
