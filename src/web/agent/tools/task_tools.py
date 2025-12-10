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


# ============================================================================
# QUERY TOOLS - Read-only operations for task information
# ============================================================================

@tool
@register_tool(ToolCategory.TASK)
async def list_tasks(
    skip: int = 0,
    limit: int = 20,
    status_filter: Optional[str] = None,
    db: AsyncSession = None
) -> str:
    """
    List all autotuning tasks with optional status filtering.

    Use this instead of query_records(table_name="tasks") for better formatting and error handling.

    Args:
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 20, max: 100)
        status_filter: Optional status filter - one of: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED

    Returns:
        JSON string with list of tasks including ID, name, status, created time, and brief config summary
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Validate and cap limit
    if limit > 100:
        limit = 100

    # Validate status filter if provided
    if status_filter:
        valid_statuses = ["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
        status_upper = status_filter.upper()
        if status_upper not in valid_statuses:
            return json.dumps({
                "error": f"Invalid status '{status_filter}'. Must be one of: {', '.join(valid_statuses)}"
            })
        status_filter = status_upper

    # Use TaskService for consistency
    tasks = await TaskService.list_tasks(db, status=status_filter, skip=skip, limit=limit)

    # Format tasks for agent display
    task_list = []
    for task in tasks:
        task_info = {
            "id": task.id,
            "name": task.task_name,
            "status": task.status.value if hasattr(task.status, 'value') else task.status,
            "created_at": task.created_at.isoformat() if task.created_at else None,
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "description": task.description,
            "model": task.model_config.get("id_or_path") if task.model_config else None,
            "runtime": task.base_runtime,
            "optimization_strategy": task.optimization_config.get("strategy") if task.optimization_config else None,
        }
        task_list.append(task_info)

    return json.dumps({
        "success": True,
        "count": len(task_list),
        "skip": skip,
        "limit": limit,
        "status_filter": status_filter,
        "tasks": task_list
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def get_task_by_id(
    task_id: int,
    db: AsyncSession = None
) -> str:
    """
    Get detailed information about a specific task by ID.

    Use this instead of query_records() when you need full task configuration and status.

    Args:
        task_id: Task ID to retrieve

    Returns:
        JSON string with complete task details including full configuration, status, timing, and experiment count
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    task = await TaskService.get_task_by_id(db, task_id)

    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Get experiment count for this task
    from web.services import ExperimentService
    experiments = await ExperimentService.list_experiments(db, task_id=task_id)

    return json.dumps({
        "success": True,
        "task": task.to_dict(include_full_config=True),
        "experiment_count": len(experiments)
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def get_task_by_name(
    task_name: str,
    db: AsyncSession = None
) -> str:
    """
    Get detailed information about a specific task by name.

    Useful when user refers to task by its name instead of ID.

    Args:
        task_name: Unique task name to retrieve

    Returns:
        JSON string with complete task details including full configuration, status, timing, and experiment count
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    task = await TaskService.get_task_by_name(db, task_name)

    if not task:
        return json.dumps({"error": f"Task '{task_name}' not found"})

    # Get experiment count for this task
    from web.services import ExperimentService
    experiments = await ExperimentService.list_experiments(db, task_id=task.id)

    return json.dumps({
        "success": True,
        "task": task.to_dict(include_full_config=True),
        "experiment_count": len(experiments)
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def list_task_experiments(
    task_id: int,
    skip: int = 0,
    limit: int = 50,
    status_filter: Optional[str] = None,
    db: AsyncSession = None
) -> str:
    """
    List all experiments for a specific task.

    Shows parameter combinations tested and their results (metrics, scores).
    Use this to analyze task progress and find best configurations.

    Args:
        task_id: Task ID to get experiments for
        skip: Number of records to skip (default: 0)
        limit: Maximum number of records to return (default: 50, max: 200)
        status_filter: Optional status filter - one of: PENDING, DEPLOYING, RUNNING, SUCCESS, FAILED

    Returns:
        JSON string with list of experiments including parameters, metrics, and scores
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Verify task exists
    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Validate and cap limit
    if limit > 200:
        limit = 200

    # Validate status filter if provided
    if status_filter:
        valid_statuses = ["PENDING", "DEPLOYING", "RUNNING", "SUCCESS", "FAILED"]
        status_upper = status_filter.upper()
        if status_upper not in valid_statuses:
            return json.dumps({
                "error": f"Invalid status '{status_filter}'. Must be one of: {', '.join(valid_statuses)}"
            })
        status_filter = status_upper

    # Use ExperimentService for consistency
    from web.services import ExperimentService
    experiments = await ExperimentService.list_experiments(
        db,
        task_id=task_id,
        status=status_filter,
        skip=skip,
        limit=limit
    )

    # Format experiments for agent display
    exp_list = []
    for exp in experiments:
        exp_info = {
            "id": exp.id,
            "status": exp.status.value if hasattr(exp.status, 'value') else exp.status,
            "parameters": exp.parameters,
            "metrics": exp.metrics,
            "objective_score": exp.objective_score,
            "is_best": (exp.id == task.best_experiment_id),
            "created_at": exp.created_at.isoformat() if exp.created_at else None,
            "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
        }
        exp_list.append(exp_info)

    # Find best experiment
    best_exp = next((e for e in exp_list if e.get("is_best")), None)

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "task_name": task.task_name,
        "count": len(exp_list),
        "skip": skip,
        "limit": limit,
        "status_filter": status_filter,
        "best_experiment": best_exp,
        "experiments": exp_list
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def get_task_results(
    task_id: int,
    include_all_experiments: bool = False,
    db: AsyncSession = None
) -> str:
    """
    Get comprehensive results for a completed task.

    Returns the best experiment and optionally all experiments with detailed metrics.
    Use this to analyze task outcomes and compare parameter configurations.

    Args:
        task_id: Task ID to get results for
        include_all_experiments: If True, include all experiments (default: False, only returns best)

    Returns:
        JSON string with task status, best experiment, and optionally all experiments with full metrics
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Get task
    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Get experiments
    from web.services import ExperimentService
    experiments = await ExperimentService.list_experiments(db, task_id=task_id)

    # Find best experiment (task.best_experiment_id)
    best_exp = next((exp for exp in experiments if exp.id == task.best_experiment_id), None)

    result = {
        "success": True,
        "task_id": task_id,
        "task_name": task.task_name,
        "task_status": task.status.value if hasattr(task.status, 'value') else task.status,
        "total_experiments": task.total_experiments,
        "successful_experiments": task.successful_experiments,
        "optimization_strategy": task.optimization_config.get("strategy") if task.optimization_config else None,
        "optimization_objective": task.optimization_config.get("objective") if task.optimization_config else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "elapsed_time": task.elapsed_time,
    }

    # Add best experiment details
    if best_exp:
        result["best_experiment"] = {
            "id": best_exp.id,
            "parameters": best_exp.parameters,
            "metrics": best_exp.metrics,
            "objective_score": best_exp.objective_score,
            "completed_at": best_exp.completed_at.isoformat() if best_exp.completed_at else None,
        }
    else:
        result["best_experiment"] = None

    # Add all experiments if requested
    if include_all_experiments:
        exp_list = []
        for exp in experiments:
            exp_list.append({
                "id": exp.id,
                "status": exp.status.value if hasattr(exp.status, 'value') else exp.status,
                "parameters": exp.parameters,
                "metrics": exp.metrics,
                "objective_score": exp.objective_score,
                "is_best": (exp.id == task.best_experiment_id),
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
                "completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
            })
        result["all_experiments"] = exp_list
        result["experiment_count"] = len(exp_list)

    return json.dumps(result, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def get_task_logs(
    task_id: int,
    tail_lines: Optional[int] = None,
    db: AsyncSession = None
) -> str:
    """
    Retrieve execution logs for a task.

    Returns the log file content for debugging and monitoring task execution.
    Useful for diagnosing failures or understanding task progress.

    Args:
        task_id: Task ID to get logs for
        tail_lines: If specified, only return last N lines (default: None, returns all)

    Returns:
        JSON string with log content and metadata
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Verify task exists
    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Locate log file
    from pathlib import Path
    log_dir = Path.home() / ".local/share/inference-autotuner/logs"
    log_file = log_dir / f"task_{task_id}.log"

    if not log_file.exists():
        return json.dumps({
            "success": True,
            "task_id": task_id,
            "task_name": task.task_name,
            "log_exists": False,
            "message": "No log file found for this task",
            "log_path": str(log_file)
        })

    try:
        # Read log file
        log_content = log_file.read_text()

        # Get tail if requested
        if tail_lines and tail_lines > 0:
            lines = log_content.splitlines()
            log_content = "\n".join(lines[-tail_lines:])

        # Get file size
        file_size = log_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "task_name": task.task_name,
            "log_exists": True,
            "log_path": str(log_file),
            "file_size_mb": round(file_size_mb, 2),
            "tail_lines": tail_lines,
            "log_content": log_content
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to read log file: {str(e)}",
            "task_id": task_id,
            "log_path": str(log_file)
        })


@tool
@register_tool(ToolCategory.TASK)
async def get_experiment_logs(
    task_id: int,
    experiment_id: int,
    tail_lines: Optional[int] = None,
    db: AsyncSession = None
) -> str:
    """
    Retrieve detailed logs for a specific experiment.

    Each experiment has its own log file with detailed execution output including:
    - Container startup logs
    - Model loading progress
    - Benchmark execution details
    - Error messages and stack traces

    Args:
        task_id: Task ID the experiment belongs to
        experiment_id: Experiment ID (the number in the experiment sequence, e.g., 1, 2, 3...)
        tail_lines: If specified, only return last N lines (default: None, returns all)

    Returns:
        JSON string with experiment log content
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Verify task exists
    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Locate experiment log file
    from pathlib import Path
    log_dir = Path.home() / ".local/share/inference-autotuner/logs"
    log_file = log_dir / f"task_{task_id}_exp_{experiment_id}.log"

    # Also check for alternative naming patterns
    alt_patterns = [
        log_dir / f"task_{task_id}_exp_{experiment_id}.log",
        log_dir / f"task{task_id}_exp{experiment_id}.log",
    ]

    found_log = None
    for pattern in alt_patterns:
        if pattern.exists():
            found_log = pattern
            break

    if not found_log:
        # List available experiment logs for this task
        available_logs = list(log_dir.glob(f"task_{task_id}_exp_*.log"))
        available_exp_ids = []
        for log in available_logs:
            # Extract experiment ID from filename
            import re
            match = re.search(r'exp_(\d+)\.log$', log.name)
            if match:
                available_exp_ids.append(int(match.group(1)))
        available_exp_ids.sort()

        return json.dumps({
            "success": False,
            "task_id": task_id,
            "experiment_id": experiment_id,
            "log_exists": False,
            "message": f"No log file found for experiment {experiment_id}",
            "tried_paths": [str(p) for p in alt_patterns],
            "available_experiment_ids": available_exp_ids,
            "hint": f"Available experiment IDs for task {task_id}: {available_exp_ids}" if available_exp_ids else "No experiment logs found for this task"
        }, indent=2)

    try:
        # Read log file
        log_content = found_log.read_text()

        # Get tail if requested
        if tail_lines and tail_lines > 0:
            lines = log_content.splitlines()
            log_content = "\n".join(lines[-tail_lines:])

        # Get file size
        file_size = found_log.stat().st_size
        file_size_kb = file_size / 1024

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "task_name": task.task_name,
            "experiment_id": experiment_id,
            "log_exists": True,
            "log_path": str(found_log),
            "file_size_kb": round(file_size_kb, 2),
            "tail_lines": tail_lines,
            "log_content": log_content
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to read log file: {str(e)}",
            "task_id": task_id,
            "experiment_id": experiment_id,
            "log_path": str(found_log)
        })


@tool
@register_tool(ToolCategory.TASK)
async def get_experiment_details(
    experiment_id: int,
    db: AsyncSession = None
) -> str:
    """
    Get detailed information about a specific experiment.

    Returns full experiment configuration, metrics, and status.
    Use this to deep-dive into a specific parameter combination.

    Args:
        experiment_id: Experiment ID to retrieve

    Returns:
        JSON string with complete experiment details including parameters, metrics, and timing
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Get experiment
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    experiment = result.scalar_one_or_none()

    if not experiment:
        return json.dumps({"error": f"Experiment {experiment_id} not found"})

    # Get parent task
    task = await TaskService.get_task_by_id(db, experiment.task_id)

    return json.dumps({
        "success": True,
        "experiment": {
            "id": experiment.id,
            "task_id": experiment.task_id,
            "task_name": task.task_name if task else None,
            "status": experiment.status.value if hasattr(experiment.status, 'value') else experiment.status,
            "parameters": experiment.parameters,
            "metrics": experiment.metrics,
            "objective_score": experiment.objective_score,
            "is_best": (experiment.id == task.best_experiment_id) if task else False,
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "started_at": experiment.started_at.isoformat() if experiment.started_at else None,
            "completed_at": experiment.completed_at.isoformat() if experiment.completed_at else None,
            "elapsed_time": experiment.elapsed_time,
            "error_message": experiment.error_message,
        }
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def analyze_slo_violations(
    task_id: int,
    db: AsyncSession = None
) -> str:
    """
    Analyze SLO (Service Level Objective) violations for all experiments in a task.

    Returns detailed analysis of which experiments violated SLO constraints,
    which metrics were violated most frequently, and violation statistics.

    Args:
        task_id: Task ID to analyze

    Returns:
        JSON string with SLO violation analysis including:
        - Total violations count
        - Hard fail vs soft penalty breakdown
        - Most violated metrics
        - List of violating experiments with details
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Get task
    task = await TaskService.get_task_by_id(db, task_id)
    if not task:
        return json.dumps({"error": f"Task {task_id} not found"})

    # Get SLO config from task
    slo_config = task.task_config.get("slo", {})
    if not slo_config:
        return json.dumps({
            "success": True,
            "task_id": task_id,
            "has_slo": False,
            "message": "This task does not have SLO constraints configured"
        })

    # Get all experiments
    from web.services import ExperimentService
    experiments = await ExperimentService.list_experiments(db, task_id=task_id)

    if not experiments:
        return json.dumps({
            "success": True,
            "task_id": task_id,
            "has_slo": True,
            "message": "No experiments found for this task"
        })

    # Analyze violations
    violation_stats = {
        "total_experiments": len(experiments),
        "successful_experiments": 0,
        "failed_experiments": 0,
        "experiments_with_violations": 0,
        "hard_fail_count": 0,
        "soft_penalty_count": 0,
        "metric_violations": {},  # metric_name -> count
        "violating_experiments": []
    }

    for exp in experiments:
        if exp.status and hasattr(exp.status, 'value'):
            status = exp.status.value
        else:
            status = exp.status

        if status == "SUCCESS":
            violation_stats["successful_experiments"] += 1
        elif status == "FAILED":
            violation_stats["failed_experiments"] += 1

        metrics = exp.metrics or {}
        has_violation = False
        violated_metrics = []

        # Check TTFT violations
        if "ttft" in slo_config:
            ttft_slo = slo_config["ttft"]
            threshold = ttft_slo.get("threshold")
            if threshold and metrics.get("ttft") is not None:
                actual = metrics["ttft"]
                if actual > threshold:
                    has_violation = True
                    violation_ratio = (actual - threshold) / threshold
                    violated_metrics.append({
                        "metric": "ttft",
                        "threshold": threshold,
                        "actual": actual,
                        "violation_ratio": violation_ratio,
                        "hard_fail": ttft_slo.get("hard_fail", False)
                    })
                    violation_stats["metric_violations"]["ttft"] = violation_stats["metric_violations"].get("ttft", 0) + 1

        # Check TPOT violations
        if "tpot" in slo_config:
            tpot_slo = slo_config["tpot"]
            threshold = tpot_slo.get("threshold")
            if threshold and metrics.get("tpot") is not None:
                actual = metrics["tpot"]
                if actual > threshold:
                    has_violation = True
                    violation_ratio = (actual - threshold) / threshold
                    violated_metrics.append({
                        "metric": "tpot",
                        "threshold": threshold,
                        "actual": actual,
                        "violation_ratio": violation_ratio,
                        "hard_fail": tpot_slo.get("hard_fail", False)
                    })
                    violation_stats["metric_violations"]["tpot"] = violation_stats["metric_violations"].get("tpot", 0) + 1

        # Check latency violations (p50, p90, p99)
        if "latency" in slo_config:
            for percentile in ["p50", "p90", "p99"]:
                if percentile in slo_config["latency"]:
                    latency_slo = slo_config["latency"][percentile]
                    threshold = latency_slo.get("threshold")
                    metric_key = f"latency_{percentile}"
                    if threshold and metrics.get(metric_key) is not None:
                        actual = metrics[metric_key]
                        if actual > threshold:
                            has_violation = True
                            violation_ratio = (actual - threshold) / threshold
                            violated_metrics.append({
                                "metric": metric_key,
                                "threshold": threshold,
                                "actual": actual,
                                "violation_ratio": violation_ratio,
                                "hard_fail": latency_slo.get("hard_fail", False)
                            })
                            violation_stats["metric_violations"][metric_key] = violation_stats["metric_violations"].get(metric_key, 0) + 1

        # Check throughput violations
        if "throughput" in slo_config:
            throughput_slo = slo_config["throughput"]
            threshold = throughput_slo.get("threshold")
            if threshold and metrics.get("throughput") is not None:
                actual = metrics["throughput"]
                # Throughput is "higher is better", so violation is actual < threshold
                if actual < threshold:
                    has_violation = True
                    violation_ratio = (threshold - actual) / threshold
                    violated_metrics.append({
                        "metric": "throughput",
                        "threshold": threshold,
                        "actual": actual,
                        "violation_ratio": violation_ratio,
                        "hard_fail": throughput_slo.get("hard_fail", False)
                    })
                    violation_stats["metric_violations"]["throughput"] = violation_stats["metric_violations"].get("throughput", 0) + 1

        if has_violation:
            violation_stats["experiments_with_violations"] += 1

            # Classify as hard fail or soft penalty
            is_hard_fail = any(v.get("hard_fail") for v in violated_metrics)
            if is_hard_fail:
                violation_stats["hard_fail_count"] += 1
            else:
                violation_stats["soft_penalty_count"] += 1

            violation_stats["violating_experiments"].append({
                "experiment_id": exp.id,
                "status": status,
                "parameters": exp.parameters,
                "objective_score": exp.objective_score,
                "violated_metrics": violated_metrics,
                "is_hard_fail": is_hard_fail
            })

    # Sort violated metrics by frequency
    most_violated = sorted(
        violation_stats["metric_violations"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "task_name": task.task_name,
        "has_slo": True,
        "slo_config": slo_config,
        "statistics": {
            "total_experiments": violation_stats["total_experiments"],
            "successful_experiments": violation_stats["successful_experiments"],
            "failed_experiments": violation_stats["failed_experiments"],
            "experiments_with_violations": violation_stats["experiments_with_violations"],
            "violation_rate": f"{violation_stats['experiments_with_violations'] / max(1, violation_stats['total_experiments']) * 100:.1f}%",
            "hard_fail_count": violation_stats["hard_fail_count"],
            "soft_penalty_count": violation_stats["soft_penalty_count"]
        },
        "most_violated_metrics": [{"metric": m, "count": c} for m, c in most_violated],
        "violating_experiments": violation_stats["violating_experiments"][:10]  # Limit to first 10
    }, indent=2)


@tool
@register_tool(ToolCategory.TASK)
async def search_experiment_logs(
    task_id: int,
    experiment_id: int,
    context_lines: int = 10,
    db: AsyncSession = None
) -> str:
    """
    Search task logs for entries related to a specific experiment.

    This is useful for debugging why a particular experiment failed.
    Returns log entries surrounding the experiment's execution.

    Args:
        task_id: Task ID the experiment belongs to
        experiment_id: Experiment ID to search for in logs
        context_lines: Number of lines to show before/after matches (default: 10)

    Returns:
        JSON string with log entries related to this experiment
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Verify experiment exists and belongs to task
    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    experiment = result.scalar_one_or_none()

    if not experiment:
        return json.dumps({"error": f"Experiment {experiment_id} not found"})

    if experiment.task_id != task_id:
        return json.dumps({
            "error": f"Experiment {experiment_id} belongs to task {experiment.task_id}, not task {task_id}"
        })

    # Get log file
    from pathlib import Path
    log_dir = Path.home() / ".local/share/inference-autotuner/logs"
    log_file = log_dir / f"task_{task_id}.log"

    if not log_file.exists():
        return json.dumps({
            "success": True,
            "log_exists": False,
            "message": f"No log file found for task {task_id}"
        })

    try:
        log_content = log_file.read_text()
        lines = log_content.splitlines()

        # Search for lines mentioning this experiment ID
        matching_sections = []
        search_patterns = [
            f"experiment {experiment_id}",
            f"experiment_id={experiment_id}",
            f"exp_id: {experiment_id}",
            f"Experiment {experiment_id}"
        ]

        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(pattern.lower() in line_lower for pattern in search_patterns):
                # Found a match, extract context
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                matching_sections.append({
                    "line_number": i + 1,
                    "matching_line": line,
                    "context": {
                        "before": lines[start:i],
                        "after": lines[i+1:end]
                    }
                })

        # Get experiment error message if exists
        error_message = experiment.error_message if experiment.error_message else None

        return json.dumps({
            "success": True,
            "task_id": task_id,
            "experiment_id": experiment_id,
            "experiment_status": experiment.status.value if hasattr(experiment.status, 'value') else experiment.status,
            "experiment_error": error_message,
            "log_exists": True,
            "matches_found": len(matching_sections),
            "matching_sections": matching_sections[:5],  # Limit to first 5 matches
            "note": "Use get_task_logs() to see the full log file if needed"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to search log file: {str(e)}",
            "task_id": task_id,
            "experiment_id": experiment_id
        })
