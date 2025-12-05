"""
Task management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete
from typing import List
import asyncio
from pathlib import Path
import os

from web.db.session import get_db
from web.db.models import Task, TaskStatus, Experiment
from web.schemas import TaskCreate, TaskUpdate, TaskResponse, TaskListResponse
from web.services import TaskService

router = APIRouter()


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(task_data: TaskCreate, db: AsyncSession = Depends(get_db)):
	"""Create a new autotuning task."""
	# Check if task name already exists
	result = await db.execute(select(Task).where(Task.task_name == task_data.task_name))
	existing_task = result.scalar_one_or_none()

	if existing_task:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST, detail=f"Task '{task_data.task_name}' already exists"
		)

	# Create new task
	db_task = Task(
		task_name=task_data.task_name,
		description=task_data.description,
		model_config=task_data.model,
		base_runtime=task_data.base_runtime,
		runtime_image_tag=task_data.runtime_image_tag,
		parameters=task_data.parameters,
		optimization_config=task_data.optimization,
		benchmark_config=task_data.benchmark,
		slo_config=task_data.slo,
		quant_config=task_data.quant_config,
		parallel_config=task_data.parallel_config,
		clusterbasemodel_config=task_data.clusterbasemodel_config,
		clusterservingruntime_config=task_data.clusterservingruntime_config,
		deployment_mode=task_data.deployment_mode,
		status=TaskStatus.PENDING,
	)

	db.add(db_task)
	await db.commit()
	await db.refresh(db_task)

	return db_task


@router.get("/", response_model=List[TaskListResponse])
async def list_tasks(skip: int = 0, limit: int = 100, status_filter: str = None, db: AsyncSession = Depends(get_db)):
	"""List all autotuning tasks."""
	# Use TaskService for business logic
	tasks = await TaskService.list_tasks(db, status=status_filter, skip=skip, limit=limit)
	return tasks


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Get task by ID."""
	# Use TaskService for business logic
	task = await TaskService.get_task_by_id(db, task_id)

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	return task


@router.get("/name/{task_name}", response_model=TaskResponse)
async def get_task_by_name(task_name: str, db: AsyncSession = Depends(get_db)):
	"""Get task by name."""
	# Use TaskService for business logic
	task = await TaskService.get_task_by_name(db, task_name)

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_name}' not found")

	return task


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(task_id: int, task_update: TaskUpdate, db: AsyncSession = Depends(get_db)):
	"""Update task."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	# Update fields
	if task_update.description is not None:
		task.description = task_update.description
	if task_update.status is not None:
		task.status = task_update.status

	await db.commit()
	await db.refresh(task)

	return task


@router.put("/{task_id}", response_model=TaskResponse)
async def replace_task(task_id: int, task_data: TaskCreate, db: AsyncSession = Depends(get_db)):
	"""Replace task configuration (for editing)."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	# Check if new task name conflicts with another task (not this one)
	if task_data.task_name != task.task_name:
		result = await db.execute(select(Task).where(Task.task_name == task_data.task_name))
		existing_task = result.scalar_one_or_none()

		if existing_task:
			raise HTTPException(
				status_code=status.HTTP_400_BAD_REQUEST,
				detail=f"Task '{task_data.task_name}' already exists"
			)

	# Only allow editing if task is not running
	if task.status == TaskStatus.RUNNING:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Cannot edit a running task"
		)

	# Update all fields
	task.task_name = task_data.task_name
	task.description = task_data.description
	task.model_config = task_data.model
	task.base_runtime = task_data.base_runtime
	task.runtime_image_tag = task_data.runtime_image_tag
	task.parameters = task_data.parameters
	task.optimization_config = task_data.optimization
	task.benchmark_config = task_data.benchmark
	task.slo_config = task_data.slo
	task.quant_config = task_data.quant_config
	task.parallel_config = task_data.parallel_config
	task.clusterbasemodel_config = task_data.clusterbasemodel_config
	task.clusterservingruntime_config = task_data.clusterservingruntime_config
	task.deployment_mode = task_data.deployment_mode

	# Reset status to pending and clear timestamps when task is edited
	task.status = TaskStatus.PENDING
	task.started_at = None
	task.completed_at = None
	task.elapsed_time = None

	# Explicitly mark as modified and commit
	db.add(task)
	await db.commit()
	await db.refresh(task)
	return task


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Delete task from database (does not clean up log files - use /clear endpoint instead)."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	# Don't allow deletion of running tasks
	if task.status == TaskStatus.RUNNING:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete running task. Cancel it first."
		)

	# Delete task from database (cascades to experiments)
	await db.delete(task)
	await db.commit()

	print(f"[API] Deleted task {task_id} from database")


@router.post("/{task_id}/start", response_model=TaskResponse)
async def start_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Start autotuning task execution."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	if task.status != TaskStatus.PENDING:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Task must be in PENDING status to start")

	# Update status to RUNNING
	task.status = TaskStatus.RUNNING
	from datetime import datetime

	task.started_at = datetime.utcnow()

	await db.commit()
	await db.refresh(task)

	# Enqueue ARQ job
	from web.workers import enqueue_autotuning_task

	job_id = await enqueue_autotuning_task(task.id)
	print(f"[API] Enqueued task {task.id} with job_id: {job_id}")

	return task


@router.post("/{task_id}/cancel", response_model=TaskResponse)
async def cancel_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Cancel running task."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	if task.status != TaskStatus.RUNNING:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Task is not running")

	task.status = TaskStatus.CANCELLED
	await db.commit()
	await db.refresh(task)

	# TODO: Cancel ARQ job (will be implemented in ARQ setup)

	return task


@router.post("/{task_id}/restart", response_model=TaskResponse)
async def restart_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Restart a completed, failed, or cancelled task and immediately start it."""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	# Only allow restart for completed, failed, or cancelled tasks
	if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail=f"Task must be completed, failed, or cancelled to restart. Current status: {task.status}"
		)

	# Delete old experiments from previous runs
	await db.execute(delete(Experiment).where(Experiment.task_id == task_id))

	# Reset task fields
	from datetime import datetime
	task.completed_at = None
	task.elapsed_time = None
	# Reset experiment counters
	task.total_experiments = 0
	task.successful_experiments = 0
	task.best_experiment_id = None
	# Clear checkpoint metadata to prevent resume
	task.task_metadata = None

	# Set status to RUNNING and start immediately
	task.status = TaskStatus.RUNNING
	task.started_at = datetime.utcnow()

	await db.commit()
	await db.refresh(task)

	# Enqueue ARQ job
	from web.workers import enqueue_autotuning_task

	job_id = await enqueue_autotuning_task(task.id)
	print(f"[API] Restarted and enqueued task {task.id} with job_id: {job_id}")

	return task


@router.post("/{task_id}/clear", response_model=TaskResponse)
async def clear_task(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Clear task experiments and logs without deleting the task configuration.

	This endpoint:
	- Deletes all experiments for the task
	- Clears the task log file
	- Resets task status and counters
	- Keeps the task configuration for future runs
	"""
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task {task_id} not found")

	# Don't allow clearing running tasks
	if task.status == TaskStatus.RUNNING:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST,
			detail="Cannot clear running task. Cancel it first."
		)

	# Delete all experiments for this task
	delete_result = await db.execute(delete(Experiment).where(Experiment.task_id == task_id))
	experiments_deleted = delete_result.rowcount
	print(f"[API] Deleted {experiments_deleted} experiments for task {task_id}")

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
	log_file = get_task_log_file(task_id)
	if log_file.exists():
		try:
			log_file.unlink()
			print(f"[API] Deleted log file for task {task_id}: {log_file}")
		except Exception as e:
			print(f"[API] Warning: Failed to delete log file {log_file}: {e}")

	print(f"[API] Cleared task {task_id}: deleted {experiments_deleted} experiments and reset task state")
	return task


def get_task_log_file(task_id: int) -> Path:
	"""Get the log file path for a task."""
	log_dir = Path.home() / ".local/share/inference-autotuner/logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	return log_dir / f"task_{task_id}.log"


def get_experiment_log_file(task_id: int, experiment_id: int) -> Path:
	"""Get the log file path for a specific experiment."""
	log_dir = Path.home() / ".local/share/inference-autotuner/logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	return log_dir / f"task_{task_id}_exp_{experiment_id}.log"


async def stream_log_file(log_file: Path, follow: bool = False):
	"""Stream log file contents, optionally following new lines."""
	try:
		# If file doesn't exist yet, wait for it (up to 30 seconds)
		if not log_file.exists():
			if follow:
				yield "data: Waiting for log file to be created...\n\n"
				for _ in range(30):
					await asyncio.sleep(1)
					if log_file.exists():
						break
				else:
					yield "data: Log file not found. Task may not have started yet.\n\n"
					return
			else:
				yield "data: Log file not found.\n\n"
				return

		# Stream existing content
		with open(log_file, "r") as f:
			for line in f:
				yield f"data: {line.rstrip()}\n\n"

		# If follow mode, watch for new lines
		if follow:
			last_pos = log_file.stat().st_size
			while True:
				await asyncio.sleep(0.5)  # Poll every 500ms
				
				# Check if file still exists
				if not log_file.exists():
					yield "data: [Log file removed]\n\n"
					break
					
				current_size = log_file.stat().st_size
				if current_size > last_pos:
					with open(log_file, "r") as f:
						f.seek(last_pos)
						for line in f:
							yield f"data: {line.rstrip()}\n\n"
					last_pos = current_size
				elif current_size < last_pos:
					# File was truncated, start from beginning
					last_pos = 0
					yield "data: [Log file was truncated]\n\n"
	except Exception as e:
		yield f"data: Error reading log: {str(e)}\n\n"


@router.get("/{task_id}/logs")
async def get_task_logs(
	task_id: int,
	follow: bool = False,
	db: AsyncSession = Depends(get_db)
):
	"""
	Get task execution logs.
	
	Args:
		task_id: Task ID
		follow: If True, streams logs in real-time (Server-Sent Events)
	
	Returns:
		Log content as text or streaming response
	"""
	# Verify task exists
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()
	
	if not task:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Task {task_id} not found"
		)
	
	log_file = get_task_log_file(task_id)
	
	# If follow mode, return streaming response (Server-Sent Events)
	if follow:
		return StreamingResponse(
			stream_log_file(log_file, follow=True),
			media_type="text/event-stream",
			headers={
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
				"X-Accel-Buffering": "no"  # Disable nginx buffering
			}
		)
	
	# Otherwise return static log content
	if not log_file.exists():
		return {"logs": "No logs available yet."}
	
	with open(log_file, "r") as f:
		logs = f.read()
	
	return {"logs": logs}


@router.delete("/{task_id}/logs", status_code=status.HTTP_204_NO_CONTENT)
async def clear_task_logs(task_id: int, db: AsyncSession = Depends(get_db)):
	"""Clear task logs (empty the file content, but keep the file)."""
	# Verify task exists
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()

	if not task:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Task {task_id} not found"
		)

	log_file = get_task_log_file(task_id)
	if log_file.exists():
		# Clear file content by opening in write mode with truncation
		with open(log_file, 'w') as f:
			pass  # Empty write truncates the file

@router.get("/{task_id}/experiments/{experiment_id}/logs")
async def get_experiment_logs(
	task_id: int,
	experiment_id: int,
	follow: bool = False,
	db: AsyncSession = Depends(get_db)
):
	"""
	Get experiment-specific execution logs.
	
	Args:
		task_id: Task ID
		experiment_id: Experiment ID
		follow: If True, streams logs in real-time (Server-Sent Events)
	
	Returns:
		Log content as text or streaming response
	"""
	# Verify task exists
	result = await db.execute(select(Task).where(Task.id == task_id))
	task = result.scalar_one_or_none()
	
	if not task:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Task {task_id} not found"
		)
	
	# Verify experiment exists
	from db.models import Experiment
	result = await db.execute(
		select(Experiment).where(
			Experiment.task_id == task_id,
			Experiment.experiment_id == experiment_id
		).order_by(Experiment.created_at.desc())
	)
	experiment = result.scalars().first()
	
	if not experiment:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=f"Experiment {experiment_id} not found for task {task_id}"
		)
	
	log_file = get_experiment_log_file(task_id, experiment_id)
	
	# If follow mode, return streaming response (Server-Sent Events)
	if follow:
		return StreamingResponse(
			stream_log_file(log_file, follow=True),
			media_type="text/event-stream",
			headers={
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
				"X-Accel-Buffering": "no"
			}
		)
	
	# Otherwise return static log content
	if not log_file.exists():
		return {"logs": "No logs available yet for this experiment."}
	
	with open(log_file, "r") as f:
		logs = f.read()
	
	return {"logs": logs}
