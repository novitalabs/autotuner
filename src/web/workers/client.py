"""
ARQ client for enqueuing jobs.
"""

from typing import Optional, List
from arq import create_pool
from arq.connections import RedisSettings, ArqRedis
from web.config import get_settings

settings = get_settings()

# Global Redis pool
_redis_pool: ArqRedis = None


async def get_arq_pool() -> ArqRedis:
	"""Get or create ARQ Redis pool."""
	global _redis_pool
	if _redis_pool is None:
		_redis_pool = await create_pool(
			RedisSettings(
				host=settings.redis_host,
				port=settings.redis_port,
				database=settings.redis_db,
			)
		)
	return _redis_pool


async def select_worker_for_task(
	deployment_mode: Optional[str] = None,
	gpu_type: Optional[str] = None
) -> Optional[str]:
	"""Select the best available worker based on deployment_mode and gpu_type.

	Workers are filtered by:
	1. deployment_mode (if specified) - must match exactly
	2. gpu_type (if specified) - partial match (case-insensitive)

	Workers are sorted by name (alias or hostname) for priority selection.

	Args:
		deployment_mode: Required deployment mode (docker, local, ome)
		gpu_type: Optional GPU type filter (e.g., "RTX 4090", "A100")

	Returns:
		worker_id of selected worker, or None if no suitable worker found
	"""
	from web.workers.registry import get_worker_registry, WorkerStatus

	registry = await get_worker_registry()
	workers = await registry.get_available_workers()

	if not workers:
		return None

	# Filter by deployment_mode
	if deployment_mode:
		workers = [w for w in workers if w.deployment_mode == deployment_mode]

	# Filter by gpu_type (partial match, case-insensitive)
	if gpu_type:
		gpu_type_lower = gpu_type.lower()
		workers = [
			w for w in workers
			if w.gpu_model and gpu_type_lower in w.gpu_model.lower()
		]

	if not workers:
		return None

	# Sort by name (alias if set, otherwise hostname) for priority
	def get_sort_key(worker):
		name = worker.alias or worker.hostname or worker.worker_id
		return name.lower()

	workers.sort(key=get_sort_key)

	# Return the first (highest priority) worker
	return workers[0].worker_id


async def check_worker_queue_active(worker_id: str) -> bool:
	"""Check if a worker is actively polling its worker-specific queue.

	Workers with the queue_name fix will have a health-check key for their queue.
	If no health-check exists, the worker is likely polling the default queue.

	Args:
		worker_id: Worker ID to check

	Returns:
		True if worker has an active health-check for its queue
	"""
	import redis.asyncio as redis

	client = redis.Redis(
		host=settings.redis_host,
		port=settings.redis_port,
		db=settings.redis_db,
	)
	try:
		# Check for worker-specific queue health-check
		health_key = f"autotuner:{worker_id}:health-check"
		exists = await client.exists(health_key)
		return exists > 0
	finally:
		await client.aclose()


async def enqueue_autotuning_task(task_id: int, task_config: dict = None) -> str:
	"""Enqueue an autotuning task.

	Args:
	    task_id: Database task ID
	    task_config: Optional full task configuration for distributed workers

	Returns:
	    Job ID
	"""
	pool = await get_arq_pool()

	# Select worker based on deployment_mode and gpu_type
	deployment_mode = task_config.get("deployment_mode") if task_config else None
	gpu_type = task_config.get("gpu_type") if task_config else None

	worker_id = await select_worker_for_task(deployment_mode, gpu_type)

	if worker_id:
		# Check if worker is actively polling its worker-specific queue
		queue_active = await check_worker_queue_active(worker_id)

		if queue_active:
			# Worker has queue_name fix - use worker-specific queue
			job = await pool.enqueue_job(
				"run_autotuning_task",
				task_id,
				task_config,
				_queue_name=f"autotuner:{worker_id}"
			)
		else:
			# Worker using default queue - route there instead
			# This handles workers without the queue_name fix
			job = await pool.enqueue_job("run_autotuning_task", task_id, task_config)
	else:
		# Fallback to default queue (any available worker)
		job = await pool.enqueue_job("run_autotuning_task", task_id, task_config)

	return job.job_id


async def get_job_status(job_id: str) -> dict:
	"""Get job status.

	Args:
	    job_id: ARQ job ID

	Returns:
	    Job status dict
	"""
	pool = await get_arq_pool()
	job = await pool.get_job(job_id)

	if job is None:
		return {"status": "not_found"}

	result = await job.result()
	return {"status": await job.status(), "result": result}
