"""
ARQ client for enqueuing jobs.
"""

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


async def enqueue_autotuning_task(task_id: int, task_config: dict = None) -> str:
	"""Enqueue an autotuning task.

	Args:
	    task_id: Database task ID
	    task_config: Optional full task configuration for distributed workers

	Returns:
	    Job ID
	"""
	pool = await get_arq_pool()
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
