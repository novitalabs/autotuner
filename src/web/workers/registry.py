"""
Redis-based worker registry for distributed ARQ workers.

Workers register themselves on startup and send periodic heartbeats.
The manager uses this registry to track available workers and their status.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import redis.asyncio as redis
from web.config import get_settings
from web.schemas.worker import (
	WorkerInfo,
	WorkerRegister,
	WorkerHeartbeat,
	WorkerStatus,
	WorkerResponse,
)

logger = logging.getLogger(__name__)

settings = get_settings()

# Redis key patterns
WORKER_KEY_PREFIX = "worker:"
WORKERS_SET_KEY = "workers:active"
WORKER_JOBS_PREFIX = "worker:jobs:"

# Heartbeat configuration
HEARTBEAT_INTERVAL = 30  # seconds
HEARTBEAT_TTL = 90  # seconds (3 missed heartbeats = offline)


class WorkerRegistry:
	"""Redis-based worker registry."""

	def __init__(self, redis_client: redis.Redis):
		self.redis = redis_client

	@classmethod
	async def create(cls) -> "WorkerRegistry":
		"""Create a new WorkerRegistry with Redis connection."""
		client = redis.Redis(
			host=settings.redis_host,
			port=settings.redis_port,
			db=settings.redis_db,
			decode_responses=True,
		)
		return cls(client)

	async def close(self):
		"""Close Redis connection."""
		if self.redis:
			await self.redis.close()

	def _worker_key(self, worker_id: str) -> str:
		"""Get Redis key for a worker."""
		return f"{WORKER_KEY_PREFIX}{worker_id}"

	def _worker_jobs_key(self, worker_id: str) -> str:
		"""Get Redis key for worker's current jobs."""
		return f"{WORKER_JOBS_PREFIX}{worker_id}"

	async def register(self, registration: WorkerRegister) -> WorkerInfo:
		"""Register a new worker or update existing registration.

		Args:
			registration: Worker registration data

		Returns:
			WorkerInfo with full worker state
		"""
		now = datetime.utcnow()
		worker_key = self._worker_key(registration.worker_id)

		# Convert nested models to dicts for proper serialization
		gpus_data = None
		if registration.gpus:
			gpus_data = [g.model_dump() for g in registration.gpus]

		capabilities_data = None
		if registration.capabilities:
			capabilities_data = registration.capabilities.model_dump()

		# Check if worker already exists
		existing = await self.redis.get(worker_key)
		if existing:
			existing_info = WorkerInfo.model_validate_json(existing)
			# Update existing worker
			worker_info = WorkerInfo(
				worker_id=registration.worker_id,
				hostname=registration.hostname,
				ip_address=registration.ip_address,
				gpu_count=registration.gpu_count,
				gpu_model=registration.gpu_model,
				gpu_memory_gb=registration.gpu_memory_gb,
				gpus=gpus_data,
				deployment_mode=registration.deployment_mode,
				max_parallel=registration.max_parallel,
				capabilities=capabilities_data,
				current_jobs=existing_info.current_jobs,
				current_job_ids=existing_info.current_job_ids,
				status=WorkerStatus.ONLINE,
				registered_at=existing_info.registered_at,
				last_heartbeat=now,
			)
		else:
			# New worker
			worker_info = WorkerInfo(
				worker_id=registration.worker_id,
				hostname=registration.hostname,
				ip_address=registration.ip_address,
				gpu_count=registration.gpu_count,
				gpu_model=registration.gpu_model,
				gpu_memory_gb=registration.gpu_memory_gb,
				gpus=gpus_data,
				deployment_mode=registration.deployment_mode,
				max_parallel=registration.max_parallel,
				capabilities=capabilities_data,
				current_jobs=0,
				current_job_ids=[],
				status=WorkerStatus.ONLINE,
				registered_at=now,
				last_heartbeat=now,
			)

		# Store worker info with TTL
		await self.redis.setex(
			worker_key,
			HEARTBEAT_TTL,
			worker_info.model_dump_json(),
		)

		# Add to active workers set
		await self.redis.sadd(WORKERS_SET_KEY, registration.worker_id)

		logger.info(f"Worker registered: {registration.worker_id} ({registration.hostname})")
		return worker_info

	async def heartbeat(self, heartbeat: WorkerHeartbeat) -> Optional[WorkerInfo]:
		"""Update worker heartbeat.

		Args:
			heartbeat: Heartbeat data with current status

		Returns:
			Updated WorkerInfo or None if worker not found
		"""
		worker_key = self._worker_key(heartbeat.worker_id)

		# Get current worker info
		existing = await self.redis.get(worker_key)
		if not existing:
			logger.warning(f"Heartbeat for unknown worker: {heartbeat.worker_id}")
			return None

		worker_info = WorkerInfo.model_validate_json(existing)

		# Update heartbeat timestamp and job info
		worker_info.last_heartbeat = datetime.utcnow()
		worker_info.current_jobs = heartbeat.current_jobs
		worker_info.current_job_ids = heartbeat.current_job_ids

		# Update GPU status if provided
		if heartbeat.gpus:
			worker_info.gpus = heartbeat.gpus

		# Update status based on jobs
		if heartbeat.status:
			worker_info.status = heartbeat.status
		elif heartbeat.current_jobs > 0:
			worker_info.status = WorkerStatus.BUSY
		else:
			worker_info.status = WorkerStatus.ONLINE

		# Refresh TTL
		await self.redis.setex(
			worker_key,
			HEARTBEAT_TTL,
			worker_info.model_dump_json(),
		)

		return worker_info

	async def deregister(self, worker_id: str) -> bool:
		"""Deregister a worker (graceful shutdown).

		Args:
			worker_id: Worker identifier

		Returns:
			True if worker was found and removed
		"""
		worker_key = self._worker_key(worker_id)

		# Remove from active set
		await self.redis.srem(WORKERS_SET_KEY, worker_id)

		# Delete worker key
		deleted = await self.redis.delete(worker_key)

		# Clean up jobs key
		await self.redis.delete(self._worker_jobs_key(worker_id))

		if deleted:
			logger.info(f"Worker deregistered: {worker_id}")
			return True

		return False

	async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
		"""Get worker information.

		Args:
			worker_id: Worker identifier

		Returns:
			WorkerInfo or None if not found
		"""
		worker_key = self._worker_key(worker_id)
		data = await self.redis.get(worker_key)

		if not data:
			return None

		return WorkerInfo.model_validate_json(data)

	async def get_all_workers(self, include_offline: bool = False) -> List[WorkerInfo]:
		"""Get all registered workers.

		Args:
			include_offline: Include workers that missed heartbeats

		Returns:
			List of WorkerInfo
		"""
		# Get all worker IDs from set
		worker_ids = await self.redis.smembers(WORKERS_SET_KEY)

		workers = []
		stale_ids = []

		for worker_id in worker_ids:
			worker_key = self._worker_key(worker_id)
			data = await self.redis.get(worker_key)

			if data:
				worker = WorkerInfo.model_validate_json(data)

				# Check if worker is stale (TTL expired but still in set)
				if (datetime.utcnow() - worker.last_heartbeat).total_seconds() > HEARTBEAT_TTL:
					worker.status = WorkerStatus.OFFLINE

				if include_offline or worker.status != WorkerStatus.OFFLINE:
					workers.append(worker)
			else:
				# Worker key expired, mark for cleanup
				stale_ids.append(worker_id)

		# Clean up stale IDs from set
		if stale_ids:
			await self.redis.srem(WORKERS_SET_KEY, *stale_ids)
			logger.info(f"Cleaned up stale workers: {stale_ids}")

		return workers

	async def get_available_workers(self) -> List[WorkerInfo]:
		"""Get workers available for new jobs.

		Returns:
			List of workers with capacity for new jobs
		"""
		all_workers = await self.get_all_workers(include_offline=False)
		return [
			w
			for w in all_workers
			if w.status in (WorkerStatus.ONLINE, WorkerStatus.BUSY) and w.current_jobs < w.max_parallel
		]

	async def update_worker_jobs(self, worker_id: str, job_ids: List[int]) -> bool:
		"""Update the list of jobs a worker is processing.

		Args:
			worker_id: Worker identifier
			job_ids: List of task IDs being processed

		Returns:
			True if updated successfully
		"""
		worker = await self.get_worker(worker_id)
		if not worker:
			return False

		worker.current_job_ids = job_ids
		worker.current_jobs = len(job_ids)
		worker.status = WorkerStatus.BUSY if job_ids else WorkerStatus.ONLINE

		worker_key = self._worker_key(worker_id)
		await self.redis.setex(
			worker_key,
			HEARTBEAT_TTL,
			worker.model_dump_json(),
		)
		return True

	async def set_worker_alias(self, worker_id: str, alias: Optional[str]) -> Optional[WorkerInfo]:
		"""Set or clear worker alias.

		Args:
			worker_id: Worker identifier
			alias: New alias or None to clear

		Returns:
			Updated WorkerInfo or None if worker not found
		"""
		worker = await self.get_worker(worker_id)
		if not worker:
			return None

		worker.alias = alias.strip() if alias else None

		worker_key = self._worker_key(worker_id)
		# Get current TTL to preserve it
		ttl = await self.redis.ttl(worker_key)
		if ttl < 0:
			ttl = HEARTBEAT_TTL

		await self.redis.setex(
			worker_key,
			ttl,
			worker.model_dump_json(),
		)

		logger.info(f"Worker alias updated: {worker_id} -> {alias}")
		return worker


def worker_info_to_response(worker: WorkerInfo) -> WorkerResponse:
	"""Convert WorkerInfo to WorkerResponse with computed fields."""
	now = datetime.utcnow()
	seconds_since = (now - worker.last_heartbeat).total_seconds()

	return WorkerResponse(
		worker_id=worker.worker_id,
		hostname=worker.hostname,
		alias=worker.alias,
		ip_address=worker.ip_address,
		gpu_count=worker.gpu_count,
		gpu_model=worker.gpu_model,
		gpu_memory_gb=worker.gpu_memory_gb,
		gpus=worker.gpus,
		deployment_mode=worker.deployment_mode,
		max_parallel=worker.max_parallel,
		current_jobs=worker.current_jobs,
		status=worker.status,
		registered_at=worker.registered_at,
		last_heartbeat=worker.last_heartbeat,
		seconds_since_heartbeat=seconds_since,
	)


# Global registry instance
_registry: Optional[WorkerRegistry] = None


async def get_worker_registry() -> WorkerRegistry:
	"""Get or create global worker registry."""
	global _registry
	if _registry is None:
		_registry = await WorkerRegistry.create()
	return _registry
