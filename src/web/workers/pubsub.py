"""
Redis Pub/Sub module for distributed worker result streaming.

Workers publish experiment results to Redis channels.
The manager subscribes to receive real-time updates.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import redis.asyncio as redis
from pydantic import BaseModel, Field
from web.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# Channel patterns
RESULTS_CHANNEL_PREFIX = "channel:results:"
WORKER_CHANNEL_PREFIX = "channel:worker:"
CONFIG_CHANNEL_PREFIX = "channel:config:"  # For worker config updates
LOG_CHANNEL_PREFIX = "channel:logs:"  # For worker log streaming
TASK_STATUS_CHANNEL_PREFIX = "channel:task_status:"  # For task status updates
ALL_RESULTS_CHANNEL = "channel:results:*"
ALL_LOGS_CHANNEL = "channel:logs:*"
ALL_TASK_STATUS_CHANNEL = "channel:task_status:*"

# Log storage
LOG_BUFFER_KEY_PREFIX = "logs:buffer:"  # Circular buffer for recent logs
LOG_BUFFER_MAX_SIZE = 500  # Keep last 500 log lines per worker


class ExperimentResult(BaseModel):
	"""Result message published when an experiment completes."""

	task_id: int
	experiment_id: int
	worker_id: str
	status: str  # "success", "failed", "error"
	metrics: Dict[str, Any] = Field(default_factory=dict)
	objective_score: Optional[float] = None
	error_message: Optional[str] = None
	elapsed_time: float = 0.0
	parameters: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=datetime.utcnow)

	class Config:
		json_encoders = {
			datetime: lambda v: v.isoformat()
		}


class WorkerEvent(BaseModel):
	"""Event message for worker status changes."""

	worker_id: str
	event_type: str  # "registered", "heartbeat", "job_started", "job_completed", "offline"
	data: Dict[str, Any] = Field(default_factory=dict)
	timestamp: datetime = Field(default_factory=datetime.utcnow)

	class Config:
		json_encoders = {
			datetime: lambda v: v.isoformat()
		}


class ConfigUpdate(BaseModel):
	"""Config update message sent from manager to worker."""

	worker_id: str
	updates: Dict[str, Any] = Field(default_factory=dict)  # Fields to update
	timestamp: datetime = Field(default_factory=datetime.utcnow)

	class Config:
		json_encoders = {
			datetime: lambda v: v.isoformat()
		}


class LogEntry(BaseModel):
	"""Log entry from a worker."""

	worker_id: str
	level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
	message: str
	source: str = "worker"  # worker, task, benchmark
	task_id: Optional[int] = None
	experiment_id: Optional[int] = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)

	class Config:
		json_encoders = {
			datetime: lambda v: v.isoformat()
		}


class TaskStatusUpdate(BaseModel):
	"""Task status update message from worker to manager."""

	task_id: int
	worker_id: str
	status: str  # "COMPLETED", "FAILED", "CANCELLED"
	total_experiments: int = 0
	successful_experiments: int = 0
	best_experiment_id: Optional[int] = None
	best_score: Optional[float] = None
	elapsed_time: Optional[float] = None
	error_message: Optional[str] = None
	timestamp: datetime = Field(default_factory=datetime.utcnow)

	class Config:
		json_encoders = {
			datetime: lambda v: v.isoformat()
		}


class ResultPublisher:
	"""Publisher for experiment results (used by workers)."""

	def __init__(self, redis_client: redis.Redis):
		self.redis = redis_client
		self.worker_id: Optional[str] = None

	@classmethod
	async def create(cls, worker_id: Optional[str] = None) -> "ResultPublisher":
		"""Create a new ResultPublisher with Redis connection."""
		client = redis.Redis(
			host=settings.redis_host,
			port=settings.redis_port,
			db=settings.redis_db,
			decode_responses=True,
		)
		publisher = cls(client)
		publisher.worker_id = worker_id
		return publisher

	async def _get_fresh_connection(self) -> redis.Redis:
		"""Get a fresh Redis connection to avoid event loop issues.

		This is needed because the original connection's lock may be bound
		to a different event loop when called from different async contexts.
		"""
		return redis.Redis(
			host=settings.redis_host,
			port=settings.redis_port,
			db=settings.redis_db,
			decode_responses=True,
		)

	async def close(self):
		"""Close Redis connection."""
		if self.redis:
			await self.redis.close()

	def _results_channel(self, task_id: int) -> str:
		"""Get channel name for task results."""
		return f"{RESULTS_CHANNEL_PREFIX}{task_id}"

	def _worker_channel(self, worker_id: str) -> str:
		"""Get channel name for worker events."""
		return f"{WORKER_CHANNEL_PREFIX}{worker_id}"

	def _config_channel(self, worker_id: str) -> str:
		"""Get channel name for worker config updates."""
		return f"{CONFIG_CHANNEL_PREFIX}{worker_id}"

	def _log_channel(self, worker_id: str) -> str:
		"""Get channel name for worker logs."""
		return f"{LOG_CHANNEL_PREFIX}{worker_id}"

	def _log_buffer_key(self, worker_id: str) -> str:
		"""Get Redis key for worker's log buffer."""
		return f"{LOG_BUFFER_KEY_PREFIX}{worker_id}"

	def _task_status_channel(self, task_id: int) -> str:
		"""Get channel name for task status updates."""
		return f"{TASK_STATUS_CHANNEL_PREFIX}{task_id}"

	async def publish_task_status(
		self,
		task_id: int,
		status: str,
		total_experiments: int = 0,
		successful_experiments: int = 0,
		best_experiment_id: Optional[int] = None,
		best_score: Optional[float] = None,
		elapsed_time: Optional[float] = None,
		error_message: Optional[str] = None,
	) -> int:
		"""Publish task status update (completion, failure).

		Args:
			task_id: Task ID
			status: Task status ("COMPLETED", "FAILED", "CANCELLED")
			total_experiments: Total number of experiments
			successful_experiments: Number of successful experiments
			best_experiment_id: ID of best experiment
			best_score: Best objective score
			elapsed_time: Total elapsed time in seconds
			error_message: Error message if failed

		Returns:
			Number of subscribers that received the message
		"""
		worker_id = self.worker_id or "unknown"

		update = TaskStatusUpdate(
			task_id=task_id,
			worker_id=worker_id,
			status=status,
			total_experiments=total_experiments,
			successful_experiments=successful_experiments,
			best_experiment_id=best_experiment_id,
			best_score=best_score,
			elapsed_time=elapsed_time,
			error_message=error_message,
		)

		channel = self._task_status_channel(task_id)
		message = update.model_dump_json()

		# Use fresh connection to avoid event loop issues
		# The original connection's lock may be bound to a different event loop
		try:
			subscribers = await self.redis.publish(channel, message)
		except RuntimeError as e:
			if "different event loop" in str(e):
				logger.warning(f"Event loop mismatch, using fresh connection for task status publish")
				fresh_conn = await self._get_fresh_connection()
				try:
					subscribers = await fresh_conn.publish(channel, message)
				finally:
					await fresh_conn.close()
			else:
				raise

		logger.info(
			f"Published task status for task {task_id}: status={status} "
			f"to {subscribers} subscribers"
		)
		return subscribers

	async def publish_log(
		self,
		message: str,
		level: str = "INFO",
		source: str = "worker",
		task_id: Optional[int] = None,
		experiment_id: Optional[int] = None,
	) -> int:
		"""Publish a log entry from this worker.

		Args:
			message: Log message
			level: Log level (DEBUG, INFO, WARNING, ERROR)
			source: Log source (worker, task, benchmark)
			task_id: Associated task ID (optional)
			experiment_id: Associated experiment ID (optional)

		Returns:
			Number of subscribers that received the message
		"""
		if not self.worker_id:
			return 0

		entry = LogEntry(
			worker_id=self.worker_id,
			level=level,
			message=message,
			source=source,
			task_id=task_id,
			experiment_id=experiment_id,
		)

		channel = self._log_channel(self.worker_id)
		entry_json = entry.model_dump_json()

		# Publish to channel for real-time streaming
		subscribers = await self.redis.publish(channel, entry_json)

		# Store in circular buffer for recent logs retrieval
		buffer_key = self._log_buffer_key(self.worker_id)
		await self.redis.lpush(buffer_key, entry_json)
		await self.redis.ltrim(buffer_key, 0, LOG_BUFFER_MAX_SIZE - 1)
		await self.redis.expire(buffer_key, 86400)  # Expire after 24 hours

		return subscribers

	async def get_recent_logs(self, worker_id: str, count: int = 100) -> list:
		"""Get recent log entries for a worker.

		Args:
			worker_id: Worker ID
			count: Number of entries to retrieve

		Returns:
			List of LogEntry dicts (newest first)
		"""
		buffer_key = self._log_buffer_key(worker_id)
		entries = await self.redis.lrange(buffer_key, 0, count - 1)
		return [LogEntry.model_validate_json(e).model_dump() for e in entries]

	async def publish_config_update(self, worker_id: str, updates: Dict[str, Any]) -> int:
		"""Publish a config update to a specific worker.

		Args:
			worker_id: Target worker ID
			updates: Config fields to update

		Returns:
			Number of subscribers that received the message
		"""
		channel = self._config_channel(worker_id)
		update = ConfigUpdate(worker_id=worker_id, updates=updates)
		message = update.model_dump_json()

		subscribers = await self.redis.publish(channel, message)
		logger.info(f"Published config update to worker {worker_id}: {updates} ({subscribers} subscribers)")
		return subscribers

	async def publish_result(self, result: ExperimentResult) -> int:
		"""Publish an experiment result.

		Args:
			result: Experiment result to publish

		Returns:
			Number of subscribers that received the message
		"""
		channel = self._results_channel(result.task_id)
		message = result.model_dump_json()

		# Use fresh connection if event loop mismatch
		try:
			subscribers = await self.redis.publish(channel, message)
		except RuntimeError as e:
			if "different event loop" in str(e):
				logger.warning(f"Event loop mismatch, using fresh connection for result publish")
				fresh_conn = await self._get_fresh_connection()
				try:
					subscribers = await fresh_conn.publish(channel, message)
				finally:
					await fresh_conn.close()
			else:
				raise

		logger.info(
			f"Published result for task {result.task_id} exp {result.experiment_id} "
			f"to {subscribers} subscribers"
		)
		return subscribers

	async def publish_worker_event(self, event: WorkerEvent) -> int:
		"""Publish a worker status event.

		Args:
			event: Worker event to publish

		Returns:
			Number of subscribers that received the message
		"""
		channel = self._worker_channel(event.worker_id)
		message = event.model_dump_json()

		# Use fresh connection if event loop mismatch
		try:
			subscribers = await self.redis.publish(channel, message)
		except RuntimeError as e:
			if "different event loop" in str(e):
				logger.warning(f"Event loop mismatch, using fresh connection for worker event publish")
				fresh_conn = await self._get_fresh_connection()
				try:
					subscribers = await fresh_conn.publish(channel, message)
				finally:
					await fresh_conn.close()
			else:
				raise

		logger.debug(f"Published worker event {event.event_type} to {subscribers} subscribers")
		return subscribers

	async def publish_experiment_started(
		self,
		task_id: int,
		experiment_id: int,
		parameters: Dict[str, Any],
	) -> int:
		"""Convenience method to publish experiment started event."""
		if not self.worker_id:
			logger.warning("Worker ID not set, skipping event publish")
			return 0

		event = WorkerEvent(
			worker_id=self.worker_id,
			event_type="job_started",
			data={
				"task_id": task_id,
				"experiment_id": experiment_id,
				"parameters": parameters,
			},
		)
		return await self.publish_worker_event(event)

	async def publish_experiment_completed(
		self,
		task_id: int,
		experiment_id: int,
		status: str,
		metrics: Dict[str, Any],
		objective_score: Optional[float] = None,
		error_message: Optional[str] = None,
		elapsed_time: float = 0.0,
		parameters: Optional[Dict[str, Any]] = None,
	) -> int:
		"""Convenience method to publish experiment completion.

		This publishes both:
		1. Result to the task results channel
		2. Worker event to the worker channel
		"""
		worker_id = self.worker_id or "unknown"

		# Publish result
		result = ExperimentResult(
			task_id=task_id,
			experiment_id=experiment_id,
			worker_id=worker_id,
			status=status,
			metrics=metrics,
			objective_score=objective_score,
			error_message=error_message,
			elapsed_time=elapsed_time,
			parameters=parameters or {},
		)
		result_subs = await self.publish_result(result)

		# Publish worker event
		event = WorkerEvent(
			worker_id=worker_id,
			event_type="job_completed",
			data={
				"task_id": task_id,
				"experiment_id": experiment_id,
				"status": status,
				"objective_score": objective_score,
			},
		)
		event_subs = await self.publish_worker_event(event)

		return result_subs + event_subs


# Global publisher instance (for worker use)
_publisher: Optional[ResultPublisher] = None


async def get_result_publisher(worker_id: Optional[str] = None) -> ResultPublisher:
	"""Get or create global result publisher."""
	global _publisher
	if _publisher is None:
		_publisher = await ResultPublisher.create(worker_id)
	elif worker_id and _publisher.worker_id != worker_id:
		_publisher.worker_id = worker_id
	return _publisher


async def close_result_publisher():
	"""Close global result publisher."""
	global _publisher
	if _publisher:
		await _publisher.close()
		_publisher = None
