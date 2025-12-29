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
ALL_RESULTS_CHANNEL = "channel:results:*"


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

		subscribers = await self.redis.publish(channel, message)
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

		subscribers = await self.redis.publish(channel, message)
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
