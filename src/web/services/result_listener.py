"""
Result listener service for the manager.

Subscribes to Redis Pub/Sub channels to receive real-time experiment results
from distributed workers. Updates the database and broadcasts to WebSocket clients.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

import redis.asyncio as redis
from web.config import get_settings
from web.workers.pubsub import (
	ExperimentResult,
	WorkerEvent,
	LogEntry,
	RESULTS_CHANNEL_PREFIX,
	WORKER_CHANNEL_PREFIX,
	LOG_CHANNEL_PREFIX,
)

logger = logging.getLogger(__name__)

settings = get_settings()


class ResultListener:
	"""Subscriber for experiment results (used by manager)."""

	def __init__(self, redis_client: redis.Redis):
		self.redis = redis_client
		self.pubsub: Optional[redis.client.PubSub] = None
		self._running = False
		self._task: Optional[asyncio.Task] = None
		self._subscribed_tasks: Set[int] = set()

		# Callbacks
		self._result_callbacks: List[Callable[[ExperimentResult], Any]] = []
		self._worker_event_callbacks: List[Callable[[WorkerEvent], Any]] = []
		self._log_callbacks: List[Callable[[LogEntry], Any]] = []

	@classmethod
	async def create(cls) -> "ResultListener":
		"""Create a new ResultListener with Redis connection."""
		client = redis.Redis(
			host=settings.redis_host,
			port=settings.redis_port,
			db=settings.redis_db,
			decode_responses=True,
		)
		listener = cls(client)
		listener.pubsub = client.pubsub()
		return listener

	async def close(self):
		"""Close Redis connection and stop listening."""
		self._running = False
		if self._task:
			self._task.cancel()
			try:
				await self._task
			except asyncio.CancelledError:
				pass
		if self.pubsub:
			await self.pubsub.close()
		if self.redis:
			await self.redis.close()

	def on_result(self, callback: Callable[[ExperimentResult], Any]):
		"""Register a callback for experiment results.

		Args:
			callback: Function to call with ExperimentResult
		"""
		self._result_callbacks.append(callback)

	def on_worker_event(self, callback: Callable[[WorkerEvent], Any]):
		"""Register a callback for worker events.

		Args:
			callback: Function to call with WorkerEvent
		"""
		self._worker_event_callbacks.append(callback)

	def on_log(self, callback: Callable[[LogEntry], Any]):
		"""Register a callback for worker log entries.

		Args:
			callback: Function to call with LogEntry
		"""
		self._log_callbacks.append(callback)

	async def subscribe_task(self, task_id: int):
		"""Subscribe to results for a specific task.

		Args:
			task_id: Task ID to subscribe to
		"""
		if task_id in self._subscribed_tasks:
			return

		channel = f"{RESULTS_CHANNEL_PREFIX}{task_id}"
		await self.pubsub.subscribe(channel)
		self._subscribed_tasks.add(task_id)
		logger.info(f"Subscribed to results channel for task {task_id}")

	async def unsubscribe_task(self, task_id: int):
		"""Unsubscribe from results for a specific task.

		Args:
			task_id: Task ID to unsubscribe from
		"""
		if task_id not in self._subscribed_tasks:
			return

		channel = f"{RESULTS_CHANNEL_PREFIX}{task_id}"
		await self.pubsub.unsubscribe(channel)
		self._subscribed_tasks.discard(task_id)
		logger.info(f"Unsubscribed from results channel for task {task_id}")

	async def subscribe_all_results(self):
		"""Subscribe to all result channels using pattern matching."""
		pattern = f"{RESULTS_CHANNEL_PREFIX}*"
		await self.pubsub.psubscribe(pattern)
		logger.info("Subscribed to all result channels")

	async def subscribe_all_workers(self):
		"""Subscribe to all worker event channels using pattern matching."""
		pattern = f"{WORKER_CHANNEL_PREFIX}*"
		await self.pubsub.psubscribe(pattern)
		logger.info("Subscribed to all worker event channels")

	async def subscribe_all_logs(self):
		"""Subscribe to all worker log channels using pattern matching."""
		pattern = f"{LOG_CHANNEL_PREFIX}*"
		await self.pubsub.psubscribe(pattern)
		logger.info("Subscribed to all worker log channels")

	async def start(self):
		"""Start listening for messages in background."""
		if self._running:
			return

		self._running = True
		self._task = asyncio.create_task(self._listen_loop())
		logger.info("Result listener started")

	async def _listen_loop(self):
		"""Main listening loop."""
		try:
			while self._running:
				try:
					message = await self.pubsub.get_message(
						ignore_subscribe_messages=True,
						timeout=1.0,
					)
					if message:
						await self._handle_message(message)
				except asyncio.CancelledError:
					break
				except Exception as e:
					logger.error(f"Error in listen loop: {e}")
					await asyncio.sleep(1)
		finally:
			logger.info("Result listener stopped")

	async def _handle_message(self, message: Dict[str, Any]):
		"""Handle incoming Pub/Sub message.

		Args:
			message: Redis Pub/Sub message dict
		"""
		msg_type = message.get("type")
		if msg_type not in ("message", "pmessage"):
			return

		channel = message.get("channel", "")
		data = message.get("data", "")

		try:
			if channel.startswith(RESULTS_CHANNEL_PREFIX) or (
				msg_type == "pmessage" and RESULTS_CHANNEL_PREFIX in channel
			):
				# Parse experiment result
				result = ExperimentResult.model_validate_json(data)
				await self._dispatch_result(result)

			elif channel.startswith(WORKER_CHANNEL_PREFIX) or (
				msg_type == "pmessage" and WORKER_CHANNEL_PREFIX in channel
			):
				# Parse worker event
				event = WorkerEvent.model_validate_json(data)
				await self._dispatch_worker_event(event)

			elif channel.startswith(LOG_CHANNEL_PREFIX) or (
				msg_type == "pmessage" and LOG_CHANNEL_PREFIX in channel
			):
				# Parse log entry
				log_entry = LogEntry.model_validate_json(data)
				await self._dispatch_log(log_entry)

		except Exception as e:
			logger.error(f"Failed to parse message from {channel}: {e}")

	async def _dispatch_result(self, result: ExperimentResult):
		"""Dispatch result to all registered callbacks.

		Args:
			result: Parsed experiment result
		"""
		logger.info(
			f"Received result: task={result.task_id} exp={result.experiment_id} "
			f"status={result.status} score={result.objective_score}"
		)

		for callback in self._result_callbacks:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback(result)
				else:
					callback(result)
			except Exception as e:
				logger.error(f"Error in result callback: {e}")

	async def _dispatch_worker_event(self, event: WorkerEvent):
		"""Dispatch worker event to all registered callbacks.

		Args:
			event: Parsed worker event
		"""
		logger.debug(f"Received worker event: {event.worker_id} - {event.event_type}")

		for callback in self._worker_event_callbacks:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback(event)
				else:
					callback(event)
			except Exception as e:
				logger.error(f"Error in worker event callback: {e}")

	async def _dispatch_log(self, log_entry: LogEntry):
		"""Dispatch log entry to all registered callbacks.

		Args:
			log_entry: Parsed log entry from worker
		"""
		logger.debug(f"Received log: [{log_entry.level}] {log_entry.worker_id}: {log_entry.message[:50]}...")

		for callback in self._log_callbacks:
			try:
				if asyncio.iscoroutinefunction(callback):
					await callback(log_entry)
				else:
					callback(log_entry)
			except Exception as e:
				logger.error(f"Error in log callback: {e}")


# Global listener instance
_listener: Optional[ResultListener] = None


async def get_result_listener() -> ResultListener:
	"""Get or create global result listener."""
	global _listener
	if _listener is None:
		_listener = await ResultListener.create()
	return _listener


async def start_result_listener():
	"""Start the global result listener with default subscriptions."""
	listener = await get_result_listener()

	# Subscribe to all channels by default
	await listener.subscribe_all_results()
	await listener.subscribe_all_workers()
	await listener.subscribe_all_logs()

	# Start listening
	await listener.start()

	return listener


async def stop_result_listener():
	"""Stop and close the global result listener."""
	global _listener
	if _listener:
		await _listener.close()
		_listener = None
