"""
Event broadcasting system for WebSocket real-time updates.

Provides a simple in-memory pub/sub system for broadcasting task and experiment
events to connected WebSocket clients.
"""

import asyncio
import logging
from typing import Dict, Set, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class EventBroadcaster:
	"""
	In-memory event broadcaster for WebSocket clients.

	Uses asyncio queues to broadcast events to all subscribed clients.
	Thread-safe for use with FastAPI and ARQ workers.
	"""

	def __init__(self):
		# Map of task_id -> set of asyncio queues
		self._subscribers: Dict[int, Set[asyncio.Queue]] = defaultdict(set)
		self._lock = asyncio.Lock()

	async def subscribe(self, task_id: int) -> asyncio.Queue:
		"""
		Subscribe to events for a specific task.

		Args:
		    task_id: Task ID to subscribe to

		Returns:
		    asyncio.Queue that will receive events
		"""
		queue = asyncio.Queue(maxsize=100)  # Buffer up to 100 events

		async with self._lock:
			self._subscribers[task_id].add(queue)
			logger.info(
				f"[EventBroadcaster] Client subscribed to task {task_id} "
				f"(total subscribers: {len(self._subscribers[task_id])})"
			)

		return queue

	async def unsubscribe(self, task_id: int, queue: asyncio.Queue):
		"""
		Unsubscribe from events for a specific task.

		Args:
		    task_id: Task ID to unsubscribe from
		    queue: Queue to remove
		"""
		async with self._lock:
			if task_id in self._subscribers:
				self._subscribers[task_id].discard(queue)
				logger.info(
					f"[EventBroadcaster] Client unsubscribed from task {task_id} "
					f"(remaining subscribers: {len(self._subscribers[task_id])})"
				)

				# Clean up empty subscriber sets
				if not self._subscribers[task_id]:
					del self._subscribers[task_id]

	async def broadcast(self, task_id: int, event: Dict[str, Any]):
		"""
		Broadcast an event to all subscribers of a task.

		Args:
		    task_id: Task ID to broadcast to
		    event: Event data dictionary
		"""
		async with self._lock:
			subscribers = self._subscribers.get(task_id, set())

			if not subscribers:
				logger.debug(f"[EventBroadcaster] No subscribers for task {task_id}, skipping broadcast")
				return

			logger.info(
				f"[EventBroadcaster] Broadcasting event to {len(subscribers)} clients for task {task_id}: {event.get('type', 'unknown')}"
			)

			# Send to all subscribers
			dead_queues = set()
			for queue in subscribers:
				try:
					# Use put_nowait to avoid blocking
					# If queue is full, oldest item is dropped
					if queue.full():
						try:
							queue.get_nowait()  # Drop oldest
						except asyncio.QueueEmpty:
							pass

					queue.put_nowait(event)
				except Exception as e:
					logger.error(f"[EventBroadcaster] Error sending to queue: {e}")
					dead_queues.add(queue)

			# Clean up dead queues
			for queue in dead_queues:
				subscribers.discard(queue)

	def broadcast_sync(self, task_id: int, event: Dict[str, Any]):
		"""
		Synchronous wrapper for broadcast (for use in non-async contexts).

		Creates a new event loop if necessary and runs broadcast.
		Safe to call from ARQ workers or other sync contexts.

		Args:
		    task_id: Task ID to broadcast to
		    event: Event data dictionary
		"""
		try:
			# Try to get running event loop
			loop = asyncio.get_event_loop()
			if loop.is_running():
				# If loop is running, schedule broadcast as task
				asyncio.create_task(self.broadcast(task_id, event))
			else:
				# If no loop running, run broadcast synchronously
				loop.run_until_complete(self.broadcast(task_id, event))
		except RuntimeError:
			# No event loop in current thread, create new one
			loop = asyncio.new_event_loop()
			asyncio.set_event_loop(loop)
			try:
				loop.run_until_complete(self.broadcast(task_id, event))
			finally:
				loop.close()

	async def get_subscriber_count(self, task_id: int) -> int:
		"""Get number of subscribers for a task."""
		async with self._lock:
			return len(self._subscribers.get(task_id, set()))

	async def clear_task_subscribers(self, task_id: int):
		"""Clear all subscribers for a task (e.g., when task completes)."""
		async with self._lock:
			if task_id in self._subscribers:
				count = len(self._subscribers[task_id])
				del self._subscribers[task_id]
				logger.info(f"[EventBroadcaster] Cleared {count} subscribers for task {task_id}")


# Global broadcaster instance
_broadcaster: Optional[EventBroadcaster] = None


def get_broadcaster() -> EventBroadcaster:
	"""Get the global EventBroadcaster instance."""
	global _broadcaster
	if _broadcaster is None:
		_broadcaster = EventBroadcaster()
	return _broadcaster


# Event type constants
class EventType:
	"""Event type constants for WebSocket messages."""

	TASK_STARTED = "task_started"
	TASK_PROGRESS = "task_progress"
	TASK_COMPLETED = "task_completed"
	TASK_FAILED = "task_failed"

	EXPERIMENT_STARTED = "experiment_started"
	EXPERIMENT_PROGRESS = "experiment_progress"
	EXPERIMENT_COMPLETED = "experiment_completed"
	EXPERIMENT_FAILED = "experiment_failed"

	BENCHMARK_STARTED = "benchmark_started"
	BENCHMARK_PROGRESS = "benchmark_progress"

	LOG_MESSAGE = "log_message"
	ERROR = "error"


def create_event(
	event_type: str,
	task_id: int,
	data: Optional[Dict[str, Any]] = None,
	experiment_id: Optional[int] = None,
	message: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	Create a standardized event dictionary.

	Args:
	    event_type: Event type (use EventType constants)
	    task_id: Task ID
	    data: Optional event data
	    experiment_id: Optional experiment ID
	    message: Optional message

	Returns:
	    Event dictionary
	"""
	import time

	event = {"type": event_type, "task_id": task_id, "timestamp": time.time()}

	if experiment_id is not None:
		event["experiment_id"] = experiment_id

	if message is not None:
		event["message"] = message

	if data is not None:
		event["data"] = data

	return event
