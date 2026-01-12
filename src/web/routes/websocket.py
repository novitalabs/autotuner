"""
WebSocket endpoints for real-time task and experiment updates.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import logging
import asyncio

from web.db.session import get_db
from web.db.models import Task
from web.events.broadcaster import get_broadcaster

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/tasks/{task_id}")
async def task_updates_websocket(websocket: WebSocket, task_id: int):
	"""
	WebSocket endpoint for real-time task progress updates.

	Clients connect to this endpoint to receive live updates about:
	- Task status changes
	- Experiment progress
	- Benchmark metrics
	- Errors and failures

	Args:
	    websocket: FastAPI WebSocket connection
	    task_id: Task ID to subscribe to
	"""
	# Accept WebSocket connection
	await websocket.accept()
	logger.info(f"[WebSocket] Client connected for task {task_id}")

	# Get broadcaster instance
	broadcaster = get_broadcaster()

	# Subscribe to task events
	queue = await broadcaster.subscribe(task_id)

	try:
		# Send initial connection confirmation
		await websocket.send_json(
			{"type": "connection_established", "task_id": task_id, "message": f"Subscribed to task {task_id} updates"}
		)

		# Event loop: receive events from broadcaster and send to client
		while True:
			# Wait for event from broadcaster
			event = await queue.get()

			# Send event to WebSocket client
			try:
				await websocket.send_json(event)
				logger.debug(f"[WebSocket] Sent event to task {task_id} client: {event.get('type')}")
			except Exception as e:
				logger.error(f"[WebSocket] Error sending event to client: {e}")
				break

	except WebSocketDisconnect:
		logger.info(f"[WebSocket] Client disconnected from task {task_id}")
	except Exception as e:
		logger.error(f"[WebSocket] Unexpected error for task {task_id}: {e}")
	finally:
		# Cleanup: unsubscribe from broadcaster
		await broadcaster.unsubscribe(task_id, queue)
		logger.info(f"[WebSocket] Cleaned up subscription for task {task_id}")


@router.websocket("/ws/experiments/{experiment_id}")
async def experiment_updates_websocket(websocket: WebSocket, experiment_id: int):
	"""
	WebSocket endpoint for real-time experiment-specific updates.

	Provides granular updates for a single experiment:
	- Deployment status
	- Benchmark progress
	- Metric updates
	- Log messages

	Args:
	    websocket: FastAPI WebSocket connection
	    experiment_id: Experiment ID to subscribe to
	"""
	# Accept WebSocket connection
	await websocket.accept()
	logger.info(f"[WebSocket] Client connected for experiment {experiment_id}")

	# Get broadcaster instance
	broadcaster = get_broadcaster()

	# Subscribe to experiment events (using special experiment ID prefix)
	# Note: Experiments use negative task_id to avoid collision with task events
	subscription_key = -(experiment_id + 1000000)  # Unique negative key
	queue = await broadcaster.subscribe(subscription_key)

	try:
		# Send initial connection confirmation
		await websocket.send_json(
			{
				"type": "connection_established",
				"experiment_id": experiment_id,
				"message": f"Subscribed to experiment {experiment_id} updates",
			}
		)

		# Event loop
		while True:
			event = await queue.get()

			try:
				await websocket.send_json(event)
				logger.debug(f"[WebSocket] Sent event to experiment {experiment_id} client: {event.get('type')}")
			except Exception as e:
				logger.error(f"[WebSocket] Error sending event to client: {e}")
				break

	except WebSocketDisconnect:
		logger.info(f"[WebSocket] Client disconnected from experiment {experiment_id}")
	except Exception as e:
		logger.error(f"[WebSocket] Unexpected error for experiment {experiment_id}: {e}")
	finally:
		await broadcaster.unsubscribe(subscription_key, queue)
		logger.info(f"[WebSocket] Cleaned up subscription for experiment {experiment_id}")


@router.get("/ws/tasks/{task_id}/subscribers")
async def get_task_subscribers(task_id: int):
	"""
	Get the number of active WebSocket subscribers for a task.

	Useful for debugging and monitoring WebSocket connections.

	Args:
	    task_id: Task ID to check

	Returns:
	    Dictionary with subscriber count
	"""
	broadcaster = get_broadcaster()
	count = await broadcaster.get_subscriber_count(task_id)

	return {"task_id": task_id, "subscriber_count": count, "status": "active" if count > 0 else "inactive"}
