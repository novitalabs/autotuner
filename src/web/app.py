"""
FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
import orjson
import logging

# Configure logging for the application
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from web.config import get_settings
from web.db.session import init_db, get_db
from web.db.seed_presets import seed_system_presets
from web.routes import tasks, experiments, system, docker, presets, runtime_params, dashboard, websocket, ome_resources, agent, workers
from web.services.result_listener import start_result_listener, stop_result_listener, get_result_listener
from web.workers.pubsub import ExperimentResult, WorkerEvent, TaskStatusUpdate
from web.db.models import Task, Experiment, TaskStatus, ExperimentStatus
from web.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


async def sync_experiment_to_local_db(result: ExperimentResult) -> bool:
	"""Sync experiment result from remote worker to local database.

	This function creates or updates experiment records in the local database
	based on results published via Redis Pub/Sub from distributed workers.

	Args:
		result: ExperimentResult from remote worker

	Returns:
		True if sync succeeded, False otherwise
	"""
	try:
		async with AsyncSessionLocal() as db:
			from sqlalchemy import select, update

			# Check if task exists locally
			task_result = await db.execute(
				select(Task).where(Task.id == result.task_id)
			)
			task = task_result.scalar_one_or_none()

			if not task:
				logger.warning(
					f"Task {result.task_id} not found in local database, "
					"cannot sync experiment result"
				)
				return False

			# Check if experiment already exists (by task_id and experiment_id)
			exp_result = await db.execute(
				select(Experiment).where(
					Experiment.task_id == result.task_id,
					Experiment.experiment_id == result.experiment_id
				)
			)
			experiment = exp_result.scalar_one_or_none()

			# Map status string to enum
			status_map = {
				"success": ExperimentStatus.SUCCESS,
				"failed": ExperimentStatus.FAILED,
				"pending": ExperimentStatus.PENDING,
				"deploying": ExperimentStatus.DEPLOYING,
				"benchmarking": ExperimentStatus.BENCHMARKING,
			}
			exp_status = status_map.get(result.status, ExperimentStatus.FAILED)

			if experiment:
				# Update existing experiment
				experiment.status = exp_status
				experiment.metrics = result.metrics
				experiment.objective_score = result.objective_score
				experiment.error_message = result.error_message
				experiment.completed_at = result.timestamp
				experiment.elapsed_time = result.elapsed_time
				logger.info(
					f"Updated experiment {result.experiment_id} for task {result.task_id} "
					f"(status={result.status}, score={result.objective_score})"
				)
			else:
				# Create new experiment
				experiment = Experiment(
					task_id=result.task_id,
					experiment_id=result.experiment_id,
					parameters=result.parameters,
					status=exp_status,
					metrics=result.metrics,
					objective_score=result.objective_score,
					error_message=result.error_message,
					started_at=result.timestamp,
					completed_at=result.timestamp,
					elapsed_time=result.elapsed_time,
				)
				db.add(experiment)
				logger.info(
					f"Created experiment {result.experiment_id} for task {result.task_id} "
					f"(status={result.status}, score={result.objective_score})"
				)

			await db.commit()

			# Update task counters
			# Count experiments for this task
			count_result = await db.execute(
				select(Experiment).where(Experiment.task_id == result.task_id)
			)
			all_experiments = count_result.scalars().all()

			total = len(all_experiments)
			successful = len([e for e in all_experiments if e.status == ExperimentStatus.SUCCESS])

			# Find best experiment (lowest score among successful)
			best_exp_id = None
			best_score = float("inf")
			for exp in all_experiments:
				if exp.status == ExperimentStatus.SUCCESS and exp.objective_score is not None:
					if exp.objective_score < best_score:
						best_score = exp.objective_score
						best_exp_id = exp.id

			# Update task
			await db.execute(
				update(Task).where(Task.id == result.task_id).values(
					total_experiments=total,
					successful_experiments=successful,
					best_experiment_id=best_exp_id,
				)
			)
			await db.commit()

			logger.info(
				f"Task {result.task_id} updated: total={total}, successful={successful}, "
				f"best_exp_id={best_exp_id}"
			)

			return True

	except Exception as e:
		logger.error(f"Failed to sync experiment to local database: {e}", exc_info=True)
		return False


async def sync_task_status_to_local_db(task_status: TaskStatusUpdate) -> bool:
	"""Sync task status from remote worker to local database.

	This function updates the task status in the local database
	based on status updates published via Redis Pub/Sub from distributed workers.

	Args:
		task_status: TaskStatusUpdate from remote worker

	Returns:
		True if sync succeeded, False otherwise
	"""
	try:
		async with AsyncSessionLocal() as db:
			from sqlalchemy import select, update

			# Check if task exists locally
			task_result = await db.execute(
				select(Task).where(Task.id == task_status.task_id)
			)
			task = task_result.scalar_one_or_none()

			if not task:
				logger.warning(
					f"Task {task_status.task_id} not found in local database, "
					"cannot sync task status"
				)
				return False

			# Map status string to enum (case-insensitive)
			status_map = {
				"completed": TaskStatus.COMPLETED,
				"failed": TaskStatus.FAILED,
				"cancelled": TaskStatus.CANCELLED,
				"running": TaskStatus.RUNNING,
				"pending": TaskStatus.PENDING,
			}
			new_status = status_map.get(task_status.status.lower())

			if not new_status:
				logger.warning(f"Unknown task status: {task_status.status}")
				return False

			# Update task status
			update_values = {
				"status": new_status,
				"total_experiments": task_status.total_experiments,
				"successful_experiments": task_status.successful_experiments,
			}

			if task_status.best_experiment_id:
				update_values["best_experiment_id"] = task_status.best_experiment_id

			if task_status.elapsed_time:
				update_values["elapsed_time"] = task_status.elapsed_time

			if new_status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
				update_values["completed_at"] = task_status.timestamp

			await db.execute(
				update(Task).where(Task.id == task_status.task_id).values(**update_values)
			)
			await db.commit()

			logger.info(
				f"Task {task_status.task_id} status synced: status={task_status.status}, "
				f"total={task_status.total_experiments}, successful={task_status.successful_experiments}"
			)

			return True

	except Exception as e:
		logger.error(f"Failed to sync task status to local database: {e}", exc_info=True)
		return False


class CustomORJSONResponse(ORJSONResponse):
	"""Custom ORJSON response with UTC timezone handling."""
	
	@staticmethod
	def orjson_default(obj):
		"""Custom serializer for types not handled by orjson."""
		if isinstance(obj, datetime):
			# Add 'Z' suffix to indicate UTC timezone
			return obj.isoformat() + 'Z'
		raise TypeError(f"Type {type(obj)} not serializable")
	
	def render(self, content) -> bytes:
		"""Render with custom default serializer."""
		return orjson.dumps(content, default=self.orjson_default)

@asynccontextmanager
async def lifespan(app: FastAPI):
	"""Application lifespan events."""
	# Startup
	print("🚀 Starting LLM Autotuner API...")
	await init_db()
	print("✅ Database initialized")

	# Seed system presets
	async for db in get_db():
		await seed_system_presets(db)
		break

	# Start result listener for Pub/Sub
	try:
		listener = await start_result_listener()

		# Register callbacks for result handling
		async def on_result(result: ExperimentResult):
			"""Handle experiment result from distributed workers."""
			logger.info(
				f"📥 Received result via Pub/Sub: task={result.task_id} "
				f"exp={result.experiment_id} status={result.status} "
				f"worker={result.worker_id}"
			)
			# Sync result to local database for distributed workers
			synced = await sync_experiment_to_local_db(result)
			if synced:
				logger.info(f"✅ Synced experiment {result.experiment_id} to local database")
			else:
				logger.warning(f"⚠️ Failed to sync experiment {result.experiment_id} to local database")

		async def on_worker_event(event: WorkerEvent):
			"""Handle worker status events."""
			logger.debug(
				f"📥 Worker event: {event.worker_id} - {event.event_type}"
			)

		async def on_task_status(task_status: TaskStatusUpdate):
			"""Handle task status updates from distributed workers."""
			logger.info(
				f"📥 Received task status via Pub/Sub: task={task_status.task_id} "
				f"status={task_status.status} worker={task_status.worker_id}"
			)
			# Sync task status to local database for distributed workers
			synced = await sync_task_status_to_local_db(task_status)
			if synced:
				logger.info(f"✅ Synced task {task_status.task_id} status to local database")
			else:
				logger.warning(f"⚠️ Failed to sync task {task_status.task_id} status to local database")

		listener.on_result(on_result)
		listener.on_worker_event(on_worker_event)
		listener.on_task_status(on_task_status)

		print("📡 Result listener started (Redis Pub/Sub)")
	except Exception as e:
		print(f"⚠️ Failed to start result listener: {e}")

	yield
	# Shutdown
	print("👋 Shutting down...")

	# Stop result listener
	try:
		await stop_result_listener()
		print("📡 Result listener stopped")
	except Exception as e:
		print(f"⚠️ Error stopping result listener: {e}")


# Create FastAPI app with custom JSON serialization
settings = get_settings()
app = FastAPI(
	title=settings.app_name,
	version=settings.app_version,
	description="API for automated LLM inference parameter tuning",
	lifespan=lifespan,
	default_response_class=CustomORJSONResponse,
)

# CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=settings.cors_origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Include routers
app.include_router(tasks.router, prefix="/api/tasks", tags=["tasks"])
app.include_router(experiments.router, prefix="/api/experiments", tags=["experiments"])
app.include_router(system.router, prefix="/api/system", tags=["system"])
app.include_router(docker.router, prefix="/api/docker", tags=["docker"])
app.include_router(presets.router)
app.include_router(runtime_params.router)
app.include_router(dashboard.router)
app.include_router(websocket.router, prefix="/api", tags=["websocket"])
app.include_router(ome_resources.router)
app.include_router(agent.router)
app.include_router(workers.router, prefix="/api/workers", tags=["workers"])


@app.get("/")
async def root():
	"""Root endpoint."""
	return {
		"name": settings.app_name,
		"version": settings.app_version,
		"status": "running",
		"docs": "/docs",
	}


@app.get("/health")
async def health():
	"""Health check endpoint."""
	return {"status": "healthy"}
