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
from web.workers.pubsub import ExperimentResult, WorkerEvent

logger = logging.getLogger(__name__)





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
			# Results are already saved to DB by the worker
			# This callback is for additional processing like notifications

		async def on_worker_event(event: WorkerEvent):
			"""Handle worker status events."""
			logger.debug(
				f"📥 Worker event: {event.worker_id} - {event.event_type}"
			)

		listener.on_result(on_result)
		listener.on_worker_event(on_worker_event)

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
