"""
FastAPI application entry point.
"""

import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime
import orjson

from web.config import get_settings
from web.db.session import init_db, get_db
from web.db.seed_presets import seed_system_presets
from web.routes import tasks, experiments, system, docker, presets, runtime_params, dashboard, websocket, ome_resources, agent


# Detect frontend dist directory early (before app creation)
def _find_frontend_dist() -> Path | None:
	"""Find the frontend dist directory."""
	# Project-relative path (src/web/app.py -> project root)
	project_root = Path(__file__).parent.parent.parent
	frontend_dist = project_root / "frontend" / "dist"

	if frontend_dist.exists() and (frontend_dist / "index.html").exists():
		return frontend_dist

	# Container paths
	for path in [Path("/app/frontend/dist"), Path("/opt/autotuner/frontend/dist")]:
		if path.exists() and (path / "index.html").exists():
			return path

	return None

# Detect frontend before creating routes
FRONTEND_PATH = _find_frontend_dist()





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

	yield
	# Shutdown
	print("👋 Shutting down...")


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


# Root endpoint - serve frontend if available, otherwise API info
if FRONTEND_PATH:
	print(f"📦 Serving frontend from: {FRONTEND_PATH}")

	# Mount assets directory for JS/CSS files
	assets_path = FRONTEND_PATH / "assets"
	if assets_path.exists():
		app.mount("/assets", StaticFiles(directory=str(assets_path)), name="static-assets")

	@app.get("/", response_class=HTMLResponse)
	async def serve_frontend_root():
		"""Serve frontend index.html for root path."""
		index_file = FRONTEND_PATH / "index.html"
		if index_file.exists():
			return FileResponse(index_file, media_type="text/html")
		return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
else:
	print("⚠️  Frontend dist not found, serving API only")

	@app.get("/")
	async def root():
		"""Root endpoint - API info."""
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


# SPA catch-all route for client-side routing (only if frontend exists)
if FRONTEND_PATH:
	from starlette.exceptions import HTTPException as StarletteHTTPException

	@app.exception_handler(404)
	async def spa_not_found_handler(request: Request, exc: StarletteHTTPException):
		"""Handle 404s by serving index.html for SPA routes."""
		path = request.url.path

		# Skip API routes - return proper JSON 404
		if path.startswith("/api/"):
			return CustomORJSONResponse(
				status_code=404,
				content={"detail": "Not Found"}
			)

		# For non-API routes, serve index.html for SPA routing
		index_file = FRONTEND_PATH / "index.html"
		if index_file.exists():
			return FileResponse(index_file, media_type="text/html")

		return HTMLResponse("<h1>Not found</h1>", status_code=404)
