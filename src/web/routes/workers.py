"""
Worker management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from web.schemas.worker import (
	WorkerRegister,
	WorkerHeartbeat,
	WorkerResponse,
	WorkerListResponse,
	WorkerRenameRequest,
	WorkerStatus,
	WorkerSlotCreate,
	WorkerSlotResponse,
	WorkerSlotListResponse,
	WorkerSlotRestoreRequest,
	WorkerSlotRestoreResponse,
	WorkerSlotDeployRequest,
	WorkerSlotStatus,
)
from web.workers.registry import (
	get_worker_registry,
	worker_info_to_response,
)
from web.workers.pubsub import get_result_publisher
from web.db.session import get_db
from web.services.worker_service import WorkerService, DeploymentConfig

router = APIRouter()


# ============== Worker Slot Endpoints (must be BEFORE /{worker_id} routes) ==============


def _slot_to_response(slot) -> WorkerSlotResponse:
	"""Convert WorkerSlot model to response schema."""
	from web.db.models import WorkerSlotStatus as DBStatus

	# Map DB status to schema status
	status_map = {
		DBStatus.ONLINE: WorkerSlotStatus.ONLINE,
		DBStatus.OFFLINE: WorkerSlotStatus.OFFLINE,
		DBStatus.UNKNOWN: WorkerSlotStatus.UNKNOWN,
	}

	return WorkerSlotResponse(
		id=slot.id,
		worker_id=slot.worker_id,
		name=slot.name,
		controller_type=slot.controller_type,
		ssh_command=slot.ssh_command,
		ssh_forward_tunnel=slot.ssh_forward_tunnel,
		ssh_reverse_tunnel=slot.ssh_reverse_tunnel,
		project_path=slot.project_path,
		manager_ssh=slot.manager_ssh,
		current_status=status_map.get(slot.current_status, WorkerSlotStatus.UNKNOWN),
		last_seen_at=slot.last_seen_at,
		last_error=slot.last_error,
		hostname=slot.hostname,
		gpu_count=slot.gpu_count,
		gpu_model=slot.gpu_model,
		created_at=slot.created_at,
		updated_at=slot.updated_at,
	)


@router.get("/slots", response_model=WorkerSlotListResponse)
async def list_worker_slots(db: AsyncSession = Depends(get_db)):
	"""List all worker slots (persistent deployment configurations).

	Returns both online and offline workers, allowing dashboard to show
	all registered workers and provide restore functionality.
	"""
	service = WorkerService(db)
	slots = await service.get_all_slots()

	responses = [_slot_to_response(s) for s in slots]

	# Count by status
	online_count = sum(1 for s in responses if s.current_status == WorkerSlotStatus.ONLINE)
	offline_count = sum(1 for s in responses if s.current_status == WorkerSlotStatus.OFFLINE)
	unknown_count = sum(1 for s in responses if s.current_status == WorkerSlotStatus.UNKNOWN)

	return WorkerSlotListResponse(
		slots=responses,
		total_count=len(responses),
		online_count=online_count,
		offline_count=offline_count,
		unknown_count=unknown_count,
	)


@router.post("/slots", response_model=WorkerSlotResponse)
async def create_worker_slot(
	request: WorkerSlotCreate,
	db: AsyncSession = Depends(get_db)
):
	"""Create a new worker slot (without deploying).

	This stores the deployment configuration but doesn't start the worker.
	Use /slots/{id}/restore to start the worker later.
	"""
	service = WorkerService(db)

	config = DeploymentConfig(
		ssh_command=request.ssh_command,
		name=request.name,
		controller_type=request.controller_type,
		project_path=request.project_path,
		manager_ssh=request.manager_ssh,
		ssh_forward_tunnel=request.ssh_forward_tunnel,
		ssh_reverse_tunnel=request.ssh_reverse_tunnel,
		auto_install=False,  # Just create slot, don't deploy
	)

	slot = await service.create_or_update_slot(config)
	return _slot_to_response(slot)


@router.post("/slots/deploy", response_model=WorkerSlotRestoreResponse)
async def deploy_new_worker(
	request: WorkerSlotDeployRequest,
	db: AsyncSession = Depends(get_db)
):
	"""Deploy a new worker (create slot + start worker).

	This is the main endpoint for adding a new remote worker:
	1. Creates a worker slot in the database
	2. Connects to the remote machine via SSH
	3. Optionally installs the project if not found
	4. Starts the worker process
	5. Waits for the worker to register with the manager
	"""
	service = WorkerService(db)

	config = DeploymentConfig(
		ssh_command=request.ssh_command,
		name=request.name,
		controller_type=request.controller_type,
		project_path=request.project_path,
		manager_ssh=request.manager_ssh,
		ssh_forward_tunnel=request.ssh_forward_tunnel,
		ssh_reverse_tunnel=request.ssh_reverse_tunnel,
		auto_install=request.auto_install,
	)

	result = await service.deploy_worker(config)

	return WorkerSlotRestoreResponse(
		success=result.success,
		message=result.message,
		worker_id=result.worker_id,
		slot_id=result.slot_id or 0,
		error=result.error,
		logs=result.logs,
		worker_info=result.worker_info,
	)


@router.get("/slots/{slot_id}", response_model=WorkerSlotResponse)
async def get_worker_slot(slot_id: int, db: AsyncSession = Depends(get_db)):
	"""Get a specific worker slot's details."""
	service = WorkerService(db)
	slot = await service.get_slot_by_id(slot_id)

	if not slot:
		raise HTTPException(status_code=404, detail=f"Worker slot not found: {slot_id}")

	return _slot_to_response(slot)


@router.post("/slots/{slot_id}/restore", response_model=WorkerSlotRestoreResponse)
async def restore_worker_slot(
	slot_id: int,
	request: WorkerSlotRestoreRequest = WorkerSlotRestoreRequest(),
	db: AsyncSession = Depends(get_db)
):
	"""Restore an offline worker by its slot ID.

	This connects to the remote machine via SSH and starts the worker process.
	If auto_install is True, it will also install the project if not found.
	"""
	service = WorkerService(db)
	result = await service.restore_worker(slot_id, auto_install=request.auto_install)

	return WorkerSlotRestoreResponse(
		success=result.success,
		message=result.message,
		worker_id=result.worker_id,
		slot_id=slot_id,
		error=result.error,
		logs=result.logs,
		worker_info=result.worker_info,
	)


@router.delete("/slots/{slot_id}")
async def delete_worker_slot(slot_id: int, db: AsyncSession = Depends(get_db)):
	"""Delete a worker slot configuration.

	Note: This only removes the persistent configuration. If the worker is
	currently running, it will continue to run but won't be restorable.
	"""
	service = WorkerService(db)
	success = await service.delete_slot(slot_id)

	if not success:
		raise HTTPException(status_code=404, detail=f"Worker slot not found: {slot_id}")

	return {"status": "ok", "message": f"Worker slot {slot_id} deleted"}


# ============== Worker Registry Endpoints (from Redis) ==============


@router.get("", response_model=WorkerListResponse)
async def list_workers(include_offline: bool = False):
	"""List all registered workers.

	Args:
		include_offline: Include workers that missed heartbeats
	"""
	registry = await get_worker_registry()
	workers = await registry.get_all_workers(include_offline=include_offline)

	responses = [worker_info_to_response(w) for w in workers]
	# Sort by worker_id for consistent ordering
	responses.sort(key=lambda w: w.worker_id)

	# Count by status
	online_count = sum(1 for w in responses if w.status == WorkerStatus.ONLINE)
	busy_count = sum(1 for w in responses if w.status == WorkerStatus.BUSY)
	offline_count = sum(1 for w in responses if w.status == WorkerStatus.OFFLINE)

	return WorkerListResponse(
		workers=responses,
		total_count=len(responses),
		online_count=online_count,
		busy_count=busy_count,
		offline_count=offline_count,
	)


@router.get("/{worker_id}", response_model=WorkerResponse)
async def get_worker(worker_id: str):
	"""Get a specific worker's information."""
	registry = await get_worker_registry()
	worker = await registry.get_worker(worker_id)

	if not worker:
		raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

	return worker_info_to_response(worker)


@router.post("/register", response_model=WorkerResponse)
async def register_worker(registration: WorkerRegister):
	"""Register a new worker or update existing registration.

	Workers should call this endpoint on startup to announce their availability.
	"""
	registry = await get_worker_registry()
	worker = await registry.register(registration)
	return worker_info_to_response(worker)


@router.post("/heartbeat", response_model=WorkerResponse)
async def worker_heartbeat(heartbeat: WorkerHeartbeat):
	"""Update worker heartbeat.

	Workers should call this every 30 seconds to maintain their registration.
	"""
	registry = await get_worker_registry()
	worker = await registry.heartbeat(heartbeat)

	if not worker:
		raise HTTPException(status_code=404, detail=f"Worker not found: {heartbeat.worker_id}")

	return worker_info_to_response(worker)


@router.delete("/{worker_id}")
async def deregister_worker(worker_id: str):
	"""Deregister a worker (graceful shutdown).

	Workers should call this when shutting down gracefully.
	"""
	registry = await get_worker_registry()
	success = await registry.deregister(worker_id)

	if not success:
		raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

	return {"status": "ok", "message": f"Worker {worker_id} deregistered"}


@router.get("/available", response_model=List[WorkerResponse])
async def get_available_workers():
	"""Get workers available for new jobs.

	Returns workers that have capacity for additional experiments.
	"""
	registry = await get_worker_registry()
	workers = await registry.get_available_workers()
	return [worker_info_to_response(w) for w in workers]


@router.patch("/{worker_id}/alias", response_model=WorkerResponse)
async def rename_worker(worker_id: str, request: WorkerRenameRequest):
	"""Set or clear a worker's alias (nickname).

	This updates both:
	1. The registry (Redis) - immediate effect
	2. The worker's local config file (via pub/sub) - persistent

	Args:
		worker_id: Worker identifier
		request: New alias or null to clear
	"""
	registry = await get_worker_registry()
	worker = await registry.set_worker_alias(worker_id, request.alias)

	if not worker:
		raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

	# Publish config update to worker so it saves to local config file
	try:
		publisher = await get_result_publisher()
		await publisher.publish_config_update(worker_id, {"alias": request.alias})
	except Exception as e:
		# Log but don't fail - registry update succeeded
		import logging
		logging.warning(f"Failed to publish config update to worker {worker_id}: {e}")

	return worker_info_to_response(worker)


@router.get("/{worker_id}/gpu-history")
async def get_worker_gpu_history(worker_id: str):
	"""Get GPU metrics history for a worker.

	Returns historical GPU utilization, memory, and temperature data
	for building charts (last 20 heartbeats, ~10 minutes at 30s interval).
	"""
	registry = await get_worker_registry()
	worker = await registry.get_worker(worker_id)

	if not worker:
		raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

	history = await registry.get_gpu_history(worker_id)
	# Reverse to get oldest first (for chart display)
	return {"worker_id": worker_id, "history": list(reversed(history))}


class WorkerConfigUpdate(BaseModel):
	"""Request body for updating worker config."""
	alias: Optional[str] = None
	deployment_mode: Optional[str] = None
	max_parallel: Optional[int] = None
	description: Optional[str] = None
	tags: Optional[List[str]] = None


@router.patch("/{worker_id}/config")
async def update_worker_config(worker_id: str, request: WorkerConfigUpdate):
	"""Update worker's local configuration file.

	This sends config updates to the worker via pub/sub.
	The worker will save updates to ~/.config/autotuner/worker.json

	Args:
		worker_id: Worker identifier
		request: Config fields to update (only non-null fields are updated)
	"""
	registry = await get_worker_registry()
	worker = await registry.get_worker(worker_id)

	if not worker:
		raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

	# Build updates dict with only non-null fields
	updates = {k: v for k, v in request.model_dump().items() if v is not None}

	if not updates:
		raise HTTPException(status_code=400, detail="No config fields to update")

	# If alias is updated, also update registry
	if "alias" in updates:
		await registry.set_worker_alias(worker_id, updates["alias"])

	# Publish config update to worker
	try:
		publisher = await get_result_publisher()
		subscribers = await publisher.publish_config_update(worker_id, updates)

		return {
			"status": "ok",
			"worker_id": worker_id,
			"updates": updates,
			"subscribers": subscribers,
			"message": f"Config update sent to {subscribers} subscriber(s)"
		}
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Failed to send config update: {e}")

