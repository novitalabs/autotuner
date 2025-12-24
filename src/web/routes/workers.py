"""
Worker management API endpoints.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from web.schemas.worker import (
	WorkerRegister,
	WorkerHeartbeat,
	WorkerResponse,
	WorkerListResponse,
	WorkerRenameRequest,
	WorkerStatus,
)
from web.workers.registry import (
	get_worker_registry,
	worker_info_to_response,
)

router = APIRouter()


@router.get("", response_model=WorkerListResponse)
async def list_workers(include_offline: bool = False):
	"""List all registered workers.

	Args:
		include_offline: Include workers that missed heartbeats
	"""
	registry = await get_worker_registry()
	workers = await registry.get_all_workers(include_offline=include_offline)

	responses = [worker_info_to_response(w) for w in workers]

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

	Args:
		worker_id: Worker identifier
		request: New alias or null to clear
	"""
	registry = await get_worker_registry()
	worker = await registry.set_worker_alias(worker_id, request.alias)

	if not worker:
		raise HTTPException(status_code=404, detail=f"Worker not found: {worker_id}")

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
