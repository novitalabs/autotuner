"""
Worker schemas for distributed ARQ worker system.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
from datetime import datetime
from enum import Enum


class WorkerStatus(str, Enum):
	"""Worker status enum."""

	ONLINE = "online"  # Worker is available and ready
	BUSY = "busy"  # Worker is processing jobs
	OFFLINE = "offline"  # Worker is not responding


class GPUInfo(BaseModel):
	"""GPU information for a worker."""

	index: int = Field(..., description="GPU device index")
	name: str = Field(..., description="GPU model name (e.g., 'NVIDIA RTX 4090')")
	memory_total_gb: float = Field(..., description="Total GPU memory in GB")
	memory_free_gb: Optional[float] = Field(None, description="Free GPU memory in GB")
	memory_used_gb: Optional[float] = Field(None, description="Used GPU memory in GB")
	utilization_percent: Optional[float] = Field(None, description="GPU utilization percentage")
	temperature_c: Optional[int] = Field(None, description="GPU temperature in Celsius")
	node_name: Optional[str] = Field(None, description="K8s node name (for OME/cluster mode)")


class WorkerCapabilities(BaseModel):
	"""Worker capabilities and supported features."""

	deployment_modes: List[str] = Field(default=["docker"], description="Supported deployment modes")
	runtimes: List[str] = Field(default=["sglang", "vllm"], description="Supported runtimes")
	docker_available: bool = Field(default=True, description="Docker with GPU support available")
	local_venvs: List[str] = Field(default=[], description="Available local venvs (e.g., '.venv-vllm', '.venv-sglang')")


class WorkerRegister(BaseModel):
	"""Schema for worker registration request."""

	worker_id: str = Field(..., description="Unique worker identifier (UUID or hostname-based)")
	hostname: str = Field(..., description="Worker hostname")
	alias: Optional[str] = Field(None, description="Worker alias from local config")
	ip_address: Optional[str] = Field(None, description="Worker IP address")
	gpu_count: int = Field(default=0, description="Number of GPUs available")
	gpu_model: Optional[str] = Field(None, description="GPU model name")
	gpu_memory_gb: Optional[float] = Field(None, description="Total GPU memory in GB")
	gpus: Optional[List[GPUInfo]] = Field(None, description="Detailed GPU information")
	deployment_mode: str = Field(default="docker", description="Default deployment mode")
	max_parallel: int = Field(default=2, description="Maximum concurrent experiments")
	capabilities: Optional[WorkerCapabilities] = Field(None, description="Worker capabilities")


class WorkerInfo(BaseModel):
	"""Full worker information stored in Redis."""

	worker_id: str
	hostname: str
	alias: Optional[str] = Field(None, description="User-defined worker nickname")
	ip_address: Optional[str] = None
	gpu_count: int = 0
	gpu_model: Optional[str] = None
	gpu_memory_gb: Optional[float] = None
	# Use Union to accept both GPUInfo objects and dicts (for Redis storage)
	gpus: Optional[List[Union[GPUInfo, dict]]] = None
	deployment_mode: str = "docker"
	max_parallel: int = 2
	current_jobs: int = 0
	current_job_ids: List[int] = Field(default_factory=list)
	status: WorkerStatus = WorkerStatus.ONLINE
	# Use Union to accept both WorkerCapabilities objects and dicts (for Redis storage)
	capabilities: Optional[Union[WorkerCapabilities, dict]] = None
	registered_at: datetime = Field(default_factory=datetime.utcnow)
	last_heartbeat: datetime = Field(default_factory=datetime.utcnow)


class WorkerHeartbeat(BaseModel):
	"""Schema for worker heartbeat update."""

	worker_id: str
	current_jobs: int = 0
	current_job_ids: List[int] = Field(default_factory=list)
	gpus: Optional[List[GPUInfo]] = Field(None, description="Current GPU status with metrics")
	status: Optional[WorkerStatus] = None


class WorkerResponse(BaseModel):
	"""Schema for worker API response."""

	worker_id: str
	hostname: str
	alias: Optional[str] = None
	ip_address: Optional[str] = None
	gpu_count: int = 0
	gpu_model: Optional[str] = None
	gpu_memory_gb: Optional[float] = None
	gpus: Optional[List[GPUInfo]] = None
	deployment_mode: str
	max_parallel: int
	current_jobs: int
	status: WorkerStatus
	registered_at: datetime
	last_heartbeat: datetime
	seconds_since_heartbeat: float = Field(..., description="Seconds since last heartbeat")


class WorkerRenameRequest(BaseModel):
	"""Schema for renaming a worker."""

	alias: Optional[str] = Field(None, description="New alias for the worker (null to clear)")


class WorkerListResponse(BaseModel):
	"""Schema for worker list API response."""

	workers: List[WorkerResponse]
	total_count: int
	online_count: int
	busy_count: int
	offline_count: int


# ============== Worker Slot Schemas ==============


class WorkerSlotStatus(str, Enum):
	"""Worker slot status enum."""

	ONLINE = "online"  # Worker is currently running
	OFFLINE = "offline"  # Worker is not responding
	UNKNOWN = "unknown"  # Never successfully connected


class WorkerSlotCreate(BaseModel):
	"""Schema for creating a worker slot."""

	name: Optional[str] = Field(None, description="Human-readable name/alias for the worker")
	ssh_command: str = Field(..., description="Full SSH command (e.g., 'ssh -p 18022 root@host')")
	controller_type: str = Field(default="docker", description="Controller type: docker, local, or ome")
	project_path: str = Field(default="/opt/inference-autotuner", description="Remote project path")
	manager_ssh: Optional[str] = Field(None, description="SSH command for worker to tunnel back to manager")
	ssh_forward_tunnel: Optional[str] = Field(None, description="SSH forward tunnel for Redis access")
	ssh_reverse_tunnel: Optional[str] = Field(None, description="SSH reverse tunnel configuration")


class WorkerSlotResponse(BaseModel):
	"""Schema for worker slot API response."""

	id: int
	worker_id: Optional[str] = None
	name: str
	controller_type: str
	ssh_command: str
	ssh_forward_tunnel: Optional[str] = None
	ssh_reverse_tunnel: Optional[str] = None
	project_path: str
	manager_ssh: Optional[str] = None
	current_status: WorkerSlotStatus
	last_seen_at: Optional[datetime] = None
	last_error: Optional[str] = None
	hostname: Optional[str] = None
	gpu_count: Optional[int] = None
	gpu_model: Optional[str] = None
	created_at: Optional[datetime] = None
	updated_at: Optional[datetime] = None

	class Config:
		from_attributes = True


class WorkerSlotListResponse(BaseModel):
	"""Schema for worker slot list API response."""

	slots: List[WorkerSlotResponse]
	total_count: int
	online_count: int
	offline_count: int
	unknown_count: int


class WorkerSlotRestoreRequest(BaseModel):
	"""Schema for restore worker request."""

	auto_install: bool = Field(default=False, description="Auto-install project if not found on remote")


class WorkerSlotRestoreResponse(BaseModel):
	"""Schema for restore worker response."""

	success: bool
	message: str
	worker_id: Optional[str] = None
	slot_id: int
	error: Optional[str] = None
	suggestion: Optional[str] = Field(None, description="Suggested fix for errors")
	logs: Optional[str] = None
	worker_info: Optional[dict] = None


class WorkerSlotDeployRequest(BaseModel):
	"""Schema for deploying a new worker (create slot + start)."""

	name: Optional[str] = Field(None, description="Human-readable name/alias for the worker")
	ssh_command: str = Field(..., description="Full SSH command (e.g., 'ssh -p 18022 root@host')")
	controller_type: str = Field(default="docker", description="Controller type: docker, local, or ome")
	project_path: str = Field(default="/opt/inference-autotuner", description="Remote project path")
	manager_ssh: Optional[str] = Field(None, description="SSH command for worker to tunnel back to manager")
	ssh_forward_tunnel: Optional[str] = Field(None, description="SSH forward tunnel for Redis access")
	ssh_reverse_tunnel: Optional[str] = Field(None, description="SSH reverse tunnel configuration")
	auto_install: bool = Field(default=True, description="Auto-install project if not found on remote")
