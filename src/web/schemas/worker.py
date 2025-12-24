"""
Worker schemas for distributed ARQ worker system.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
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
	gpus: Optional[List[GPUInfo]] = None
	deployment_mode: str = "docker"
	max_parallel: int = 2
	current_jobs: int = 0
	current_job_ids: List[int] = Field(default_factory=list)
	status: WorkerStatus = WorkerStatus.ONLINE
	capabilities: Optional[WorkerCapabilities] = None
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
