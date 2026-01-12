"""
Pydantic schemas for API request/response models.
"""

import os
from pydantic import BaseModel, Field, field_serializer
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# Default deployment mode from environment
DEFAULT_DEPLOYMENT_MODE = os.environ.get("DEPLOYMENT_MODE", "docker")


class TaskStatusEnum(str, Enum):
	"""Task status enum."""

	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class ExperimentStatusEnum(str, Enum):
	"""Experiment status enum."""

	PENDING = "pending"
	DEPLOYING = "deploying"
	BENCHMARKING = "benchmarking"
	SUCCESS = "success"
	FAILED = "failed"


class OptimizationStrategyEnum(str, Enum):
	"""Optimization strategy enum."""

	GRID_SEARCH = "grid_search"
	BAYESIAN = "bayesian"
	RANDOM = "random"


# Task schemas
class TaskCreate(BaseModel):
	"""Schema for creating a task."""

	task_name: str = Field(..., description="Unique task name")
	description: Optional[str] = Field(None, description="Task description")
	model: Dict[str, Any] = Field(..., description="Model configuration")
	base_runtime: str = Field(..., description="Runtime (sglang, vllm)")
	runtime_image_tag: Optional[str] = Field(None, description="Docker image tag")
	parameters: Dict[str, Any] = Field(..., description="Parameter grid")
	optimization: Dict[str, Any] = Field(..., description="Optimization settings")
	benchmark: Dict[str, Any] = Field(..., description="Benchmark configuration")
	slo: Optional[Dict[str, Any]] = Field(None, description="SLO configuration")
	quant_config: Optional[Dict[str, Any]] = Field(None, description="Quantization configuration")
	parallel_config: Optional[Dict[str, Any]] = Field(None, description="Parallel execution configuration")
	clusterbasemodel_config: Optional[Dict[str, Any]] = Field(
		None, description="ClusterBaseModel preset or custom config (OME mode)"
	)
	clusterservingruntime_config: Optional[Dict[str, Any]] = Field(
		None, description="ClusterServingRuntime preset or custom config (OME mode)"
	)
	deployment_mode: str = Field(default_factory=lambda: DEFAULT_DEPLOYMENT_MODE, description="Deployment mode")


class TaskUpdate(BaseModel):
	"""Schema for updating a task."""

	description: Optional[str] = None
	status: Optional[TaskStatusEnum] = None


class TaskResponse(BaseModel):
	"""Schema for task response."""

	model_config = {"from_attributes": True, "populate_by_name": True}

	id: int
	task_name: str
	description: Optional[str]
	status: TaskStatusEnum
	model: Dict[str, Any] = Field(alias="model_config", serialization_alias="model")
	base_runtime: str
	runtime_image_tag: Optional[str]
	parameters: Dict[str, Any]
	optimization: Dict[str, Any] = Field(alias="optimization_config", serialization_alias="optimization")
	benchmark: Dict[str, Any] = Field(alias="benchmark_config", serialization_alias="benchmark")
	slo: Optional[Dict[str, Any]] = Field(None, alias="slo_config", serialization_alias="slo")
	quant_config: Optional[Dict[str, Any]] = None
	parallel_config: Optional[Dict[str, Any]] = None
	clusterbasemodel_config: Optional[Dict[str, Any]] = None
	clusterservingruntime_config: Optional[Dict[str, Any]] = None
	created_clusterbasemodel: Optional[str] = None
	created_clusterservingruntime: Optional[str] = None
	deployment_mode: str
	total_experiments: int
	successful_experiments: int
	best_experiment_id: Optional[int]
	created_at: datetime
	started_at: Optional[datetime]
	completed_at: Optional[datetime]
	elapsed_time: Optional[float]

	# Datetime serializers to add 'Z' suffix for UTC timezone
	@field_serializer('created_at', 'started_at', 'completed_at', when_used='json')
	def serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
		if dt is None:
			return None
		return dt.isoformat() + 'Z'


class TaskListResponse(BaseModel):
	"""Schema for task list response."""

	model_config = {"from_attributes": True, "populate_by_name": True}

	id: int
	task_name: str
	description: Optional[str]
	status: TaskStatusEnum
	base_runtime: str
	total_experiments: int
	successful_experiments: int
	best_experiment_id: Optional[int]
	created_at: datetime
	elapsed_time: Optional[float]
	slo: Optional[Dict[str, Any]] = Field(None, alias="slo_config", serialization_alias="slo")

	# Datetime serializers to add 'Z' suffix for UTC timezone
	@field_serializer('created_at', when_used='json')
	def serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
		if dt is None:
			return None
		return dt.isoformat() + 'Z'


# Experiment schemas
class ExperimentResponse(BaseModel):
	"""Schema for experiment response."""

	model_config = {"from_attributes": True}

	id: int
	task_id: int
	experiment_id: int
	parameters: Dict[str, Any]
	status: ExperimentStatusEnum
	error_message: Optional[str]
	metrics: Optional[Dict[str, Any]]
	objective_score: Optional[float]
	gpu_info: Optional[Dict[str, Any]]  # GPU information: model, count, device_ids, world_size
	service_name: Optional[str]
	service_url: Optional[str]
	created_at: datetime
	started_at: Optional[datetime]
	completed_at: Optional[datetime]
	elapsed_time: Optional[float]

	# Datetime serializers to add 'Z' suffix for UTC timezone
	@field_serializer('created_at', 'started_at', 'completed_at', when_used='json')
	def serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
		if dt is None:
			return None
		return dt.isoformat() + 'Z'


# Progress update schema
class ProgressUpdate(BaseModel):
	"""Schema for progress updates (SSE)."""

	task_id: int
	task_name: str
	status: TaskStatusEnum
	current_experiment: Optional[int] = None
	total_experiments: int
	successful_experiments: int
	progress_percent: float
	message: str
	experiment: Optional[ExperimentResponse] = None


# System schemas
class HealthResponse(BaseModel):
	"""Health check response."""

	status: str
	database: str = "ok"
	redis: str = "ok"


class SystemInfoResponse(BaseModel):
	"""System information response."""

	app_name: str
	version: str
	deployment_mode: str
	available_runtimes: List[str]
	timezone: str = "UTC"
