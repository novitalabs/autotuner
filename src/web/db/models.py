"""
Database models for tasks, experiments, and parameter presets.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, JSON, DateTime, Float, ForeignKey, Enum as SQLEnum, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

Base = declarative_base()


class TaskStatus(str, enum.Enum):
	"""Task status enum."""

	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class Task(Base):
	"""Autotuning task model."""

	__tablename__ = "tasks"

	id = Column(Integer, primary_key=True, index=True)
	task_name = Column(String, unique=True, index=True, nullable=False)
	description = Column(String)
	status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, index=True)

	# Configuration
	model_config = Column(JSON, nullable=False)  # model name, namespace
	base_runtime = Column(String, nullable=False)  # sglang, vllm
	runtime_image_tag = Column(String)
	parameters = Column(JSON, nullable=False)  # parameter grid
	optimization_config = Column(JSON, nullable=False)  # strategy, objective
	benchmark_config = Column(JSON, nullable=False)  # benchmark settings
	slo_config = Column(JSON, nullable=True)  # SLO configuration (ttft, tpot, latency, steepness)
	quant_config = Column(JSON, nullable=True)  # runtime quantization config (gemm_dtype, kvcache_dtype, attention_dtype, moe_dtype)
	parallel_config = Column(JSON, nullable=True)  # parallel execution config (tp, pp, dp, cp, moe_tp, moe_ep)

	# OME Resource Configuration (for auto-creation)
	clusterbasemodel_config = Column(JSON, nullable=True)  # ClusterBaseModel preset or custom config
	clusterservingruntime_config = Column(JSON, nullable=True)  # ClusterServingRuntime preset or custom config
	created_clusterbasemodel = Column(String, nullable=True)  # Name of CBM if auto-created by task
	created_clusterservingruntime = Column(String, nullable=True)  # Name of CSR if auto-created by task

	# Deployment mode and worker selection
	deployment_mode = Column(String, default="docker")  # docker, ome
	gpu_type = Column(String, nullable=True)  # GPU type filter for worker selection (e.g., "RTX 4090", "A100")

	# Metadata for checkpoints and other task state (using task_metadata as Python attribute name)
	task_metadata = Column("metadata", JSON, nullable=True)

	# Results
	total_experiments = Column(Integer, default=0)
	successful_experiments = Column(Integer, default=0)
	best_experiment_id = Column(Integer, ForeignKey("experiments.id"), nullable=True)

	# Timing
	created_at = Column(DateTime, default=datetime.utcnow)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	elapsed_time = Column(Float, nullable=True)

	# Relationships
	experiments = relationship("Experiment", back_populates="task", foreign_keys="Experiment.task_id")
	best_experiment = relationship("Experiment", foreign_keys=[best_experiment_id], post_update=True)

	def to_dict(self, include_full_config=False):
		"""
		Convert task to dictionary.

		Args:
			include_full_config: If True, include all configuration details.
							   If False, return summary view (for list endpoints).
		"""
		base = {
			"id": self.id,
			"task_name": self.task_name,
			"status": self.status.value if hasattr(self.status, 'value') else str(self.status),
			"deployment_mode": self.deployment_mode,
			"gpu_type": self.gpu_type,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"total_experiments": self.total_experiments,
			"successful_experiments": self.successful_experiments,
			"best_experiment_id": self.best_experiment_id,
		}

		if include_full_config:
			base.update({
				"description": self.description,
				"base_runtime": self.base_runtime,
				"runtime_image_tag": self.runtime_image_tag,
				"model_config": self.model_config,
				"parameters": self.parameters,
				"optimization_config": self.optimization_config,
				"benchmark_config": self.benchmark_config,
				"slo_config": self.slo_config,
				"quant_config": self.quant_config,
				"parallel_config": self.parallel_config,
				"clusterbasemodel_config": self.clusterbasemodel_config,
				"clusterservingruntime_config": self.clusterservingruntime_config,
				"created_clusterbasemodel": self.created_clusterbasemodel,
				"created_clusterservingruntime": self.created_clusterservingruntime,
				"task_metadata": self.task_metadata,
				"successful_experiments": self.successful_experiments,
				"failed_experiments": getattr(self, 'failed_experiments', 0),
				"started_at": self.started_at.isoformat() if self.started_at else None,
				"completed_at": self.completed_at.isoformat() if self.completed_at else None,
				"elapsed_time": self.elapsed_time,
			})

		return base


class ExperimentStatus(str, enum.Enum):
	"""Experiment status enum."""

	PENDING = "pending"
	DEPLOYING = "deploying"
	BENCHMARKING = "benchmarking"
	SUCCESS = "success"
	FAILED = "failed"


class Experiment(Base):
	"""Individual experiment (single parameter configuration) model."""

	__tablename__ = "experiments"

	id = Column(Integer, primary_key=True, index=True)
	task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
	experiment_id = Column(Integer, nullable=False)  # Sequential ID within task

	# Configuration
	parameters = Column(JSON, nullable=False)

	# Status
	status = Column(SQLEnum(ExperimentStatus), default=ExperimentStatus.PENDING)
	error_message = Column(String, nullable=True)

	# Results
	metrics = Column(JSON, nullable=True)
	objective_score = Column(Float, nullable=True, index=True)

	# GPU info
	gpu_info = Column(JSON, nullable=True)  # {model: str, count: int, device_ids: list, world_size: int}

	# Service info
	service_name = Column(String, nullable=True)
	service_url = Column(String, nullable=True)

	# Timing
	created_at = Column(DateTime, default=datetime.utcnow)
	started_at = Column(DateTime, nullable=True)
	completed_at = Column(DateTime, nullable=True)
	elapsed_time = Column(Float, nullable=True)

	# Relationship
	task = relationship("Task", back_populates="experiments", foreign_keys=[task_id])

	def to_dict(self, include_logs=False):
		"""
		Convert experiment to dictionary.

		Args:
			include_logs: If True, include benchmark_logs (can be large).
		"""
		data = {
			"id": self.id,
			"task_id": self.task_id,
			"experiment_id": self.experiment_id,
			"experiment_name": getattr(self, 'experiment_name', f"exp-{self.id}"),
			"status": self.status.value if hasattr(self.status, 'value') else str(self.status),
			"parameters": self.parameters,
			"metrics": self.metrics,
			"objective_score": self.objective_score,
			"slo_violations": getattr(self, 'slo_violations', None),
			"gpu_info": self.gpu_info,
			"service_name": self.service_name,
			"service_url": self.service_url,
			"error_message": self.error_message,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"started_at": self.started_at.isoformat() if self.started_at else None,
			"completed_at": self.completed_at.isoformat() if self.completed_at else None,
			"elapsed_time": self.elapsed_time,
		}

		if include_logs:
			data["benchmark_logs"] = getattr(self, 'benchmark_logs', None)

		return data


class ParameterPreset(Base):
	"""Parameter preset model for reusable parameter configurations."""

	__tablename__ = "parameter_presets"

	id = Column(Integer, primary_key=True, autoincrement=True)
	name = Column(String(255), nullable=False, unique=True, index=True)
	description = Column(Text)
	category = Column(String(100), index=True)
	runtime = Column(String(50), index=True)  # Runtime: sglang, vllm, or None for universal
	is_system = Column(Boolean, default=False, index=True)
	parameters = Column(JSON, nullable=False)
	preset_metadata = Column("metadata", JSON)  # Use different Python name
	created_at = Column(DateTime(timezone=True), server_default=func.now())
	updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())

	def to_dict(self):
		"""Convert model to dictionary."""
		return {
			"id": self.id,
			"name": self.name,
			"description": self.description,
			"category": self.category,
			"runtime": self.runtime,
			"is_system": self.is_system,
			"parameters": self.parameters,
			"metadata": self.preset_metadata,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
		}


class MessageRole(str, enum.Enum):
	"""Chat message role enum."""

	USER = "user"
	ASSISTANT = "assistant"
	SYSTEM = "system"


class ChatSession(Base):
	"""Agent chat session model."""

	__tablename__ = "chat_sessions"

	id = Column(Integer, primary_key=True, index=True)
	session_id = Column(String, unique=True, index=True, nullable=False)  # UUID
	user_id = Column(String, nullable=True, index=True)  # For future multi-user support
	title = Column(String, nullable=True)  # Session title (auto-generated or user-edited)
	context_summary = Column(Text, nullable=True)  # Long-term context summary
	is_active = Column(Boolean, default=True, index=True)
	session_metadata = Column("metadata", JSON, nullable=True)  # Session metadata including tool authorizations
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

	# Relationships
	messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
	subscriptions = relationship("AgentEventSubscription", back_populates="session", cascade="all, delete-orphan")


class ChatMessage(Base):
	"""Agent chat message model."""

	__tablename__ = "chat_messages"

	id = Column(Integer, primary_key=True, index=True)
	session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=False, index=True)
	role = Column(SQLEnum(MessageRole), nullable=False)
	content = Column(Text, nullable=False)
	tool_calls = Column(JSON, nullable=True)  # Record of tool executions
	message_metadata = Column("metadata", JSON, nullable=True)  # task_id, experiment_id references
	token_count = Column(Integer, nullable=True)  # For context management
	created_at = Column(DateTime, default=datetime.utcnow, index=True)

	# Relationship
	session = relationship("ChatSession", back_populates="messages")


class AgentEventSubscription(Base):
	"""Agent event subscription model for auto-triggering analysis."""

	__tablename__ = "agent_event_subscriptions"

	id = Column(Integer, primary_key=True, index=True)
	session_id = Column(String, ForeignKey("chat_sessions.session_id"), nullable=False, index=True)
	task_id = Column(Integer, ForeignKey("tasks.id"), nullable=False, index=True)
	event_types = Column(JSON, nullable=False)  # List of event types to monitor
	is_active = Column(Boolean, default=True, index=True)
	created_at = Column(DateTime, default=datetime.utcnow)
	expires_at = Column(DateTime, nullable=True)

	# Relationships
	session = relationship("ChatSession", back_populates="subscriptions")
	task = relationship("Task")


class WorkerSlotStatus(str, enum.Enum):
	"""Worker slot status enum."""

	ONLINE = "online"      # Worker is currently running
	OFFLINE = "offline"    # Worker is not responding
	UNKNOWN = "unknown"    # Never successfully connected


class WorkerSlot(Base):
	"""Persistent worker deployment configuration.

	This table stores deployment configurations for workers, allowing:
	- Persistence of worker configs across restarts
	- Display of offline workers in dashboard
	- One-click worker recovery
	"""

	__tablename__ = "worker_slots"

	id = Column(Integer, primary_key=True, index=True)

	# Worker identification
	worker_id = Column(String, unique=True, nullable=True, index=True)  # Set after first successful registration
	name = Column(String, nullable=False, index=True)  # Human-readable name/alias

	# Connection configuration
	controller_type = Column(String, nullable=False, default="docker")  # docker, local, ome
	ssh_command = Column(String, nullable=False)  # e.g., "ssh -p 18022 root@host"
	ssh_forward_tunnel = Column(String, nullable=True)  # For Redis access tunnel
	ssh_reverse_tunnel = Column(String, nullable=True)  # Optional reverse tunnel

	# Deployment configuration
	project_path = Column(String, default="/opt/inference-autotuner")  # Remote project path
	manager_ssh = Column(String, nullable=True)  # SSH command for worker to tunnel back to manager

	# Status tracking
	current_status = Column(SQLEnum(WorkerSlotStatus), default=WorkerSlotStatus.UNKNOWN)
	last_seen_at = Column(DateTime, nullable=True)  # Last heartbeat from Redis
	last_error = Column(Text, nullable=True)  # Last deployment/connection error

	# Hardware info (cached from last registration)
	hostname = Column(String, nullable=True)
	gpu_count = Column(Integer, nullable=True)
	gpu_model = Column(String, nullable=True)

	# Metadata
	created_at = Column(DateTime, default=datetime.utcnow)
	updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

	def to_dict(self) -> dict:
		"""Convert to dictionary."""
		return {
			"id": self.id,
			"worker_id": self.worker_id,
			"name": self.name,
			"controller_type": self.controller_type,
			"ssh_command": self.ssh_command,
			"ssh_forward_tunnel": self.ssh_forward_tunnel,
			"ssh_reverse_tunnel": self.ssh_reverse_tunnel,
			"project_path": self.project_path,
			"manager_ssh": self.manager_ssh,
			"current_status": self.current_status.value if self.current_status else "unknown",
			"last_seen_at": self.last_seen_at.isoformat() if self.last_seen_at else None,
			"last_error": self.last_error,
			"hostname": self.hostname,
			"gpu_count": self.gpu_count,
			"gpu_model": self.gpu_model,
			"created_at": self.created_at.isoformat() if self.created_at else None,
			"updated_at": self.updated_at.isoformat() if self.updated_at else None,
		}
