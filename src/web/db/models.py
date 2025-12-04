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

	# Deployment mode
	deployment_mode = Column(String, default="docker")  # docker, ome

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
	context_summary = Column(Text, nullable=True)  # Long-term context summary
	is_active = Column(Boolean, default=True, index=True)
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
