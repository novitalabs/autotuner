"""
ARQ worker configuration and task functions.
"""

import sys
import logging
import socket
import uuid
import subprocess
from pathlib import Path

# Add project root to path for imports
# workers/autotuner_worker.py -> web -> src -> project_root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from arq import create_pool
from arq.connections import RedisSettings
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, update
from datetime import datetime
from typing import Dict, Any, Optional, List
import io
import asyncio

from web.config import get_settings
from web.db.models import Task, Experiment, TaskStatus, ExperimentStatus
from src.orchestrator import AutotunerOrchestrator
from src.utils.optimizer import generate_parameter_grid, create_optimization_strategy, restore_optimization_strategy
from src.utils.quantization_integration import merge_parameters_with_quant_config
from src.utils.gpu_scheduler import estimate_gpu_requirements, check_gpu_availability, wait_for_gpu_availability
from web.workers.checkpoint import TaskCheckpoint
from web.events.broadcaster import get_broadcaster, EventType, create_event
from web.workers.pubsub import get_result_publisher, close_result_publisher

settings = get_settings()

# Create database session for workers
engine = create_async_engine(settings.database_url, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class StreamToLogger:
	"""File-like stream object that redirects writes to a logger instance."""

	def __init__(self, logger, log_level=logging.INFO):
		self.logger = logger
		self.log_level = log_level
		self.linebuf = ""

	def write(self, buf):
		for line in buf.rstrip().splitlines():
			self.logger.log(self.log_level, line.rstrip())

	def flush(self):
		pass


def setup_task_logging(task_id: int):
	"""Setup logging for a specific task.

	Args:
	    task_id: Task ID

	Returns:
	    Logger instance configured for this task
	"""
	# Create log directory
	log_dir = Path.home() / ".local/share/autotuner/logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	log_file = log_dir / f"task_{task_id}.log"

	# Create logger for this task
	logger = logging.getLogger(f"task_{task_id}")
	logger.setLevel(logging.DEBUG)
	logger.handlers.clear()  # Remove any existing handlers
	logger.propagate = False  # CRITICAL: Don't propagate to parent loggers to avoid recursion

	# Create file handler
	file_handler = logging.FileHandler(log_file, mode="a")
	file_handler.setLevel(logging.DEBUG)

	# Create console handler - IMPORTANT: Use sys.__stdout__ (the true original)
	# sys.__stdout__ is saved by Python at startup and is never redirected
	console_handler = logging.StreamHandler(sys.__stdout__)
	console_handler.setLevel(logging.INFO)

	# Create formatter
	formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
	file_handler.setFormatter(formatter)
	console_handler.setFormatter(formatter)

	# Add handlers to logger
	logger.addHandler(file_handler)
	logger.addHandler(console_handler)

	# Redirect stdout and stderr to logger
	sys.stdout = StreamToLogger(logger, logging.INFO)
	sys.stderr = StreamToLogger(logger, logging.ERROR)

	return logger


def setup_experiment_logging(task_id: int, experiment_id: int):
	"""Setup logging for a specific experiment.

	This creates a separate log file for each experiment to prevent log pollution
	when experiments timeout but their subprocesses continue running.

	Args:
	    task_id: Task ID
	    experiment_id: Experiment ID

	Returns:
	    Logger instance configured for this experiment
	"""
	# Create log directory
	log_dir = Path.home() / ".local/share/autotuner/logs"
	log_dir.mkdir(parents=True, exist_ok=True)

	# Create separate log files for task and experiment
	task_log_file = log_dir / f"task_{task_id}.log"
	experiment_log_file = log_dir / f"task_{task_id}_exp_{experiment_id}.log"

	# Create logger for this experiment
	logger_name = f"task_{task_id}_exp_{experiment_id}"
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG)
	logger.handlers.clear()
	logger.propagate = False

	# Create file handler for experiment-specific log
	exp_file_handler = logging.FileHandler(experiment_log_file, mode="w")  # Overwrite mode
	exp_file_handler.setLevel(logging.DEBUG)

	# Also write to task log for aggregated view
	task_file_handler = logging.FileHandler(task_log_file, mode="a")  # Append mode
	task_file_handler.setLevel(logging.DEBUG)

	# Create console handler
	console_handler = logging.StreamHandler(sys.__stdout__)
	console_handler.setLevel(logging.INFO)

	# Create formatter
	formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
	exp_file_handler.setFormatter(formatter)
	task_file_handler.setFormatter(formatter)
	console_handler.setFormatter(formatter)

	# Add handlers to logger
	logger.addHandler(exp_file_handler)
	logger.addHandler(task_file_handler)
	logger.addHandler(console_handler)

	# Redirect stdout and stderr to logger
	sys.stdout = StreamToLogger(logger, logging.INFO)
	sys.stderr = StreamToLogger(logger, logging.ERROR)

	return logger


def calculate_failure_penalty(
	started_at: datetime,
	failed_at: datetime,
	timeout_seconds: int,
	experiment_status: ExperimentStatus,
	error_message: str,
	objective_name: str
) -> float:
	"""Calculate penalty score for failed experiment based on failure timing.

	The earlier the failure, the worse the penalty. This provides gradient information
	to the Bayesian optimizer even when all experiments fail.

	IMPORTANT: All objectives use minimize direction (throughput is negated).
	Failed experiments should return POSITIVE penalty values (higher = worse).

	Args:
		started_at: When experiment started
		failed_at: When experiment failed
		timeout_seconds: Maximum allowed time
		experiment_status: Experiment status (DEPLOYING, BENCHMARKING, FAILED)
		error_message: Error message describing failure
		objective_name: Optimization objective (e.g., "maximize_throughput")

	Returns:
		Penalty score (positive value, higher = worse failure)

	Penalty scale (all objectives use minimize, so positive penalties):
		- Early failure (0-20% completion): 1000 (worst)
		- Mid failure (20-60% completion): 500
		- Late failure (60-100% completion): 200
		- Timeout (100% time used): 100 (least bad)
	"""
	# Calculate completion percentage
	elapsed = (failed_at - started_at).total_seconds()
	completion_pct = min(elapsed / timeout_seconds, 1.0) if timeout_seconds > 0 else 0.0

	# Base penalty depends on completion percentage
	# Using positive values because all objectives use minimize direction
	if completion_pct < 0.20:
		# Very early failure (deployment, immediate crash)
		base_penalty = 1000
	elif completion_pct < 0.60:
		# Mid-stage failure (benchmark started but failed early)
		base_penalty = 500
	elif completion_pct < 0.95:
		# Late-stage failure (benchmark mostly completed)
		base_penalty = 200
	else:
		# Timeout or near-timeout (experiment ran full duration)
		base_penalty = 100

	# Additional penalty modifiers based on error type
	error_lower = error_message.lower() if error_message else ""

	# Deployment failures are worse than benchmark failures
	if "deploy" in error_lower or "not found" in error_lower:
		base_penalty *= 1.2
	# OOM or resource errors are severe
	elif "oom" in error_lower or "memory" in error_lower or "cuda" in error_lower:
		base_penalty *= 1.5
	# Connection errors might be transient
	elif "connection" in error_lower or "timeout" in error_lower:
		base_penalty *= 0.8

	return base_penalty



async def run_experiment_with_timeout(
	orchestrator: AutotunerOrchestrator,
	task_config: Dict[str, Any],
	iteration: int,
	params: Dict[str, Any],
	timeout_seconds: int,
	logger: logging.Logger,
	on_benchmark_start=None
) -> Dict[str, Any]:
	"""
	Run a single experiment with timeout enforcement.

	Args:
		orchestrator: AutotunerOrchestrator instance
		task_config: Task configuration
		iteration: Experiment iteration number
		params: Parameter configuration for this experiment
		timeout_seconds: Maximum time allowed for this experiment
		logger: Logger instance
		on_benchmark_start: Optional callback when benchmark phase starts

	Returns:
		Result dict with status, metrics, etc.

	Raises:
		asyncio.TimeoutError: If experiment exceeds timeout
	"""
	# Wrap synchronous orchestrator.run_experiment in async
	loop = asyncio.get_event_loop()

	try:
		# Run with timeout
		result = await asyncio.wait_for(
			loop.run_in_executor(
				None,  # Use default executor
				orchestrator.run_experiment,
				task_config,
				iteration,
				params,
				on_benchmark_start  # Pass the callback
			),
			timeout=timeout_seconds
		)
		return result

	except asyncio.TimeoutError:
		logger.error(f"[Experiment {iteration}] Timed out after {timeout_seconds}s")
		raise


async def run_autotuning_task(ctx: Dict[str, Any], task_id: int, task_config: Dict[str, Any] = None) -> Dict[str, Any]:
	"""Run autotuning task in background.

	Args:
	    ctx: ARQ context
	    task_id: Database task ID
	    task_config: Optional full task configuration for distributed workers.
	                 If provided, worker can run without database access.

	Returns:
	    Task summary dict
	"""
	# Setup logging for this task
	logger = setup_task_logging(task_id)

	# Try to connect to database, but continue if task_config is provided
	db = None
	task = None
	try:
		async with AsyncSessionLocal() as db:
			result = await db.execute(select(Task).where(Task.id == task_id))
			task = result.scalar_one_or_none()
	except Exception as db_err:
		logger.warning(f"[ARQ Worker] Database not accessible: {db_err}")
		db = None

	# If no task from DB and no task_config provided, fail
	if not task and not task_config:
		error_msg = f"Task {task_id} not found and no task_config provided"
		logger.error(error_msg)
		return {"error": error_msg}

	# Build task_config from database if not provided
	if task_config is None and task:
		task_config = {
			"task_name": task.task_name,
			"description": task.description or "",
			"model": task.model_config,
			"base_runtime": task.base_runtime,
			"runtime_image_tag": task.runtime_image_tag,
			"parameters": task.parameters,
			"optimization": task.optimization_config,
			"benchmark": task.benchmark_config,
			"deployment_mode": task.deployment_mode,
			"clusterbasemodel_config": task.clusterbasemodel_config,
			"clusterservingruntime_config": task.clusterservingruntime_config,
			"slo": task.slo_config,
		}

	# Normalize task_config keys from Redis format (_config suffix) to orchestrator format
	# This handles task_config passed via Redis queue which uses model_config, optimization_config, etc.
	key_mapping = {
		"model_config": "model",
		"optimization_config": "optimization",
		"benchmark_config": "benchmark",
		"slo_config": "slo",
	}
	for old_key, new_key in key_mapping.items():
		if old_key in task_config and new_key not in task_config:
			task_config[new_key] = task_config.pop(old_key)

	# Now we have task_config, proceed with execution

	# Check if this worker can handle the task's deployment mode
	worker_deployment_mode = ctx.get("deployment_mode", "docker")
	task_deployment_mode = task_config.get("deployment_mode", "docker")

	# Docker tasks can only be processed by Docker workers
	if task_deployment_mode == "docker" and worker_deployment_mode != "docker":
		error_msg = f"Worker ({worker_deployment_mode}) cannot process docker task, skipping"
		logger.warning(f"[ARQ Worker] {error_msg}")
		# Return defer error to let another worker try
		from arq import Retry
		raise Retry(defer=5)  # Retry after 5 seconds by another worker

	# OME tasks can only be processed by OME workers
	if task_deployment_mode == "ome" and worker_deployment_mode != "ome":
		error_msg = f"Worker ({worker_deployment_mode}) cannot process OME task, skipping"
		logger.warning(f"[ARQ Worker] {error_msg}")
		from arq import Retry
		raise Retry(defer=5)

	async with AsyncSessionLocal() as db:
		# Re-fetch task if we have DB access (for status updates)
		if task is None:
			try:
				result = await db.execute(select(Task).where(Task.id == task_id))
				task = result.scalar_one_or_none()
			except:
				pass  # Continue without DB access

		try:
			task_name = task_config.get("task_name", f"task-{task_id}")
			logger.info(f"[ARQ Worker] Starting task: {task_name}")

			# Set task context for remote logging
			from web.workers.log_handler import get_log_handler
			log_handler = get_log_handler()
			if log_handler:
				log_handler.set_task_context(task_id, None)

			# Get broadcaster instance
			broadcaster = get_broadcaster()

			# Get result publisher from context (initialized in on_startup)
			publisher = ctx.get("publisher")
			worker_id = ctx.get("worker_id", "unknown")

			# Update task status if we have DB access
			if task:
				task.status = TaskStatus.RUNNING
				task.started_at = datetime.utcnow()
				await db.commit()

			# Broadcast task started event
			broadcaster.broadcast_sync(
				task_id,
				create_event(
					EventType.TASK_STARTED,
					task_id=task_id,
					message=f"Task '{task_name}' started"
				)
			)

			# Check GPU availability before starting task (only for Docker mode)
			deployment_mode = task_config.get("deployment_mode", "docker")
			if deployment_mode == "docker":
				logger.info(f"[ARQ Worker] Checking GPU availability for Docker deployment...")

				# Estimate GPU requirements from task configuration
				required_gpus, estimated_memory_mb = estimate_gpu_requirements(task_config)

				logger.info(f"[ARQ Worker] Task requires {required_gpus} GPU(s)")

				# Check if GPUs are available
				is_available, availability_message = check_gpu_availability(
					required_gpus=required_gpus,
					min_memory_mb=estimated_memory_mb
				)

				if not is_available:
					logger.warning(f"[ARQ Worker] GPUs not immediately available: {availability_message}")
					logger.info(f"[ARQ Worker] Waiting for GPUs to become available (timeout=5 minutes)...")

					# Wait for GPUs to become available
					is_available, availability_message = wait_for_gpu_availability(
						required_gpus=required_gpus,
						min_memory_mb=estimated_memory_mb,
						timeout_seconds=300,  # 5 minutes
						check_interval=30  # Check every 30 seconds
					)

					if not is_available:
						# GPUs still not available after waiting
						error_msg = f"Insufficient GPUs after waiting: {availability_message}"
						logger.error(f"[ARQ Worker] {error_msg}")

						# Mark task as failed if we have DB access
						started_at = task.started_at if task else datetime.utcnow()
						if task:
							task.status = TaskStatus.FAILED
							task.completed_at = datetime.utcnow()
							elapsed_time = (task.completed_at - started_at).total_seconds()
							task.elapsed_time = elapsed_time
							await db.commit()
						else:
							elapsed_time = 0

						# Broadcast failure event
						broadcaster.broadcast_sync(
							task_id,
							create_event(
								EventType.TASK_FAILED,
								task_id=task_id,
								message=error_msg
							)
						)

						return {
							"status": "failed",
							"error": error_msg,
							"elapsed_time": elapsed_time
						}
					else:
						logger.info(f"[ARQ Worker] ✓ GPUs became available: {availability_message}")
				else:
					logger.info(f"[ARQ Worker] ✓ GPU availability confirmed: {availability_message}")
			else:
				logger.info(f"[ARQ Worker] Skipping GPU check for {deployment_mode} mode")

			# Create optimization strategy - use task_config
			optimization_config = task_config.get("optimization") or {}
			strategy_name = optimization_config.get("strategy", "grid_search")
			max_iterations = optimization_config.get("max_iterations", 100)
			timeout_per_iteration = optimization_config.get("timeout_per_iteration", 1800)  # Default 30 minutes

			logger.info(f"[ARQ Worker] Optimization strategy: {strategy_name}")
			logger.info(f"[ARQ Worker] Max iterations: {max_iterations}")
			logger.info(f"[ARQ Worker] Timeout per experiment: {timeout_per_iteration}s")

			# Check for existing checkpoint and resume if available
			task_metadata = task.task_metadata if task else None
			checkpoint = TaskCheckpoint.load_checkpoint(task_metadata)
			if checkpoint:
				logger.info(f"[ARQ Worker] Found checkpoint at iteration {checkpoint['iteration']}")
				logger.info(f"[ARQ Worker] Resuming from checkpoint...")

				# Restore strategy from checkpoint
				try:
					strategy = restore_optimization_strategy(checkpoint["strategy_state"])
					logger.info(f"[ARQ Worker] Strategy restored from checkpoint")
				except Exception as e:
					logger.error(f"[ARQ Worker] Failed to restore strategy from checkpoint: {e}")
					logger.info(f"[ARQ Worker] Creating fresh strategy instead")
					# Merge quant_config with parameters for fresh strategy
					parameters = task_config.get("parameters") or {}
					quant_config = task.quant_config if task else None
					merged_parameters = merge_parameters_with_quant_config(parameters, quant_config)
					strategy = create_optimization_strategy(optimization_config, merged_parameters)

				# Restore progress from checkpoint
				best_score = checkpoint["best_score"]
				best_experiment_id = checkpoint.get("best_experiment_id")
				iteration = checkpoint["iteration"]

				logger.info(f"[ARQ Worker] Restored state: iteration={iteration}, best_score={best_score}, best_experiment_id={best_experiment_id}")
			else:
				logger.info(f"[ARQ Worker] No checkpoint found, starting fresh")

				# Merge quant_config and parallel_config with parameters to create full parameter spec
				# First merge quant_config
				parameters = task_config.get("parameters") or {}
				quant_config = task.quant_config if task else None
				merged_parameters = merge_parameters_with_quant_config(parameters, quant_config)
				logger.info(f"[ARQ Worker] Merged parameters (base + quant_config): {merged_parameters}")

				# Then merge parallel_config
				from utils.parallel_integration import merge_parameters_with_parallel_config
				parallel_config = task.parallel_config if task else None
				merged_parameters = merge_parameters_with_parallel_config(merged_parameters, parallel_config)
				logger.info(f"[ARQ Worker] Merged parameters (base + quant_config + parallel_config): {merged_parameters}")

				# Create fresh strategy with merged parameters
				try:
					strategy = create_optimization_strategy(optimization_config, merged_parameters)
				except Exception as e:
					logger.error(f"[ARQ Worker] Failed to create optimization strategy: {e}")
					raise

				# Initialize progress tracking
				best_score = float("inf")
				best_experiment_id = None
				iteration = 0

			# Set initial total_experiments (may be less for grid search, unknown for Bayesian)
			if strategy_name == "grid_search":
				# Grid search knows total upfront
				# Use merged parameters to calculate total
				parameters = task_config.get("parameters") or {}
				quant_config = task.quant_config if task else None
				merged_parameters = merge_parameters_with_quant_config(parameters, quant_config)
				param_grid = generate_parameter_grid(merged_parameters)
				total_experiments = min(len(param_grid), max_iterations)
			else:
				# Bayesian/random: use max_iterations as upper bound
				total_experiments = max_iterations

			if task:
				task.total_experiments = total_experiments
				await db.commit()

			logger.info(f"[ARQ Worker] Expected experiments: {total_experiments}")

			# Create orchestrator
			orchestrator = AutotunerOrchestrator(
				deployment_mode=deployment_mode,
				use_direct_benchmark=True,
				docker_model_path=settings.docker_model_path,
				verbose=False,
				http_proxy=settings.http_proxy,
				https_proxy=settings.https_proxy,
				no_proxy=settings.no_proxy,
				hf_token=settings.hf_token,
			)

			# Run experiments using strategy
			while not strategy.should_stop():
				iteration += 1

				# Get next parameter suggestion
				params = strategy.suggest_parameters()
				if params is None:
					logger.info(f"[ARQ Worker] Strategy has no more suggestions")
					break

				logger.info(f"[ARQ Worker] Running experiment {iteration} with params: {params}")

				# Switch to experiment-specific logging to prevent log pollution from zombie processes
				logger = setup_experiment_logging(task_id, iteration)
				logger.info(f"[Experiment {iteration}] Logging to experiment-specific file")

				# Set task context for remote logging
				from web.workers.log_handler import get_log_handler
				log_handler = get_log_handler()
				if log_handler:
					log_handler.set_task_context(task_id, iteration)

				# Create experiment record
				db_experiment = Experiment(
					task_id=task_id,
					experiment_id=iteration,
					parameters=params,
					status=ExperimentStatus.PENDING,
				)
				db.add(db_experiment)
				await db.commit()
				await db.refresh(db_experiment)

				# Update status to deploying
				db_experiment.status = ExperimentStatus.DEPLOYING
				db_experiment.started_at = datetime.utcnow()
				await db.commit()

				logger.info(f"[Experiment {iteration}] Status: DEPLOYING")

				# Broadcast experiment started event
				broadcaster.broadcast_sync(
					task_id,
					create_event(
						EventType.EXPERIMENT_STARTED,
						task_id=task_id,
						experiment_id=iteration,
						data={
							"parameters": params,
							"status": "deploying"
						},
						message=f"Experiment {iteration} started"
					)
				)

				# Shared flag to signal when benchmark starts
				benchmark_started = {'value': False}

				# Define callback to update status when benchmark starts
				def on_benchmark_start():
					benchmark_started['value'] = True

				# Start a background task to monitor and update status
				async def monitor_benchmark_status():
					while not benchmark_started['value']:
						await asyncio.sleep(0.1)  # Check every 100ms
					# Update status to BENCHMARKING
					db_experiment.status = ExperimentStatus.BENCHMARKING
					await db.commit()
					logger.info(f"[Experiment {iteration}] Status: BENCHMARKING")

					# Broadcast benchmark progress event
					broadcaster.broadcast_sync(
						task_id,
						create_event(
							EventType.BENCHMARK_PROGRESS,
							task_id=task_id,
							experiment_id=iteration,
							data={"status": "benchmarking"},
							message=f"Experiment {iteration} benchmarking in progress"
						)
					)

				monitor_task = asyncio.create_task(monitor_benchmark_status())

				# Run experiment using orchestrator with timeout
				try:
					result = await run_experiment_with_timeout(
						orchestrator=orchestrator,
						task_config=task_config,
						iteration=iteration,
						params=params,
						timeout_seconds=timeout_per_iteration,
						logger=logger,
						on_benchmark_start=on_benchmark_start
					)

					logger.info(f"[Experiment {iteration}] Status: {result['status'].upper()}")
					if result.get("metrics"):
						logger.info(f"[Experiment {iteration}] Metrics: {result['metrics']}")

					# Cancel monitor task since experiment is done
					if not monitor_task.done():
						monitor_task.cancel()
						try:
							await monitor_task
						except asyncio.CancelledError:
							pass

					# Save container logs if available (Docker mode)
					if result.get("container_logs"):
						logger.info(f"[Experiment {iteration}] ========== Container Logs ==========")
						for line in result["container_logs"].splitlines():
							logger.info(f"[Experiment {iteration}] {line}")
						logger.info(f"[Experiment {iteration}] ========== End Container Logs ==========")

					# Update experiment with results
					db_experiment.status = (
						ExperimentStatus.SUCCESS if result["status"] == "success" else ExperimentStatus.FAILED
					)
					db_experiment.metrics = result.get("metrics")
					db_experiment.objective_score = result.get("objective_score")
					db_experiment.gpu_info = result.get("gpu_info")  # Save GPU information
					db_experiment.error_message = result.get("error_message")  # Save error message for failed experiments
					db_experiment.completed_at = datetime.utcnow()

					# Save created resources to task (only on first experiment)
					if iteration == 1:
						created_resources = result.get("created_resources", {})
						if created_resources:
							cbm_name = created_resources.get("clusterbasemodel")
							csr_name = created_resources.get("clusterservingruntime")
							if cbm_name:
								task.created_clusterbasemodel = cbm_name
								logger.info(f"[ARQ Worker] Task created ClusterBaseModel: {cbm_name}")
							if csr_name:
								task.created_clusterservingruntime = csr_name
								logger.info(f"[ARQ Worker] Task created ClusterServingRuntime: {csr_name}")

					if db_experiment.started_at:
						elapsed = (db_experiment.completed_at - db_experiment.started_at).total_seconds()
						db_experiment.elapsed_time = elapsed
						logger.info(f"[Experiment {iteration}] Completed in {elapsed:.2f}s")

					# Update strategy with result
					if result["status"] == "success":
						if task:
							task.successful_experiments += 1
						objective_score = result.get("objective_score")

						# Tell strategy about the result
						strategy.tell_result(
							parameters=params,
							objective_score=objective_score,
							metrics=result.get("metrics", {})
						)

						# Check if this is the best experiment
						if objective_score is not None and objective_score < best_score:
							best_score = objective_score
							best_experiment_id = db_experiment.id
							logger.info(f"[Experiment {iteration}] New best score: {best_score:.4f}")
					else:
						# Experiment failed - calculate penalty based on failure timing
						objective_name = optimization_config.get("objective", "minimize_latency")

						# Calculate penalty score based on when failure occurred
						penalty_score = calculate_failure_penalty(
							started_at=db_experiment.started_at,
							failed_at=db_experiment.completed_at,
							timeout_seconds=timeout_per_iteration,
							experiment_status=db_experiment.status,
							error_message=result.get("error_message", ""),
							objective_name=objective_name
						)
						# Save penalty score to database for frontend display
						db_experiment.objective_score = penalty_score


						logger.info(f"[Experiment {iteration}] Failed with penalty score: {penalty_score:.1f}")
						logger.info(f"[Experiment {iteration}] Elapsed: {(db_experiment.completed_at - db_experiment.started_at).total_seconds():.1f}s / {timeout_per_iteration}s")

						# Tell strategy about failed experiment with graded penalty
						strategy.tell_result(
							parameters=params,
							objective_score=penalty_score,
							metrics={}
						)

					await db.commit()

					# Broadcast experiment completion event
					broadcaster.broadcast_sync(
						task_id,
						create_event(
							EventType.EXPERIMENT_COMPLETED if result["status"] == "success" else EventType.EXPERIMENT_FAILED,
							task_id=task_id,
							experiment_id=iteration,
							data={
								"status": result["status"],
								"metrics": result.get("metrics"),
								"objective_score": result.get("objective_score"),
								"elapsed_time": elapsed if db_experiment.started_at else None
							},
							message=f"Experiment {iteration} {result['status']}"
						)
					)

					# Publish result via Redis Pub/Sub for distributed listeners
					if publisher:
						try:
							await publisher.publish_experiment_completed(
								task_id=task_id,
								experiment_id=iteration,
								status=result["status"],
								metrics=result.get("metrics") or {},  # Ensure never None
								objective_score=result.get("objective_score"),
								error_message=result.get("error_message"),
								elapsed_time=elapsed if db_experiment.started_at else 0.0,
								parameters=params,
							)
							logger.debug(f"[Experiment {iteration}] Result published to Pub/Sub")
						except Exception as pub_err:
							logger.warning(f"[Experiment {iteration}] Failed to publish result: {pub_err}")

					# Save checkpoint after each experiment
					try:
						# Use a fresh session for checkpoint to avoid session expiration issues
						async with AsyncSessionLocal() as checkpoint_db:
							# Update task directly with SQL to avoid session state issues
							from sqlalchemy import update
							await checkpoint_db.execute(
								update(Task).where(Task.id == task_id).values(
									total_experiments=iteration,
									successful_experiments=task.successful_experiments if task else 0,
								)
							)
							await checkpoint_db.commit()
						logger.info(f"[ARQ Worker] Checkpoint saved at iteration {iteration}")

						# Broadcast task progress event
						broadcaster.broadcast_sync(
							task_id,
							create_event(
								EventType.TASK_PROGRESS,
								task_id=task_id,
								data={
									"current_experiment": iteration,
									"total_experiments": total_experiments,
									"successful_experiments": task.successful_experiments if task else 0,
									"progress_percent": (iteration / total_experiments * 100) if total_experiments > 0 else 0,
									"best_score": best_score if best_score != float("inf") else None
								},
								message=f"Progress: {iteration}/{total_experiments} experiments completed"
							)
						)
					except Exception as checkpoint_error:
						logger.warning(f"[ARQ Worker] Failed to save checkpoint: {checkpoint_error}")

				except asyncio.TimeoutError:
					# Cancel monitor task
					if not monitor_task.done():
						monitor_task.cancel()
						try:
							await monitor_task
						except asyncio.CancelledError:
							pass

					# Experiment timed out
					logger.error(f"[Experiment {iteration}] Timed out after {timeout_per_iteration}s")

					# CRITICAL: Force cleanup of stalled container
					task_name_for_cleanup = task_config.get("task_name", f"task-{task_id}")
					namespace = task_config.get("model", {}).get("namespace", "default")
					service_id = f"{task_name_for_cleanup}-exp{iteration}"
					logger.info(f"[Cleanup] Forcing cleanup of service '{service_id}' after timeout")
					try:
						loop = asyncio.get_event_loop()
						await loop.run_in_executor(None, orchestrator.cleanup_experiment, service_id, None, namespace, iteration)
						logger.info(f"[Cleanup] Successfully cleaned up service '{service_id}'")
					except Exception as cleanup_error:
						logger.error(f"[Cleanup] Failed to cleanup service: {cleanup_error}")

					db_experiment.status = ExperimentStatus.FAILED
					db_experiment.error_message = f"Experiment timed out after {timeout_per_iteration} seconds"
					db_experiment.completed_at = datetime.utcnow()
					await db.commit()

					# Calculate penalty for timeout failure
					objective_name = optimization_config.get("objective", "minimize_latency")
					penalty_score = calculate_failure_penalty(
						started_at=db_experiment.started_at,
						failed_at=db_experiment.completed_at,
						timeout_seconds=timeout_per_iteration,
						experiment_status=db_experiment.status,
						error_message=db_experiment.error_message,
						objective_name=objective_name
					)
					# Save penalty score to database for frontend display
					db_experiment.objective_score = penalty_score


					logger.info(f"[Experiment {iteration}] Timeout penalty score: {penalty_score:.1f}")

					# Tell strategy about timeout with graded penalty
					strategy.tell_result(
						parameters=params,
						objective_score=penalty_score,
						metrics={}
					)

					# Publish result via Redis Pub/Sub for distributed listeners (timeout case)
					if publisher:
						try:
							timeout_elapsed = (db_experiment.completed_at - db_experiment.started_at).total_seconds() if db_experiment.started_at else timeout_per_iteration
							await publisher.publish_experiment_completed(
								task_id=task_id,
								experiment_id=iteration,
								status="failed",
								metrics={},
								objective_score=penalty_score,
								error_message=db_experiment.error_message,
								elapsed_time=timeout_elapsed,
								parameters=params,
							)
							logger.debug(f"[Experiment {iteration}] Timeout result published to Pub/Sub")
						except Exception as pub_err:
							logger.warning(f"[Experiment {iteration}] Failed to publish timeout result: {pub_err}")

					# Save checkpoint after timeout
					try:
						await db.refresh(task)
						updated_metadata = TaskCheckpoint.save_checkpoint(
							task_metadata=task.task_metadata or {},
							iteration=iteration,
							best_score=best_score,
							best_experiment_id=best_experiment_id,
							strategy_state=strategy.get_state(),
						)
						task.task_metadata = updated_metadata
						await db.commit()
						logger.info(f"[ARQ Worker] Checkpoint saved at iteration {iteration} (after timeout)")
					except Exception as checkpoint_error:
						logger.warning(f"[ARQ Worker] Failed to save checkpoint: {checkpoint_error}")

				except Exception as e:
					# Cancel monitor task
					if not monitor_task.done():
						monitor_task.cancel()
						try:
							await monitor_task
						except asyncio.CancelledError:
							pass

					logger.error(f"[Experiment {iteration}] Failed: {e}", exc_info=True)
					db_experiment.status = ExperimentStatus.FAILED
					db_experiment.error_message = str(e)
					db_experiment.completed_at = datetime.utcnow()
					await db.commit()

					# Calculate penalty for exception failure
					objective_name = optimization_config.get("objective", "minimize_latency")
					penalty_score = calculate_failure_penalty(
						started_at=db_experiment.started_at,
						failed_at=db_experiment.completed_at,
						timeout_seconds=timeout_per_iteration,
						experiment_status=db_experiment.status,
						error_message=db_experiment.error_message,
						objective_name=objective_name
					)
					# Save penalty score to database for frontend display
					db_experiment.objective_score = penalty_score


					logger.info(f"[Experiment {iteration}] Exception penalty score: {penalty_score:.1f}")

					# Tell strategy about exception with graded penalty
					strategy.tell_result(
						parameters=params,
						objective_score=penalty_score,
						metrics={}
					)

					# Publish result via Redis Pub/Sub for distributed listeners (exception case)
					if publisher:
						try:
							exception_elapsed = (db_experiment.completed_at - db_experiment.started_at).total_seconds() if db_experiment.started_at else 0.0
							await publisher.publish_experiment_completed(
								task_id=task_id,
								experiment_id=iteration,
								status="failed",
								metrics={},
								objective_score=penalty_score,
								error_message=db_experiment.error_message,
								elapsed_time=exception_elapsed,
								parameters=params,
							)
							logger.debug(f"[Experiment {iteration}] Exception result published to Pub/Sub")
						except Exception as pub_err:
							logger.warning(f"[Experiment {iteration}] Failed to publish exception result: {pub_err}")

					# Save checkpoint after failed experiment
					try:
						await db.refresh(task)
						updated_metadata = TaskCheckpoint.save_checkpoint(
							task_metadata=task.task_metadata or {},
							iteration=iteration,
							best_score=best_score,
							best_experiment_id=best_experiment_id,
							strategy_state=strategy.get_state(),
						)
						task.task_metadata = updated_metadata
						await db.commit()
						logger.info(f"[ARQ Worker] Checkpoint saved at iteration {iteration} (after failure)")
					except Exception as checkpoint_error:
						logger.warning(f"[ARQ Worker] Failed to save checkpoint: {checkpoint_error}")

			# Update task with final results using a fresh session
			try:
				async with AsyncSessionLocal() as final_db:
					from sqlalchemy import update

					# Determine final status
					successful_count = task.successful_experiments if task else 0
					final_status = TaskStatus.COMPLETED if successful_count > 0 else TaskStatus.FAILED
					completed_at = datetime.utcnow()
					elapsed_time = None

					if task and task.started_at:
						elapsed_time = (completed_at - task.started_at).total_seconds()

					# Update task in database
					await final_db.execute(
						update(Task).where(Task.id == task_id).values(
							status=final_status,
							total_experiments=iteration,
							successful_experiments=successful_count,
							best_experiment_id=best_experiment_id,
							completed_at=completed_at,
							elapsed_time=elapsed_time,
						)
					)
					await final_db.commit()

					# Publish task status via Redis for distributed workers
					if publisher:
						await publisher.publish_task_status(
							task_id=task_id,
							status=final_status.value,
							total_experiments=iteration,
							successful_experiments=successful_count,
							best_experiment_id=best_experiment_id,
							best_score=best_score if best_score != float("inf") else None,
							elapsed_time=elapsed_time,
						)

					logger.info(f"[ARQ Worker] Task completed in {elapsed_time:.2f}s - Best experiment: {best_experiment_id}" if elapsed_time else f"[ARQ Worker] Task completed - Best experiment: {best_experiment_id}")

					# Broadcast task completion event
					broadcaster.broadcast_sync(
						task_id,
						create_event(
							EventType.TASK_COMPLETED if final_status == TaskStatus.COMPLETED else EventType.TASK_FAILED,
							task_id=task_id,
							data={
								"status": final_status.value,
								"total_experiments": iteration,
								"successful_experiments": successful_count,
								"best_experiment_id": best_experiment_id,
								"best_score": best_score if best_score != float("inf") else None,
								"elapsed_time": elapsed_time
							},
							message=f"Task completed: {successful_count}/{iteration} experiments successful"
						)
					)
			except Exception as final_update_error:
				logger.error(f"[ARQ Worker] Failed to update final task status: {final_update_error}")

			logger.info(
				f"[ARQ Worker] Task finished: {task_name} - {task.successful_experiments if task else 0}/{total_experiments} successful"
			)
			return {"task_id": task_id, "task_name": task_name, "status": "completed"}

		except Exception as e:
			logger.error(f"[ARQ Worker] Task failed: {e}", exc_info=True)
			if task:
				task.status = TaskStatus.FAILED
				task.completed_at = datetime.utcnow()
				await db.commit()

				# Publish task failure via Redis for distributed workers
				if publisher:
					elapsed_time = None
					if task.started_at:
						elapsed_time = (task.completed_at - task.started_at).total_seconds()
					await publisher.publish_task_status(
						task_id=task_id,
						status=TaskStatus.FAILED.value,
						total_experiments=task.total_experiments or 0,
						successful_experiments=task.successful_experiments or 0,
						elapsed_time=elapsed_time,
						error_message=str(e),
					)
			return {"task_id": task_id, "error": str(e)}
		finally:
			# Restore stdout and stderr
			sys.stdout = sys.__stdout__
			sys.stderr = sys.__stderr__
			# Remove handlers to prevent memory leaks
			logger.handlers.clear()


# ARQ worker settings
class WorkerSettings:
	"""ARQ worker configuration."""

	redis_settings = RedisSettings(
		host=settings.redis_host,
		port=settings.redis_port,
		database=settings.redis_db,
	)

	functions = [run_autotuning_task]

	# Worker config
	max_jobs = 5  # Maximum concurrent jobs
	job_timeout = 86400 * 30  # 720 hours timeout for entire task (rely on per-experiment timeout instead)
	keep_result = 3600  # Keep results for 1 hour

	# Worker identity (can be overridden via environment variable WORKER_ID)
	worker_id: Optional[str] = None

	@staticmethod
	def get_worker_id() -> str:
		"""Get or generate worker ID."""
		import os
		worker_id = os.environ.get("WORKER_ID")
		if not worker_id:
			# Generate from hostname + short UUID
			hostname = socket.gethostname()
			short_uuid = str(uuid.uuid4())[:8]
			worker_id = f"{hostname}-{short_uuid}"
		return worker_id

	@staticmethod
	def get_gpu_info() -> tuple[int, Optional[str], Optional[float], Optional[List[Dict]]]:
		"""Get GPU information using nvidia-smi.

		Returns:
			Tuple of (gpu_count, gpu_model, total_memory_gb, gpu_details)
		"""
		try:
			result = subprocess.run(
				["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
				capture_output=True,
				text=True,
				timeout=10
			)
			if result.returncode != 0:
				return 0, None, None, None

			gpus = []
			total_memory = 0
			gpu_model = None

			for line in result.stdout.strip().split("\n"):
				if not line.strip():
					continue
				parts = [p.strip() for p in line.split(",")]
				if len(parts) >= 3:
					idx = int(parts[0])
					name = parts[1]
					memory_mb = float(parts[2])
					memory_gb = memory_mb / 1024

					if gpu_model is None:
						gpu_model = name

					gpus.append({
						"index": idx,
						"name": name,
						"memory_total_gb": round(memory_gb, 2)
					})
					total_memory += memory_gb

			return len(gpus), gpu_model, round(total_memory, 2), gpus

		except Exception as e:
			logging.warning(f"Failed to get GPU info: {e}")
			return 0, None, None, None

	@staticmethod
	def get_cluster_gpu_info() -> tuple[int, Optional[str], Optional[float], Optional[List[Dict]]]:
		"""Get cluster-wide GPU information using kubectl for OME mode.

		Returns:
			Tuple of (total_gpu_count, gpu_model, total_memory_gb, gpu_details)
		"""
		try:
			# Get nodes with GPU info from Kubernetes
			result = subprocess.run(
				["kubectl", "get", "nodes", "-o", "json"],
				capture_output=True,
				text=True,
				timeout=15
			)
			if result.returncode != 0:
				logging.warning(f"kubectl failed: {result.stderr}")
				return WorkerSettings.get_gpu_info()  # Fallback to local

			import json as json_module
			nodes_data = json_module.loads(result.stdout)

			# Get local GPU info for model name and memory (fallback for missing labels)
			local_gpu_count, local_gpu_model, local_gpu_memory, local_gpu_details = WorkerSettings.get_gpu_info()
			local_hostname = socket.gethostname()

			total_gpus = 0
			gpu_model = None
			gpus = []
			gpu_idx = 0

			for node in nodes_data.get("items", []):
				node_name = node["metadata"]["name"]
				capacity = node["status"].get("capacity", {})
				labels = node["metadata"].get("labels", {})

				# Find GPU count
				node_gpu_count = 0
				for key in capacity:
					if "nvidia.com/gpu" in key:
						node_gpu_count = int(capacity[key])
						break

				if node_gpu_count > 0:
					# Get GPU model from node labels, fallback to local nvidia-smi
					node_gpu_model = labels.get("nvidia.com/gpu.product")
					if not node_gpu_model or node_gpu_model == "Unknown GPU":
						node_gpu_model = local_gpu_model or "GPU"

					if gpu_model is None:
						gpu_model = node_gpu_model

					# For local node, use detailed GPU info from nvidia-smi
					if node_name == local_hostname and local_gpu_details:
						for g in local_gpu_details:
							gpus.append({
								"index": gpu_idx,
								"name": g.get("name", node_gpu_model),
								"memory_total_gb": g.get("memory_total_gb", 0),
								"node_name": node_name
							})
							gpu_idx += 1
					else:
						# For remote nodes, use count from kubectl
						for i in range(node_gpu_count):
							gpus.append({
								"index": gpu_idx,
								"name": node_gpu_model,
								"memory_total_gb": 0,  # Not available from kubectl
								"node_name": node_name
							})
							gpu_idx += 1

					total_gpus += node_gpu_count

			if total_gpus == 0:
				logging.warning("No GPUs found in cluster, falling back to local")
				return WorkerSettings.get_gpu_info()

			logging.info(f"Cluster GPU info: {total_gpus} GPUs across {len(set(g['node_name'] for g in gpus))} nodes")
			return total_gpus, gpu_model, None, gpus

		except FileNotFoundError:
			logging.warning("kubectl not found, falling back to local GPU info")
			return WorkerSettings.get_gpu_info()
		except Exception as e:
			logging.warning(f"Failed to get cluster GPU info: {e}, falling back to local")
			return WorkerSettings.get_gpu_info()

	@staticmethod
	async def on_startup(ctx: Dict[str, Any]) -> None:
		"""Called when worker starts. Register with the manager."""
		import os
		from web.schemas.worker import WorkerRegister, WorkerCapabilities, GPUInfo
		from web.workers.registry import get_worker_registry, HEARTBEAT_INTERVAL
		from web.workers.worker_config import (
			load_worker_config,
			get_worker_alias,
			get_deployment_mode,
		)

		worker_id = WorkerSettings.get_worker_id()
		hostname = socket.gethostname()

		# Setup remote logging to stream logs to manager
		try:
			from web.workers.log_handler import setup_remote_logging
			log_handler = setup_remote_logging(worker_id, level=logging.INFO)
			ctx["log_handler"] = log_handler
			logging.info(f"📡 Remote logging enabled for worker: {worker_id}")
		except Exception as e:
			logging.warning(f"Failed to setup remote logging (logs will be local only): {e}")

		# Load local config file
		local_config = load_worker_config()
		ctx["local_config"] = local_config

		# Get deployment mode and alias (config file with env var override)
		deployment_mode = get_deployment_mode()
		worker_alias = get_worker_alias()

		# Log config source
		from web.workers.worker_config import get_config_path
		config_path = get_config_path()
		logging.info(f"📁 Worker config: {config_path} (alias={worker_alias}, mode={deployment_mode})")

		# Get GPU information - use cluster info for OME mode
		if deployment_mode == "ome":
			gpu_count, gpu_model, gpu_memory_gb, gpu_details = WorkerSettings.get_cluster_gpu_info()
		else:
			gpu_count, gpu_model, gpu_memory_gb, gpu_details = WorkerSettings.get_gpu_info()

		# Convert GPU details to GPUInfo objects
		gpus = None
		if gpu_details:
			gpus = [GPUInfo(
				index=g["index"],
				name=g["name"],
				memory_total_gb=g.get("memory_total_gb", 0),
				node_name=g.get("node_name")  # Include node_name for OME mode
			) for g in gpu_details]

		# Create registration
		registration = WorkerRegister(
			worker_id=worker_id,
			hostname=hostname,
			alias=worker_alias,
			ip_address=None,  # Could detect local IP if needed
			gpu_count=gpu_count,
			gpu_model=gpu_model,
			gpu_memory_gb=gpu_memory_gb,
			gpus=gpus,
			deployment_mode=deployment_mode,
			max_parallel=WorkerSettings.max_jobs,
			capabilities=WorkerCapabilities(
				deployment_modes=[deployment_mode],
				docker_available=deployment_mode == "docker"
			)
		)

		# Register with manager
		try:
			registry = await get_worker_registry()
			worker_info = await registry.register(registration)
			ctx["worker_id"] = worker_id
			ctx["registry"] = registry
			ctx["deployment_mode"] = deployment_mode  # Store for heartbeat loop
			alias_info = f" alias={worker_alias}" if worker_alias else ""
			logging.info(f"✅ Worker registered: {worker_id} ({hostname},{alias_info} {gpu_count} GPUs)")
		except Exception as e:
			logging.error(f"❌ Failed to register worker: {e}")
			# Continue anyway - worker can still process jobs

		# Initialize result publisher for Pub/Sub
		try:
			publisher = await get_result_publisher(worker_id)
			ctx["publisher"] = publisher
			logging.info(f"📡 Result publisher initialized for worker: {worker_id}")
		except Exception as e:
			logging.error(f"❌ Failed to initialize result publisher: {e}")

		# Start heartbeat task
		ctx["heartbeat_task"] = asyncio.create_task(
			WorkerSettings._heartbeat_loop(ctx)
		)

		# Start config listener task
		ctx["config_listener_task"] = asyncio.create_task(
			WorkerSettings._config_listener(ctx)
		)

	@staticmethod
	async def on_shutdown(ctx: Dict[str, Any]) -> None:
		"""Called when worker shuts down. Deregister from the manager."""
		# Cancel heartbeat task
		heartbeat_task = ctx.get("heartbeat_task")
		if heartbeat_task:
			heartbeat_task.cancel()
			try:
				await heartbeat_task
			except asyncio.CancelledError:
				pass

		# Cancel config listener task
		config_listener_task = ctx.get("config_listener_task")
		if config_listener_task:
			config_listener_task.cancel()
			try:
				await config_listener_task
			except asyncio.CancelledError:
				pass

		# Close result publisher
		try:
			await close_result_publisher()
			logging.info("📡 Result publisher closed")
		except Exception as e:
			logging.error(f"Failed to close result publisher: {e}")

		# Shutdown remote logging
		log_handler = ctx.get("log_handler")
		if log_handler:
			try:
				from web.workers.log_handler import shutdown_remote_logging
				shutdown_remote_logging()
				logging.info("📡 Remote logging shutdown")
			except Exception as e:
				logging.error(f"Failed to shutdown remote logging: {e}")

		# Deregister worker
		worker_id = ctx.get("worker_id")
		registry = ctx.get("registry")
		if worker_id and registry:
			try:
				await registry.deregister(worker_id)
				logging.info(f"👋 Worker deregistered: {worker_id}")
			except Exception as e:
				logging.error(f"Failed to deregister worker: {e}")

	@staticmethod
	async def _heartbeat_loop(ctx: Dict[str, Any]) -> None:
		"""Background task to send periodic heartbeats."""
		from web.schemas.worker import WorkerHeartbeat, GPUInfo
		from web.workers.registry import HEARTBEAT_INTERVAL

		worker_id = ctx.get("worker_id")
		registry = ctx.get("registry")
		deployment_mode = ctx.get("deployment_mode", "docker")

		if not worker_id or not registry:
			return

		while True:
			try:
				await asyncio.sleep(HEARTBEAT_INTERVAL)

				# Get current job count from ARQ context
				# Note: ARQ doesn't expose this directly, so we track manually
				current_jobs = ctx.get("current_jobs", 0)
				current_job_ids = ctx.get("current_job_ids", [])

				# Collect GPU metrics - use cluster metrics for OME mode
				if deployment_mode == "ome":
					gpu_metrics = WorkerSettings._get_cluster_gpu_metrics()
				else:
					gpu_metrics = WorkerSettings._get_gpu_metrics()

				heartbeat = WorkerHeartbeat(
					worker_id=worker_id,
					current_jobs=current_jobs,
					current_job_ids=current_job_ids,
					gpus=gpu_metrics,
				)

				await registry.heartbeat(heartbeat)
				logging.debug(f"💓 Heartbeat sent: {worker_id} (jobs: {current_jobs}, gpus: {len(gpu_metrics) if gpu_metrics else 0})")

			except asyncio.CancelledError:
				break
			except Exception as e:
				logging.warning(f"Heartbeat failed: {e}")

	@staticmethod
	async def _config_listener(ctx: Dict[str, Any]) -> None:
		"""Background task to listen for config updates from manager."""
		import redis.asyncio as redis
		from web.workers.pubsub import CONFIG_CHANNEL_PREFIX
		from web.workers.worker_config import update_worker_config, get_config_path

		worker_id = ctx.get("worker_id")
		if not worker_id:
			return

		channel = f"{CONFIG_CHANNEL_PREFIX}{worker_id}"
		logging.info(f"📡 Config listener started, subscribing to: {channel}")

		try:
			# Create separate Redis connection for pubsub
			client = redis.Redis(
				host=settings.redis_host,
				port=settings.redis_port,
				db=settings.redis_db,
				decode_responses=True,
			)
			pubsub = client.pubsub()
			await pubsub.subscribe(channel)

			async for message in pubsub.listen():
				if message["type"] == "message":
					try:
						import json
						data = json.loads(message["data"])
						updates = data.get("updates", {})

						if updates:
							logging.info(f"📥 Received config update: {updates}")
							config = update_worker_config(**updates)
							logging.info(f"✅ Config updated and saved to {get_config_path()}")

							# If alias changed, update the registration
							if "alias" in updates:
								registry = ctx.get("registry")
								if registry:
									await registry.set_worker_alias(worker_id, updates["alias"])
									logging.info(f"✅ Worker alias updated in registry: {updates['alias']}")

					except Exception as e:
						logging.error(f"Failed to process config update: {e}")

		except asyncio.CancelledError:
			logging.info("📡 Config listener stopped")
			raise
		except Exception as e:
			logging.error(f"Config listener error: {e}")
		finally:
			try:
				await pubsub.unsubscribe(channel)
				await client.close()
			except Exception:
				pass

	@staticmethod
	def _get_gpu_metrics() -> Optional[List["GPUInfo"]]:
		"""Get current GPU metrics using nvidia-smi.

		Returns:
			List of GPUInfo objects or None if unavailable
		"""
		from web.schemas.worker import GPUInfo
		try:
			result = subprocess.run(
				[
					"nvidia-smi",
					"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
					"--format=csv,noheader,nounits"
				],
				capture_output=True,
				text=True,
				timeout=10
			)
			if result.returncode != 0:
				return None

			hostname = socket.gethostname()
			gpus = []
			for line in result.stdout.strip().split("\n"):
				if not line.strip():
					continue
				parts = [p.strip() for p in line.split(",")]
				if len(parts) >= 7:
					try:
						gpus.append(GPUInfo(
							index=int(parts[0]),
							name=parts[1],
							memory_total_gb=round(float(parts[2]) / 1024, 2),
							memory_used_gb=round(float(parts[3]) / 1024, 2),
							memory_free_gb=round(float(parts[4]) / 1024, 2),
							utilization_percent=float(parts[5]) if parts[5] != "[N/A]" else None,
							temperature_c=int(parts[6]) if parts[6] != "[N/A]" else None,
							node_name=hostname,  # Include node_name for cluster mode
						))
					except (ValueError, IndexError):
						continue

			return gpus if gpus else None

		except Exception as e:
			logging.debug(f"Failed to get GPU metrics: {e}")
			return None

	@staticmethod
	def _get_cluster_gpu_metrics() -> Optional[List["GPUInfo"]]:
		"""Get GPU metrics from all cluster nodes using kubectl exec.

		For OME/cluster mode, this method:
		1. Gets all nodes with GPUs
		2. Finds a GPU pod on each node
		3. Executes nvidia-smi in those pods to collect metrics

		Returns:
			List of GPUInfo objects from all cluster nodes, or None if unavailable
		"""
		from web.schemas.worker import GPUInfo
		import json as json_module

		try:
			# First get local metrics as fallback and for local node
			local_gpus = WorkerSettings._get_gpu_metrics() or []
			local_hostname = socket.gethostname()

			# Get all nodes with GPU capacity
			result = subprocess.run(
				["kubectl", "get", "nodes", "-o", "json"],
				capture_output=True,
				text=True,
				timeout=15
			)
			if result.returncode != 0:
				logging.warning(f"kubectl get nodes failed: {result.stderr}")
				return local_gpus if local_gpus else None

			nodes_data = json_module.loads(result.stdout)
			gpu_nodes = []

			for node in nodes_data.get("items", []):
				node_name = node["metadata"]["name"]
				capacity = node["status"].get("capacity", {})

				# Check for GPU capacity
				for key in capacity:
					if "nvidia.com/gpu" in key:
						gpu_count = int(capacity[key])
						if gpu_count > 0:
							gpu_nodes.append(node_name)
						break

			if not gpu_nodes:
				return local_gpus if local_gpus else None

			# Get all pods across all namespaces that might have GPUs
			result = subprocess.run(
				["kubectl", "get", "pods", "-A", "-o", "json"],
				capture_output=True,
				text=True,
				timeout=15
			)
			if result.returncode != 0:
				logging.warning(f"kubectl get pods failed: {result.stderr}")
				return local_gpus if local_gpus else None

			pods_data = json_module.loads(result.stdout)

			# Build a map of node -> pod that can run nvidia-smi
			node_to_pod = {}
			for pod in pods_data.get("items", []):
				pod_name = pod["metadata"]["name"]
				namespace = pod["metadata"]["namespace"]
				node_name = pod["spec"].get("nodeName")
				phase = pod["status"].get("phase")

				# Skip if not running or no node assigned
				if phase != "Running" or not node_name:
					continue

				# Skip if node already has a pod assigned
				if node_name in node_to_pod:
					continue

				# Check if pod has GPU resources
				containers = pod["spec"].get("containers", [])
				for container in containers:
					resources = container.get("resources", {})
					limits = resources.get("limits", {})
					requests = resources.get("requests", {})

					has_gpu = any(
						"nvidia.com/gpu" in key
						for key in list(limits.keys()) + list(requests.keys())
					)

					if has_gpu:
						node_to_pod[node_name] = {
							"pod": pod_name,
							"namespace": namespace,
							"container": container["name"]
						}
						break

			# Collect metrics from all nodes
			all_gpus = []
			gpu_idx = 0

			for node_name in sorted(gpu_nodes):
				# For local node, use local nvidia-smi (already collected)
				if node_name == local_hostname:
					for gpu in local_gpus:
						# Re-index GPUs for consistent cluster-wide indexing
						all_gpus.append(GPUInfo(
							index=gpu_idx,
							name=gpu.name,
							memory_total_gb=gpu.memory_total_gb,
							memory_used_gb=gpu.memory_used_gb,
							memory_free_gb=gpu.memory_free_gb,
							utilization_percent=gpu.utilization_percent,
							temperature_c=gpu.temperature_c,
							node_name=node_name,
						))
						gpu_idx += 1
					continue

				# For remote nodes, use kubectl exec
				pod_info = node_to_pod.get(node_name)
				if not pod_info:
					logging.debug(f"No GPU pod found on node {node_name}, skipping metrics")
					# Add placeholder GPUs without metrics
					# We need to know how many GPUs this node has
					for node in nodes_data.get("items", []):
						if node["metadata"]["name"] == node_name:
							capacity = node["status"].get("capacity", {})
							for key in capacity:
								if "nvidia.com/gpu" in key:
									node_gpu_count = int(capacity[key])
									for i in range(node_gpu_count):
										all_gpus.append(GPUInfo(
											index=gpu_idx,
											name="GPU",
											memory_total_gb=0,
											node_name=node_name,
										))
										gpu_idx += 1
									break
							break
					continue

				# Execute nvidia-smi in the pod
				try:
					exec_result = subprocess.run(
						[
							"kubectl", "exec", "-n", pod_info["namespace"],
							pod_info["pod"], "-c", pod_info["container"],
							"--", "nvidia-smi",
							"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
							"--format=csv,noheader,nounits"
						],
						capture_output=True,
						text=True,
						timeout=10
					)

					if exec_result.returncode == 0:
						for line in exec_result.stdout.strip().split("\n"):
							if not line.strip():
								continue
							parts = [p.strip() for p in line.split(",")]
							if len(parts) >= 7:
								try:
									all_gpus.append(GPUInfo(
										index=gpu_idx,
										name=parts[1],
										memory_total_gb=round(float(parts[2]) / 1024, 2),
										memory_used_gb=round(float(parts[3]) / 1024, 2),
										memory_free_gb=round(float(parts[4]) / 1024, 2),
										utilization_percent=float(parts[5]) if parts[5] != "[N/A]" else None,
										temperature_c=int(parts[6]) if parts[6] != "[N/A]" else None,
										node_name=node_name,
									))
									gpu_idx += 1
								except (ValueError, IndexError):
									continue
					else:
						logging.debug(f"nvidia-smi exec failed on {node_name}: {exec_result.stderr}")
						# Add placeholder GPUs without metrics for this node
						for node in nodes_data.get("items", []):
							if node["metadata"]["name"] == node_name:
								capacity = node["status"].get("capacity", {})
								for key in capacity:
									if "nvidia.com/gpu" in key:
										node_gpu_count = int(capacity[key])
										for i in range(node_gpu_count):
											all_gpus.append(GPUInfo(
												index=gpu_idx,
												name="GPU",
												memory_total_gb=0,
												node_name=node_name,
											))
											gpu_idx += 1
										break
								break

				except subprocess.TimeoutExpired:
					logging.warning(f"kubectl exec timed out for node {node_name}")
				except Exception as e:
					logging.warning(f"Failed to get GPU metrics from {node_name}: {e}")

			return all_gpus if all_gpus else local_gpus

		except FileNotFoundError:
			logging.debug("kubectl not found, falling back to local metrics")
			return WorkerSettings._get_gpu_metrics()
		except Exception as e:
			logging.warning(f"Failed to get cluster GPU metrics: {e}")
			return WorkerSettings._get_gpu_metrics()
