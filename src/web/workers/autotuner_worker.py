"""
ARQ worker configuration and task functions.
"""

import sys
import logging
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
from typing import Dict, Any
import io
import asyncio

from src.web.config import get_settings
from src.web.db.models import Task, Experiment, TaskStatus, ExperimentStatus
from src.orchestrator import AutotunerOrchestrator
from src.utils.optimizer import generate_parameter_grid, create_optimization_strategy, restore_optimization_strategy
from src.utils.quantization_integration import merge_parameters_with_quant_config
from src.utils.gpu_scheduler import estimate_gpu_requirements, check_gpu_availability, wait_for_gpu_availability
from src.web.workers.checkpoint import TaskCheckpoint
from src.web.events.broadcaster import get_broadcaster, EventType, create_event

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


async def run_autotuning_task(ctx: Dict[str, Any], task_id: int) -> Dict[str, Any]:
	"""Run autotuning task in background.

	Args:
	    ctx: ARQ context
	    task_id: Database task ID

	Returns:
	    Task summary dict
	"""
	# Setup logging for this task
	logger = setup_task_logging(task_id)

	async with AsyncSessionLocal() as db:
		# Get task from database
		result = await db.execute(select(Task).where(Task.id == task_id))
		task = result.scalar_one_or_none()

		if not task:
			error_msg = f"Task {task_id} not found"
			logger.error(error_msg)
			return {"error": error_msg}

		try:
			logger.info(f"[ARQ Worker] Starting task: {task.task_name}")

			# Get broadcaster instance
			broadcaster = get_broadcaster()

			# Update task status
			task.status = TaskStatus.RUNNING
			task.started_at = datetime.utcnow()
			await db.commit()

			# Broadcast task started event
			broadcaster.broadcast_sync(
				task_id,
				create_event(
					EventType.TASK_STARTED,
					task_id=task_id,
					message=f"Task '{task.task_name}' started"
				)
			)

			# Create task configuration dict (similar to JSON task file)
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

			# Check GPU availability before starting task (only for Docker mode)
			if task.deployment_mode == "docker":
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

						# Mark task as failed
						task.status = TaskStatus.FAILED
						task.completed_at = datetime.utcnow()
						elapsed_time = (task.completed_at - task.started_at).total_seconds()
						task.elapsed_time = elapsed_time
						await db.commit()

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
				logger.info(f"[ARQ Worker] Skipping GPU check for {task.deployment_mode} mode")

			# Create optimization strategy
			optimization_config = task.optimization_config or {}
			strategy_name = optimization_config.get("strategy", "grid_search")
			max_iterations = optimization_config.get("max_iterations", 100)
			timeout_per_iteration = optimization_config.get("timeout_per_iteration", 1800)  # Default 30 minutes

			logger.info(f"[ARQ Worker] Optimization strategy: {strategy_name}")
			logger.info(f"[ARQ Worker] Max iterations: {max_iterations}")
			logger.info(f"[ARQ Worker] Timeout per experiment: {timeout_per_iteration}s")

			# Check for existing checkpoint and resume if available
			checkpoint = TaskCheckpoint.load_checkpoint(task.task_metadata)
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
					merged_parameters = merge_parameters_with_quant_config(
						task.parameters or {},
						task.quant_config
					)
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
				merged_parameters = merge_parameters_with_quant_config(
					task.parameters or {},
					task.quant_config
				)
				logger.info(f"[ARQ Worker] Merged parameters (base + quant_config): {merged_parameters}")

				# Then merge parallel_config
				from utils.parallel_integration import merge_parameters_with_parallel_config
				merged_parameters = merge_parameters_with_parallel_config(
					merged_parameters,
					task.parallel_config
				)
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
				merged_parameters = merge_parameters_with_quant_config(
					task.parameters or {},
					task.quant_config
				)
				param_grid = generate_parameter_grid(merged_parameters)
				total_experiments = min(len(param_grid), max_iterations)
			else:
				# Bayesian/random: use max_iterations as upper bound
				total_experiments = max_iterations

			task.total_experiments = total_experiments
			await db.commit()

			logger.info(f"[ARQ Worker] Expected experiments: {total_experiments}")

			# Create orchestrator
			orchestrator = AutotunerOrchestrator(
				deployment_mode=task.deployment_mode,
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

					# Save checkpoint after each experiment
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
									"successful_experiments": task.successful_experiments,
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
					task_name = task.task_name
					namespace = task.model_config.get("namespace", "default")
					service_id = f"{task_name}-exp{iteration}"
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

			# Update task total_experiments with actual count
			task.total_experiments = iteration

			# Update task with final results
			# Refresh task object to ensure it's properly tracked by the session
			await db.refresh(task)

			# Check if any experiments succeeded
			if task.successful_experiments > 0:
				task.status = TaskStatus.COMPLETED
			else:
				# All experiments failed
				task.status = TaskStatus.FAILED
				logger.warning(f"[ARQ Worker] Task {task.task_name} - All {iteration} experiments failed")

			task.completed_at = datetime.utcnow()
			task.best_experiment_id = best_experiment_id

			# Clear checkpoint after successful completion
			task.task_metadata = TaskCheckpoint.clear_checkpoint(task.task_metadata)

			if task.started_at:
				elapsed = (task.completed_at - task.started_at).total_seconds()
				task.elapsed_time = elapsed
				logger.info(f"[ARQ Worker] Task completed in {elapsed:.2f}s - Best experiment: {best_experiment_id}")

			await db.commit()
			await db.refresh(task)  # Ensure changes are reflected

			# Broadcast task completion event
			broadcaster.broadcast_sync(
				task_id,
				create_event(
					EventType.TASK_COMPLETED if task.status == TaskStatus.COMPLETED else EventType.TASK_FAILED,
					task_id=task_id,
					data={
						"status": task.status.value,
						"total_experiments": iteration,
						"successful_experiments": task.successful_experiments,
						"best_experiment_id": best_experiment_id,
						"best_score": best_score if best_score != float("inf") else None,
						"elapsed_time": elapsed if task.started_at else None
					},
					message=f"Task completed: {task.successful_experiments}/{iteration} experiments successful"
				)
			)

			logger.info(
				f"[ARQ Worker] Task finished: {task.task_name} - {task.successful_experiments}/{total_experiments} successful"
			)
			return {"task_id": task_id, "task_name": task.task_name, "status": "completed"}

		except Exception as e:
			logger.error(f"[ARQ Worker] Task failed: {e}", exc_info=True)
			task.status = TaskStatus.FAILED
			task.completed_at = datetime.utcnow()
			await db.commit()
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
