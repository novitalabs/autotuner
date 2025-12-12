"""
Autotuner Orchestrator

Main orchestration logic for running parameter tuning experiments.
Coordinates deployment controllers and benchmark execution.
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for relative imports
sys.path.insert(0, str(Path(__file__).parent))

# Lazy imports - only import controllers when needed to avoid dependency issues
# This allows running in local mode without docker/kubernetes dependencies
def _import_ome_controller():
	from controllers.ome_controller import OMEController
	return OMEController

def _import_docker_controller():
	from controllers.docker_controller import DockerController
	return DockerController

def _import_local_controller():
	from controllers.local_controller import LocalController
	return LocalController

def _import_benchmark_controller():
	from controllers.benchmark_controller import BenchmarkController
	return BenchmarkController

def _import_direct_benchmark_controller():
	from controllers.direct_benchmark_controller import DirectBenchmarkController
	return DirectBenchmarkController
from utils.optimizer import generate_parameter_grid, calculate_objective_score, create_optimization_strategy
from utils.quantization_integration import prepare_runtime_parameters
from config import clusterbasemodel_presets, clusterservingruntime_presets


class AutotunerOrchestrator:
	"""Main orchestrator for the autotuning process."""

	def __init__(
		self,
		deployment_mode: str = "ome",
		kubeconfig_path: str = None,
		use_direct_benchmark: bool = False,
		docker_model_path: str = "/mnt/data/models",
		verbose: bool = False,
		http_proxy: str = "",
		https_proxy: str = "",
		no_proxy: str = "",
		hf_token: str = "",
	):
		"""Initialize the orchestrator.

		Args:
		    deployment_mode: Deployment mode - 'ome' (Kubernetes), 'docker' (standalone), or 'local' (subprocess)
		    kubeconfig_path: Path to kubeconfig file (for OME mode)
		    use_direct_benchmark: If True, use direct genai-bench CLI instead of K8s BenchmarkJob
		    docker_model_path: Base path for models in Docker/Local mode
		    verbose: If True, stream genai-bench output in real-time
		    http_proxy: HTTP proxy URL for containers (optional)
		    https_proxy: HTTPS proxy URL for containers (optional)
		    no_proxy: Comma-separated list of hosts to bypass proxy (optional)
		"""
		self.deployment_mode = deployment_mode.lower()
		self.use_direct_benchmark = use_direct_benchmark

		# Initialize model deployment controller based on mode
		if self.deployment_mode == "local":
			print("[Config] Deployment mode: Local subprocess")
			# Use .venv-sglang python if available, otherwise fall back to system python
			sglang_python = Path(__file__).parent.parent / ".venv-sglang" / "bin" / "python"
			if sglang_python.exists():
				python_path = str(sglang_python)
			else:
				python_path = "python3"
			LocalController = _import_local_controller()
			DirectBenchmarkController = _import_direct_benchmark_controller()
			self.model_controller = LocalController(
				model_base_path=docker_model_path,
				python_path=python_path,
				http_proxy=http_proxy,
				https_proxy=https_proxy,
				no_proxy=no_proxy,
				hf_token=hf_token
			)
			# Local mode always uses direct benchmark
			self.use_direct_benchmark = True
			self.benchmark_controller = DirectBenchmarkController(verbose=verbose)
			print("[Config] Benchmark mode: Direct CLI (automatic for Local mode)")
		elif self.deployment_mode == "docker":
			print("[Config] Deployment mode: Standalone Docker")
			DockerController = _import_docker_controller()
			DirectBenchmarkController = _import_direct_benchmark_controller()
			self.model_controller = DockerController(
				model_base_path=docker_model_path,
				http_proxy=http_proxy,
				https_proxy=https_proxy,
				no_proxy=no_proxy,
				hf_token=hf_token
			)
			# Docker mode always uses direct benchmark (no K8s)
			self.use_direct_benchmark = True
			self.benchmark_controller = DirectBenchmarkController(verbose=verbose)
			print("[Config] Benchmark mode: Direct CLI (automatic for Docker mode)")
			print("[Config] Containers will be auto-removed after stop")
		elif self.deployment_mode == "ome":
			print("[Config] Deployment mode: OME (Kubernetes)")
			OMEController = _import_ome_controller()
			self.model_controller = OMEController(kubeconfig_path)
			if use_direct_benchmark:
				DirectBenchmarkController = _import_direct_benchmark_controller()
				self.benchmark_controller = DirectBenchmarkController(verbose=verbose)
				print("[Config] Benchmark mode: Direct genai-bench CLI")
			else:
				BenchmarkController = _import_benchmark_controller()
				self.benchmark_controller = BenchmarkController(kubeconfig_path)
				print("[Config] Benchmark mode: Kubernetes BenchmarkJob CRD")
		else:
			raise ValueError(f"Unknown deployment mode: {deployment_mode}. Use 'ome', 'docker', or 'local'")

		self.results = []

	def run_experiment(self, task: Dict[str, Any], experiment_id: int, parameters: Dict[str, Any], on_benchmark_start=None) -> Dict[str, Any]:
		"""Run a single tuning experiment.

		Args:
		    task: Task configuration
		    experiment_id: Unique experiment identifier
		    parameters: Parameter values for this experiment
		    on_benchmark_start: Optional callback function called when benchmark phase starts

		Returns:
		    Experiment results dictionary
		"""
		task_name = task["task_name"]
		namespace = task["model"]["namespace"]
		model_name = task["model"]["id_or_path"]
		runtime_name = task["base_runtime"]
		timeout = task["optimization"].get("timeout_per_iteration", 1800)  # Default 30 minutes

		# Dynamically adjust timeout for torch-compile + multi-GPU scenarios
		# Triton kernel autotuning can take 10-20 minutes with TP > 1
		enable_compile = parameters.get("enable-torch-compile", False)
		tp_size = parameters.get("__parallel__tp", 1)

		if enable_compile and tp_size > 1:
			# TP > 1 with torch-compile: increase timeout significantly
			adjusted_timeout = max(timeout, 1200)  # At least 20 minutes
			if adjusted_timeout > timeout:
				print(f"[Timeout] Increased from {timeout}s to {adjusted_timeout}s (torch-compile + TP={tp_size})")
				timeout = adjusted_timeout
		elif enable_compile:
			# TP = 1 with torch-compile: moderate increase
			adjusted_timeout = max(timeout, 900)  # At least 15 minutes
			if adjusted_timeout > timeout:
				print(f"[Timeout] Increased from {timeout}s to {adjusted_timeout}s (torch-compile)")
				timeout = adjusted_timeout

		# Optional: custom Docker image tag
		image_tag = task.get("runtime_image_tag")

		# Step 0: Ensure ClusterBaseModel and ClusterServingRuntime exist (OME mode only)
		created_resources = {"clusterbasemodel": None, "clusterservingruntime": None}

		if self.deployment_mode == "ome":
			# Handle ClusterBaseModel creation
			cbm_config = task.get("clusterbasemodel_config")
			if cbm_config:
				print(f"\n[Step 0a/4] Ensuring ClusterBaseModel exists...")
				cbm_name, cbm_created = self._ensure_clusterbasemodel(cbm_config, model_name)
				if cbm_created:
					created_resources["clusterbasemodel"] = cbm_name
					print(f"ClusterBaseModel '{cbm_name}' is ready")
				elif cbm_name:
					print(f"Using existing ClusterBaseModel '{cbm_name}'")
				else:
					print("Warning: Failed to ensure ClusterBaseModel, using model_name as fallback")

			# Handle ClusterServingRuntime creation
			csr_config = task.get("clusterservingruntime_config")
			if csr_config:
				print(f"\n[Step 0b/4] Ensuring ClusterServingRuntime exists...")
				csr_name, csr_created = self._ensure_clusterservingruntime(csr_config, runtime_name)
				if csr_created:
					created_resources["clusterservingruntime"] = csr_name
					print(f"ClusterServingRuntime '{csr_name}' is ready")
				elif csr_name:
					print(f"Using existing ClusterServingRuntime '{csr_name}'")
				else:
					print("Warning: Failed to ensure ClusterServingRuntime, using runtime_name as fallback")

				# Update runtime_name to use the created/ensured runtime
				if csr_name:
					runtime_name = csr_name

		print(f"\n{'='*80}")
		print(f"Experiment {experiment_id}")
		print(f"Parameters: {parameters}")
		print(f"{'='*80}\n")

		# Convert __quant__ prefixed parameters to runtime-specific CLI args
		runtime_parameters = prepare_runtime_parameters(
			base_runtime=runtime_name,
			params=parameters,
			model_path=model_name,
			model_config=task.get("model")
		)

		print(f"Runtime-specific parameters: {runtime_parameters}")

		experiment_result = {
			"experiment_id": experiment_id,
			"parameters": parameters,  # Keep original for database
			"status": "failed",
			"metrics": None,
			"container_logs": None,  # Will store container logs for Docker mode
		"error_message": None,  # Will store error details for failed experiments
		"created_resources": created_resources,  # Track created resources
		}

		# Step 1: Deploy InferenceService
		print(f"[Step 1/4] Deploying InferenceService...")

		# Extract storage configuration if present (for PVC support)
		storage_config = task.get("storage")

		# Call deploy based on deployment mode
		if self.deployment_mode == "docker":
			# DockerController
			isvc_name = self.model_controller.deploy_inference_service(
				task_name=task_name,
				experiment_id=experiment_id,
				namespace=namespace,
				model_name=model_name,
				runtime_name=runtime_name,
				parameters=runtime_parameters,
				image_tag=image_tag,
			)
		elif self.deployment_mode == "local":
			# LocalController
			isvc_name = self.model_controller.deploy_inference_service(
				task_name=task_name,
				experiment_id=experiment_id,
				namespace=namespace,
				model_name=model_name,
				runtime_name=runtime_name,
				parameters=runtime_parameters,
			)
		else:
			# OMEController
			isvc_name = self.model_controller.deploy_inference_service(
				task_name=task_name,
				experiment_id=experiment_id,
				namespace=namespace,
				model_name=model_name,
				runtime_name=runtime_name,
				parameters=runtime_parameters,
				storage=storage_config,
				enable_gpu_selection=False,
			)

		if not isvc_name:
			error_msg = "Failed to deploy InferenceService"
			print(error_msg)
			experiment_result["error_message"] = error_msg
			return experiment_result

		# Step 2: Wait for InferenceService to be ready
		print(f"\n[Step 2/4] Waiting for InferenceService to be ready...")
		if not self.model_controller.wait_for_ready(isvc_name, namespace, timeout=timeout):
			error_msg = "InferenceService did not become ready in time"
			print(error_msg)
			# Capture container logs for debugging
			container_logs = self.cleanup_experiment(isvc_name, None, namespace)
			if container_logs:
				experiment_result["container_logs"] = container_logs
				# Extract key error info from container logs
				if "ValueError" in container_logs or "Error" in container_logs:
					# Get last few lines of error for error_message
					log_lines = container_logs.strip().split('\n')
					error_lines = [line for line in log_lines if "Error" in line or "failed" in line.lower()]
					if error_lines:
						error_msg += f" - {error_lines[-1][:200]}"  # Last error, truncated
			experiment_result["error_message"] = error_msg
			return experiment_result

		# Get GPU information if available (Docker mode)
		gpu_info = None
		if self.deployment_mode == "docker" and hasattr(self.model_controller, "get_gpu_info"):
			gpu_info = self.model_controller.get_gpu_info(isvc_name, namespace)
			if gpu_info:
				experiment_result["gpu_info"] = gpu_info

		# Step 3: Run benchmark
		print(f"\n[Step 3/4] Running benchmark...")

		# Notify that benchmark phase is starting
		if on_benchmark_start:
			on_benchmark_start()

		if self.use_direct_benchmark:
			# Get endpoint URL (differs between Docker, Local and OME modes)
			endpoint_url = None
			gpu_indices = None

			if self.deployment_mode == "docker":
				# Docker mode: Get direct URL from controller
				endpoint_url = self.model_controller.get_service_url(isvc_name, namespace)
				if not endpoint_url:
					error_msg = "Failed to get service URL from Docker controller"
					print(error_msg)
					experiment_result["error_message"] = error_msg
					self.cleanup_experiment(isvc_name, None, namespace, experiment_id)
					return experiment_result

				# Extract GPU indices from gpu_info for monitoring
				if gpu_info and "indices" in gpu_info.get("gpu_info", {}):
					gpu_indices = gpu_info["gpu_info"]["indices"]
					print(f"[GPU Monitor] Will monitor GPUs: {gpu_indices} during benchmark")

			elif self.deployment_mode == "local":
				# Local mode: Get direct URL from controller
				endpoint_url = self.model_controller.get_service_url(isvc_name, namespace)
				if not endpoint_url:
					error_msg = "Failed to get service URL from Local controller"
					print(error_msg)
					experiment_result["error_message"] = error_msg
					self.cleanup_experiment(isvc_name, None, namespace, experiment_id)
					return experiment_result

				# Extract GPU indices from controller's process info for monitoring
				local_gpu_info = self.model_controller.get_gpu_info(isvc_name, namespace)
				if local_gpu_info and local_gpu_info.get("device_ids"):
					gpu_indices = [int(idx) for idx in local_gpu_info["device_ids"]]
					print(f"[GPU Monitor] Will monitor GPUs: {gpu_indices} during benchmark")

			# Direct CLI execution with automatic port forwarding (OME) or direct URL (Docker)
			# Merge slo_config into benchmark_config for SLO-aware filtering
			benchmark_config_with_slo = task["benchmark"].copy()
			if "slo" in task:
				benchmark_config_with_slo["slo_config"] = task["slo"]
				slo_value = benchmark_config_with_slo.get("slo_config")
				print(f"[DEBUG ORCHESTRATOR] slo_config value: {slo_value}")

			metrics = self.benchmark_controller.run_benchmark(
				task_name=task_name,
				experiment_id=experiment_id,
				service_name=isvc_name,
				namespace=namespace,
				benchmark_config=benchmark_config_with_slo,
				timeout=timeout,
				endpoint_url=endpoint_url,
				gpu_indices=gpu_indices,
			)
			benchmark_name = None  # No K8s resource to track

			# Step 4: Process results
			print(f"\n[Step 4/4] Processing results...")
			if metrics:
				# Add per-GPU throughput metrics BEFORE calculating objective score
				# This ensures per-GPU values are available for optimization
				if gpu_info and gpu_info.get("count", 0) > 0:
					gpu_count = gpu_info["count"]
					# Add per-GPU throughput for all throughput metrics
					if "mean_output_throughput" in metrics:
						metrics["mean_output_throughput_per_gpu"] = metrics["mean_output_throughput"] / gpu_count
					if "max_output_throughput" in metrics:
						metrics["max_output_throughput_per_gpu"] = metrics["max_output_throughput"] / gpu_count
					if "mean_total_throughput" in metrics:
						metrics["mean_total_throughput_per_gpu"] = metrics["mean_total_throughput"] / gpu_count
					if "max_total_throughput" in metrics:
						metrics["max_total_throughput_per_gpu"] = metrics["max_total_throughput"] / gpu_count

					# Also add per-GPU metrics to raw results
					for raw_result in metrics.get("raw_results", []):
						if "mean_output_throughput_tokens_per_s" in raw_result:
							raw_result["mean_output_throughput_per_gpu"] = raw_result["mean_output_throughput_tokens_per_s"] / gpu_count
						if "mean_input_throughput_tokens_per_s" in raw_result:
							raw_result["mean_input_throughput_per_gpu"] = raw_result["mean_input_throughput_tokens_per_s"] / gpu_count
						if "mean_total_tokens_throughput_tokens_per_s" in raw_result:
							raw_result["mean_total_throughput_per_gpu"] = raw_result["mean_total_tokens_throughput_tokens_per_s"] / gpu_count

				# Get SLO configuration from task if present
				slo_config = task.get("slo")

				# Calculate objective score with SLO penalties (per-GPU metrics now available)
				score = calculate_objective_score(metrics, task["optimization"]["objective"], slo_config)

				# Check if this is a hard SLO failure (score = inf/-inf)
				is_slo_failure = (score == float("inf") or score == float("-inf"))

				if is_slo_failure:
					experiment_result["status"] = "failed"
					experiment_result["slo_violation"] = True
					experiment_result["error_message"] = "Hard SLO violation - experiment exceeded threshold limits"
					print(f"Experiment {experiment_id} FAILED due to hard SLO violation")
				else:
					experiment_result["status"] = "success"
					experiment_result["metrics"] = metrics
					experiment_result["objective_score"] = score
					print(f"Experiment {experiment_id} completed. Score: {score}")
			else:
				error_msg = "Failed to retrieve benchmark results"
				print(error_msg)
				experiment_result["error_message"] = error_msg
		else:
			# K8s BenchmarkJob execution
			benchmark_name = self.benchmark_controller.create_benchmark_job(
				task_name=task_name,
				experiment_id=experiment_id,
				namespace=namespace,
				isvc_name=isvc_name,
				benchmark_config=task["benchmark"],
			)

			if not benchmark_name:
				error_msg = "Failed to create BenchmarkJob"
				print(error_msg)
				experiment_result["error_message"] = error_msg
				self.cleanup_experiment(isvc_name, None, namespace)
				return experiment_result

			# Wait for benchmark to complete
			if not self.benchmark_controller.wait_for_completion(benchmark_name, namespace, timeout=timeout):
				error_msg = "Benchmark did not complete in time"
				print(error_msg)
				experiment_result["error_message"] = error_msg
				self.cleanup_experiment(isvc_name, benchmark_name, namespace)
				return experiment_result

			# Step 4: Collect results
			print(f"\n[Step 4/4] Collecting results...")
			metrics = self.benchmark_controller.get_benchmark_results(benchmark_name, namespace)

			if metrics:
				# Get SLO configuration from task if present
				slo_config = task.get("slo")

				# Calculate objective score with SLO penalties
				score = calculate_objective_score(metrics, task["optimization"]["objective"], slo_config)

				# Check if this is a hard SLO failure (score = inf/-inf)
				is_slo_failure = (score == float("inf") or score == float("-inf"))

				if is_slo_failure:
					experiment_result["status"] = "failed"
					experiment_result["slo_violation"] = True
					experiment_result["error_message"] = "Hard SLO violation - experiment exceeded threshold limits"
					print(f"Experiment {experiment_id} FAILED due to hard SLO violation")
				else:
					experiment_result["status"] = "success"
					experiment_result["metrics"] = metrics
					experiment_result["objective_score"] = score
					print(f"Experiment {experiment_id} completed. Score: {score}")
			else:
				error_msg = "Failed to retrieve benchmark results"
				print(error_msg)
				experiment_result["error_message"] = error_msg

		# Cleanup
		print(f"\n[Cleanup] Removing experiment resources...")
		container_logs = self.cleanup_experiment(isvc_name, benchmark_name, namespace, experiment_id)

		# Store container logs in result if available
		if container_logs:
			experiment_result["container_logs"] = container_logs

		return experiment_result

	def cleanup_experiment(self, isvc_name: str, benchmark_name: str, namespace: str, experiment_id: int = None) -> str:
		"""Clean up experiment resources.

		Args:
		    isvc_name: InferenceService name
		    benchmark_name: BenchmarkJob name (can be None)
		    namespace: K8s namespace
		    experiment_id: Experiment ID (for direct benchmark cleanup)

		Returns:
		    Container logs if available (Docker mode only), None otherwise
		"""
		container_logs = None

		# Retrieve container logs before deletion (Docker mode only)
		if self.deployment_mode == "docker" and hasattr(self.model_controller, "get_container_logs"):
			print(f"[Cleanup] Retrieving container logs before deletion...")
			container_logs = self.model_controller.get_container_logs(isvc_name, namespace, tail=0)  # Get ALL logs
			if container_logs:
				print(f"[Cleanup] Retrieved {len(container_logs)} bytes of container logs")

		# Continue with cleanup
		if self.use_direct_benchmark and experiment_id:
			self.benchmark_controller.cleanup_results(
				task_name=isvc_name.rsplit("-exp", 1)[0], experiment_id=experiment_id
			)
		elif benchmark_name:
			self.benchmark_controller.delete_benchmark_job(benchmark_name, namespace)

		if isvc_name:
			self.model_controller.delete_inference_service(isvc_name, namespace)

		return container_logs

	def run_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
		"""Run a complete autotuning task using optimization strategy.

		Args:
		    task: Task configuration dictionary

		Returns:
		    Summary of all experiments
		"""
		task_name = task["task_name"]
		print(f"Starting task: {task_name}")
		if "description" in task:
			print(f"Description: {task['description']}")

		# Create optimization strategy
		optimization_config = task.get("optimization", {})
		strategy_name = optimization_config.get("strategy", "grid_search")
		objective = optimization_config.get("objective", "minimize_latency")

		print(f"\nOptimization strategy: {strategy_name}")
		print(f"Objective: {objective}")

		# Merge parameters and parallel_config for optimization
		# This allows parallel parameters (tp, pp, dp, etc.) to be tuned alongside other parameters
		combined_parameters = dict(task.get("parameters", {}))
		parallel_config = task.get("parallel_config", {})

		if parallel_config:
			print(f"\nParallel configuration detected:")
			for key, value in parallel_config.items():
				print(f"  {key}: {value}")
				# Add parallel config to combined parameters for optimization
				combined_parameters[key] = value

		try:
			strategy = create_optimization_strategy(optimization_config, combined_parameters)
		except Exception as e:
			print(f"Error creating optimization strategy: {e}")
			raise

		# Run experiments using strategy
		start_time = time.time()
		iteration = 0
		max_iterations = optimization_config.get("max_iterations", 100)

		print(f"\nStarting optimization (max {max_iterations} iterations)...")

		while not strategy.should_stop():
			iteration += 1

			# Get next parameter suggestion from strategy
			parameters = strategy.suggest_parameters()
			if parameters is None:
				print(f"[Orchestrator] Strategy has no more suggestions")
				break

			# Run experiment
			print(f"\n{'='*80}")
			print(f"Experiment {iteration}")
			print(f"{'='*80}")
			result = self.run_experiment(task, iteration, parameters)
			self.results.append(result)

			# Update strategy with result
			if result["status"] == "success":
				strategy.tell_result(
					parameters=parameters,
					objective_score=result["objective_score"],
					metrics=result.get("metrics", {})
				)
			else:
				# For failed experiments, report worst possible score
				worst_score = float("inf") if "minimize" in objective else float("-inf")
				strategy.tell_result(
					parameters=parameters,
					objective_score=worst_score,
					metrics={}
				)

		elapsed = time.time() - start_time

		# Find best result
		successful_results = [r for r in self.results if r["status"] == "success"]
		if successful_results:
			best_result = min(successful_results, key=lambda r: r["objective_score"])
			print(f"\n{'='*80}")
			print(f"AUTOTUNING COMPLETE")
			print(f"{'='*80}")
			print(f"Strategy: {strategy_name}")
			print(f"Total experiments: {len(self.results)}")
			print(f"Successful: {len(successful_results)}")
			print(f"Failed: {len(self.results) - len(successful_results)}")
			print(f"Total time: {elapsed:.1f}s")
			print(f"\nBest configuration:")
			print(f"  Parameters: {best_result['parameters']}")
			print(f"  Score: {best_result['objective_score']:.4f}")
		else:
			best_result = None
			print("\nNo successful experiments!")

		return {
			"task_name": task_name,
			"strategy": strategy_name,
			"total_experiments": len(self.results),
			"successful_experiments": len(successful_results),
			"elapsed_time": elapsed,
			"best_result": best_result,
			"all_results": self.results,
		}

	def _ensure_clusterbasemodel(self, config: Dict[str, Any], fallback_name: str) -> tuple[str, bool]:
		"""Ensure ClusterBaseModel exists, create if needed.

		Args:
		    config: ClusterBaseModel configuration (preset or spec)
		    fallback_name: Fallback name if config doesn't specify

		Returns:
		    Tuple of (resource_name, was_created)
		"""
		if self.deployment_mode != "ome":
			print("Warning: ClusterBaseModel creation only supported in OME mode")
			return (fallback_name, False)

		# Check if config specifies a preset
		preset_name = config.get("preset")
		if preset_name:
			try:
				preset = clusterbasemodel_presets.get_preset(preset_name)
				name = preset["name"]
				spec = preset["spec"]
				print(f"Using ClusterBaseModel preset: {preset['display_name']}")
			except ValueError as e:
				print(f"Error loading preset: {e}")
				return (fallback_name, False)
		else:
			# Custom configuration
			name = config.get("name", fallback_name)
			spec = config.get("spec")
			if not spec:
				print("Error: ClusterBaseModel config must have 'preset' or 'spec'")
				return (fallback_name, False)

		# Apply any overrides
		overrides = config.get("overrides", {})
		if overrides:
			# Deep merge overrides into spec
			for key, value in overrides.items():
				if isinstance(value, dict) and key in spec and isinstance(spec[key], dict):
					spec[key] = {**spec[key], **value}
				else:
					spec[key] = value

		# Ensure resource exists
		success = self.model_controller.ensure_clusterbasemodel(
			name=name,
			spec=spec,
			labels=config.get("labels"),
			annotations=config.get("annotations")
		)

		return (name if success else fallback_name, success)

	def _ensure_clusterservingruntime(self, config: Dict[str, Any], fallback_name: str) -> tuple[str, bool]:
		"""Ensure ClusterServingRuntime exists, create if needed.

		Args:
		    config: ClusterServingRuntime configuration (preset or spec)
		    fallback_name: Fallback name if config doesn't specify

		Returns:
		    Tuple of (resource_name, was_created)
		"""
		if self.deployment_mode != "ome":
			print("Warning: ClusterServingRuntime creation only supported in OME mode")
			return (fallback_name, False)

		# Check if config specifies a preset
		preset_name = config.get("preset")
		if preset_name:
			try:
				preset = clusterservingruntime_presets.get_preset(preset_name)
				name = preset["name"]
				spec = preset["spec"]
				print(f"Using ClusterServingRuntime preset: {preset['display_name']}")
			except ValueError as e:
				print(f"Error loading preset: {e}")
				return (fallback_name, False)
		else:
			# Custom configuration
			name = config.get("name", fallback_name)
			spec = config.get("spec")
			if not spec:
				print("Error: ClusterServingRuntime config must have 'preset' or 'spec'")
				return (fallback_name, False)

		# Apply any overrides
		overrides = config.get("overrides", {})
		if overrides:
			# Deep merge overrides into spec
			def deep_merge(base, override):
				result = base.copy()
				for key, value in override.items():
					if isinstance(value, dict) and key in result and isinstance(result[key], dict):
						result[key] = deep_merge(result[key], value)
					else:
						result[key] = value
				return result

			spec = deep_merge(spec, overrides)

		# Ensure resource exists
		success = self.model_controller.ensure_clusterservingruntime(
			name=name,
			spec=spec,
			labels=config.get("labels"),
			annotations=config.get("annotations")
		)

		return (name if success else fallback_name, success)
