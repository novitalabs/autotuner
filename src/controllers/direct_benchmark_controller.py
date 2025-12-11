"""
Direct GenAI-Bench Controller

Runs genai-bench directly using the CLI instead of Kubernetes BenchmarkJob CRDs.
This bypasses the genai-bench v251014 image issues by using the local installation.
"""

import json
import subprocess
import time
import signal
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys

# Add src to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gpu_monitor import get_gpu_monitor, GPUSnapshot
from utils.optimizer import check_batch_slo_compliance


class DirectBenchmarkController:
	"""Controller for running genai-bench directly via CLI."""

	def __init__(self, genai_bench_path: str = "env/bin/genai-bench", verbose: bool = False):
		"""Initialize the direct benchmark controller.

		Args:
		    genai_bench_path: Path to genai-bench executable (can be relative or absolute)
		    verbose: If True, stream genai-bench output in real-time
		"""
		# Convert to Path and resolve to absolute path
		genai_bench_path_obj = Path(genai_bench_path)

		# If relative path, resolve relative to project root
		if not genai_bench_path_obj.is_absolute():
			# Try to find project root (where src/ directory is located)
			current_file = Path(__file__).resolve()  # controllers/direct_benchmark_controller.py
			project_root = current_file.parent.parent.parent  # Go up to inference-autotuner/
			genai_bench_path_obj = project_root / genai_bench_path_obj

		self.genai_bench_path = genai_bench_path_obj
		if not self.genai_bench_path.exists():
			raise FileNotFoundError(f"genai-bench not found at {self.genai_bench_path}")

		self.verbose = verbose

		# Results directory - always resolve relative to project root
		current_file = Path(__file__).resolve()  # controllers/direct_benchmark_controller.py
		project_root = current_file.parent.parent.parent  # Go up to inference-autotuner/
		self.results_dir = project_root / "benchmark_results"
		self.results_dir.mkdir(exist_ok=True)

		# Port forward process tracking
		self.port_forward_proc = None
		self.local_port = None

	def _monitor_gpus_during_benchmark(
		self,
		gpu_indices: List[int],
		duration_seconds: float,
		interval_seconds: float = 1.0,
		stop_event: Optional[threading.Event] = None
	) -> List[GPUSnapshot]:
		"""Monitor GPUs during benchmark execution (runs in background thread).

		Args:
		    gpu_indices: List of GPU indices to monitor
		    duration_seconds: Maximum monitoring duration
		    interval_seconds: Sampling interval
		    stop_event: Threading event to signal early termination

		Returns:
		    List of GPU snapshots
		"""
		gpu_monitor = get_gpu_monitor()
		snapshots = []
		start_time = time.time()

		print(f"[GPU Monitor] Starting monitoring for GPUs {gpu_indices} (interval={interval_seconds}s)")

		while (time.time() - start_time) < duration_seconds:
			# Check if early termination requested
			if stop_event and stop_event.is_set():
				print(f"[GPU Monitor] Stopped early (benchmark completed)")
				break

			snapshot = gpu_monitor.query_gpus(use_cache=False)
			if snapshot:
				# Filter to requested GPUs
				filtered_gpus = [gpu for gpu in snapshot.gpus if gpu.index in gpu_indices]
				if filtered_gpus:
					from utils.gpu_monitor import GPUSnapshot
					from datetime import datetime
					filtered_snapshot = GPUSnapshot(
						timestamp=datetime.now(),
						gpus=filtered_gpus,
						total_gpus=len(filtered_gpus),
						available_gpus=sum(1 for gpu in filtered_gpus if gpu.is_available)
					)
					snapshots.append(filtered_snapshot)

			time.sleep(interval_seconds)

		print(f"[GPU Monitor] Collected {len(snapshots)} snapshots over {time.time() - start_time:.1f}s")
		return snapshots

	def _ensure_tokenizer_cached(self, model_tokenizer: str) -> tuple:
		"""Ensure tokenizer is cached locally, downloading with proxy if needed.

		Args:
		    model_tokenizer: HuggingFace tokenizer ID (e.g., "meta-llama/Llama-3.2-3B-Instruct")

		Returns:
		    Tuple of (success: bool, is_fully_cached: bool)
		    - success: True if tokenizer is cached or successfully downloaded
		    - is_fully_cached: True if tokenizer was already fully cached (can use offline mode)
		"""
		import os
		from pathlib import Path

		# Check if tokenizer is already cached
		cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
		# HuggingFace cache uses format: models--org--model-name
		cache_name = "models--" + model_tokenizer.replace("/", "--")
		cached_path = cache_dir / cache_name

		if cached_path.exists():
			# Check if tokenizer_config.json exists (indicates complete tokenizer cache)
			# Look for tokenizer files in snapshots directory
			snapshots_dir = cached_path / "snapshots"
			has_tokenizer_config = False
			if snapshots_dir.exists():
				for snapshot_dir in snapshots_dir.iterdir():
					if snapshot_dir.is_dir():
						tokenizer_config = snapshot_dir / "tokenizer_config.json"
						if tokenizer_config.exists():
							has_tokenizer_config = True
							break

			if has_tokenizer_config:
				print(f"[Tokenizer] Fully cached (offline mode OK): {model_tokenizer}")
				return True, True
			else:
				print(f"[Tokenizer] Partially cached (needs online access): {model_tokenizer}")
				return True, False

		print(f"[Tokenizer] Not cached, downloading: {model_tokenizer}")

		# Try to download with proxy if configured
		proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')

		try:
			# Setup environment for download
			env = os.environ.copy()
			if proxy_url:
				env['HTTP_PROXY'] = proxy_url
				env['http_proxy'] = proxy_url
				env['HTTPS_PROXY'] = proxy_url
				env['https_proxy'] = proxy_url
				print(f"[Tokenizer] Using proxy for download: {proxy_url}")

			# Pass through HF_TOKEN if set
			if 'HF_TOKEN' in os.environ:
				env['HF_TOKEN'] = os.environ['HF_TOKEN']

			# Download tokenizer using Python subprocess
			import subprocess
			code = f"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{model_tokenizer}')"
			result = subprocess.run(
				["python3", "-c", code],
				env=env,
				capture_output=True,
				text=True,
				timeout=120  # 2 minutes timeout
			)

			if result.returncode != 0:
				print(f"[Tokenizer] Download failed: {result.stderr}")
				return False, False

			print(f"[Tokenizer] Successfully downloaded and cached: {model_tokenizer}")
			return True, True

		except subprocess.TimeoutExpired:
			print(f"[Tokenizer] Download timeout after 120s")
			return False, False
		except Exception as e:
			print(f"[Tokenizer] Error downloading tokenizer: {e}")
			return False, False

	def setup_port_forward(
		self, service_name: str, namespace: str, remote_port: int = 8080, local_port: int = 8080
	) -> Optional[str]:
		"""Setup kubectl port-forward for accessing InferenceService.

		Args:
		    service_name: InferenceService name (used to find pods)
		    namespace: K8s namespace
		    remote_port: Remote service port (default 8080 for OME InferenceServices)
		    local_port: Local port to forward to

		Returns:
		    Local endpoint URL if successful, None otherwise
		"""
		print(f"[Port Forward] Setting up port forward for {service_name}.{namespace}...")

		# First, find the pod for this InferenceService
		# OME creates pods with labels matching the InferenceService
		try:
			import subprocess as sp

			result = sp.run(
				[
					"kubectl",
					"get",
					"pods",
					"-n",
					namespace,
					"-l",
					f"ome.io/inferenceservice={service_name}",
					"-o",
					"jsonpath={.items[0].metadata.name}",
				],
				capture_output=True,
				text=True,
				timeout=10,
			)

			if result.returncode != 0 or not result.stdout.strip():
				print(f"[Port Forward] No pods found for InferenceService {service_name}")
				print(f"[Port Forward] Trying direct service name with -engine suffix...")
				# Fallback to service name with -engine suffix
				pod_or_svc = f"svc/{service_name}-engine"
			else:
				pod_name = result.stdout.strip()
				print(f"[Port Forward] Found pod: {pod_name}")
				pod_or_svc = f"pod/{pod_name}"

		except Exception as e:
			print(f"[Port Forward] Error finding pod: {e}")
			pod_or_svc = f"svc/{service_name}-engine"

		# Start port-forward in background
		cmd = [
			"kubectl",
			"port-forward",
			pod_or_svc,
			f"{local_port}:{remote_port}",
			"-n",
			namespace,
		]

		try:
			self.port_forward_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
			self.local_port = local_port

			# Wait a bit for port forward to establish
			time.sleep(5)

			# Check if process is still running
			if self.port_forward_proc.poll() is not None:
				stderr = self.port_forward_proc.stderr.read()
				print(f"[Port Forward] Failed to establish: {stderr}")
				return None

			endpoint = f"http://localhost:{local_port}"
			print(f"[Port Forward] Established: {endpoint}")
			return endpoint

		except Exception as e:
			print(f"[Port Forward] Error: {e}")
			return None

	def cleanup_port_forward(self):
		"""Stop port forward process."""
		if self.port_forward_proc:
			print(f"[Port Forward] Stopping port forward on localhost:{self.local_port}")
			try:
				self.port_forward_proc.send_signal(signal.SIGTERM)
				self.port_forward_proc.wait(timeout=5)
			except subprocess.TimeoutExpired:
				self.port_forward_proc.kill()
			except Exception as e:
				print(f"[Port Forward] Error stopping: {e}")
			finally:
				self.port_forward_proc = None
				self.local_port = None


	def _warmup_service(self, endpoint_url: str, model_name: str, num_requests: int = 3):
		"""Warmup the inference service by sending a few requests.
		
		This triggers torch compile, CUDA graph capture, and other JIT optimizations
		that would otherwise impact the first benchmark batch.
		
		Args:
		    endpoint_url: Service endpoint URL
		    model_name: Model name for the API request
		    num_requests: Number of warmup requests to send (default: 3)
		"""
		import requests
		import time
		
		print(f"[Warmup] Sending {num_requests} warmup requests to trigger JIT compilation...")
		
		warmup_prompt = "Hello, this is a warmup request."
		
		for i in range(num_requests):
			try:
				response = requests.post(
					f"{endpoint_url}/v1/completions",
					json={
						"model": model_name,
						"prompt": warmup_prompt,
						"max_tokens": 10,
						"temperature": 0.0
					},
					timeout=30
				)
				
				if response.status_code == 200:
					print(f"[Warmup] Request {i+1}/{num_requests} completed")
				else:
					print(f"[Warmup] Request {i+1}/{num_requests} returned status {response.status_code}")
					
			except Exception as e:
				print(f"[Warmup] Request {i+1}/{num_requests} failed: {e}")
			
			# Small delay between requests
			if i < num_requests - 1:
				time.sleep(0.5)
		
		# Wait a bit for compilation to fully complete
		print(f"[Warmup] Waiting 2 seconds for JIT compilation to complete...")
		time.sleep(2)
		print(f"[Warmup] Warmup phase completed")

	def run_benchmark(
		self,
		task_name: str,
		experiment_id: int,
		service_name: str,
		namespace: str,
		benchmark_config: Dict[str, Any],
		timeout: int = 1800,
		local_port: int = 8080,
		endpoint_url: Optional[str] = None,
		gpu_indices: Optional[List[int]] = None,
	) -> Optional[Dict[str, Any]]:
		"""Run benchmark against an inference endpoint with automatic port forwarding.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Unique experiment identifier
		    service_name: K8s service name (or Docker container name)
		    namespace: K8s namespace (ignored in Docker mode)
		    benchmark_config: Benchmark configuration from input JSON
		    timeout: Maximum execution time in seconds
		    local_port: Local port for port forwarding (ignored if endpoint_url is provided)
		    endpoint_url: Optional direct endpoint URL (skips port-forward setup for Docker mode)
		    gpu_indices: Optional list of GPU indices to monitor during benchmark

		Returns:
		    Dict containing benchmark metrics and GPU statistics, or None if failed
		"""
		benchmark_name = f"{task_name}-exp{experiment_id}"
		output_dir = self.results_dir / benchmark_name

		# Ensure tokenizer is cached before running benchmark
		model_tokenizer = benchmark_config.get("model_tokenizer", "gpt2")
		tokenizer_success, is_fully_cached = self._ensure_tokenizer_cached(model_tokenizer)
		if not tokenizer_success:
			print(f"[Benchmark] Failed to ensure tokenizer is cached: {model_tokenizer}")
			print(f"[Benchmark] Continuing anyway, offline mode may fail if tokenizer not cached")
		
		# Track offline mode for later use
		use_offline_mode = is_fully_cached

		# Setup endpoint URL
		if endpoint_url:
			# Direct URL provided (Docker mode)
			print(f"[Benchmark] Using direct endpoint: {endpoint_url}")
			need_cleanup = False
		else:
			# Setup port forward (Kubernetes mode)
			endpoint_url = self.setup_port_forward(service_name, namespace, 8080, local_port)
			if not endpoint_url:
				print(f"[Benchmark] Failed to setup port forward")
				return None
			need_cleanup = True

		print(f"[Benchmark] Running genai-bench for experiment {experiment_id}")
		print(f"[Benchmark] Endpoint: {endpoint_url}")
		print(f"[Benchmark] Output directory: {output_dir}")

		# Get model name first (needed for warmup)
		model_name = benchmark_config.get("model_name", "unknown")

		# Warmup phase: Send requests to trigger torch compile and CUDA graph capture
		self._warmup_service(endpoint_url, model_name)

		# Build genai-bench command
		cmd = [
			str(self.genai_bench_path),
			"benchmark",
			"--api-backend",
			"openai",
			"--api-base",
			endpoint_url,
			"--api-key",
			"dummy",  # Required but not used for local servers
			"--task",
			benchmark_config.get("task", "text-to-text"),
			"--experiment-base-dir",
			str(output_dir.parent),
			"--experiment-folder-name",
			output_dir.name,
		]

		# Add model name to command (already extracted above)
		cmd.extend(["--api-model-name", model_name])

		# Add model tokenizer (required by genai-bench)
		model_tokenizer = benchmark_config.get("model_tokenizer", "gpt2")
		cmd.extend(["--model-tokenizer", model_tokenizer])

		# Add traffic scenarios
		traffic_scenarios = benchmark_config.get("traffic_scenarios", ["D(100,100)"])
		for scenario in traffic_scenarios:
			cmd.extend(["--traffic-scenario", scenario])

		# Add concurrency levels
		num_concurrency = benchmark_config.get("num_concurrency", [1])
		for concurrency in num_concurrency:
			cmd.extend(["--num-concurrency", str(concurrency)])

		# Add iteration limits (note: different parameter names than K8s BenchmarkJob)
		max_time = benchmark_config.get("max_time_per_iteration", 10)
		max_requests = benchmark_config.get("max_requests_per_iteration", 50)
		cmd.extend(["--max-time-per-run", str(max_time), "--max-requests-per-run", str(max_requests)])

		# Add additional request params (needs to be JSON string)
		additional_params = benchmark_config.get("additional_params", {})
		if additional_params:
			import json

			params_json = json.dumps(additional_params)
			cmd.extend(["--additional-request-params", params_json])

		# Handle custom dataset from URL
		dataset_path = None
		if "dataset_url" in benchmark_config:
			from utils.dataset_manager import DatasetManager
			dm = DatasetManager()
			try:
				dataset_path = dm.get_dataset(
					url=benchmark_config["dataset_url"],
					deduplicate=benchmark_config.get("dataset_deduplicate", True),
				)
				print(f"[Benchmark] Using custom dataset: {dataset_path}")
				cmd.extend(["--dataset-path", str(dataset_path)])
				cmd.extend(["--dataset-prompt-column", "prompt"])
			except Exception as e:
				print(f"[Benchmark] Failed to load dataset from URL: {e}")
				return None

		print(f"[Benchmark] Command: {' '.join(cmd)}")

		# Setup environment with proxy settings for HuggingFace downloads
		import os
		env = os.environ.copy()

		# Check if proxy is configured in environment or use default from config
		from web.config import get_settings
		settings = get_settings()
		proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or settings.https_proxy
		if proxy_url:
			# Only set proxy env vars if proxy is configured (env vars must be strings, not None)
			env['HTTP_PROXY'] = proxy_url
			env['http_proxy'] = proxy_url
			env['HTTPS_PROXY'] = proxy_url
			env['https_proxy'] = proxy_url
			print(f"[Benchmark] Using proxy: {proxy_url}")
		else:
			print(f"[Benchmark] No proxy configured")
		env['NO_PROXY'] = 'localhost,127.0.0.1,.local'
		env['no_proxy'] = 'localhost,127.0.0.1,.local'

		# Pass through HF_TOKEN if set (for gated models like Llama)
		if 'HF_TOKEN' in os.environ:
			env['HF_TOKEN'] = os.environ['HF_TOKEN']
			print(f"[Benchmark] HF_TOKEN is set (for accessing gated models)")
		else:
			print(f"[Benchmark] HF_TOKEN not set (only public models accessible)")

		# Set HF_HUB_OFFLINE=1 if tokenizer is fully cached to avoid network access
		# This fixes the "Network is unreachable" error when HuggingFace API is not accessible
		if use_offline_mode:
			env['HF_HUB_OFFLINE'] = '1'
			print(f"[Benchmark] HF_HUB_OFFLINE=1 (tokenizer fully cached, skipping HuggingFace API)")
		else:
			print(f"[Benchmark] HuggingFace online mode (tokenizer needs metadata from API)")

		# Filter out None values from environment (subprocess requires all values to be strings)
		env = {k: v for k, v in env.items() if v is not None}

		# Setup GPU monitoring if GPU indices provided
		gpu_monitor_thread = None
		gpu_snapshots = []
		stop_gpu_monitoring = threading.Event()

		if gpu_indices:
			print(f"[GPU Monitor] GPU monitoring enabled for GPUs: {gpu_indices}")
			# Start GPU monitoring in background thread
			gpu_monitor_thread = threading.Thread(
				target=lambda: gpu_snapshots.extend(
					self._monitor_gpus_during_benchmark(
						gpu_indices=gpu_indices,
						duration_seconds=timeout + 60,  # Give extra buffer
						interval_seconds=1.0,
						stop_event=stop_gpu_monitoring
					)
				),
				daemon=True
			)
			gpu_monitor_thread.start()
		else:
			print(f"[GPU Monitor] GPU monitoring disabled (no GPU indices provided)")

		# Run benchmark
		try:
			start_time = time.time()

			# Use results directory as working directory (writable by all users)
			# This prevents permission errors when genai-bench creates log files
			work_dir = output_dir.parent
			print(f"[Benchmark] Working directory: {work_dir}")

			if self.verbose:
				# Stream output in real-time
				print(f"[Benchmark] Starting genai-bench (streaming output)...")
				process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=str(work_dir))

				stdout_lines = []
				for line in process.stdout:
					print(f"[genai-bench] {line.rstrip()}")
					stdout_lines.append(line)

				process.wait(timeout=timeout)
				result_returncode = process.returncode
				result_stdout = "".join(stdout_lines)
				result_stderr = ""
			else:
				# Capture output
				result = subprocess.run(
					cmd,
					capture_output=True,
					text=True,
					timeout=timeout,
					check=False,  # Don't raise exception on non-zero exit
					env=env,
					cwd=str(work_dir),
				)
				result_returncode = result.returncode
				result_stdout = result.stdout
				result_stderr = result.stderr

			elapsed_time = time.time() - start_time

			# Stop GPU monitoring if running
			if gpu_monitor_thread:
				stop_gpu_monitoring.set()  # Signal thread to stop
				gpu_monitor_thread.join(timeout=5)  # Wait for thread to complete
				print(f"[GPU Monitor] Stopped monitoring")

			print(f"[Benchmark] Completed in {elapsed_time:.1f}s")
			print(f"[Benchmark] Exit code: {result_returncode}")

			# Show genai-bench output for debugging (only if not in verbose mode)
			if not self.verbose:
				if result_stdout:
					print(f"[Benchmark] STDOUT:\n{result_stdout}")
				if result_stderr:
					print(f"[Benchmark] STDERR:\n{result_stderr}")

			# Even if genai-bench exits with error, try to parse partial results
			# This allows partial batch success (e.g., some concurrency levels succeed before OOM)
			if result_returncode != 0:
				print(f"[Benchmark] WARNING: genai-bench exited with non-zero code: {result_returncode}")
				print(f"[Benchmark] Will attempt to parse partial results from successful batches...")

			# Parse results from output directory (may contain partial results)
			metrics = self._parse_results(output_dir, slo_config=benchmark_config.get("slo_config"))
			if metrics:
				metrics["elapsed_time"] = elapsed_time
				metrics["benchmark_name"] = benchmark_name

				# Add GPU monitoring statistics if available
				if gpu_snapshots:
					print(f"[GPU Monitor] Processing {len(gpu_snapshots)} GPU snapshots...")
					gpu_monitor = get_gpu_monitor()
					gpu_stats = gpu_monitor.get_summary_stats(gpu_snapshots)
					if gpu_stats:
						metrics["gpu_monitoring"] = gpu_stats
						print(f"[GPU Monitor] GPU statistics added to metrics")

						# Print summary for debugging
						for gpu_idx, stats in gpu_stats.get("gpu_stats", {}).items():
							util_mean = stats["utilization"]["mean"]
							mem_mean = stats["memory_usage_percent"]["mean"]
							print(f"[GPU Monitor]   GPU {gpu_idx}: Avg Util={util_mean:.1f}%, Avg Mem={mem_mean:.1f}%")

				return metrics
			else:
				print(f"[Benchmark] No results found in {output_dir}")
				# If genai-bench failed AND no results exist, this is a complete failure
				if result_returncode != 0:
					print(f"[Benchmark] FAILED: No successful batches (exit code: {result_returncode})")
				return None

		except subprocess.TimeoutExpired:
			print(f"[Benchmark] Timeout after {timeout}s")
			return None
		except Exception as e:
			print(f"[Benchmark] Error running genai-bench: {e}")
			return None
		finally:
			# Stop GPU monitoring if still running
			if gpu_monitor_thread and gpu_monitor_thread.is_alive():
				stop_gpu_monitoring.set()
				gpu_monitor_thread.join(timeout=3)

			# Only cleanup port forward if we set it up
			if need_cleanup:
				self.cleanup_port_forward()

	def _parse_results(self, output_dir: Path, slo_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
		"""Parse benchmark results from output directory.

		Args:
		    output_dir: Directory containing benchmark results

		Returns:
		    Dict containing parsed metrics, or None if unavailable
		"""
		# Look for JSON results files in the experiment directory
		# genai-bench creates files like:
		# - experiment_metadata.json (metadata)
		# - D100_100_text-to-text_num_concurrency_1_time_9s.json (actual results per concurrency)
		result_files = list(output_dir.glob("D*.json"))  # Only get result files, not metadata
		if not result_files:
			print(f"[Benchmark] No result JSON files found in {output_dir}")
			print(f"[Benchmark] Directory contents:")
			if output_dir.exists():
				for item in output_dir.rglob("*"):
					print(f"  {item}")
			return None

		print(f"[Benchmark] Found {len(result_files)} result file(s)")

		try:
			# Parse all result files and aggregate metrics
			all_metrics = []
			for result_file in result_files:
				print(f"[Benchmark] Parsing {result_file.name}")
				try:
					with open(result_file, "r") as f:
						data = json.load(f)

					# genai-bench result structure: {"aggregated_metrics": {...}}
					if "aggregated_metrics" in data:
						all_metrics.append(data["aggregated_metrics"])
					else:
						print(f"[Benchmark] Warning: No aggregated_metrics in {result_file.name}")
				except Exception as e:
					print(f"[Benchmark] Error parsing {result_file.name}: {e}")
					continue

			# Filter batches by SLO compliance if slo_config is provided
			print(f"[DEBUG] _parse_results: slo_config type={type(slo_config)}, value={slo_config}")
			print(f"[DEBUG] _parse_results: all_metrics count={len(all_metrics) if all_metrics else 0}")
			if slo_config and all_metrics:
				print(f"[Benchmark] Filtering {len(all_metrics)} batches by SLO compliance...")
				slo_compliant_metrics = []
				slo_violated_metrics = []
				
				for batch_metrics in all_metrics:
					is_compliant, violations = check_batch_slo_compliance(batch_metrics, slo_config)
					if is_compliant:
						slo_compliant_metrics.append(batch_metrics)
					else:
						slo_violated_metrics.append(batch_metrics)
						concurrency = batch_metrics.get("num_concurrency", "?")
						print(f"[Benchmark] ✗ Batch concurrency={concurrency} violated SLO: {violations}")
				
				if slo_compliant_metrics:
					print(f"[Benchmark] ✓ {len(slo_compliant_metrics)}/{len(all_metrics)} batches passed SLO")
					# Use only SLO-compliant batches for aggregation
					all_metrics = slo_compliant_metrics
				else:
					print(f"[Benchmark] ✗ No batches passed SLO! Using all {len(all_metrics)} batches with penalties")
					# Keep all batches - let penalty system handle it

			if not all_metrics:
				print(f"[Benchmark] No valid metrics found in result files")
				return None

			# Aggregate metrics across all concurrency levels
			# Use the best (lowest latency or highest throughput) from all runs
			aggregated = {
				"num_result_files": len(all_metrics),
				"concurrency_levels": [m.get("num_concurrency") for m in all_metrics],
				"raw_results": all_metrics,  # Keep all raw results for reference
			}

			# Extract key performance metrics
			# Average across all concurrency levels
			if all_metrics:
				# E2E Latency stats (mean across all concurrencies)
				e2e_latencies = [m["stats"]["e2e_latency"]["mean"] for m in all_metrics if "stats" in m]
				if e2e_latencies:
					aggregated["mean_e2e_latency"] = sum(e2e_latencies) / len(e2e_latencies)
					aggregated["min_e2e_latency"] = min(e2e_latencies)
					aggregated["max_e2e_latency"] = max(e2e_latencies)

				# P50, P90, P99 latencies (average across concurrencies)
				p50_latencies = [m["stats"]["e2e_latency"]["p50"] for m in all_metrics if "stats" in m]
				p90_latencies = [m["stats"]["e2e_latency"]["p90"] for m in all_metrics if "stats" in m]
				p99_latencies = [m["stats"]["e2e_latency"]["p99"] for m in all_metrics if "stats" in m]
				if p50_latencies:
					aggregated["p50_e2e_latency"] = sum(p50_latencies) / len(p50_latencies)
				if p90_latencies:
					aggregated["p90_e2e_latency"] = sum(p90_latencies) / len(p90_latencies)
				if p99_latencies:
					aggregated["p99_e2e_latency"] = sum(p99_latencies) / len(p99_latencies)

				# TTFT (Time to First Token) stats
				ttft_means = [m["stats"]["ttft"]["mean"] for m in all_metrics if "stats" in m]
				if ttft_means:
					aggregated["mean_ttft"] = sum(ttft_means) / len(ttft_means)

				# TPOT (Time Per Output Token) stats
				tpot_means = [m["stats"]["tpot"]["mean"] for m in all_metrics if "stats" in m]
				if tpot_means:
					aggregated["mean_tpot"] = sum(tpot_means) / len(tpot_means)

				# Throughput (tokens/s) - calculated from SLO-compliant batches only
				output_throughputs = [m.get("mean_output_throughput_tokens_per_s", 0) for m in all_metrics]
				if output_throughputs:
					aggregated["mean_output_throughput"] = sum(output_throughputs) / len(output_throughputs)
					aggregated["max_output_throughput"] = max(output_throughputs)

				total_throughputs = [m.get("mean_total_tokens_throughput_tokens_per_s", 0) for m in all_metrics]
				if total_throughputs:
					aggregated["mean_total_throughput"] = sum(total_throughputs) / len(total_throughputs)
					aggregated["max_total_throughput"] = max(total_throughputs)

				# Request stats
				total_requests = sum(m.get("num_requests", 0) for m in all_metrics)
				total_completed = sum(m.get("num_completed_requests", 0) for m in all_metrics)
				total_errors = sum(m.get("num_error_requests", 0) for m in all_metrics)
				aggregated["total_requests"] = total_requests
				aggregated["total_completed_requests"] = total_completed
				aggregated["total_error_requests"] = total_errors
				if total_requests > 0:
					aggregated["success_rate"] = total_completed / total_requests

			return aggregated

		except Exception as e:
			print(f"[Benchmark] Error parsing results: {e}")
			import traceback

			traceback.print_exc()
			return None

	def cleanup_results(self, task_name: str, experiment_id: int) -> bool:
		"""Clean up benchmark result files.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Experiment identifier

		Returns:
		    True if cleaned up successfully
		"""
		benchmark_name = f"{task_name}-exp{experiment_id}"
		output_dir = self.results_dir / benchmark_name

		if output_dir.exists():
			import shutil

			try:
				shutil.rmtree(output_dir)
				print(f"[Benchmark] Cleaned up results: {output_dir}")
				return True
			except Exception as e:
				print(f"[Benchmark] Error cleaning up results: {e}")
				return False
		return True
