"""
Local Subprocess Controller

Manages the lifecycle of model inference services using local subprocess.
No Docker or Kubernetes required - direct process management.
"""

import os
import re
import signal
import socket
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List

from controllers.base_controller import BaseModelController
from utils.gpu_monitor import get_gpu_monitor


def sanitize_service_name(name: str) -> str:
	"""
	Sanitize a name for service identification.

	Args:
	    name: The name to sanitize

	Returns:
	    Sanitized name
	"""
	name = name.lower()
	name = re.sub(r'[^a-z0-9-._]', '-', name)
	name = re.sub(r'^[^a-z0-9]+', '', name)
	name = re.sub(r'[^a-z0-9]+$', '', name)
	name = re.sub(r'-+', '-', name)
	return name


class LocalController(BaseModelController):
	"""Controller for managing local subprocess deployments."""

	def __init__(
		self,
		model_base_path: str = "/mnt/data/models",
		python_path: str = "python3",
		http_proxy: str = "",
		https_proxy: str = "",
		no_proxy: str = "",
		hf_token: str = "",
	):
		"""Initialize the local subprocess controller.

		Args:
		    model_base_path: Base path where models are stored
		    python_path: Path to python executable with sglang installed
		    http_proxy: HTTP proxy URL (optional)
		    https_proxy: HTTPS proxy URL (optional)
		    no_proxy: Comma-separated list of hosts to bypass proxy (optional)
		    hf_token: HuggingFace access token for gated models (optional)
		"""
		self.model_base_path = Path(model_base_path)
		self.python_path = python_path
		self.processes: Dict[str, Dict[str, Any]] = {}

		# Store proxy settings
		self.http_proxy = http_proxy
		self.https_proxy = https_proxy
		self.no_proxy = no_proxy
		self.hf_token = hf_token

		# Log directory
		self.log_dir = Path.home() / ".local/share/inference-autotuner/logs"
		self.log_dir.mkdir(parents=True, exist_ok=True)

		print(f"[Local] Initialized LocalController")
		print(f"[Local] Python path: {self.python_path}")
		print(f"[Local] Model base path: {self.model_base_path}")

		if self.http_proxy or self.https_proxy:
			print(f"[Local] Proxy configured - HTTP: {self.http_proxy or 'None'}, HTTPS: {self.https_proxy or 'None'}")

	def deploy_inference_service(
		self,
		task_name: str,
		experiment_id: int,
		namespace: str,
		model_name: str,
		runtime_name: str,
		parameters: Dict[str, Any],
		image_tag: Optional[str] = None,
	) -> Optional[str]:
		"""Deploy a model inference service using local subprocess.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Unique experiment identifier
		    namespace: Namespace identifier (used for naming)
		    model_name: Model name (HuggingFace model ID or local path)
		    runtime_name: Runtime identifier (e.g., 'sglang', 'vllm')
		    parameters: Runtime parameters (tp_size, mem_frac, etc.)
		    image_tag: Unused in local mode, kept for compatibility

		Returns:
		    Service ID if successful, None otherwise
		"""
		# Sanitize names
		safe_task_name = sanitize_service_name(task_name)
		safe_namespace = sanitize_service_name(namespace)
		service_id = f"{safe_namespace}-{safe_task_name}-exp{experiment_id}"

		# Determine model path
		model_path = self._resolve_model_path(model_name)
		if model_path is None:
			print(f"[Local] ERROR: Could not resolve model path for '{model_name}'")
			return None

		# Find available port
		host_port = self._find_available_port(8000, 8100)
		if not host_port:
			print(f"[Local] Could not find available port in range 8000-8100")
			return None

		# Calculate GPU requirements
		tp = parameters.get("tensor-parallel-size", parameters.get("tp-size", parameters.get("tp_size", 1)))
		pp = parameters.get("pipeline-parallel-size", parameters.get("pp-size", parameters.get("pp_size", 1)))
		dp = parameters.get("data-parallel-size", parameters.get("dp-size", parameters.get("dp_size", 1)))

		tp = int(tp) if isinstance(tp, (int, float, str)) else 1
		pp = int(pp) if isinstance(pp, (int, float, str)) else 1
		dp = int(dp) if isinstance(dp, (int, float, str)) else 1

		num_gpus = tp * pp * dp
		print(f"[Local] Parallel configuration: TP={tp}, PP={pp}, DP={dp}, Total GPUs={num_gpus}")

		# Select GPUs
		gpu_info = self._select_gpus(num_gpus)
		if not gpu_info:
			print(f"[Local] Failed to allocate {num_gpus} GPU(s)")
			return None

		gpu_devices = gpu_info["device_ids"]
		gpu_model = gpu_info["gpu_model"]

		print(f"[Local] Deploying service '{service_id}'")
		print(f"[Local] Model: {model_path}")
		print(f"[Local] GPUs: {gpu_devices} (Model: {gpu_model})")
		print(f"[Local] Port: {host_port}")

		# Build command
		cmd = self._build_command(runtime_name, model_path, host_port, parameters)
		if not cmd:
			print(f"[Local] Unsupported runtime: {runtime_name}")
			return None

		print(f"[Local] Command: {self.python_path} {' '.join(cmd)}")

		# Prepare environment
		env = os.environ.copy()
		env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)

		if self.http_proxy:
			env["HTTP_PROXY"] = self.http_proxy
			env["http_proxy"] = self.http_proxy
		if self.https_proxy:
			env["HTTPS_PROXY"] = self.https_proxy
			env["https_proxy"] = self.https_proxy
		if self.no_proxy:
			env["NO_PROXY"] = self.no_proxy
			env["no_proxy"] = self.no_proxy
		if self.hf_token:
			env["HF_TOKEN"] = self.hf_token
			env["HUGGING_FACE_HUB_TOKEN"] = self.hf_token

		# Log file
		log_file_path = self.log_dir / f"{service_id}.log"

		try:
			# Start subprocess
			log_file = open(log_file_path, 'w')
			proc = subprocess.Popen(
				[self.python_path] + cmd,
				stdout=log_file,
				stderr=subprocess.STDOUT,
				env=env,
				start_new_session=True,  # Create new process group
			)

			# Store process info
			self.processes[service_id] = {
				"process": proc,
				"port": host_port,
				"gpu_devices": gpu_devices,
				"gpu_model": gpu_model,
				"world_size": num_gpus,
				"log_file": log_file,
				"log_file_path": str(log_file_path),
			}

			print(f"[Local] Service '{service_id}' started (PID: {proc.pid})")
			print(f"[Local] Log file: {log_file_path}")
			print(f"[Local] Service URL: http://localhost:{host_port}")

			return service_id

		except Exception as e:
			print(f"[Local] Error starting process: {e}")
			return None

	def wait_for_ready(self, service_id: str, namespace: str, timeout: int = 600, poll_interval: int = 5) -> bool:
		"""Wait for the local subprocess service to become ready.

		Args:
		    service_id: Service identifier
		    namespace: Namespace identifier
		    timeout: Maximum wait time in seconds
		    poll_interval: Polling interval in seconds

		Returns:
		    True if service is ready, False if timeout or error
		"""
		if service_id not in self.processes:
			print(f"[Local] Service '{service_id}' not found")
			return False

		proc_info = self.processes[service_id]
		proc = proc_info["process"]
		host_port = proc_info["port"]

		health_url = f"http://localhost:{host_port}/health"
		models_url = f"http://localhost:{host_port}/v1/models"

		start_time = time.time()
		print(f"[Local] Waiting for service to be ready...")

		while time.time() - start_time < timeout:
			# Check if process is still running
			poll_result = proc.poll()
			if poll_result is not None:
				# Process has exited
				print(f"[Local] Process exited with code: {poll_result}")
				self._print_logs(service_id, tail=100)
				return False

			# Check health endpoints
			try:
				health_response = requests.get(health_url, timeout=5)
				if health_response.status_code == 200:
					print(f"[Local] Service is ready! (via /health) URL: http://localhost:{host_port}")
					return True
			except requests.RequestException:
				pass

			try:
				models_response = requests.get(models_url, timeout=5)
				if models_response.status_code == 200:
					print(f"[Local] Service is ready! (via /v1/models) URL: http://localhost:{host_port}")
					return True
			except requests.RequestException:
				pass

			elapsed = int(time.time() - start_time)
			print(f"[Local] Waiting for service... ({elapsed}s)")
			time.sleep(poll_interval)

		# Timeout
		print(f"[Local] Timeout waiting for service '{service_id}' after {timeout}s")
		self._print_logs(service_id, tail=100)
		return False

	def delete_inference_service(self, service_id: str, namespace: str) -> bool:
		"""Delete a local subprocess service.

		Args:
		    service_id: Service identifier
		    namespace: Namespace identifier

		Returns:
		    True if deleted successfully
		"""
		if service_id not in self.processes:
			print(f"[Local] Service '{service_id}' not found (already deleted?)")
			return True

		try:
			proc_info = self.processes[service_id]
			proc = proc_info["process"]
			log_file = proc_info.get("log_file")

			print(f"[Local] Stopping service '{service_id}' (PID: {proc.pid})...")

			# Try graceful termination first
			try:
				os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
				proc.wait(timeout=10)
				print(f"[Local] Service terminated gracefully")
			except subprocess.TimeoutExpired:
				# Force kill
				print(f"[Local] Forcing termination...")
				os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
				proc.wait(timeout=5)
			except ProcessLookupError:
				print(f"[Local] Process already terminated")

			# Close log file
			if log_file:
				try:
					log_file.close()
				except:
					pass

			del self.processes[service_id]
			print(f"[Local] Service '{service_id}' deleted")
			return True

		except Exception as e:
			print(f"[Local] Error deleting service: {e}")
			# Clean up from tracking anyway
			if service_id in self.processes:
				del self.processes[service_id]
			return False

	def get_service_url(self, service_id: str, namespace: str) -> Optional[str]:
		"""Get the service URL for a local subprocess.

		Args:
		    service_id: Service identifier
		    namespace: Namespace identifier

		Returns:
		    Service URL if available, None otherwise
		"""
		if service_id not in self.processes:
			return None

		host_port = self.processes[service_id]["port"]
		return f"http://localhost:{host_port}"

	def get_container_logs(self, service_id: str, namespace: str, tail: int = 1000) -> Optional[str]:
		"""Get logs from a local subprocess.

		Args:
		    service_id: Service identifier
		    namespace: Namespace identifier
		    tail: Number of lines to retrieve

		Returns:
		    Log content as string, None if not found
		"""
		if service_id not in self.processes:
			return None

		log_file_path = self.processes[service_id].get("log_file_path")
		if not log_file_path or not Path(log_file_path).exists():
			return None

		try:
			with open(log_file_path, 'r') as f:
				lines = f.readlines()
				if tail > 0:
					lines = lines[-tail:]
				return ''.join(lines)
		except Exception as e:
			print(f"[Local] Error reading logs: {e}")
			return None

	def get_gpu_info(self, service_id: str, namespace: str) -> Optional[Dict[str, Any]]:
		"""Get GPU information for a deployed service.

		Args:
		    service_id: Service identifier
		    namespace: Namespace identifier

		Returns:
		    Dict with GPU info, or None if not found
		"""
		if service_id not in self.processes:
			return None

		proc_info = self.processes[service_id]
		return {
			"model": proc_info.get("gpu_model", "Unknown"),
			"count": len(proc_info.get("gpu_devices", [])),
			"device_ids": proc_info.get("gpu_devices", []),
			"world_size": proc_info.get("world_size", 1)
		}

	def _resolve_model_path(self, model_name: str) -> Optional[str]:
		"""Resolve model name to actual path.

		Args:
		    model_name: Model name or path

		Returns:
		    Resolved model path or HuggingFace ID
		"""
		# If it's an absolute path
		if model_name.startswith("/"):
			if Path(model_name).exists():
				return model_name
			return None

		# If it looks like a HuggingFace ID (contains /)
		if "/" in model_name and not model_name.startswith("."):
			# Return as-is, let sglang download it
			print(f"[Local] Using HuggingFace model ID: {model_name}")
			return model_name

		# Try as relative to model_base_path
		local_path = self.model_base_path / model_name
		if local_path.exists():
			print(f"[Local] Using local model at {local_path}")
			return str(local_path)

		# Not found locally, return as-is (might be a HuggingFace model)
		print(f"[Local] Model '{model_name}' not found locally, treating as HuggingFace ID")
		return model_name

	def _build_command(self, runtime_name: str, model_path: str, port: int, parameters: Dict[str, Any]) -> Optional[List[str]]:
		"""Build command line for the inference server.

		Args:
		    runtime_name: Runtime name (sglang, vllm)
		    model_path: Path to model
		    port: Port to listen on
		    parameters: Runtime parameters

		Returns:
		    Command list or None if unsupported runtime
		"""
		runtime_lower = runtime_name.lower()

		if "sglang" in runtime_lower:
			cmd = [
				"-m", "sglang.launch_server",
				"--model-path", model_path,
				"--host", "0.0.0.0",
				"--port", str(port),
			]
		elif "vllm" in runtime_lower:
			cmd = [
				"-m", "vllm.entrypoints.openai.api_server",
				"--model", model_path,
				"--host", "0.0.0.0",
				"--port", str(port),
			]
		else:
			return None

		# Add parameters
		for param_name, param_value in parameters.items():
			# Skip internal parameters
			if param_name.startswith("__"):
				continue

			# Convert to CLI format
			if not param_name.startswith("--"):
				cli_param = f"--{param_name}"
			else:
				cli_param = param_name

			# Handle boolean parameters
			if isinstance(param_value, bool):
				if param_value:
					cmd.append(cli_param)
			else:
				cmd.extend([cli_param, str(param_value)])

		return cmd

	def _select_gpus(self, num_gpus: int) -> Optional[Dict[str, Any]]:
		"""Select GPU devices using intelligent allocation.

		Args:
		    num_gpus: Number of GPUs required

		Returns:
		    Dict with device_ids and gpu_model, or None if not enough GPUs
		"""
		gpu_monitor = get_gpu_monitor()

		if not gpu_monitor.is_available():
			print("[Local] nvidia-smi not available. Using fallback GPU allocation.")
			return {
				"device_ids": [str(i) for i in range(num_gpus)],
				"gpu_model": "Unknown",
			}

		try:
			min_memory_mb = 8000  # Minimum 8GB free per GPU
			allocated_gpus, success = gpu_monitor.allocate_gpus(
				count=num_gpus,
				min_memory_mb=min_memory_mb
			)

			if not success or len(allocated_gpus) < num_gpus:
				print(f"[Local] Retrying allocation without memory constraint...")
				allocated_gpus, success = gpu_monitor.allocate_gpus(count=num_gpus, min_memory_mb=None)

				if not success:
					return None

			# Get GPU info
			snapshot = gpu_monitor.query_gpus(use_cache=False)
			gpu_model = None
			if snapshot:
				for gpu in snapshot.gpus:
					if gpu.index in allocated_gpus:
						gpu_model = gpu.name
						break

			print(f"[Local] Selected GPUs: {allocated_gpus}")
			return {
				"device_ids": [str(idx) for idx in allocated_gpus],
				"gpu_model": gpu_model or "Unknown",
			}

		except Exception as e:
			print(f"[Local] Error in GPU selection: {e}, using fallback")
			return {
				"device_ids": [str(i) for i in range(num_gpus)],
				"gpu_model": "Unknown",
			}

	def _find_available_port(self, start_port: int, end_port: int) -> Optional[int]:
		"""Find an available port in the specified range.

		Args:
		    start_port: Start of port range
		    end_port: End of port range

		Returns:
		    Available port number, or None if no ports available
		"""
		for port in range(start_port, end_port + 1):
			try:
				with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
					s.bind(("", port))
					return port
			except OSError:
				continue
		return None

	def _print_logs(self, service_id: str, tail: int = 50):
		"""Print logs for debugging.

		Args:
		    service_id: Service identifier
		    tail: Number of lines to print
		"""
		logs = self.get_container_logs(service_id, "", tail=tail)
		if logs:
			print(f"[Local] === Last {tail} lines of logs ===")
			print(logs)
			print(f"[Local] === End of logs ===")
