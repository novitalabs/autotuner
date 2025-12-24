"""
Base Controller Interface

Abstract base class for model deployment controllers.
Supports multiple deployment modes (OME/Kubernetes, Docker, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from .utils import parse_parallel_config
from utils.gpu_monitor import get_gpu_monitor


class BaseModelController(ABC):
	"""Abstract base class for model deployment controllers."""

	@abstractmethod
	def deploy_inference_service(
		self,
		task_name: str,
		experiment_id: int,
		namespace: str,
		model_name: str,
		runtime_name: str,
		parameters: Dict[str, Any],
	) -> Optional[str]:
		"""Deploy a model inference service with specified parameters.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Unique experiment identifier
		    namespace: Namespace/resource group identifier
		    model_name: Model name/path
		    runtime_name: Runtime identifier (e.g., 'sglang')
		    parameters: Deployment parameters (tp_size, mem_frac, etc.)

		Returns:
		    Service identifier (name/ID) if successful, None otherwise
		"""
		pass

	def _get_parallel_config(self, parameters: Dict[str, Any]) -> Dict[str, int]:
		"""
		Get parallel configuration from parameters using shared utility.

		This is a convenience wrapper for the parse_parallel_config utility
		that can be overridden by subclasses if needed.

		Args:
		    parameters: Runtime parameters dictionary

		Returns:
		    Dictionary with normalized keys: {tp, pp, dp, cp, dcp, world_size}
		"""
		return parse_parallel_config(parameters)

	def _select_gpus_intelligent(self, num_gpus: int, min_memory_mb: int = 8000, log_prefix: str = "") -> Optional[Dict[str, Any]]:
		"""
		Intelligent GPU selection with fallback logic.

		This method provides a consistent GPU selection strategy across controllers:
		1. Try intelligent allocation with memory constraint
		2. Retry without memory constraint if needed
		3. Fall back to simple sequential allocation if all else fails

		Args:
		    num_gpus: Number of GPUs required
		    min_memory_mb: Minimum memory per GPU (default: 8000 MB)
		    log_prefix: Prefix for log messages (e.g., "[Docker]", "[Local]")

		Returns:
		    Dict with 'device_ids' (list of GPU IDs), 'gpu_model' (str), and 'gpu_info' (detailed info),
		    or None if not enough GPUs
		"""
		gpu_monitor = get_gpu_monitor()

		if not gpu_monitor.is_available():
			print(f"{log_prefix} nvidia-smi not available. Using fallback GPU allocation.")
			# Fallback: use first N GPUs
			return {
				"device_ids": [str(i) for i in range(num_gpus)],
				"gpu_model": "Unknown",
				"gpu_info": {
					"count": num_gpus,
					"indices": list(range(num_gpus)),
					"allocation_method": "fallback"
				}
			}

		# Use intelligent GPU allocation
		try:
			allocated_gpus, success = gpu_monitor.allocate_gpus(
				count=num_gpus,
				min_memory_mb=min_memory_mb
			)

			if not success or len(allocated_gpus) < num_gpus:
				print(f"{log_prefix} Could not allocate {num_gpus} GPU(s) with {min_memory_mb}MB free memory")
				print(f"{log_prefix} Available GPUs: {len(allocated_gpus)}")

				# Try without memory constraint
				print(f"{log_prefix} Retrying allocation without memory constraint...")
				allocated_gpus, success = gpu_monitor.allocate_gpus(count=num_gpus, min_memory_mb=None)

				if not success:
					print(f"{log_prefix} Failed to allocate any GPUs")
					return None

			# Get detailed GPU info for allocated devices
			snapshot = gpu_monitor.query_gpus(use_cache=False)
			if not snapshot:
				print(f"{log_prefix} Failed to query GPU information")
				return None

			gpu_details = []
			gpu_model = None
			for gpu in snapshot.gpus:
				if gpu.index in allocated_gpus:
					gpu_details.append({
						"index": gpu.index,
						"name": gpu.name,
						"uuid": gpu.uuid,
						"memory_total_mb": gpu.memory_total_mb,
						"memory_free_mb": gpu.memory_free_mb,
						"memory_usage_percent": gpu.memory_usage_percent,
						"utilization_percent": gpu.utilization_percent,
						"temperature_c": gpu.temperature_c,
						"power_draw_w": gpu.power_draw_w,
						"availability_score": gpu.score
					})
					if gpu_model is None:
						gpu_model = gpu.name

			print(f"{log_prefix} Selected GPUs: {allocated_gpus}")
			print(f"{log_prefix} GPU Model: {gpu_model}")
			for detail in gpu_details:
				print(f"{log_prefix}   GPU {detail['index']}: {detail['memory_free_mb']}/{detail['memory_total_mb']}MB free, "
					  f"{detail['utilization_percent']}% utilized, Score: {detail['availability_score']:.2f}")

			return {
				"device_ids": [str(idx) for idx in allocated_gpus],
				"gpu_model": gpu_model or "Unknown",
				"gpu_info": {
					"count": len(allocated_gpus),
					"indices": allocated_gpus,
					"allocation_method": "intelligent",
					"details": gpu_details,
					"allocated_at": snapshot.timestamp.isoformat()
				}
			}

		except Exception as e:
			print(f"{log_prefix} Error in intelligent GPU selection: {e}")
			print(f"{log_prefix} Falling back to simple allocation")

			# Final fallback: simple first-N allocation
			return {
				"device_ids": [str(i) for i in range(num_gpus)],
				"gpu_model": "Unknown",
				"gpu_info": {
					"count": num_gpus,
					"indices": list(range(num_gpus)),
					"allocation_method": "fallback_error"
				}
			}

	@abstractmethod
	def wait_for_ready(self, service_id: str, namespace: str, timeout: int = 600, poll_interval: int = 10) -> bool:
		"""Wait for the inference service to become ready.

		Args:
		    service_id: Service identifier returned by deploy_inference_service
		    namespace: Namespace/resource group identifier
		    timeout: Maximum wait time in seconds
		    poll_interval: Polling interval in seconds

		Returns:
		    True if service is ready, False if timeout or error
		"""
		pass

	@abstractmethod
	def delete_inference_service(self, service_id: str, namespace: str) -> bool:
		"""Delete an inference service.

		Args:
		    service_id: Service identifier
		    namespace: Namespace/resource group identifier

		Returns:
		    True if deleted successfully
		"""
		pass

	@abstractmethod
	def get_service_url(self, service_id: str, namespace: str) -> Optional[str]:
		"""Get the service URL/endpoint for the inference service.

		Args:
		    service_id: Service identifier
		    namespace: Namespace/resource group identifier

		Returns:
		    Service URL/endpoint if available, None otherwise
		"""
		pass
