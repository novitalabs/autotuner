"""
OME Deployment Controller

Manages the lifecycle of InferenceService resources for autotuning experiments.
"""

import time
import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from jinja2 import Template
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from .base_controller import BaseModelController
from .utils import sanitize_dns_name

# Import GPU discovery utility with absolute import
# Add parent directory to path if needed
if str(Path(__file__).parent.parent) not in sys.path:
	sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.gpu_discovery import find_best_node_for_deployment


class OMEController(BaseModelController):
	"""Controller for managing OME InferenceService deployments."""

	def __init__(self, kubeconfig_path: Optional[str] = None):
		"""Initialize the OME controller.

		Args:
		    kubeconfig_path: Path to kubeconfig file. If None, uses in-cluster config.
		"""
		try:
			if kubeconfig_path:
				config.load_kube_config(config_file=kubeconfig_path)
			else:
				config.load_kube_config()
		except Exception:
			# Fallback to in-cluster config
			config.load_incluster_config()

		self.api_client = client.ApiClient()
		self.custom_api = client.CustomObjectsApi(self.api_client)
		self.core_api = client.CoreV1Api(self.api_client)

		# Load templates
		template_dir = Path(__file__).parent.parent / "templates"
		with open(template_dir / "inference_service.yaml.j2") as f:
			self.isvc_template = Template(f.read())
		with open(template_dir / "clusterbasemodel.yaml.j2") as f:
			self.cbm_template = Template(f.read())
		with open(template_dir / "clusterservingruntime.yaml.j2") as f:
			self.csr_template = Template(f.read())

	def create_namespace(self, namespace: str) -> bool:
		"""Create namespace if it doesn't exist.

		Args:
		    namespace: Namespace name

		Returns:
		    True if created or already exists
		"""
		try:
			self.core_api.read_namespace(namespace)
			print(f"Namespace '{namespace}' already exists")
			return True
		except ApiException as e:
			if e.status == 404:
				# Create namespace
				ns_body = client.V1Namespace(metadata=client.V1ObjectMeta(name=namespace))
				self.core_api.create_namespace(ns_body)
				print(f"Created namespace '{namespace}'")
				return True
			else:
				print(f"Error checking namespace: {e}")
				return False

	def deploy_inference_service(
		self,
		task_name: str,
		experiment_id: int,
		namespace: str,
		model_name: str,
		runtime_name: str,
		parameters: Dict[str, Any],
		storage: Optional[Dict[str, Any]] = None,
		enable_gpu_selection: bool = True,
	) -> Optional[str]:
		"""Deploy an InferenceService with specified parameters.

		Args:
		    task_name: Autotuning task name
		    experiment_id: Unique experiment identifier (will be converted to string internally)
		    namespace: K8s namespace
		    model_name: Model name
		    runtime_name: ServingRuntime name
		    parameters: SGLang parameters (tp_size, mem_frac, etc.)
		    storage: Optional storage configuration for PVC support
		             {
		                 'type': 'pvc',
		                 'pvc_name': 'model-storage-pvc',
		                 'pvc_subpath': 'meta/llama-3-2-1b-instruct',
		                 'mount_path': '/raid/models/meta/llama-3-2-1b-instruct'
		             }
		    enable_gpu_selection: If True, intelligently select node with idle GPUs (default: True)

		Returns:
		    InferenceService name if successful, None otherwise
		"""
		# Convert experiment_id to string for Kubernetes resource naming
		experiment_id_str = str(experiment_id)

		# Sanitize task_name to be DNS-1123 compliant
		safe_task_name = sanitize_dns_name(task_name)
		isvc_name = f"{safe_task_name}-exp{experiment_id_str}"

		# Sanitize model_name to match ClusterBaseModel naming convention
		# Convert "meta-llama/Llama-3.2-3B-Instruct" to "llama-3-2-3b-instruct"
		# Strip namespace prefix (everything before and including the slash) first
		model_basename = model_name.split('/')[-1] if '/' in model_name else model_name
		safe_model_name = sanitize_dns_name(model_basename)

		# Determine required GPUs from parameters
		required_gpus = parameters.get('tpsize', parameters.get('tp_size', parameters.get('tp-size', 1)))

		# Find best node for deployment if enabled
		selected_node = None
		if enable_gpu_selection:
			print(f"\n=== GPU Node Selection ===")
			print(f"Looking for node with {required_gpus} idle GPU(s)...")
			selected_node = find_best_node_for_deployment(required_gpus=required_gpus)
			if selected_node:
				print(f"✓ Selected node: {selected_node}")
			else:
				print("⚠ No specific node selected (will use Kubernetes scheduler)")
			print("=" * 26 + "\n")

		# Render template
		rendered = self.isvc_template.render(
			namespace=namespace,
			isvc_name=isvc_name,
			task_name=task_name,
			experiment_id=experiment_id_str,
			model_name=safe_model_name,
			runtime_name=runtime_name,
			params=parameters,
			storage=storage,
			selected_node=selected_node,  # Pass to template
		)

		# Parse YAML (contains namespace + InferenceService)
		resources = list(yaml.safe_load_all(rendered))

		# Create namespace
		self.create_namespace(namespace)

		# Create InferenceService
		try:
			isvc_resource = resources[1]  # Second resource is the InferenceService
			self.custom_api.create_namespaced_custom_object(
				group="ome.io",
				version="v1beta1",
				namespace=namespace,
				plural="inferenceservices",
				body=isvc_resource,
			)
			node_info = f" on node '{selected_node}'" if selected_node else ""
			print(f"Created InferenceService '{isvc_name}' in namespace '{namespace}'{node_info}")
			return isvc_name
		except ApiException as e:
			print(f"Error creating InferenceService: {e}")
			return None

	def wait_for_ready(self, isvc_name: str, namespace: str, timeout: int = 600, poll_interval: int = 10) -> bool:
		"""Wait for InferenceService to become ready.

		Args:
		    isvc_name: InferenceService name
		    namespace: K8s namespace
		    timeout: Maximum wait time in seconds
		    poll_interval: Polling interval in seconds

		Returns:
		    True if ready, False if timeout or error
		"""
		start_time = time.time()

		while time.time() - start_time < timeout:
			try:
				isvc = self.custom_api.get_namespaced_custom_object(
					group="ome.io",
					version="v1beta1",
					namespace=namespace,
					plural="inferenceservices",
					name=isvc_name,
				)

				# Check status conditions
				status = isvc.get("status", {})
				conditions = status.get("conditions", [])

				for condition in conditions:
					if condition.get("type") == "Ready":
						if condition.get("status") == "True":
							url = status.get("url", "N/A")
							print(f"InferenceService '{isvc_name}' is ready! URL: {url}")
							return True
						elif condition.get("status") == "False":
							reason = condition.get("reason", "Unknown")
							message = condition.get("message", "No details")
							print(f"InferenceService not ready - {reason}: {message}")

				elapsed = int(time.time() - start_time)
				print(f"Waiting for InferenceService '{isvc_name}' to be ready... ({elapsed}s)")
				time.sleep(poll_interval)

			except ApiException as e:
				print(f"Error checking InferenceService status: {e}")
				time.sleep(poll_interval)

		print(f"Timeout waiting for InferenceService '{isvc_name}' to be ready")
		return False

	def delete_inference_service(self, isvc_name: str, namespace: str) -> bool:
		"""Delete an InferenceService.

		Args:
		    isvc_name: InferenceService name
		    namespace: K8s namespace

		Returns:
		    True if deleted successfully
		"""
		try:
			self.custom_api.delete_namespaced_custom_object(
				group="ome.io",
				version="v1beta1",
				namespace=namespace,
				plural="inferenceservices",
				name=isvc_name,
			)
			print(f"Deleted InferenceService '{isvc_name}' from namespace '{namespace}'")
			return True
		except ApiException as e:
			if e.status == 404:
				print(f"InferenceService '{isvc_name}' not found (already deleted?)")
				return True
			else:
				print(f"Error deleting InferenceService: {e}")
				return False

	def get_service_url(self, isvc_name: str, namespace: str) -> Optional[str]:
		"""Get the service URL for an InferenceService.

		Args:
		    isvc_name: InferenceService name
		    namespace: K8s namespace

		Returns:
		    Service URL if available, None otherwise
		"""
		try:
			isvc = self.custom_api.get_namespaced_custom_object(
				group="ome.io",
				version="v1beta1",
				namespace=namespace,
				plural="inferenceservices",
				name=isvc_name,
			)
			return isvc.get("status", {}).get("url")
		except ApiException as e:
			print(f"Error getting service URL: {e}")
			return None

	# ClusterBaseModel Management
	def ensure_clusterbasemodel(
		self,
		name: str,
		spec: Dict[str, Any],
		labels: Optional[Dict[str, str]] = None,
		annotations: Optional[Dict[str, str]] = None,
	) -> bool:
		"""Ensure ClusterBaseModel exists, create if missing.

		Args:
		    name: ClusterBaseModel name
		    spec: ClusterBaseModel specification
		    labels: Optional labels to add
		    annotations: Optional annotations to add

		Returns:
		    True if exists or created successfully, False otherwise
		"""
		try:
			# Check if ClusterBaseModel already exists
			self.custom_api.get_cluster_custom_object(
				group="ome.io", version="v1beta1", plural="clusterbasemodels", name=name
			)
			print(f"ClusterBaseModel '{name}' already exists")
			return True
		except ApiException as e:
			if e.status == 404:
				# Create ClusterBaseModel
				return self._create_clusterbasemodel(name, spec, labels, annotations)
			else:
				print(f"Error checking ClusterBaseModel '{name}': {e}")
				return False

	def _create_clusterbasemodel(
		self,
		name: str,
		spec: Dict[str, Any],
		labels: Optional[Dict[str, str]] = None,
		annotations: Optional[Dict[str, str]] = None,
	) -> bool:
		"""Create ClusterBaseModel from spec.

		Args:
		    name: ClusterBaseModel name
		    spec: ClusterBaseModel specification
		    labels: Optional labels to add
		    annotations: Optional annotations to add

		Returns:
		    True if created successfully, False otherwise
		"""
		try:
			# Render template
			rendered = self.cbm_template.render(
				name=name, spec=spec, labels=labels or {}, annotations=annotations or {}
			)

			# Parse YAML and create resource
			resource = yaml.safe_load(rendered)
			self.custom_api.create_cluster_custom_object(
				group="ome.io", version="v1beta1", plural="clusterbasemodels", body=resource
			)
			print(f"Created ClusterBaseModel '{name}'")
			return True
		except ApiException as e:
			print(f"Error creating ClusterBaseModel '{name}': {e}")
			return False

	def list_clusterbasemodels(self) -> Optional[Dict[str, Any]]:
		"""List all ClusterBaseModels in the cluster.

		Returns:
		    List of ClusterBaseModels or None on error
		"""
		try:
			result = self.custom_api.list_cluster_custom_object(
				group="ome.io", version="v1beta1", plural="clusterbasemodels"
			)
			return result
		except ApiException as e:
			print(f"Error listing ClusterBaseModels: {e}")
			return None

	# ClusterServingRuntime Management
	def ensure_clusterservingruntime(
		self,
		name: str,
		spec: Dict[str, Any],
		labels: Optional[Dict[str, str]] = None,
		annotations: Optional[Dict[str, str]] = None,
	) -> bool:
		"""Ensure ClusterServingRuntime exists, create if missing.

		Args:
		    name: ClusterServingRuntime name
		    spec: ClusterServingRuntime specification
		    labels: Optional labels to add
		    annotations: Optional annotations to add

		Returns:
		    True if exists or created successfully, False otherwise
		"""
		try:
			# Check if ClusterServingRuntime already exists
			self.custom_api.get_cluster_custom_object(
				group="ome.io", version="v1beta1", plural="clusterservingruntimes", name=name
			)
			print(f"ClusterServingRuntime '{name}' already exists")
			return True
		except ApiException as e:
			if e.status == 404:
				# Create ClusterServingRuntime
				return self._create_clusterservingruntime(name, spec, labels, annotations)
			else:
				print(f"Error checking ClusterServingRuntime '{name}': {e}")
				return False

	def _create_clusterservingruntime(
		self,
		name: str,
		spec: Dict[str, Any],
		labels: Optional[Dict[str, str]] = None,
		annotations: Optional[Dict[str, str]] = None,
	) -> bool:
		"""Create ClusterServingRuntime from spec.

		Args:
		    name: ClusterServingRuntime name
		    spec: ClusterServingRuntime specification
		    labels: Optional labels to add
		    annotations: Optional annotations to add

		Returns:
		    True if created successfully, False otherwise
		"""
		try:
			# Render template
			rendered = self.csr_template.render(
				name=name, spec=spec, labels=labels or {}, annotations=annotations or {}
			)

			# Parse YAML and create resource
			resource = yaml.safe_load(rendered)
			self.custom_api.create_cluster_custom_object(
				group="ome.io", version="v1beta1", plural="clusterservingruntimes", body=resource
			)
			print(f"Created ClusterServingRuntime '{name}'")
			return True
		except ApiException as e:
			print(f"Error creating ClusterServingRuntime '{name}': {e}")
			return False

	def list_clusterservingruntimes(self) -> Optional[Dict[str, Any]]:
		"""List all ClusterServingRuntimes in the cluster.

		Returns:
		    List of ClusterServingRuntimes or None on error
		"""
		try:
			result = self.custom_api.list_cluster_custom_object(
				group="ome.io", version="v1beta1", plural="clusterservingruntimes"
			)
			return result
		except ApiException as e:
			print(f"Error listing ClusterServingRuntimes: {e}")
			return None
