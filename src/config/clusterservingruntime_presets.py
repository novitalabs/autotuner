"""
ClusterServingRuntime preset configurations.

This module defines presets for common inference runtimes (SGLang, vLLM, TensorRT-LLM)
that can be auto-deployed when creating autotuning tasks in OME mode.
"""

from typing import Dict, Any, List


# Preset definitions for ClusterServingRuntime
CLUSTERSERVINGRUNTIME_PRESETS: Dict[str, Dict[str, Any]] = {
	# SGLang Runtimes
	"sglang-llama-small": {
		"name": "sglang-llama-small",
		"display_name": "SGLang - Llama Small (1B-3B)",
		"description": "SGLang runtime optimized for small Llama models (1B-3B parameters)",
		"runtime_type": "sglang",
		"model_architecture": "LlamaForCausalLM",
		"model_size_min": "500M",
		"model_size_max": "4B",
		"spec": {
			"disabled": False,
			"supportedModelFormats": [
				{
					"modelFramework": {"name": "transformers", "version": "4.45.0.dev0"},
					"modelFormat": {"name": "safetensors", "version": "1.0.0"},
					"modelArchitecture": "LlamaForCausalLM",
					"autoSelect": False,
					"priority": 1,
				}
			],
			"protocolVersions": ["openAI"],
			"modelSizeRange": {"min": "500M", "max": "4B"},
			"engineConfig": {
				"annotations": {
					"prometheus.io/scrape": "true",
					"prometheus.io/port": "8080",
					"prometheus.io/path": "/metrics",
				},
				"labels": {"logging-forward": "enabled"},
				"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
				"volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
				"runner": {
					"name": "ome-container",
					"image": "docker.io/lmsysorg/sglang:v0.4.8.post1-cu126",
					"ports": [{"containerPort": 8080, "name": "http1", "protocol": "TCP"}],
					"command": ["python3", "-m", "sglang.launch_server"],
					"args": [
						"--host=0.0.0.0",
						"--port=8080",
						"--model-path=$(MODEL_PATH)",
						"--tp-size=1",
						"--mem-frac=0.9",
					],
					"volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
					"resources": {
						"requests": {"cpu": 10, "memory": "30Gi", "nvidia.com/gpu": 1},
						"limits": {"cpu": 10, "memory": "30Gi", "nvidia.com/gpu": 1},
					},
					"readinessProbe": {
						"httpGet": {"path": "/health_generate", "port": 8080},
						"failureThreshold": 3,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 200,
					},
					"livenessProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 5,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 60,
					},
					"startupProbe": {
						"httpGet": {"path": "/health_generate", "port": 8080},
						"failureThreshold": 150,
						"successThreshold": 1,
						"periodSeconds": 6,
						"initialDelaySeconds": 60,
						"timeoutSeconds": 30,
					},
				},
			},
		},
	},
	"sglang-llama-large": {
		"name": "sglang-llama-large",
		"display_name": "SGLang - Llama Large (70B+)",
		"description": "SGLang runtime optimized for large Llama models (70B+ parameters, multi-GPU)",
		"runtime_type": "sglang",
		"model_architecture": "LlamaForCausalLM",
		"model_size_min": "50B",
		"model_size_max": "100B",
		"spec": {
			"disabled": False,
			"supportedModelFormats": [
				{
					"modelFramework": {"name": "transformers", "version": "4.45.0.dev0"},
					"modelFormat": {"name": "safetensors", "version": "1.0.0"},
					"modelArchitecture": "LlamaForCausalLM",
					"autoSelect": False,
					"priority": 1,
				}
			],
			"protocolVersions": ["openAI"],
			"modelSizeRange": {"min": "50B", "max": "100B"},
			"engineConfig": {
				"annotations": {
					"prometheus.io/scrape": "true",
					"prometheus.io/port": "8080",
					"prometheus.io/path": "/metrics",
				},
				"labels": {"logging-forward": "enabled"},
				"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
				"volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
				"runner": {
					"name": "ome-container",
					"image": "docker.io/lmsysorg/sglang:v0.4.8.post1-cu126",
					"ports": [{"containerPort": 8080, "name": "http1", "protocol": "TCP"}],
					"command": ["python3", "-m", "sglang.launch_server"],
					"args": [
						"--host=0.0.0.0",
						"--port=8080",
						"--model-path=$(MODEL_PATH)",
						"--tp-size=4",
						"--mem-frac=0.85",
					],
					"volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
					"resources": {
						"requests": {"cpu": 40, "memory": "200Gi", "nvidia.com/gpu": 4},
						"limits": {"cpu": 40, "memory": "200Gi", "nvidia.com/gpu": 4},
					},
					"readinessProbe": {
						"httpGet": {"path": "/health_generate", "port": 8080},
						"failureThreshold": 3,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 200,
					},
					"livenessProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 5,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 60,
					},
					"startupProbe": {
						"httpGet": {"path": "/health_generate", "port": 8080},
						"failureThreshold": 150,
						"successThreshold": 1,
						"periodSeconds": 6,
						"initialDelaySeconds": 60,
						"timeoutSeconds": 30,
					},
				},
			},
		},
	},
	# vLLM Runtimes
	"vllm-llama-small": {
		"name": "vllm-llama-small",
		"display_name": "vLLM - Llama Small (1B-3B)",
		"description": "vLLM runtime optimized for small Llama models (1B-3B parameters)",
		"runtime_type": "vllm",
		"model_architecture": "LlamaForCausalLM",
		"model_size_min": "500M",
		"model_size_max": "4B",
		"spec": {
			"disabled": False,
			"supportedModelFormats": [
				{
					"modelFramework": {"name": "transformers", "version": "4.45.0.dev0"},
					"modelFormat": {"name": "safetensors", "version": "1.0.0"},
					"modelArchitecture": "LlamaForCausalLM",
					"autoSelect": True,
					"priority": 1,
					"version": "1.0.0",
				}
			],
			"protocolVersions": ["openAI"],
			"modelSizeRange": {"min": "500M", "max": "4B"},
			"engineConfig": {
				"annotations": {
					"prometheus.io/scrape": "true",
					"prometheus.io/port": "8080",
					"prometheus.io/path": "/metrics",
				},
				"labels": {"logging-forward": "enabled"},
				"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
				"volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
				"runner": {
					"name": "ome-container",
					"image": "docker.io/vllm/vllm-openai:v0.9.0.1",
					"ports": [{"containerPort": 8080, "name": "http1", "protocol": "TCP"}],
					"command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
					"args": [
						"--port=8080",
						"--model=$(MODEL_PATH)",
						"--middleware=vllm.entrypoints.openai.middleware.log_opc_header",
						"--max-log-len=0",
						"--served-model-name=vllm-model",
						"--tensor-parallel-size=1",
						"--preemption-mode=swap",
						"--max-model-len=131072",
						"--gpu-memory-utilization=0.90",
					],
					"volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
					"resources": {
						"requests": {"cpu": 10, "memory": "30Gi", "nvidia.com/gpu": 1},
						"limits": {"cpu": 10, "memory": "30Gi", "nvidia.com/gpu": 1},
					},
					"readinessProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 3,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 200,
					},
					"livenessProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 5,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 60,
					},
					"startupProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 150,
						"successThreshold": 1,
						"periodSeconds": 6,
						"initialDelaySeconds": 60,
						"timeoutSeconds": 30,
					},
				},
			},
		},
	},
	"vllm-llama-large": {
		"name": "vllm-llama-large",
		"display_name": "vLLM - Llama Large (70B+)",
		"description": "vLLM runtime optimized for large Llama models (70B+ parameters, multi-GPU)",
		"runtime_type": "vllm",
		"model_architecture": "LlamaForCausalLM",
		"model_size_min": "50B",
		"model_size_max": "100B",
		"spec": {
			"disabled": False,
			"supportedModelFormats": [
				{
					"modelFramework": {"name": "transformers", "version": "4.45.0.dev0"},
					"modelFormat": {"name": "safetensors", "version": "1.0.0"},
					"modelArchitecture": "LlamaForCausalLM",
					"autoSelect": True,
					"priority": 1,
					"version": "1.0.0",
				}
			],
			"protocolVersions": ["openAI"],
			"modelSizeRange": {"min": "50B", "max": "100B"},
			"engineConfig": {
				"annotations": {
					"prometheus.io/scrape": "true",
					"prometheus.io/port": "8080",
					"prometheus.io/path": "/metrics",
				},
				"labels": {"logging-forward": "enabled"},
				"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
				"volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
				"runner": {
					"name": "ome-container",
					"image": "docker.io/vllm/vllm-openai:v0.9.0.1",
					"ports": [{"containerPort": 8080, "name": "http1", "protocol": "TCP"}],
					"command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
					"args": [
						"--port=8080",
						"--model=$(MODEL_PATH)",
						"--middleware=vllm.entrypoints.openai.middleware.log_opc_header",
						"--max-log-len=0",
						"--served-model-name=vllm-model",
						"--tensor-parallel-size=4",
						"--preemption-mode=swap",
						"--max-model-len=131072",
						"--gpu-memory-utilization=0.85",
					],
					"volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
					"resources": {
						"requests": {"cpu": 40, "memory": "200Gi", "nvidia.com/gpu": 4},
						"limits": {"cpu": 40, "memory": "200Gi", "nvidia.com/gpu": 4},
					},
					"readinessProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 3,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 200,
					},
					"livenessProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 5,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 60,
					},
					"startupProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 150,
						"successThreshold": 1,
						"periodSeconds": 6,
						"initialDelaySeconds": 60,
						"timeoutSeconds": 30,
					},
				},
			},
		},
	},
	# MoE Runtimes
	"sglang-mixtral-moe": {
		"name": "sglang-mixtral-moe",
		"display_name": "SGLang - Mixtral MoE (8x7B)",
		"description": "SGLang runtime optimized for Mixtral mixture-of-experts models",
		"runtime_type": "sglang",
		"model_architecture": "MixtralForCausalLM",
		"model_size_min": "40B",
		"model_size_max": "60B",
		"spec": {
			"disabled": False,
			"supportedModelFormats": [
				{
					"modelFramework": {"name": "transformers", "version": "4.45.0.dev0"},
					"modelFormat": {"name": "safetensors", "version": "1.0.0"},
					"modelArchitecture": "MixtralForCausalLM",
					"autoSelect": False,
					"priority": 1,
				}
			],
			"protocolVersions": ["openAI"],
			"modelSizeRange": {"min": "40B", "max": "60B"},
			"engineConfig": {
				"annotations": {
					"prometheus.io/scrape": "true",
					"prometheus.io/port": "8080",
					"prometheus.io/path": "/metrics",
				},
				"labels": {"logging-forward": "enabled"},
				"tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"}],
				"volumes": [{"name": "dshm", "emptyDir": {"medium": "Memory"}}],
				"runner": {
					"name": "ome-container",
					"image": "docker.io/lmsysorg/sglang:v0.4.8.post1-cu126",
					"ports": [{"containerPort": 8080, "name": "http1", "protocol": "TCP"}],
					"command": ["python3", "-m", "sglang.launch_server"],
					"args": [
						"--host=0.0.0.0",
						"--port=8080",
						"--model-path=$(MODEL_PATH)",
						"--tp-size=2",
						"--mem-frac=0.85",
					],
					"volumeMounts": [{"mountPath": "/dev/shm", "name": "dshm"}],
					"resources": {
						"requests": {"cpu": 20, "memory": "100Gi", "nvidia.com/gpu": 2},
						"limits": {"cpu": 20, "memory": "100Gi", "nvidia.com/gpu": 2},
					},
					"readinessProbe": {
						"httpGet": {"path": "/health_generate", "port": 8080},
						"failureThreshold": 3,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 200,
					},
					"livenessProbe": {
						"httpGet": {"path": "/health", "port": 8080},
						"failureThreshold": 5,
						"successThreshold": 1,
						"periodSeconds": 60,
						"timeoutSeconds": 60,
					},
					"startupProbe": {
						"httpGet": {"path": "/health_generate", "port": 8080},
						"failureThreshold": 150,
						"successThreshold": 1,
						"periodSeconds": 6,
						"initialDelaySeconds": 60,
						"timeoutSeconds": 30,
					},
				},
			},
		},
	},
}


def get_preset(preset_name: str) -> Dict[str, Any]:
	"""
	Get a ClusterServingRuntime preset by name.

	Args:
	    preset_name: Name of the preset

	Returns:
	    Preset configuration dictionary

	Raises:
	    ValueError: If preset not found
	"""
	if preset_name not in CLUSTERSERVINGRUNTIME_PRESETS:
		raise ValueError(
			f"ClusterServingRuntime preset '{preset_name}' not found. Available presets: {list(CLUSTERSERVINGRUNTIME_PRESETS.keys())}"
		)

	return CLUSTERSERVINGRUNTIME_PRESETS[preset_name].copy()


def list_presets() -> List[Dict[str, Any]]:
	"""
	List all available ClusterServingRuntime presets.

	Returns:
	    List of preset configurations
	"""
	return [
		{
			"name": name,
			"display_name": preset["display_name"],
			"description": preset["description"],
			"runtime_type": preset["runtime_type"],
			"model_architecture": preset["model_architecture"],
		}
		for name, preset in CLUSTERSERVINGRUNTIME_PRESETS.items()
	]


def get_presets_by_runtime(runtime_type: str) -> List[Dict[str, Any]]:
	"""
	Get presets filtered by runtime type (sglang, vllm, tensorrt-llm).

	Args:
	    runtime_type: Runtime type to filter by

	Returns:
	    List of matching preset configurations
	"""
	return [
		{
			"name": name,
			"display_name": preset["display_name"],
			"description": preset["description"],
			"model_architecture": preset["model_architecture"],
		}
		for name, preset in CLUSTERSERVINGRUNTIME_PRESETS.items()
		if preset["runtime_type"] == runtime_type
	]


def validate_custom_config(config: Dict[str, Any]) -> bool:
	"""
	Validate a custom ClusterServingRuntime configuration.

	Args:
	    config: Custom configuration dictionary

	Returns:
	    True if valid, raises ValueError if invalid

	Raises:
	    ValueError: If configuration is invalid
	"""
	required_fields = ["supportedModelFormats", "protocolVersions", "engineConfig"]
	for field in required_fields:
		if field not in config:
			raise ValueError(f"ClusterServingRuntime config missing required field: {field}")

	engine_config = config["engineConfig"]
	if "runner" not in engine_config:
		raise ValueError("ClusterServingRuntime engineConfig must contain 'runner'")

	runner = engine_config["runner"]
	required_runner_fields = ["name", "image", "resources"]
	for field in required_runner_fields:
		if field not in runner:
			raise ValueError(f"ClusterServingRuntime runner missing required field: {field}")

	return True


def merge_preset_with_overrides(preset_name: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Merge a preset with custom overrides.

	Args:
	    preset_name: Name of the preset to use as base
	    overrides: Dictionary of fields to override

	Returns:
	    Merged configuration
	"""
	preset = get_preset(preset_name)
	spec = preset["spec"].copy()

	# Deep merge overrides
	def deep_merge(base: Dict, override: Dict) -> Dict:
		result = base.copy()
		for key, value in override.items():
			if isinstance(value, dict) and key in result and isinstance(result[key], dict):
				result[key] = deep_merge(result[key], value)
			else:
				result[key] = value
		return result

	spec = deep_merge(spec, overrides)

	return {
		"name": preset["name"],
		"display_name": preset["display_name"],
		"description": preset["description"],
		"runtime_type": preset["runtime_type"],
		"model_architecture": preset["model_architecture"],
		"spec": spec,
	}
