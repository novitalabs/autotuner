"""
ClusterBaseModel preset configurations.

This module defines presets for common AI models that can be auto-deployed
when creating autotuning tasks in OME mode.
"""

from typing import Dict, Any, List


# Preset definitions for ClusterBaseModel
CLUSTERBASEMODEL_PRESETS: Dict[str, Dict[str, Any]] = {
	"llama-3-2-1b-instruct": {
		"name": "llama-3-2-1b-instruct",
		"display_name": "Llama 3.2 1B Instruct",
		"description": "Meta's Llama 3.2 1B Instruct model - small, efficient model for testing and development",
		"spec": {
			"displayName": "meta.llama-3.2-1b-instruct",
			"vendor": "meta",
			"disabled": False,
			"version": "1.0.0",
			"modelFormat": {"name": "safetensors", "operator": "Equal", "version": "1.0.0", "weight": 1},
			"modelArchitecture": "LlamaForCausalLM",
			"modelFramework": {"name": "transformers", "operator": "Equal", "version": "4.45.0.dev0"},
			"storage": {
				"storageUri": "hf://meta-llama/Llama-3.2-1B-Instruct",
				"path": "/raid/models/meta/llama-3-2-1b-instruct",
				"key": "hf-token",
			},
		},
	},
	"llama-3-2-3b-instruct": {
		"name": "llama-3-2-3b-instruct",
		"display_name": "Llama 3.2 3B Instruct",
		"description": "Meta's Llama 3.2 3B Instruct model - balanced performance and efficiency",
		"spec": {
			"displayName": "meta.llama-3.2-3b-instruct",
			"vendor": "meta",
			"disabled": False,
			"version": "1.0.0",
			"modelFormat": {"name": "safetensors", "operator": "Equal", "version": "1.0.0", "weight": 1},
			"modelArchitecture": "LlamaForCausalLM",
			"modelFramework": {"name": "transformers", "operator": "Equal", "version": "4.45.0.dev0"},
			"storage": {
				"storageUri": "hf://meta-llama/Llama-3.2-3B-Instruct",
				"path": "/raid/models/meta/llama-3-2-3b-instruct",
				"key": "hf-token",
			},
		},
	},
	"llama-3-1-70b-instruct": {
		"name": "llama-3-1-70b-instruct",
		"display_name": "Llama 3.1 70B Instruct",
		"description": "Meta's Llama 3.1 70B Instruct model - high-performance large language model",
		"spec": {
			"displayName": "meta.llama-3.1-70b-instruct",
			"vendor": "meta",
			"disabled": False,
			"version": "1.0.0",
			"modelFormat": {"name": "safetensors", "operator": "Equal", "version": "1.0.0", "weight": 1},
			"modelArchitecture": "LlamaForCausalLM",
			"modelFramework": {"name": "transformers", "operator": "Equal", "version": "4.45.0.dev0"},
			"storage": {
				"storageUri": "hf://meta-llama/Meta-Llama-3.1-70B-Instruct",
				"path": "/raid/models/meta/llama-3-1-70b-instruct",
				"key": "hf-token",
			},
		},
	},
	"llama-3-3-70b-instruct": {
		"name": "llama-3-3-70b-instruct",
		"display_name": "Llama 3.3 70B Instruct",
		"description": "Meta's Llama 3.3 70B Instruct model - latest high-performance model",
		"spec": {
			"displayName": "meta.llama-3.3-70b-instruct",
			"vendor": "meta",
			"disabled": False,
			"version": "1.0.0",
			"modelFormat": {"name": "safetensors", "operator": "Equal", "version": "1.0.0", "weight": 1},
			"modelArchitecture": "LlamaForCausalLM",
			"modelFramework": {"name": "transformers", "operator": "Equal", "version": "4.45.0.dev0"},
			"storage": {
				"storageUri": "hf://meta-llama/Llama-3.3-70B-Instruct",
				"path": "/raid/models/meta/llama-3-3-70b-instruct",
				"key": "hf-token",
			},
		},
	},
	"mistral-7b-instruct": {
		"name": "mistral-7b-instruct",
		"display_name": "Mistral 7B Instruct",
		"description": "Mistral AI's 7B Instruct model - efficient open-source model",
		"spec": {
			"displayName": "mistralai.mistral-7b-instruct",
			"vendor": "mistralai",
			"disabled": False,
			"version": "1.0.0",
			"modelFormat": {"name": "safetensors", "operator": "Equal", "version": "1.0.0", "weight": 1},
			"modelArchitecture": "LlamaForCausalLM",
			"modelFramework": {"name": "transformers", "operator": "Equal", "version": "4.45.0.dev0"},
			"storage": {
				"storageUri": "hf://mistralai/Mistral-7B-Instruct-v0.3",
				"path": "/raid/models/mistralai/mistral-7b-instruct",
				"key": "hf-token",
			},
		},
	},
	"mixtral-8x7b-instruct": {
		"name": "mixtral-8x7b-instruct",
		"display_name": "Mixtral 8x7B Instruct",
		"description": "Mistral AI's Mixtral 8x7B MoE model - high-performance mixture of experts",
		"spec": {
			"displayName": "mistralai.mixtral-8x7b-instruct",
			"vendor": "mistralai",
			"disabled": False,
			"version": "1.0.0",
			"modelFormat": {"name": "safetensors", "operator": "Equal", "version": "1.0.0", "weight": 1},
			"modelArchitecture": "LlamaForCausalLM",
			"modelFramework": {"name": "transformers", "operator": "Equal", "version": "4.45.0.dev0"},
			"storage": {
				"storageUri": "hf://mistralai/Mixtral-8x7B-Instruct-v0.1",
				"path": "/raid/models/mistralai/mixtral-8x7b-instruct",
				"key": "hf-token",
			},
		},
	},
	"deepseek-v3": {
		"name": "deepseek-v3",
		"display_name": "DeepSeek V3",
		"description": "DeepSeek's V3 model - advanced large language model",
		"spec": {
			"displayName": "deepseek-ai.deepseek-v3",
			"vendor": "deepseek-ai",
			"disabled": False,
			"version": "1.0.0",
			"modelFormat": {"name": "safetensors", "operator": "Equal", "version": "1.0.0", "weight": 1},
			"modelArchitecture": "LlamaForCausalLM",
			"modelFramework": {"name": "transformers", "operator": "Equal", "version": "4.45.0.dev0"},
			"storage": {
				"storageUri": "hf://deepseek-ai/DeepSeek-V3",
				"path": "/raid/models/deepseek-ai/deepseek-v3",
				"key": "hf-token",
			},
		},
	},
}


def get_preset(preset_name: str) -> Dict[str, Any]:
	"""
	Get a ClusterBaseModel preset by name.

	Args:
	    preset_name: Name of the preset

	Returns:
	    Preset configuration dictionary

	Raises:
	    ValueError: If preset not found
	"""
	if preset_name not in CLUSTERBASEMODEL_PRESETS:
		raise ValueError(
			f"ClusterBaseModel preset '{preset_name}' not found. Available presets: {list(CLUSTERBASEMODEL_PRESETS.keys())}"
		)

	return CLUSTERBASEMODEL_PRESETS[preset_name].copy()


def list_presets() -> List[Dict[str, Any]]:
	"""
	List all available ClusterBaseModel presets.

	Returns:
	    List of preset configurations
	"""
	return [
		{"name": name, "display_name": preset["display_name"], "description": preset["description"]}
		for name, preset in CLUSTERBASEMODEL_PRESETS.items()
	]


def validate_custom_config(config: Dict[str, Any]) -> bool:
	"""
	Validate a custom ClusterBaseModel configuration.

	Args:
	    config: Custom configuration dictionary

	Returns:
	    True if valid, raises ValueError if invalid

	Raises:
	    ValueError: If configuration is invalid
	"""
	required_fields = ["displayName", "vendor", "version", "storage"]
	for field in required_fields:
		if field not in config:
			raise ValueError(f"ClusterBaseModel config missing required field: {field}")

	storage = config["storage"]
	if "storageUri" not in storage or "path" not in storage:
		raise ValueError("ClusterBaseModel storage must contain 'storageUri' and 'path'")

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
	for key, value in overrides.items():
		if isinstance(value, dict) and key in spec and isinstance(spec[key], dict):
			spec[key] = {**spec[key], **value}
		else:
			spec[key] = value

	return {
		"name": preset["name"],
		"display_name": preset["display_name"],
		"description": preset["description"],
		"spec": spec,
	}
