"""
Parallel execution configuration mapper.

Maps parallel configuration to runtime-specific CLI arguments.
Handles presets and validates engine-specific constraints.
"""

from typing import Dict, Any, Optional, List


# Parallel configuration presets per engine
PARALLEL_PRESETS = {
	"vllm": {
		"single-gpu": {
			"description": "Single GPU execution (no parallelism)",
			"tp": 1,
			"pp": 1,
			"dp": 1,
		},
		"high-throughput": {
			"description": "Maximize throughput with data parallelism (8 GPUs)",
			"tp": 1,
			"pp": 1,
			"dp": 8,
		},
		"large-model-tp": {
			"description": "Tensor parallelism for large models (8 GPUs)",
			"tp": 8,
			"pp": 1,
			"dp": 1,
		},
		"large-model-tp-pp": {
			"description": "TP + PP for very large models (16 GPUs: 8 TP × 2 PP)",
			"tp": 8,
			"pp": 2,
			"dp": 1,
		},
		"moe-optimized": {
			"description": "MoE with expert parallelism (16 GPUs: 2 TP × 8 DP)",
			"tp": 2,
			"pp": 1,
			"dp": 8,
			"enable_expert_parallel": True,
		},
		"long-context": {
			"description": "Long context with decode context parallel (16 GPUs: 4 TP × 4 DCP)",
			"tp": 4,
			"pp": 1,
			"dp": 1,
			"dcp": 4,
		},
		"balanced": {
			"description": "Balanced TP and DP (8 GPUs: 2 TP × 4 DP)",
			"tp": 2,
			"pp": 1,
			"dp": 4,
		},
	},
	"sglang": {
		"single-gpu": {
			"description": "Single GPU execution (no parallelism)",
			"tp": 1,
			"pp": 1,
			"dp": 1,
		},
		"high-throughput": {
			"description": "Maximize throughput with data parallelism (8 GPUs)",
			"tp": 1,
			"pp": 1,
			"dp": 8,
		},
		"large-model-tp": {
			"description": "Tensor parallelism for large models (8 GPUs)",
			"tp": 8,
			"pp": 1,
			"dp": 1,
		},
		"large-model-tp-pp": {
			"description": "TP + PP for very large models (16 GPUs: 8 TP × 2 PP)",
			"tp": 8,
			"pp": 2,
			"dp": 1,
		},
		"moe-optimized": {
			"description": "MoE with automatic expert distribution (16 GPUs: 2 TP × 8 DP)",
			"tp": 2,
			"pp": 1,
			"dp": 8,
			"moe_dense_tp": 2,
		},
		"balanced": {
			"description": "Balanced TP and DP (8 GPUs: 2 TP × 4 DP)",
			"tp": 2,
			"pp": 1,
			"dp": 4,
		},
	},
	"tensorrt-llm": {
		"single-gpu": {
			"description": "Single GPU execution (no parallelism)",
			"tp": 1,
			"pp": 1,
		},
		"large-model-tp": {
			"description": "Tensor parallelism for large models (8 GPUs)",
			"tp": 8,
			"pp": 1,
		},
		"large-model-tp-pp": {
			"description": "TP + PP for very large models (16 GPUs: 8 TP × 2 PP)",
			"tp": 8,
			"pp": 2,
		},
		"moe-optimized": {
			"description": "MoE with explicit EP configuration (16 GPUs)",
			"tp": 4,
			"pp": 1,
			"moe_tp": 2,
			"moe_ep": 8,
		},
		"long-context": {
			"description": "Long context with context parallel (16 GPUs: 8 TP × 2 CP)",
			"tp": 8,
			"pp": 1,
			"cp": 2,
		},
	},
}


def get_preset_names(base_runtime: str) -> List[str]:
	"""
	Get available preset names for a given runtime.

	Args:
	    base_runtime: Runtime name (vllm, sglang, tensorrt-llm)

	Returns:
	    List of preset names
	"""
	runtime_key = base_runtime.lower().replace("-", "").replace("_", "")
	if runtime_key == "tensorrtllm":
		runtime_key = "tensorrt-llm"

	return list(PARALLEL_PRESETS.get(runtime_key, {}).keys())


def resolve_parallel_preset(base_runtime: str, preset_name: str) -> Optional[Dict[str, Any]]:
	"""
	Resolve a parallel preset name to its configuration.

	Args:
	    base_runtime: Runtime name (vllm, sglang, tensorrt-llm)
	    preset_name: Name of the preset

	Returns:
	    Resolved configuration dict, or None if preset not found

	Example:
	    >>> resolve_parallel_preset("vllm", "high-throughput")
	    {"tp": 1, "pp": 1, "dp": 8}
	"""
	runtime_key = base_runtime.lower().replace("-", "").replace("_", "")
	if runtime_key == "tensorrtllm":
		runtime_key = "tensorrt-llm"

	presets = PARALLEL_PRESETS.get(runtime_key, {})
	preset = presets.get(preset_name)

	if preset:
		# Return copy without description
		return {k: v for k, v in preset.items() if k != "description"}

	return None


def map_to_vllm_parallel_args(parallel_config: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Map parallel configuration to vLLM CLI arguments.

	Args:
	    parallel_config: Parallel config with tp, pp, dp, dcp, enable_expert_parallel fields

	Returns:
	    Dictionary of vLLM CLI arguments

	Example:
	    >>> map_to_vllm_parallel_args({"tp": 4, "pp": 2, "dp": 1})
	    {"--tensor-parallel-size": "4", "--pipeline-parallel-size": "2", "--data-parallel-size": "1"}
	"""
	args = {}

	# Tensor parallel
	if "tp" in parallel_config and parallel_config["tp"] != 1:
		args["--tensor-parallel-size"] = str(parallel_config["tp"])

	# Pipeline parallel
	if "pp" in parallel_config and parallel_config["pp"] != 1:
		args["--pipeline-parallel-size"] = str(parallel_config["pp"])

	# Data parallel
	if "dp" in parallel_config and parallel_config["dp"] != 1:
		args["--data-parallel-size"] = str(parallel_config["dp"])

	# Decode context parallel (long context)
	if "dcp" in parallel_config and parallel_config["dcp"] != 1:
		args["--decode-context-parallel-size"] = str(parallel_config["dcp"])

	# Expert parallel (MoE)
	if parallel_config.get("enable_expert_parallel"):
		args["--enable-expert-parallel"] = None  # Boolean flag

	return args


def map_to_sglang_parallel_args(parallel_config: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Map parallel configuration to SGLang CLI arguments.

	Args:
	    parallel_config: Parallel config with tp, pp, dp, moe_dense_tp fields

	Returns:
	    Dictionary of SGLang CLI arguments

	Example:
	    >>> map_to_sglang_parallel_args({"tp": 4, "pp": 1, "dp": 2})
	    {"--tp-size": "4", "--dp-size": "2"}
	"""
	args = {}

	# Tensor parallel
	if "tp" in parallel_config and parallel_config["tp"] != 1:
		args["--tp-size"] = str(parallel_config["tp"])

	# Pipeline parallel
	if "pp" in parallel_config and parallel_config["pp"] != 1:
		args["--pp-size"] = str(parallel_config["pp"])

	# Data parallel
	if "dp" in parallel_config and parallel_config["dp"] != 1:
		args["--dp-size"] = str(parallel_config["dp"])

	# MoE dense TP (for MoE models)
	if "moe_dense_tp" in parallel_config:
		args["--moe-dense-tp-size"] = str(parallel_config["moe_dense_tp"])

	return args


def map_to_tensorrt_llm_parallel_args(parallel_config: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Map parallel configuration to TensorRT-LLM build-time arguments.

	Note: TensorRT-LLM requires build-time parallelism configuration.
	These parameters are used during engine building, not runtime.

	Args:
	    parallel_config: Parallel config with tp, pp, cp, moe_tp, moe_ep fields

	Returns:
	    Dictionary of TensorRT-LLM Mapping parameters

	Example:
	    >>> map_to_tensorrt_llm_parallel_args({"tp": 8, "pp": 2})
	    {"tp_size": 8, "pp_size": 2}
	"""
	args = {}

	# Tensor parallel
	if "tp" in parallel_config:
		args["tp_size"] = parallel_config["tp"]

	# Pipeline parallel
	if "pp" in parallel_config:
		args["pp_size"] = parallel_config["pp"]

	# Context parallel (long context)
	if "cp" in parallel_config and parallel_config["cp"] != 1:
		args["cp_size"] = parallel_config["cp"]

	# MoE parallelism
	if "moe_tp" in parallel_config:
		args["moe_tp_size"] = parallel_config["moe_tp"]

	if "moe_ep" in parallel_config:
		args["moe_ep_size"] = parallel_config["moe_ep"]

	if "moe_cluster" in parallel_config:
		args["moe_cluster_size"] = parallel_config["moe_cluster"]

	# Note: No data parallelism for TensorRT-LLM
	# Users should launch multiple engine instances for DP

	return args


def get_runtime_parallel_args(
	base_runtime: str, parallel_config: Dict[str, Any], model_config: Optional[Dict] = None
) -> Dict[str, Any]:
	"""
	Get runtime-specific parallel arguments from parallel configuration.

	This is the main entry point for converting parallel_config to CLI arguments.

	Args:
	    base_runtime: Runtime name (vllm, sglang, tensorrt-llm)
	    parallel_config: Parallel configuration dict
	    model_config: Optional model configuration (for context)

	Returns:
	    Dictionary of runtime-specific CLI arguments

	Example:
	    >>> get_runtime_parallel_args("vllm", {"tp": 4, "dp": 2})
	    {"--tensor-parallel-size": "4", "--data-parallel-size": "2"}
	"""
	if not parallel_config:
		return {}

	# Normalize runtime name
	runtime = base_runtime.lower()

	if runtime == "vllm":
		return map_to_vllm_parallel_args(parallel_config)
	elif runtime == "sglang":
		return map_to_sglang_parallel_args(parallel_config)
	elif runtime in ["tensorrt-llm", "tensorrt_llm", "tensorrtllm"]:
		return map_to_tensorrt_llm_parallel_args(parallel_config)
	else:
		# Unknown runtime, return empty dict
		return {}


def validate_parallel_config(base_runtime: str, parallel_config: Dict[str, Any]) -> tuple[bool, str]:
	"""
	Validate parallel configuration for a given runtime.

	Checks for:
	- Unsupported parameters for the runtime
	- Invalid parameter values
	- Constraint violations (e.g., tp % dp != 0 for SGLang)

	Args:
	    base_runtime: Runtime name
	    parallel_config: Parallel configuration to validate

	Returns:
	    (is_valid, error_message) tuple

	Example:
	    >>> validate_parallel_config("sglang", {"tp": 4, "dp": 3})
	    (False, "SGLang requires tp_size to be divisible by dp_size (4 % 3 != 0)")
	"""
	if not parallel_config:
		return True, "Valid (no parallel configuration)"

	runtime = base_runtime.lower()

	# Check for preset mode
	if "presets" in parallel_config:
		presets = parallel_config.get("presets")
		if isinstance(presets, list):
			for preset_name in presets:
				preset = resolve_parallel_preset(base_runtime, preset_name)
				if not preset:
					return False, f"Unknown preset '{preset_name}' for runtime '{base_runtime}'"
		else:
			preset = resolve_parallel_preset(base_runtime, presets)
			if not preset:
				return False, f"Unknown preset '{presets}' for runtime '{base_runtime}'"

	# Runtime-specific validation
	if runtime == "vllm":
		return _validate_vllm_parallel(parallel_config)
	elif runtime == "sglang":
		return _validate_sglang_parallel(parallel_config)
	elif runtime in ["tensorrt-llm", "tensorrt_llm", "tensorrtllm"]:
		return _validate_tensorrt_llm_parallel(parallel_config)

	return True, "Valid"


def _validate_vllm_parallel(parallel_config: Dict[str, Any]) -> tuple[bool, str]:
	"""Validate vLLM-specific parallel configuration."""
	# vLLM supports all parallel types
	# Just check for invalid parameter names
	valid_params = {"tp", "pp", "dp", "dcp", "enable_expert_parallel"}
	invalid = set(parallel_config.keys()) - valid_params

	if invalid:
		return False, f"Invalid parameters for vLLM: {invalid}"

	return True, "Valid"


def _validate_sglang_parallel(parallel_config: Dict[str, Any]) -> tuple[bool, str]:
	"""Validate SGLang-specific parallel configuration."""
	valid_params = {"tp", "pp", "dp", "moe_dense_tp"}
	invalid = set(parallel_config.keys()) - valid_params

	if invalid:
		# Check if user tried to use CP (not supported by SGLang)
		if "cp" in invalid or "dcp" in invalid:
			return False, "SGLang does not support context parallelism (cp/dcp)"
		return False, f"Invalid parameters for SGLang: {invalid}"

	# SGLang constraint: tp_size % dp_size == 0
	tp = parallel_config.get("tp", 1)
	dp = parallel_config.get("dp", 1)

	if tp % dp != 0:
		return False, f"SGLang requires tp_size to be divisible by dp_size ({tp} % {dp} != 0)"

	return True, "Valid"


def _validate_tensorrt_llm_parallel(parallel_config: Dict[str, Any]) -> tuple[bool, str]:
	"""Validate TensorRT-LLM-specific parallel configuration."""
	valid_params = {"tp", "pp", "cp", "moe_tp", "moe_ep", "moe_cluster"}
	invalid = set(parallel_config.keys()) - valid_params

	if invalid:
		# Check if user tried to use DP (not supported by TensorRT-LLM)
		if "dp" in invalid:
			return False, "TensorRT-LLM does not support data parallelism (dp). Use multiple engine instances for DP."
		return False, f"Invalid parameters for TensorRT-LLM: {invalid}"

	# Note: TensorRT-LLM parallelism is build-time only
	# This is a limitation that should be documented

	return True, "Valid (Note: TensorRT-LLM requires engine rebuild for parallel config changes)"


def get_parallel_config_summary(parallel_config: Optional[Dict[str, Any]]) -> str:
	"""
	Get a human-readable summary of parallel configuration.

	Args:
	    parallel_config: Parallel configuration

	Returns:
	    Summary string

	Example:
	    >>> get_parallel_config_summary({"tp": 4, "pp": 2, "dp": 1})
	    "TP: 4, PP: 2, DP: 1 (Total GPUs: 8)"
	"""
	if not parallel_config:
		return "No parallelism"

	# Handle preset references (array mode only)
	if "presets" in parallel_config:
		presets = parallel_config["presets"]
		if isinstance(presets, list):
			return f"Presets: {', '.join(presets)}"
		return f"Preset: {presets}"

	parts = []

	tp = parallel_config.get("tp", 1)
	pp = parallel_config.get("pp", 1)
	dp = parallel_config.get("dp", 1)
	cp = parallel_config.get("cp") or parallel_config.get("dcp")

	if tp != 1:
		parts.append(f"TP: {tp}")
	if pp != 1:
		parts.append(f"PP: {pp}")
	if dp != 1:
		parts.append(f"DP: {dp}")
	if cp and cp != 1:
		parts.append(f"CP: {cp}")

	if parallel_config.get("enable_expert_parallel"):
		parts.append("EP: enabled")

	if "moe_tp" in parallel_config or "moe_ep" in parallel_config:
		moe_tp = parallel_config.get("moe_tp", "?")
		moe_ep = parallel_config.get("moe_ep", "?")
		parts.append(f"MoE: TP={moe_tp}, EP={moe_ep}")

	if not parts:
		return "Single GPU"

	summary = ", ".join(parts)

	# Calculate total GPUs
	total_gpus = tp * pp * dp
	if cp and cp != 1:
		total_gpus = tp * pp * cp

	summary += f" (Total GPUs: {total_gpus})"

	return summary
