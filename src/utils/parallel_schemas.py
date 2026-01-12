"""
Engine-specific parallel parameter schemas and allowed values.

This module defines what parallel parameters are supported by each engine
and what values are acceptable. Used for:
- Frontend validation and UI generation
- Backend parameter validation
- Documentation generation
"""

from typing import Dict, Any, List, Optional


# vLLM Parallel Parameter Schema
VLLM_PARALLEL_SCHEMA = {
	"tp": {
		"name": "Tensor Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8, 16],
		"default": 1,
		"cli_arg": "--tensor-parallel-size",
		"description": "Number of GPUs for tensor parallelism (splits model layers)",
	},
	"pp": {
		"name": "Pipeline Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8],
		"default": 1,
		"cli_arg": "--pipeline-parallel-size",
		"description": "Number of pipeline stages (splits model into stages)",
	},
	"dp": {
		"name": "Data Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8, 16],
		"default": 1,
		"cli_arg": "--data-parallel-size",
		"description": "Number of data parallel replicas (for throughput)",
	},
	"dcp": {
		"name": "Decode Context Parallel",
		"type": "integer",
		"allowed": [1, 2, 4, 8],
		"default": 1,
		"cli_arg": "--decode-context-parallel-size",
		"description": "Context parallelism for decode phase (long context)",
	},
	"enable_expert_parallel": {
		"name": "Enable Expert Parallelism",
		"type": "boolean",
		"allowed": [True, False],
		"default": False,
		"cli_arg": "--enable-expert-parallel",
		"description": "Enable expert parallelism for MoE models",
	},
}

# SGLang Parallel Parameter Schema
SGLANG_PARALLEL_SCHEMA = {
	"tp": {
		"name": "Tensor Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8, 16],
		"default": 1,
		"cli_arg": "--tp-size",
		"description": "Number of GPUs for tensor parallelism",
	},
	"pp": {
		"name": "Pipeline Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4],
		"default": 1,
		"cli_arg": "--pp-size",
		"description": "Number of pipeline stages",
	},
	"dp": {
		"name": "Data Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8, 16],
		"default": 1,
		"cli_arg": "--dp-size",
		"description": "Number of data parallel replicas",
		"constraint": "tp must be divisible by dp (tp % dp == 0)",
	},
	"moe_dense_tp": {
		"name": "MoE Dense TP",
		"type": "integer",
		"allowed": [1, 2, 4, 8],
		"default": None,
		"cli_arg": "--moe-dense-tp-size",
		"description": "Tensor parallel size for dense layers in MoE models (if None, uses tp)",
	},
}

# TensorRT-LLM Parallel Parameter Schema
TENSORRT_LLM_PARALLEL_SCHEMA = {
	"tp": {
		"name": "Tensor Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8, 16],
		"default": 1,
		"build_param": "tp_size",
		"description": "Number of GPUs for tensor parallelism (build-time)",
	},
	"pp": {
		"name": "Pipeline Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4],
		"default": 1,
		"build_param": "pp_size",
		"description": "Number of pipeline stages (build-time)",
	},
	"cp": {
		"name": "Context Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8],
		"default": 1,
		"build_param": "cp_size",
		"description": "Context parallelism for long context (build-time)",
	},
	"moe_tp": {
		"name": "MoE Tensor Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8],
		"default": None,
		"build_param": "moe_tp_size",
		"description": "Tensor parallel size for MoE layers (build-time)",
	},
	"moe_ep": {
		"name": "MoE Expert Parallelism",
		"type": "integer",
		"allowed": [1, 2, 4, 8, 16],
		"default": None,
		"build_param": "moe_ep_size",
		"description": "Expert parallelism size for MoE models (build-time)",
	},
	"moe_cluster": {
		"name": "MoE Cluster Size",
		"type": "integer",
		"allowed": [1, 2, 4],
		"default": None,
		"build_param": "moe_cluster_size",
		"description": "MoE cluster size for advanced MoE configuration (build-time)",
	},
}

# Mapping from runtime to schema
ENGINE_SCHEMAS = {
	"vllm": VLLM_PARALLEL_SCHEMA,
	"sglang": SGLANG_PARALLEL_SCHEMA,
	"tensorrt-llm": TENSORRT_LLM_PARALLEL_SCHEMA,
	"tensorrt_llm": TENSORRT_LLM_PARALLEL_SCHEMA,
	"tensorrtllm": TENSORRT_LLM_PARALLEL_SCHEMA,
}


def get_engine_schema(base_runtime: str) -> Dict[str, Any]:
	"""
	Get parallel parameter schema for a given engine.

	Args:
	    base_runtime: Runtime name (vllm, sglang, tensorrt-llm)

	Returns:
	    Schema dictionary for the engine

	Example:
	    >>> schema = get_engine_schema("vllm")
	    >>> schema["tp"]["allowed"]
	    [1, 2, 4, 8, 16]
	"""
	runtime_key = base_runtime.lower().replace("-", "").replace("_", "")
	if runtime_key == "tensorrtllm":
		runtime_key = "tensorrt-llm"

	schema = ENGINE_SCHEMAS.get(runtime_key)
	if not schema:
		raise ValueError(f"Unknown runtime: {base_runtime}")

	return schema


def get_supported_parameters(base_runtime: str) -> List[str]:
	"""
	Get list of supported parallel parameter names for an engine.

	Args:
	    base_runtime: Runtime name

	Returns:
	    List of parameter names (e.g., ["tp", "pp", "dp"])

	Example:
	    >>> get_supported_parameters("sglang")
	    ["tp", "pp", "dp", "moe_dense_tp"]
	"""
	schema = get_engine_schema(base_runtime)
	return list(schema.keys())


def get_allowed_values(base_runtime: str, param_name: str) -> List[Any]:
	"""
	Get allowed values for a specific parameter on an engine.

	Args:
	    base_runtime: Runtime name
	    param_name: Parameter name (e.g., "tp", "pp")

	Returns:
	    List of allowed values

	Example:
	    >>> get_allowed_values("vllm", "tp")
	    [1, 2, 4, 8, 16]
	"""
	schema = get_engine_schema(base_runtime)
	if param_name not in schema:
		raise ValueError(f"Parameter '{param_name}' not supported by {base_runtime}")

	return schema[param_name]["allowed"]


def validate_parameter_value(base_runtime: str, param_name: str, value: Any) -> tuple[bool, str]:
	"""
	Validate a parameter value against the engine schema.

	Args:
	    base_runtime: Runtime name
	    param_name: Parameter name
	    value: Value to validate

	Returns:
	    (is_valid, error_message) tuple

	Example:
	    >>> validate_parameter_value("vllm", "tp", 4)
	    (True, "Valid")
	    >>> validate_parameter_value("vllm", "tp", 3)
	    (False, "Value 3 not allowed for tp. Allowed values: [1, 2, 4, 8, 16]")
	"""
	try:
		schema = get_engine_schema(base_runtime)
	except ValueError as e:
		return False, str(e)

	if param_name not in schema:
		return False, f"Parameter '{param_name}' not supported by {base_runtime}"

	param_schema = schema[param_name]
	allowed = param_schema["allowed"]

	if value not in allowed:
		return False, f"Value {value} not allowed for {param_name}. Allowed values: {allowed}"

	return True, "Valid"


def get_parameter_constraints(base_runtime: str) -> Dict[str, str]:
	"""
	Get all parameter constraints for an engine.

	Args:
	    base_runtime: Runtime name

	Returns:
	    Dictionary of {param_name: constraint_description}

	Example:
	    >>> get_parameter_constraints("sglang")
	    {"dp": "tp must be divisible by dp (tp % dp == 0)"}
	"""
	schema = get_engine_schema(base_runtime)
	constraints = {}

	for param_name, param_info in schema.items():
		if "constraint" in param_info:
			constraints[param_name] = param_info["constraint"]

	return constraints


def get_cli_arg(base_runtime: str, param_name: str) -> Optional[str]:
	"""
	Get CLI argument name for a parameter.

	Args:
	    base_runtime: Runtime name
	    param_name: Parameter name

	Returns:
	    CLI argument string (e.g., "--tensor-parallel-size") or None

	Example:
	    >>> get_cli_arg("vllm", "tp")
	    "--tensor-parallel-size"
	"""
	schema = get_engine_schema(base_runtime)
	if param_name not in schema:
		return None

	param_schema = schema[param_name]
	return param_schema.get("cli_arg") or param_schema.get("build_param")


def get_default_value(base_runtime: str, param_name: str) -> Any:
	"""
	Get default value for a parameter.

	Args:
	    base_runtime: Runtime name
	    param_name: Parameter name

	Returns:
	    Default value or None

	Example:
	    >>> get_default_value("vllm", "tp")
	    1
	"""
	schema = get_engine_schema(base_runtime)
	if param_name not in schema:
		return None

	return schema[param_name].get("default")


def get_frontend_schema(base_runtime: str) -> Dict[str, Any]:
	"""
	Get a frontend-friendly schema with all information.

	This is used by the frontend to generate the UI form.

	Args:
	    base_runtime: Runtime name

	Returns:
	    Complete schema for frontend consumption

	Example:
	    >>> schema = get_frontend_schema("vllm")
	    >>> schema["tp"]["name"]
	    "Tensor Parallelism"
	"""
	return get_engine_schema(base_runtime)


def validate_parallel_combination(base_runtime: str, config: Dict[str, Any]) -> tuple[bool, str]:
	"""
	Validate a complete parallel configuration against engine-specific constraints.

	This checks:
	- All parameter values are in allowed lists
	- Engine-specific constraints are satisfied (e.g., SGLang tp % dp == 0)

	Args:
	    base_runtime: Runtime name
	    config: Parallel configuration dict

	Returns:
	    (is_valid, error_message) tuple

	Example:
	    >>> validate_parallel_combination("sglang", {"tp": 4, "dp": 2})
	    (True, "Valid")
	    >>> validate_parallel_combination("sglang", {"tp": 4, "dp": 3})
	    (False, "SGLang requires tp_size to be divisible by dp_size (4 % 3 != 0)")
	"""
	# Validate individual parameter values
	for param_name, value in config.items():
		is_valid, error_msg = validate_parameter_value(base_runtime, param_name, value)
		if not is_valid:
			return False, error_msg

	# Engine-specific constraint checks
	runtime_lower = base_runtime.lower()

	if runtime_lower == "sglang":
		# SGLang constraint: tp % dp == 0
		tp = config.get("tp", 1)
		dp = config.get("dp", 1)
		if tp % dp != 0:
			return False, f"SGLang requires tp_size to be divisible by dp_size ({tp} % {dp} != 0)"

	elif runtime_lower in ["tensorrt-llm", "tensorrt_llm", "tensorrtllm"]:
		# TensorRT-LLM: No data parallelism
		if "dp" in config:
			return False, "TensorRT-LLM does not support data parallelism (dp). Use multiple engine instances for DP."

	# GPU count validation
	tp = config.get("tp", 1)
	pp = config.get("pp", 1)
	dp = config.get("dp", 1)
	cp = config.get("cp") or config.get("dcp", 1)

	if runtime_lower in ["vllm", "sglang"]:
		total_gpus = tp * pp * dp
	elif runtime_lower in ["tensorrt-llm", "tensorrt_llm", "tensorrtllm"]:
		total_gpus = tp * pp * cp

	if total_gpus > 64:  # Reasonable upper limit
		return False, f"Total GPU count too high: {total_gpus}. Check your parallel configuration."

	return True, "Valid"


def get_parameter_description(base_runtime: str, param_name: str) -> Optional[str]:
	"""
	Get description for a parameter.

	Args:
	    base_runtime: Runtime name
	    param_name: Parameter name

	Returns:
	    Description string or None

	Example:
	    >>> get_parameter_description("vllm", "tp")
	    "Number of GPUs for tensor parallelism (splits model layers)"
	"""
	schema = get_engine_schema(base_runtime)
	if param_name not in schema:
		return None

	return schema[param_name].get("description")


# Export all functions
__all__ = [
	"VLLM_PARALLEL_SCHEMA",
	"SGLANG_PARALLEL_SCHEMA",
	"TENSORRT_LLM_PARALLEL_SCHEMA",
	"ENGINE_SCHEMAS",
	"get_engine_schema",
	"get_supported_parameters",
	"get_allowed_values",
	"validate_parameter_value",
	"get_parameter_constraints",
	"get_cli_arg",
	"get_default_value",
	"get_frontend_schema",
	"validate_parallel_combination",
	"get_parameter_description",
]
