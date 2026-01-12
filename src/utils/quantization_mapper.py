"""
Quantization configuration mapper for converting four-field quant config
to runtime-specific CLI arguments (vLLM, SGLang, TensorRT-LLM).
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


# Quantization presets (optional, for convenience)
QUANTIZATION_PRESETS = {
	"default": {"gemm_dtype": "auto", "kvcache_dtype": "auto", "attention_dtype": "auto", "moe_dtype": "auto"},
	"kv-cache-fp8": {"gemm_dtype": "auto", "kvcache_dtype": "fp8_e5m2", "attention_dtype": "auto", "moe_dtype": "auto"},
	"dynamic-fp8": {"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2", "attention_dtype": "fp8", "moe_dtype": "fp8"},
	"bf16-stable": {
		"gemm_dtype": "bfloat16",
		"kvcache_dtype": "fp8_e5m2",
		"attention_dtype": "auto",
		"moe_dtype": "auto",
	},
	"aggressive-moe": {
		"gemm_dtype": "bfloat16",
		"kvcache_dtype": "fp8_e5m2",
		"attention_dtype": "fp8",
		"moe_dtype": "w4afp8",
	},
}


def expand_preset(preset_name: str) -> Dict[str, str]:
	"""Expand preset name to full quantization configuration."""
	if preset_name not in QUANTIZATION_PRESETS:
		raise ValueError(
			f"Unknown quantization preset: {preset_name}. " f"Available presets: {list(QUANTIZATION_PRESETS.keys())}"
		)
	return QUANTIZATION_PRESETS[preset_name].copy()


def validate_quant_config(config: Dict[str, Any]) -> tuple[bool, str]:
	"""
	Validate quantization configuration.

	Returns:
	    (is_valid, error_message)
	"""
	if not config:
		return True, "Valid (empty config)"

	# Check if it's a preset reference (array only)
	if "presets" in config:
		presets = config["presets"]
		if not isinstance(presets, list):
			return False, "presets must be an array"
		for preset_name in presets:
			if preset_name not in QUANTIZATION_PRESETS:
				return False, f"Unknown preset: {preset_name}"
		return True, "Valid presets"

	# Validate individual fields
	valid_gemm_dtypes = ["auto", "float16", "bfloat16", "float32", "fp8", "int8"]
	valid_kvcache_dtypes = ["auto", "fp16", "bfloat16", "fp8", "fp8_e5m2", "fp8_e4m3", "int8", "int4"]
	valid_attention_dtypes = ["auto", "float16", "bfloat16", "fp8", "fp8_e5m2", "fp8_e4m3", "fp8_block"]
	valid_moe_dtypes = ["auto", "float16", "bfloat16", "fp8", "w4afp8", "mxfp4", "int8"]

	gemm_dtype = config.get("gemm_dtype", "auto")
	if gemm_dtype not in valid_gemm_dtypes:
		return False, f"Invalid gemm_dtype: {gemm_dtype}. Must be one of {valid_gemm_dtypes}"

	kvcache_dtype = config.get("kvcache_dtype", "auto")
	if kvcache_dtype not in valid_kvcache_dtypes:
		return False, f"Invalid kvcache_dtype: {kvcache_dtype}. Must be one of {valid_kvcache_dtypes}"

	attention_dtype = config.get("attention_dtype", "auto")
	if attention_dtype not in valid_attention_dtypes:
		return False, f"Invalid attention_dtype: {attention_dtype}. Must be one of {valid_attention_dtypes}"

	moe_dtype = config.get("moe_dtype", "auto")
	if moe_dtype not in valid_moe_dtypes:
		return False, f"Invalid moe_dtype: {moe_dtype}. Must be one of {valid_moe_dtypes}"

	return True, "Valid"


def resolve_quant_config(config: Optional[Dict[str, Any]]) -> Dict[str, str]:
	"""
	Resolve quantization configuration, expanding presets if needed.

	Args:
	    config: Quantization config (can contain 'presets' array or explicit fields)

	Returns:
	    Resolved config with explicit gemm_dtype, kvcache_dtype, attention_dtype, moe_dtype

	Note:
	    This function now only handles explicit field configs.
	    Preset expansion should be done before calling this function.
	"""
	if not config:
		# Default: no quantization
		return {"gemm_dtype": "auto", "kvcache_dtype": "auto", "attention_dtype": "auto", "moe_dtype": "auto"}

	# Use explicit fields with defaults
	return {
		"gemm_dtype": config.get("gemm_dtype", "auto"),
		"kvcache_dtype": config.get("kvcache_dtype", "auto"),
		"attention_dtype": config.get("attention_dtype", "auto"),
		"moe_dtype": config.get("moe_dtype", "auto"),
	}


def should_apply_dynamic_fp8(gemm_dtype: str, model_quantization: Optional[str] = None) -> bool:
	"""
	Determine if dynamic FP8 (W8A8) quantization should be applied.

	Returns False if model is already quantized with offline methods.
	"""
	if gemm_dtype != "fp8":
		return False

	# List of offline quantization methods
	offline_quant_methods = ["awq", "gptq", "gguf", "squeezellm", "marlin", "nvfp4", "fp8", "bitsandbytes", "hqq"]

	if model_quantization and model_quantization in offline_quant_methods:
		logger.warning(
			f"Ignoring gemm_dtype='fp8': model already quantized with {model_quantization}. "
			f"Dynamic FP8 only applies to unquantized FP16/BF16 models."
		)
		return False

	return True


def map_to_vllm_args(quant_config: Dict[str, str], model_quantization: Optional[str] = None) -> Dict[str, str]:
	"""
	Map four-field quantization config to vLLM CLI arguments.

	Args:
	    quant_config: Resolved quant config with gemm_dtype, kvcache_dtype, attention_dtype, moe_dtype
	    model_quantization: Detected model quantization method (e.g., "awq", "gptq", None)

	Returns:
	    Dictionary of vLLM CLI arguments (e.g., {"--dtype": "auto", "--kv-cache-dtype": "fp8"})
	"""
	args = {}

	gemm_dtype = quant_config["gemm_dtype"]
	kvcache_dtype = quant_config["kvcache_dtype"]
	attention_dtype = quant_config["attention_dtype"]
	# moe_dtype not supported by vLLM (uses gemm_dtype)

	# GEMM dtype
	if should_apply_dynamic_fp8(gemm_dtype, model_quantization):
		# Dynamic FP8 quantization (W8A8)
		args["--quantization"] = "fp8"
		args["--dtype"] = "auto"
	elif gemm_dtype == "int8":
		args["--quantization"] = "int8"
		args["--dtype"] = "auto"
	elif gemm_dtype != "auto":
		# Explicit dtype (float16, bfloat16, float32)
		args["--dtype"] = gemm_dtype
	else:
		# Auto: let vLLM decide
		args["--dtype"] = "auto"

	# KV cache dtype
	if kvcache_dtype != "auto":
		args["--kv-cache-dtype"] = kvcache_dtype

	# Attention dtype
	# vLLM doesn't support explicit attention dtype control (uses GEMM dtype)
	if attention_dtype not in ["auto", gemm_dtype]:
		logger.warning(
			f"vLLM does not support separate attention_dtype. "
			f"Requested '{attention_dtype}' will fall back to gemm_dtype '{gemm_dtype}'."
		)

	return args


def map_to_sglang_args(quant_config: Dict[str, str], model_quantization: Optional[str] = None) -> Dict[str, str]:
	"""
	Map four-field quantization config to SGLang CLI arguments.

	Args:
	    quant_config: Resolved quant config with gemm_dtype, kvcache_dtype, attention_dtype, moe_dtype
	    model_quantization: Detected model quantization method (e.g., "awq", "gptq", None)

	Returns:
	    Dictionary of SGLang CLI arguments
	"""
	args = {}

	gemm_dtype = quant_config["gemm_dtype"]
	kvcache_dtype = quant_config["kvcache_dtype"]
	attention_dtype = quant_config["attention_dtype"]
	moe_dtype = quant_config["moe_dtype"]

	# GEMM dtype
	# MoE dtype can override GEMM quantization for MoE models
	if moe_dtype in ["w4afp8", "mxfp4"]:
		# MoE-specific quantization takes precedence
		args["--quantization"] = moe_dtype
		args["--dtype"] = "auto"

		# Set MoE backend
		if moe_dtype == "w4afp8":
			args["--moe-runner-backend"] = "flashinfer_mxfp4"
		elif moe_dtype == "mxfp4":
			args["--moe-runner-backend"] = "flashinfer_mxfp4"
	elif should_apply_dynamic_fp8(gemm_dtype, model_quantization):
		# Dynamic FP8 quantization (W8A8)
		args["--quantization"] = "fp8"
		args["--dtype"] = "auto"
	elif gemm_dtype == "int8":
		# SGLang doesn't support plain "int8", skip it (use auto)
		# Valid options would be: w8a8_int8, but that's not a user-facing dtype
		args["--dtype"] = "auto"
	elif gemm_dtype != "auto":
		# Explicit dtype (float16, bfloat16)
		args["--dtype"] = gemm_dtype
	else:
		# Auto: let SGLang decide
		args["--dtype"] = "auto"

	# KV cache dtype
	# SGLang only accepts: auto, fp8_e5m2, fp8_e4m3
	# Map other values or skip them
	if kvcache_dtype == "fp8":
		# Generic fp8 -> use fp8_e5m2 (better quality)
		args["--kv-cache-dtype"] = "fp8_e5m2"
	elif kvcache_dtype in ["fp8_e5m2", "fp8_e4m3"]:
		args["--kv-cache-dtype"] = kvcache_dtype
	# For fp16, bfloat16, int8, int4: use 'auto' (omit the argument)

	# Attention dtype
	if attention_dtype in ["fp8", "fp8_e5m2", "fp8_e4m3"]:
		# Enable FP8 attention with FlashInfer backend
		args["--attention-backend"] = "flashinfer"
		# FP8 attention is enabled automatically when using FlashInfer + FP8 KV cache

	# MoE dtype (if FP8 but not w4afp8/mxfp4)
	if moe_dtype == "fp8" and "--quantization" not in args:
		# Enable FP8 MoE with FlashInfer backend
		args["--moe-runner-backend"] = "flashinfer_cutlass"

	return args


def map_to_tensorrt_llm_args(quant_config: Dict[str, str], model_quantization: Optional[str] = None) -> Dict[str, str]:
	"""
	Map four-field quantization config to TensorRT-LLM arguments.

	Args:
	    quant_config: Resolved quant config with gemm_dtype, kvcache_dtype, attention_dtype, moe_dtype
	    model_quantization: Detected model quantization method

	Returns:
	    Dictionary of TensorRT-LLM arguments
	"""
	args = {}

	gemm_dtype = quant_config["gemm_dtype"]
	kvcache_dtype = quant_config["kvcache_dtype"]
	attention_dtype = quant_config["attention_dtype"]
	# moe_dtype not supported by TensorRT-LLM (uses GEMM dtype)

	# GEMM dtype
	if should_apply_dynamic_fp8(gemm_dtype, model_quantization):
		args["--quant-algo"] = "FP8"
	elif gemm_dtype == "int8":
		args["--quant-algo"] = "INT8"
	# Note: TensorRT-LLM auto-detects model dtype, no explicit --dtype flag

	# KV cache dtype
	if "fp8" in kvcache_dtype:
		args["--kv-cache-quant-algo"] = "FP8"
	elif kvcache_dtype == "int8":
		args["--kv-cache-quant-algo"] = "INT8"
	elif kvcache_dtype == "int4":
		args["--kv-cache-quant-algo"] = "INT4"

	# Attention dtype (FMHA quantization)
	if attention_dtype in ["fp8", "fp8_e5m2", "fp8_e4m3"]:
		args["--fmha-quant-algo"] = "FP8"
	elif attention_dtype == "fp8_block":
		args["--fmha-quant-algo"] = "FP8_BLOCK"

	return args


def merge_parameters(quant_args: Dict[str, str], user_parameters: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Merge quantization-derived arguments with user-specified parameters.
	User parameters have higher priority and will override quant_config-derived values.

	Args:
	    quant_args: Arguments derived from quant_config
	    user_parameters: User-specified parameters from Task.parameters

	Returns:
	    Merged parameter dictionary with user parameters taking precedence
	"""
	# Start with quant_config-derived args
	merged = quant_args.copy()

	# Override with user parameters
	# Convert user parameters to CLI format if needed
	for key, value in user_parameters.items():
		# Handle both formats: "dtype" and "--dtype"
		cli_key = key if key.startswith("--") else f"--{key}"

		# If user explicitly set this parameter, override quant_config
		if isinstance(value, list):
			# If it's a list (parameter grid), use the first value for now
			# The orchestrator will handle grid expansion
			if value:
				merged[cli_key] = str(value[0])
		elif isinstance(value, bool):
			# Preserve boolean type - do NOT convert to string
			# Docker/OME controllers handle boolean flags specially
			merged[cli_key] = value
		else:
			merged[cli_key] = str(value)

	return merged


def get_runtime_args(
	runtime: str,
	quant_config: Optional[Dict[str, Any]],
	user_parameters: Optional[Dict[str, Any]] = None,
	model_quantization: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	Get runtime-specific CLI arguments from quantization config and user parameters.

	Args:
	    runtime: Runtime engine ("vllm", "sglang", "tensorrt_llm")
	    quant_config: Quantization configuration (can be None, preset, or explicit fields)
	    user_parameters: User-specified parameters (optional, takes priority over quant_config)
	    model_quantization: Detected offline model quantization (e.g., "awq", "gptq")

	Returns:
	    Dictionary of CLI arguments ready to be passed to the runtime

	Example:
	    >>> get_runtime_args(
	    ...     "sglang",
	    ...     {"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2"},
	    ...     {"tp-size": [1, 2], "mem-fraction-static": 0.85}
	    ... )
	    {"--quantization": "fp8", "--dtype": "auto", "--kv-cache-dtype": "fp8_e5m2",
	     "--tp-size": "1", "--mem-fraction-static": "0.85"}
	"""
	# Validate and resolve quant config
	is_valid, msg = validate_quant_config(quant_config or {})
	if not is_valid:
		raise ValueError(f"Invalid quantization config: {msg}")

	resolved_config = resolve_quant_config(quant_config)

	# Map to runtime-specific arguments
	runtime_lower = runtime.lower()
	if runtime_lower == "vllm":
		quant_args = map_to_vllm_args(resolved_config, model_quantization)
	elif runtime_lower == "sglang":
		quant_args = map_to_sglang_args(resolved_config, model_quantization)
	elif runtime_lower in ["tensorrt_llm", "trtllm", "tensorrt-llm"]:
		quant_args = map_to_tensorrt_llm_args(resolved_config, model_quantization)
	else:
		logger.warning(f"Unknown runtime: {runtime}. Returning empty quant args.")
		quant_args = {}

	# Merge with user parameters (user parameters take priority)
	if user_parameters:
		return merge_parameters(quant_args, user_parameters)
	else:
		return quant_args
