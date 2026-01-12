"""
Integration helper for quantization configuration in orchestrator.

This module provides utilities to integrate quantization configuration
with the autotuner orchestrator and parameter grid expansion.
"""

import logging
import itertools
from typing import Dict, Any, Optional, List

from utils.quantization_mapper import (
    get_runtime_args,
    resolve_quant_config,
    validate_quant_config
)

logger = logging.getLogger(__name__)


def detect_model_quantization(model_path: str, model_config: Optional[Dict] = None) -> Optional[str]:
    """
    Detect offline quantization method from model path or config.

    Args:
        model_path: Model path or HuggingFace model ID
        model_config: Optional model configuration dict

    Returns:
        Quantization method name ("awq", "gptq", "gguf", etc.) or None if not quantized
    """
    # Check model path for quantization indicators
    model_path_lower = model_path.lower()

    # Common quantization patterns in model names
    if "awq" in model_path_lower:
        return "awq"
    elif "gptq" in model_path_lower:
        return "gptq"
    elif "gguf" in model_path_lower or model_path.endswith(".gguf"):
        return "gguf"
    elif "nvfp4" in model_path_lower or "fp4" in model_path_lower:
        return "nvfp4"
    elif "fp8" in model_path_lower:
        return "fp8"

    # Check model_config for quantization_config field
    if model_config and isinstance(model_config, dict):
        if "quantization" in model_config:
            return model_config["quantization"]
        if "quantization_config" in model_config:
            quant_config = model_config["quantization_config"]
            if isinstance(quant_config, dict) and "quant_method" in quant_config:
                return quant_config["quant_method"]

    return None


def prepare_experiment_parameters(
    base_runtime: str,
    quant_config: Optional[Dict[str, Any]],
    param_combination: Dict[str, Any],
    model_path: str,
    model_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Prepare final experiment parameters by merging quant_config and user parameters.

    Args:
        base_runtime: Runtime engine ("vllm", "sglang", "tensorrt_llm")
        quant_config: Quantization configuration from task
        param_combination: Single parameter combination from grid (user parameters)
        model_path: Model path for detecting offline quantization
        model_config: Optional model configuration

    Returns:
        Merged parameter dictionary with quant_config-derived args + user overrides

    Example:
        >>> prepare_experiment_parameters(
        ...     "sglang",
        ...     {"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2"},
        ...     {"tp-size": 2, "mem-fraction-static": 0.85},
        ...     "meta-llama/Llama-3.2-1B-Instruct"
        ... )
        {"quantization": "fp8", "dtype": "auto", "kv-cache-dtype": "fp8_e5m2",
         "tp-size": 2, "mem-fraction-static": 0.85}
    """
    # Detect offline model quantization
    model_quantization = detect_model_quantization(model_path, model_config)

    if model_quantization:
        logger.info(f"Detected offline model quantization: {model_quantization}")

    # Get runtime arguments from quant_config
    try:
        runtime_args = get_runtime_args(
            runtime=base_runtime,
            quant_config=quant_config,
            user_parameters=param_combination,
            model_quantization=model_quantization
        )
    except ValueError as e:
        logger.error(f"Failed to process quantization config: {e}")
        # Fall back to user parameters only
        runtime_args = param_combination.copy()

    # Remove CLI prefix (--) for cleaner parameter dict
    # Controllers will add it back when building commands
    cleaned_params = {}
    for key, value in runtime_args.items():
        clean_key = key.lstrip("-")
        cleaned_params[clean_key] = value

    return cleaned_params


def expand_quantization_presets(
    quant_config: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Expand quantization configuration if it contains multiple presets.

    Args:
        quant_config: Quantization config (can contain "presets" list)

    Returns:
        List of expanded quantization configs (one per preset)

    Example:
        >>> expand_quantization_presets({"presets": ["default", "kv-cache-fp8"]})
        [
            {"gemm_dtype": "auto", "kvcache_dtype": "auto", ...},
            {"gemm_dtype": "auto", "kvcache_dtype": "fp8_e5m2", ...}
        ]
    """
    if not quant_config:
        return [None]

    # Check if multiple presets are specified
    if "presets" in quant_config:
        presets = quant_config["presets"]
        if not isinstance(presets, list):
            presets = [presets]

        expanded = []
        for preset_name in presets:
            preset_config = {"preset": preset_name}
            # Allow overriding preset values with explicit fields
            preset_config.update({k: v for k, v in quant_config.items() if k != "presets"})
            expanded.append(preset_config)

        return expanded

    # Single configuration
    return [quant_config]


def validate_task_quant_config(quant_config: Optional[Dict[str, Any]]) -> tuple[bool, str]:
    """
    Validate quantization configuration from task.

    Args:
        quant_config: Quantization configuration to validate

    Returns:
        (is_valid, error_message) tuple
    """
    if not quant_config:
        return True, "Valid (no quantization config)"

    return validate_quant_config(quant_config)


def get_quant_config_summary(quant_config: Optional[Dict[str, Any]]) -> str:
    """
    Get a human-readable summary of quantization configuration.

    Args:
        quant_config: Quantization configuration

    Returns:
        Summary string

    Example:
        >>> get_quant_config_summary({"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2"})
        "GEMM: fp8, KV Cache: fp8_e5m2, Attention: auto, MoE: auto"
    """
    if not quant_config:
        return "No quantization"

    # Handle preset references (array mode only)
    if "presets" in quant_config:
        presets = quant_config["presets"]
        if isinstance(presets, list):
            return f"Presets: {', '.join(presets)}"
        return f"Preset: {presets}"

    # Use explicit fields
    gemm = quant_config.get("gemm_dtype", "auto")
    kvcache = quant_config.get("kvcache_dtype", "auto")
    attention = quant_config.get("attention_dtype", "auto")
    moe = quant_config.get("moe_dtype", "auto")

    return f"GEMM: {gemm}, KV Cache: {kvcache}, Attention: {attention}, MoE: {moe}"


def expand_quant_config_to_parameter_spec(
    quant_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convert quant_config with arrays into parameter spec format for generate_parameter_grid.

    This function handles quant_config that contains arrays of values and converts them
    into the format expected by generate_parameter_grid (simple list format).

    IMPORTANT: This function keeps the dtype field names (gemm_dtype, etc.) as-is.
    They will be converted to runtime-specific CLI args later in prepare_experiment_parameters.

    Args:
        quant_config: Quantization config with possible arrays or presets
                     Example: {"gemm_dtype": ["auto", "fp8"], "kvcache_dtype": ["auto", "fp8_e5m2"]}
                     Or: {"presets": ["default", "kv-cache-fp8", "dynamic-fp8"]}

    Returns:
        Parameter spec dict in generate_parameter_grid format with original field names
        Example: {"__quant__gemm_dtype": ["auto", "fp8"], "__quant__kvcache_dtype": ["auto", "fp8_e5m2"]}
        Or for presets: {"__quant__preset": ["default", "kv-cache-fp8", "dynamic-fp8"]}

    Note:
        - Single values are wrapped in lists
        - Keys are prefixed with __quant__ to distinguish from regular parameters
        - All values (including 'auto') are kept for grid expansion
        - Presets are stored as __quant__preset for later resolution
    """
    if not quant_config:
        return {}

    # Handle preset mode
    if "presets" in quant_config:
        presets = quant_config["presets"]
        if not isinstance(presets, list):
            presets = [presets]
        return {"__quant__preset": presets}

    # Fields that should be expanded into parameter grid
    dtype_fields = ["gemm_dtype", "kvcache_dtype", "attention_dtype", "moe_dtype"]

    param_spec = {}

    for field in dtype_fields:
        if field not in quant_config:
            continue

        value = quant_config[field]

        # Convert to list if single value
        if not isinstance(value, list):
            value = [value]

        # Skip if empty list
        if not value:
            continue

        # Add with __quant__ prefix to distinguish from regular CLI parameters
        param_spec[f"__quant__{field}"] = value

    return param_spec


def merge_parameters_with_quant_config(
    base_parameters: Dict[str, Any],
    quant_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Merge base parameters with quant_config parameter spec.

    This combines user-defined parameters with quantization parameters,
    preparing a unified parameter specification for grid expansion.

    Args:
        base_parameters: User-defined parameters (e.g., {"tp-size": [1, 2]})
        quant_config: Quantization configuration with arrays

    Returns:
        Merged parameter specification

    Example:
        >>> merge_parameters_with_quant_config(
        ...     {"tp-size": [1, 2]},
        ...     {"gemm_dtype": ["auto", "fp8"], "kvcache_dtype": ["fp8_e5m2"]}
        ... )
        {"tp-size": [1, 2], "__quant__gemm_dtype": ["auto", "fp8"], "__quant__kvcache_dtype": ["fp8_e5m2"]}

    Precedence:
        - Aliased quantization keys in base_parameters (e.g., kv-cache-dtype) override
          both quant_config and any existing __quant__* entries.
    """
    # Start with base parameters
    base_parameters = base_parameters or {}

    quant_param_map = {
        "gemm-dtype": "gemm_dtype",
        "gemm_dtype": "gemm_dtype",
        "kv-cache-dtype": "kvcache_dtype",
        "kv_cache_dtype": "kvcache_dtype",
        "kvcache-dtype": "kvcache_dtype",
        "kvcache_dtype": "kvcache_dtype",
        "attention-dtype": "attention_dtype",
        "attention_dtype": "attention_dtype",
        "moe-dtype": "moe_dtype",
        "moe_dtype": "moe_dtype",
    }

    quant_overrides = {}
    merged = {}

    for key, value in base_parameters.items():
        if key.startswith("__quant__"):
            merged[key] = value
            continue
        mapped = quant_param_map.get(key)
        if mapped:
            quant_overrides[f"__quant__{mapped}"] = value
        else:
            merged[key] = value

    # Add quant_config parameters
    quant_params = expand_quant_config_to_parameter_spec(quant_config)

    # Merge (quant overrides take precedence over quant_config)
    merged.update(quant_params)
    merged.update(quant_overrides)

    return merged


def extract_quant_config_from_params(params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract quantization config from experiment parameters and separate regular parameters.

    Experiment parameters may contain __quant__ prefixed fields from grid expansion.
    This function separates them and reconstructs the quant_config.

    Args:
        params: Experiment parameters dict, may contain __quant__ prefixed keys
                Example: {"tp-size": 1, "__quant__gemm_dtype": "fp8", "__quant__kvcache_dtype": "fp8_e5m2"}
                Or: {"__quant__preset": "kv-cache-fp8"}

    Returns:
        Tuple of (regular_params, quant_config)
        Example: ({"tp-size": 1}, {"gemm_dtype": "fp8", "kvcache_dtype": "fp8_e5m2"})
        Or: ({}, {"preset": "kv-cache-fp8"})
    """
    regular_params = {}
    quant_config = {}

    for key, value in params.items():
        if key.startswith("__quant__"):
            # Extract field name
            field_name = key.replace("__quant__", "")
            quant_config[field_name] = value
        else:
            regular_params[key] = value

    return regular_params, quant_config if quant_config else None


def prepare_runtime_parameters(
    base_runtime: str,
    params: Dict[str, Any],
    model_path: str,
    model_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Prepare runtime-specific CLI parameters from experiment parameters.

    This function:
    1. Extracts quant_config from __quant__ prefixed parameters
    2. Extracts parallel_config from __parallel__ prefixed parameters
    3. Converts both configs to runtime-specific CLI arguments
    4. Merges with regular parameters

    Args:
        base_runtime: Runtime engine ("vllm", "sglang", "tensorrt_llm")
        params: Experiment parameters (may contain __quant__ and __parallel__ prefixed fields)
        model_path: Model path for detecting offline quantization
        model_config: Optional model configuration

    Returns:
        Runtime-ready parameter dict with correct CLI argument names

    Example:
        >>> prepare_runtime_parameters(
        ...     "sglang",
        ...     {"__quant__gemm_dtype": "fp8", "__parallel__tp": 2, "mem-fraction-static": 0.7},
        ...     "meta-llama/Llama-3.2-1B-Instruct"
        ... )
        {"quantization": "fp8", "dtype": "auto", "tp-size": 2, "mem-fraction-static": 0.7}
    """
    # Import here to avoid circular imports
    from utils.parallel_integration import extract_parallel_config_from_params, prepare_runtime_parallel_parameters

    # Separate regular params, quant config, and parallel config
    regular_params, quant_config = extract_quant_config_from_params(params)
    regular_params, parallel_config = extract_parallel_config_from_params(regular_params)

    # Process quantization config
    if quant_config:
        regular_params = prepare_experiment_parameters(
            base_runtime=base_runtime,
            quant_config=quant_config,
            param_combination=regular_params,
            model_path=model_path,
            model_config=model_config
        )

    # Process parallel config
    if parallel_config:
        # Reconstruct params with __parallel__ prefix for prepare_runtime_parallel_parameters
        parallel_prefixed_params = {f"__parallel__{k}": v for k, v in parallel_config.items()}
        parallel_prefixed_params.update(regular_params)
        regular_params = prepare_runtime_parallel_parameters(
            base_runtime=base_runtime,
            params=parallel_prefixed_params
        )

    return regular_params
