"""
Shared Controller Utilities

This module provides common utilities used across different controller implementations
(Docker, Local, OME) to avoid code duplication and ensure consistency.

Key utilities:
- Name sanitization for different deployment modes
- Port allocation
- Parallel configuration parsing
- Proxy environment setup
- Parameter conversion
- Runtime configuration
"""

import re
import socket
from typing import Dict, Any, Optional, List


# ============================================================================
# Name Sanitization
# ============================================================================

def sanitize_name_generic(name: str, allow_periods: bool = False) -> str:
    """
    Generic name sanitization for use across controllers.

    Rules:
    - Lowercase letters, numbers, '-', and optionally '.' only
    - Must start with a lowercase letter (or alphanumeric if allow_periods=True)
    - Must end with alphanumeric character
    - Multiple consecutive dashes replaced with single dash

    Args:
        name: The name to sanitize
        allow_periods: If True, allow periods (for Docker); if False, replace with dash (for K8s DNS)

    Returns:
        Sanitized name
    """
    # Convert to lowercase
    name = name.lower()

    # Replace invalid characters
    if allow_periods:
        # Docker allows periods
        name = re.sub(r'[^a-z0-9-._]', '-', name)
    else:
        # Kubernetes DNS-1123: no periods
        name = re.sub(r'[^a-z0-9-]', '-', name)

    # Remove leading non-letters (must start with letter for DNS compliance)
    name = re.sub(r'^[^a-z]+', '', name)

    # Remove trailing non-alphanumeric
    name = re.sub(r'[^a-z0-9]+$', '', name)

    # Replace multiple consecutive dashes with single dash
    name = re.sub(r'-+', '-', name)

    # Truncate to 253 characters (DNS limit)
    name = name[:253]

    # Ensure name starts with a letter (if empty after sanitization, use 'task')
    if not name or not name[0].isalpha():
        name = 'task-' + name

    # Remove trailing non-alphanumeric characters again (in case 'task-' prefix left trailing dash)
    name = re.sub(r'[^a-z0-9]+$', '', name)

    # Final safety check: if still empty or invalid, return 'task'
    if not name or not name[0].isalpha():
        name = 'task'

    return name


def sanitize_container_name(name: str) -> str:
    """
    Sanitize name for Docker container naming.
    Allows periods for Docker compatibility.

    Args:
        name: The name to sanitize

    Returns:
        Container-safe name
    """
    return sanitize_name_generic(name, allow_periods=True)


def sanitize_service_name(name: str) -> str:
    """
    Sanitize name for local service identification.
    No periods allowed for consistency with Kubernetes.

    Args:
        name: The name to sanitize

    Returns:
        Service-safe name
    """
    return sanitize_name_generic(name, allow_periods=False)


def sanitize_dns_name(name: str) -> str:
    """
    Sanitize name to be OME webhook compliant (DNS-1123).

    OME webhook requires names to match: [a-z]([-a-z0-9]*[a-z0-9])?
    Rules:
    - lowercase letters, numbers, '-' only (NO periods)
    - must start with a lowercase letter
    - must end with alphanumeric character
    - max 253 characters

    Args:
        name: The name to sanitize

    Returns:
        OME-compliant name
    """
    return sanitize_name_generic(name, allow_periods=False)


# ============================================================================
# Port Allocation
# ============================================================================

def find_available_port(start_port: int, end_port: int) -> Optional[int]:
    """
    Find an available port in the specified range.

    Args:
        start_port: Start of port range
        end_port: End of port range (inclusive)

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


# ============================================================================
# Parallel Configuration Parsing
# ============================================================================

def parse_parallel_config(parameters: Dict[str, Any]) -> Dict[str, int]:
    """
    Parse parallel configuration from parameters.

    Handles multiple parameter naming conventions:
    - tensor-parallel-size, tp-size, tp_size
    - pipeline-parallel-size, pp-size, pp_size
    - data-parallel-size, dp-size, dp_size
    - context-parallel-size, cp-size, cp_size
    - decode-context-parallel-size, dcp-size, dcp_size

    Args:
        parameters: Runtime parameters dictionary

    Returns:
        Dictionary with normalized keys: {tp, pp, dp, cp, dcp, world_size}
    """
    # Parse TP (tensor parallel)
    tp = parameters.get("tensor-parallel-size",
                       parameters.get("tp-size",
                                    parameters.get("tp_size", 1)))

    # Parse PP (pipeline parallel)
    pp = parameters.get("pipeline-parallel-size",
                       parameters.get("pp-size",
                                    parameters.get("pp_size", 1)))

    # Parse DP (data parallel)
    dp = parameters.get("data-parallel-size",
                       parameters.get("dp-size",
                                    parameters.get("dp_size", 1)))

    # Parse CP (context parallel)
    cp = parameters.get("context-parallel-size",
                       parameters.get("cp-size",
                                    parameters.get("cp_size", 1)))

    # Parse DCP (decode context parallel)
    dcp = parameters.get("decode-context-parallel-size",
                        parameters.get("dcp-size",
                                     parameters.get("dcp_size", 1)))

    # Convert to integers with validation
    def safe_int_conversion(value, param_name, default=1):
        """Safely convert value to integer with validation."""
        try:
            result = int(value) if isinstance(value, (int, float, str)) else default
            # Clamp to >= 1 (parallelism size must be at least 1)
            if result < 1:
                print(f"[parse_parallel_config] Warning: {param_name}={result} is invalid (< 1), clamping to 1")
                result = 1
            return result
        except (ValueError, TypeError) as e:
            print(f"[parse_parallel_config] Warning: Failed to convert {param_name}={value} to int: {e}, using default={default}")
            return default

    tp = safe_int_conversion(tp, "tp", default=1)
    pp = safe_int_conversion(pp, "pp", default=1)
    dp = safe_int_conversion(dp, "dp", default=1)
    cp = safe_int_conversion(cp, "cp", default=1)
    dcp = safe_int_conversion(dcp, "dcp", default=1)

    # Calculate world_size
    # For vLLM/SGLang: world_size = tp × pp × max(dp, dcp, cp)
    # For TensorRT-LLM: world_size = tp × pp × cp (no dp support)
    world_size = tp * pp * max(dp, dcp, cp)

    return {
        "tp": tp,
        "pp": pp,
        "dp": dp,
        "cp": cp,
        "dcp": dcp,
        "world_size": world_size
    }


# ============================================================================
# Environment Setup
# ============================================================================

def setup_proxy_environment(
    base_env: Dict[str, str],
    http_proxy: str = "",
    https_proxy: str = "",
    no_proxy: str = "",
    hf_token: str = ""
) -> Dict[str, str]:
    """
    Setup proxy and HuggingFace token environment variables.

    Args:
        base_env: Base environment dictionary to extend
        http_proxy: HTTP proxy URL (optional)
        https_proxy: HTTPS proxy URL (optional)
        no_proxy: Comma-separated list of hosts to bypass proxy (optional)
        hf_token: HuggingFace access token (optional)

    Returns:
        Extended environment dictionary
    """
    env = base_env.copy()

    # Add proxy settings if configured
    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy

    if https_proxy:
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy

    if no_proxy:
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy

    # Add HuggingFace token if configured
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token  # Alternative name some libraries use

    return env


# ============================================================================
# Parameter Conversion
# ============================================================================

def build_param_string(parameters: Dict[str, Any]) -> str:
    """
    Convert parameters dictionary to CLI argument string.

    Handles:
    - Boolean parameters (true = flag, false = skip)
    - String/numeric parameters (--key value)
    - Parameter name normalization (add -- prefix if missing)

    Args:
        parameters: Parameters dictionary

    Returns:
        CLI argument string (e.g., "--tp-size 4 --enable-mixed-chunk")
    """
    parts = []

    for param_name, param_value in parameters.items():
        # Skip internal parameters
        if param_name.startswith("__"):
            continue

        # Convert parameter name to CLI format (add -- prefix if not present)
        if not param_name.startswith("--"):
            cli_param = f"--{param_name}"
        else:
            cli_param = param_name

        # Handle boolean parameters specially
        # - If false: skip the parameter entirely (don't add to command)
        # - If true: add parameter flag without value (e.g., --enable-mixed-chunk)
        # - Otherwise: add parameter with value (e.g., --tp-size 4)
        if isinstance(param_value, bool):
            if param_value:  # Only add flag if True
                parts.append(cli_param)
            # If False, skip this parameter entirely
        else:
            parts.append(f"{cli_param} {param_value}")

    return " ".join(parts)


def build_param_list(parameters: Dict[str, Any]) -> List[str]:
    """
    Convert parameters dictionary to CLI argument list.

    Similar to build_param_string but returns a list suitable for subprocess.

    Args:
        parameters: Parameters dictionary

    Returns:
        CLI argument list (e.g., ["--tp-size", "4", "--enable-mixed-chunk"])
    """
    parts = []

    for param_name, param_value in parameters.items():
        # Skip internal parameters
        if param_name.startswith("__"):
            continue

        # Convert parameter name to CLI format
        if not param_name.startswith("--"):
            cli_param = f"--{param_name}"
        else:
            cli_param = param_name

        # Handle boolean parameters
        if isinstance(param_value, bool):
            if param_value:
                parts.append(cli_param)
        else:
            parts.extend([cli_param, str(param_value)])

    return parts


# ============================================================================
# Runtime Configuration
# ============================================================================

# Runtime configurations mapping runtime names to Docker images and command templates
RUNTIME_CONFIGS = {
    "sglang": {
        "image": "lmsysorg/sglang:v0.5.2-cu126",
        "command": "python3 -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --port {port}",
        "module": "-m sglang.launch_server",
        "model_param": "--model-path"
    },
    "vllm": {
        "image": "vllm/vllm-openai:latest",
        # New vLLM image uses 'vllm serve' as entrypoint, so just pass args
        "command": "{model_path} --host 0.0.0.0 --port {port}",
        "module": "",  # Not needed for new image
        "model_param": ""  # Model is positional arg now
    },
}


def get_runtime_config(
    runtime_name: str,
    image_tag: Optional[str] = None
) -> Optional[Dict[str, str]]:
    """
    Get runtime configuration (image, command template, etc.) for a given runtime.

    Supports:
    - Exact match or prefix match (e.g., "sglang" matches "sglang-v1.2")
    - Custom image tag override

    Args:
        runtime_name: Runtime identifier (e.g., 'sglang', 'vllm')
        image_tag: Optional Docker image tag to override default

    Returns:
        Dictionary with 'image', 'command', 'module', 'model_param' keys,
        or None if unsupported runtime
    """
    # Try exact match or prefix match
    config = None
    for key, cfg in RUNTIME_CONFIGS.items():
        if runtime_name.lower().startswith(key):
            config = cfg.copy()
            break

    if not config:
        return None

    # Override image tag if provided
    if image_tag:
        # Extract base image name (before colon)
        base_image = config["image"].split(":")[0]
        config["image"] = f"{base_image}:{image_tag}"

    return config


# ============================================================================
# GPU Configuration Helper
# ============================================================================

def format_gpu_devices_for_cuda(gpu_indices: List[int]) -> str:
    """
    Format GPU indices for CUDA_VISIBLE_DEVICES environment variable.

    For multi-GPU deployments, maps Docker's allocated GPU indices to
    consecutive indices (0, 1, 2, ...) expected by inference engines.

    Args:
        gpu_indices: List of GPU indices allocated by Docker/system

    Returns:
        Comma-separated string (e.g., "0,1,2,3")
    """
    # Map to consecutive indices starting from 0
    return ",".join(str(i) for i in range(len(gpu_indices)))
