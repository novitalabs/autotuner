"""
Environment variable building utilities.

Consolidates environment setup logic from Docker and Local controllers,
particularly for proxy settings and HuggingFace configuration.
"""

from typing import Dict, Optional, Mapping
import os


def build_inference_env(
    base_env: Optional[Mapping[str, str]] = None,
    *,
    http_proxy: str = "",
    https_proxy: str = "",
    no_proxy: str = "",
    hf_token: str = "",
    hf_home: Optional[str] = None,
    cuda_visible_devices: Optional[str] = None,
    additional: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Build environment variables for inference server processes.
    
    Consolidates common environment setup:
    - Proxy configuration (with both upper and lowercase variants)
    - HuggingFace authentication and cache
    - CUDA device selection
    - Additional custom variables
    
    Args:
        base_env: Base environment to extend (defaults to os.environ)
        http_proxy: HTTP proxy URL
        https_proxy: HTTPS proxy URL
        no_proxy: No-proxy hosts (comma-separated)
        hf_token: HuggingFace API token
        hf_home: HuggingFace cache directory
        cuda_visible_devices: CUDA device IDs (comma-separated)
        additional: Additional environment variables to set
        
    Returns:
        Dictionary of environment variables ready for subprocess
        
    Example:
        >>> env = build_inference_env(
        ...     http_proxy="http://proxy:1081",
        ...     hf_token="hf_xxxxx",
        ...     cuda_visible_devices="0,1"
        ... )
        >>> # Use in subprocess:
        >>> subprocess.run(["python", "server.py"], env=env)
    """
    # Start with base environment
    if base_env is None:
        env = dict(os.environ)
    else:
        env = dict(base_env)
    
    # Proxy settings (both upper and lowercase for compatibility)
    if http_proxy:
        env["HTTP_PROXY"] = http_proxy
        env["http_proxy"] = http_proxy
        
    if https_proxy:
        env["HTTPS_PROXY"] = https_proxy
        env["https_proxy"] = https_proxy
        
    if no_proxy:
        env["NO_PROXY"] = no_proxy
        env["no_proxy"] = no_proxy
    
    # HuggingFace configuration
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
        
    if hf_home:
        env["HF_HOME"] = hf_home
    
    # CUDA device selection
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    # Additional custom variables
    if additional:
        env.update(additional)
    
    return env


def mask_sensitive_env(env: Dict[str, str]) -> Dict[str, str]:
    """
    Create a copy of environment with sensitive values masked.
    
    Useful for logging environment without exposing secrets.
    
    Args:
        env: Environment dictionary
        
    Returns:
        New dict with sensitive values replaced with "***"
        
    Example:
        >>> env = {"HF_TOKEN": "hf_secret123", "PATH": "/usr/bin"}
        >>> safe_env = mask_sensitive_env(env)
        >>> print(safe_env)
        {'HF_TOKEN': '***', 'PATH': '/usr/bin'}
    """
    sensitive_keys = {
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "API_KEY",
        "SECRET",
        "PASSWORD",
        "TOKEN",
    }
    
    masked = {}
    for key, value in env.items():
        # Check if key contains sensitive words
        is_sensitive = any(
            sensitive_word in key.upper()
            for sensitive_word in sensitive_keys
        )
        
        if is_sensitive:
            masked[key] = "***"
        else:
            masked[key] = value
            
    return masked


def format_env_for_display(env: Dict[str, str], mask_secrets: bool = True) -> str:
    """
    Format environment variables for logging/display.
    
    Args:
        env: Environment dictionary
        mask_secrets: Whether to mask sensitive values
        
    Returns:
        Human-readable string representation
        
    Example:
        >>> env = {"HF_TOKEN": "hf_secret", "PATH": "/usr/bin"}
        >>> print(format_env_for_display(env))
        HF_TOKEN=***
        PATH=/usr/bin
    """
    if mask_secrets:
        env = mask_sensitive_env(env)
        
    lines = [f"{key}={value}" for key, value in sorted(env.items())]
    return "\n".join(lines)


def get_proxy_env() -> Dict[str, str]:
    """
    Extract proxy settings from current environment.
    
    Returns:
        Dict with proxy settings (empty if not set)
        
    Example:
        >>> proxy_env = get_proxy_env()
        >>> if proxy_env:
        ...     print("Proxy configured:", proxy_env.get("HTTP_PROXY"))
    """
    proxy_vars = ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", 
                  "http_proxy", "https_proxy", "no_proxy"]
    
    return {
        key: value
        for key, value in os.environ.items()
        if key in proxy_vars
    }


def merge_env(
    base: Dict[str, str],
    override: Dict[str, str],
    replace: bool = False
) -> Dict[str, str]:
    """
    Merge two environment dictionaries.
    
    Args:
        base: Base environment
        override: Values to add/override
        replace: If True, completely replace base. If False, merge/update.
        
    Returns:
        Merged environment dictionary
        
    Example:
        >>> base = {"PATH": "/usr/bin", "USER": "alice"}
        >>> override = {"PATH": "/opt/bin:/usr/bin", "LANG": "en_US"}
        >>> merged = merge_env(base, override)
        >>> merged["PATH"]
        '/opt/bin:/usr/bin'
        >>> merged["USER"]
        'alice'
    """
    if replace:
        return dict(override)
    else:
        result = dict(base)
        result.update(override)
        return result
