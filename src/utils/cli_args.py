"""
CLI argument building utilities for runtime parameter conversion.

Consolidates the repeated logic of converting parameter dictionaries
into command-line arguments for Docker/Local controller subprocesses.
"""

from typing import Any, Dict, List, Set


def build_cli_args(
    parameters: Dict[str, Any],
    *,
    skip_prefixes: tuple = ("__",),
    bool_flags: bool = True,
    add_prefix: bool = True
) -> List[str]:
    """
    Convert parameter dictionary to CLI argument list.
    
    Args:
        parameters: Dict of parameter names to values
        skip_prefixes: Tuple of prefixes for parameters to skip (e.g., "__internal")
        bool_flags: If True, boolean True adds flag, False skips it
        add_prefix: If True, add "--" prefix to parameter names without it
        
    Returns:
        List of CLI arguments ready for subprocess
        
    Examples:
        >>> build_cli_args({"model": "llama", "tp-size": 2, "verbose": True})
        ['--model', 'llama', '--tp-size', '2', '--verbose']
        
        >>> build_cli_args({"model": "llama", "quiet": False})
        ['--model', 'llama']
        
        >>> build_cli_args({"__internal": "skip", "public": "keep"})
        ['--public', 'keep']
    """
    args = []
    
    for key, value in parameters.items():
        # Skip internal parameters
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
            
        # Handle parameter name formatting
        if add_prefix and not key.startswith("--"):
            flag = f"--{key}"
        else:
            flag = key
            
        # Handle different value types
        if isinstance(value, bool):
            if bool_flags and value:
                args.append(flag)
            # False booleans are skipped
                
        elif value is None:
            # None values are skipped
            continue
            
        elif isinstance(value, (list, tuple)):
            # Multiple values: --flag val1 --flag val2
            for item in value:
                args.append(flag)
                args.append(str(item))
                
        else:
            # Regular value
            args.append(flag)
            args.append(str(value))
            
    return args


def build_cli_string(
    parameters: Dict[str, Any],
    **kwargs
) -> str:
    """
    Convert parameter dictionary to CLI argument string.
    
    Convenience wrapper around build_cli_args() that joins into a string.
    
    Args:
        parameters: Dict of parameter names to values
        **kwargs: Passed to build_cli_args()
        
    Returns:
        Space-separated CLI argument string
        
    Example:
        >>> build_cli_string({"model": "llama", "tp-size": 2})
        '--model llama --tp-size 2'
    """
    args = build_cli_args(parameters, **kwargs)
    return " ".join(args)


def merge_cli_args(
    base_args: List[str],
    additional_params: Dict[str, Any],
    **kwargs
) -> List[str]:
    """
    Merge base CLI arguments with additional parameters.
    
    Useful when you have a command template and want to add dynamic parameters.
    
    Args:
        base_args: Base argument list (e.g., ['python', '-m', 'vllm.serve'])
        additional_params: Additional parameters to append
        **kwargs: Passed to build_cli_args()
        
    Returns:
        Combined argument list
        
    Example:
        >>> base = ['python', '-m', 'vllm.serve', 'model-name']
        >>> params = {'tp-size': 2, 'gpu-memory-utilization': 0.9}
        >>> merge_cli_args(base, params)
        ['python', '-m', 'vllm.serve', 'model-name', '--tp-size', '2', 
         '--gpu-memory-utilization', '0.9']
    """
    additional_args = build_cli_args(additional_params, **kwargs)
    return base_args + additional_args


def normalize_param_name(name: str, format: str = "cli") -> str:
    """
    Normalize parameter name to different formats.
    
    Args:
        name: Parameter name in any format
        format: Target format ('cli', 'python', 'env')
            - 'cli': kebab-case with -- prefix (--tp-size)
            - 'python': snake_case (tp_size)
            - 'env': UPPER_SNAKE_CASE (TP_SIZE)
            
    Returns:
        Normalized parameter name
        
    Examples:
        >>> normalize_param_name("tp_size", "cli")
        '--tp-size'
        
        >>> normalize_param_name("--tp-size", "python")
        'tp_size'
        
        >>> normalize_param_name("tp-size", "env")
        'TP_SIZE'
    """
    # Remove any existing prefix
    clean_name = name.lstrip("-")
    
    if format == "cli":
        # Convert to kebab-case and add prefix
        kebab = clean_name.replace("_", "-")
        return f"--{kebab}"
        
    elif format == "python":
        # Convert to snake_case
        return clean_name.replace("-", "_")
        
    elif format == "env":
        # Convert to UPPER_SNAKE_CASE
        snake = clean_name.replace("-", "_")
        return snake.upper()
        
    else:
        raise ValueError(f"Unknown format: {format}")


def filter_cli_args(
    args: List[str],
    allowed_params: Set[str],
    prefix: str = "--"
) -> List[str]:
    """
    Filter CLI arguments to only allowed parameters.
    
    Useful when you need to pass a subset of parameters to a specific command.
    
    Args:
        args: Full argument list
        allowed_params: Set of allowed parameter names (without prefix)
        prefix: Parameter prefix to look for
        
    Returns:
        Filtered argument list
        
    Example:
        >>> args = ['--model', 'llama', '--tp-size', '2', '--verbose']
        >>> allowed = {'model', 'tp-size'}
        >>> filter_cli_args(args, allowed)
        ['--model', 'llama', '--tp-size', '2']
    """
    filtered = []
    skip_next = False
    
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
            
        if arg.startswith(prefix):
            param_name = arg[len(prefix):]
            
            if param_name in allowed_params:
                filtered.append(arg)
                # Include the value if there is one
                if i + 1 < len(args) and not args[i + 1].startswith(prefix):
                    filtered.append(args[i + 1])
                    skip_next = True
        else:
            # Non-parameter argument (e.g., positional)
            filtered.append(arg)
            
    return filtered
