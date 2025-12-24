"""
Name sanitization utilities for different deployment modes.

Standardizes naming across Docker, Local, and OME (Kubernetes) controllers
to ensure valid container/service/resource names.
"""

import re
from typing import Optional


def sanitize_name(
    name: str,
    *,
    allowed_pattern: str = r"[^a-z0-9-]",
    max_len: Optional[int] = None,
    must_start_letter: bool = False,
    allow_underscore: bool = False
) -> str:
    """
    Generic name sanitization with configurable rules.
    
    Args:
        name: Input name to sanitize
        allowed_pattern: Regex pattern for characters to replace (default: anything not a-z0-9-)
        max_len: Maximum length (None for no limit)
        must_start_letter: If True, ensure name starts with a letter
        allow_underscore: If True, allow underscores in name
        
    Returns:
        Sanitized name string
    """
    # Convert to lowercase
    name = name.lower()
    
    # Adjust pattern if underscores are allowed
    if allow_underscore:
        pattern = r"[^a-z0-9_-]"
    else:
        pattern = allowed_pattern
    
    # Replace invalid characters with hyphens
    name = re.sub(pattern, "-", name)
    
    # Trim leading invalid characters
    if must_start_letter:
        name = re.sub(r"^[^a-z]+", "", name)
    else:
        name = re.sub(r"^[^a-z0-9]+", "", name)
    
    # Trim trailing invalid characters
    name = re.sub(r"[^a-z0-9]+$", "", name)
    
    # Collapse multiple consecutive hyphens
    name = re.sub(r"-+", "-", name)
    
    # Ensure name starts with letter if required
    if must_start_letter and name and not name[0].isalpha():
        name = "task-" + name
    
    # Apply length limit
    if max_len and len(name) > max_len:
        name = name[:max_len]
        # Re-trim trailing invalid characters after truncation
        name = re.sub(r"[^a-z0-9]+$", "", name)
    
    # Ensure we have something
    if not name:
        name = "unnamed"
    
    return name


def sanitize_docker_name(name: str) -> str:
    """
    Sanitize name for Docker container/volume naming.
    
    Docker naming rules:
    - Can contain: a-z, 0-9, _, -, .
    - Can start with anything
    - Typical max length: 255 chars (generous)
    
    Args:
        name: Input name
        
    Returns:
        Docker-compatible name
        
    Examples:
        >>> sanitize_docker_name("My Task #1")
        'my-task-1'
        
        >>> sanitize_docker_name("task_with_underscores")
        'task_with_underscores'
    """
    return sanitize_name(
        name,
        allowed_pattern=r"[^a-z0-9_.-]",
        max_len=255,
        must_start_letter=False,
        allow_underscore=True
    )


def sanitize_local_id(name: str) -> str:
    """
    Sanitize name for local subprocess identification.
    
    Similar to Docker but used for process tracking.
    
    Args:
        name: Input name
        
    Returns:
        Local-compatible identifier
        
    Examples:
        >>> sanitize_local_id("My Task #1")
        'my-task-1'
    """
    return sanitize_name(
        name,
        allowed_pattern=r"[^a-z0-9_-]",
        max_len=255,
        must_start_letter=False,
        allow_underscore=True
    )


def sanitize_k8s_name(name: str) -> str:
    """
    Sanitize name for Kubernetes resource naming (DNS label format).
    
    Kubernetes DNS label rules (RFC 1123):
    - Can contain: a-z, 0-9, - (no underscores!)
    - Must start and end with alphanumeric
    - Max length: 63 characters
    
    Args:
        name: Input name
        
    Returns:
        Kubernetes DNS-compatible name
        
    Examples:
        >>> sanitize_k8s_name("My Task #1")
        'my-task-1'
        
        >>> sanitize_k8s_name("123-task")
        'task-123-task'
        
        >>> sanitize_k8s_name("task_with_underscores")
        'task-with-underscores'
    """
    return sanitize_name(
        name,
        allowed_pattern=r"[^a-z0-9-]",
        max_len=63,
        must_start_letter=True,
        allow_underscore=False
    )


# Convenient aliases
sanitize_container_name = sanitize_docker_name
sanitize_service_name = sanitize_local_id
sanitize_dns_name = sanitize_k8s_name
sanitize_ome_name = sanitize_k8s_name


def generate_unique_name(base_name: str, suffix: str, sanitizer=sanitize_name) -> str:
    """
    Generate a unique name by combining base name and suffix.
    
    Args:
        base_name: Base name (e.g., task name)
        suffix: Suffix to add (e.g., timestamp, random string)
        sanitizer: Sanitization function to use
        
    Returns:
        Sanitized unique name
        
    Examples:
        >>> generate_unique_name("my-task", "abc123", sanitize_docker_name)
        'my-task-abc123'
        
        >>> generate_unique_name("My Task", "20231224", sanitize_k8s_name)
        'task-my-task-20231224'
    """
    combined = f"{base_name}-{suffix}"
    return sanitizer(combined)
