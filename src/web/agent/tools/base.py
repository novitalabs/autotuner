"""
Base classes and utilities for agent tools.

This module defines the core abstractions for the tool system:
- Authorization scopes for privileged operations
- Tool categories for organization
- Decorator for tool registration
"""

import enum
from typing import Callable, Any
from functools import wraps


class AuthorizationScope(str, enum.Enum):
    """Authorization scopes for privileged tools."""
    BASH_COMMANDS = "bash_commands"
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    DOCKER_CONTROL = "docker_control"
    KUBECTL_OPERATIONS = "kubectl_operations"
    DATABASE_WRITE = "database_write"
    ARQ_CONTROL = "arq_control"  # Managing ARQ workers


class ToolCategory(str, enum.Enum):
    """Categories for organizing tools."""
    DATABASE = "database"
    SYSTEM = "system"
    FILE = "file"
    API = "api"
    TASK = "task"
    EXPERIMENT = "experiment"
    PRESET = "preset"


def register_tool(
    category: ToolCategory,
    requires_auth: bool = False,
    auth_scope: AuthorizationScope = None
):
    """
    Decorator to register tools with metadata.

    Args:
        category: Tool category for organization
        requires_auth: Whether tool requires user authorization
        auth_scope: Authorization scope if requires_auth is True

    Example:
        @tool
        @register_tool(ToolCategory.DATABASE)
        async def list_tasks(db: AsyncSession = None) -> str:
            ...

        @tool
        @register_tool(
            ToolCategory.SYSTEM,
            requires_auth=True,
            auth_scope=AuthorizationScope.BASH_COMMANDS
        )
        async def execute_bash(command: str) -> str:
            ...
    """
    def decorator(func: Callable) -> Callable:
        # Attach metadata to function
        func._tool_category = category
        func._requires_authorization = requires_auth
        func._authorization_scope = auth_scope

        # Validate: if requires_auth, must provide auth_scope
        if requires_auth and auth_scope is None:
            raise ValueError(
                f"Tool {func.__name__} requires authorization but no auth_scope provided"
            )

        return func

    return decorator
