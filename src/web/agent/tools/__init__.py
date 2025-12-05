"""
Tool registry for agent tools.

Provides centralized management of all available tools with automatic discovery.
"""

from typing import List, Dict
from langchain_core.tools import BaseTool
import logging

logger = logging.getLogger(__name__)


class ToolsRegistry:
    """Central registry for all agent tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._load_all_tools()

    def _load_all_tools(self):
        """Load all tools from modules."""
        # Import tool modules
        try:
            from . import database_tools, system_tools, file_tools, api_tools
        except ImportError as e:
            logger.warning(f"Failed to import tool modules: {e}")
            return

        # Auto-discover tools with @tool decorator and metadata
        for module in [database_tools, system_tools, file_tools, api_tools]:
            for name in dir(module):
                # Skip private attributes
                if name.startswith('_'):
                    continue

                obj = getattr(module, name)

                # Check if it's a LangChain StructuredTool
                if hasattr(obj, 'coroutine') and callable(obj.coroutine):
                    # The original function is in the coroutine attribute
                    original_func = obj.coroutine
                    if hasattr(original_func, '_tool_category'):
                        # Copy metadata from original function to tool object
                        obj._tool_category = original_func._tool_category
                        obj._requires_authorization = original_func._requires_authorization
                        obj._authorization_scope = getattr(original_func, '_authorization_scope', None)
                        self._tools[name] = obj
                        logger.info(f"Registered tool: {name} (category: {obj._tool_category})")
                elif hasattr(obj, '_tool_category'):
                    # Direct function with metadata (fallback)
                    self._tools[name] = obj
                    logger.info(f"Registered tool: {name} (category: {obj._tool_category})")

    def get_all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_safe_tools(self) -> List[BaseTool]:
        """Get tools that don't require authorization."""
        return [
            t for t in self._tools.values()
            if not getattr(t, '_requires_authorization', False)
        ]

    def get_privileged_tools(self) -> List[BaseTool]:
        """Get tools that require authorization."""
        return [
            t for t in self._tools.values()
            if getattr(t, '_requires_authorization', False)
        ]

    def get_tool(self, name: str) -> BaseTool:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def get_tool_count(self) -> int:
        """Get total number of registered tools."""
        return len(self._tools)


# Global singleton
_registry = None


def get_tools_registry() -> ToolsRegistry:
    """Get the global tools registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolsRegistry()
        logger.info(f"Initialized tools registry with {_registry.get_tool_count()} tools")
    return _registry


__all__ = [
    'ToolsRegistry',
    'get_tools_registry',
]
