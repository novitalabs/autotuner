"""
System tools for agent.

Provides utilities like sleep/wait for task coordination.
"""

from langchain_core.tools import tool
from web.agent.tools.base import register_tool, ToolCategory
import asyncio
import json


@tool
@register_tool(ToolCategory.SYSTEM)
async def sleep(seconds: float) -> str:
    """
    Wait for a specified number of seconds before continuing.

    Use this tool when you need to:
    - Wait for a task to make progress before checking status again
    - Pause between operations to avoid overwhelming the system
    - Give time for background processes to complete

    Args:
        seconds: Number of seconds to wait (1-300, max 5 minutes)

    Returns:
        JSON string confirming the wait completed

    Example usage:
        - After starting a task, sleep 30 seconds then check status
        - Wait between multiple API calls to avoid rate limiting
        - Pause while waiting for a deployment to stabilize
    """
    # Validate and clamp the sleep duration
    if seconds < 1:
        seconds = 1
    elif seconds > 300:
        seconds = 300  # Max 5 minutes

    await asyncio.sleep(seconds)

    return json.dumps({
        "success": True,
        "message": f"Waited {seconds} seconds",
        "seconds_waited": seconds
    })


@tool
@register_tool(ToolCategory.SYSTEM)
async def get_current_time() -> str:
    """
    Get the current server time.

    Useful for:
    - Timestamping operations
    - Calculating elapsed time between checks
    - Logging and debugging

    Returns:
        JSON string with current time in various formats
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    return json.dumps({
        "success": True,
        "utc_iso": now.isoformat(),
        "utc_timestamp": now.timestamp(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S UTC")
    })
