"""
High-level parameter preset management tools for agent.

These tools provide business-friendly operations for managing parameter presets.
They wrap REST API logic and do NOT require authorization (safe business operations).

All tools use shared Service layer (web.services) for consistency with REST API.
"""

from langchain_core.tools import tool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from web.services import PresetService
from web.agent.tools.base import register_tool, ToolCategory
from web.db.models import ParameterPreset
import json
from typing import Optional


@tool
@register_tool(ToolCategory.PRESET)
async def create_preset(
    name: str,
    parameters: dict,
    description: str = None,
    category: str = None,
    runtime: str = None,
    db: AsyncSession = None
) -> str:
    """
    Create a new parameter preset.

    Args:
        name: Unique preset name
        parameters: Parameter dict, e.g., {"tp-size": 2, "mem-fraction-static": 0.85}
        description: Preset description (optional)
        category: Category (e.g., "performance", "memory", "latency") (optional)
        runtime: Runtime ("sglang" or "vllm") or None for universal (optional)

    Returns:
        JSON string with created preset details including ID
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Check if name already exists
    result = await db.execute(select(ParameterPreset).where(ParameterPreset.name == name))
    existing = result.scalar_one_or_none()

    if existing:
        return json.dumps({"error": f"Preset '{name}' already exists"})

    # Validate parameters (basic check)
    if not isinstance(parameters, dict) or not parameters:
        return json.dumps({"error": "Parameters must be a non-empty dictionary"})

    # Create preset
    db_preset = ParameterPreset(
        name=name,
        description=description,
        category=category,
        runtime=runtime,
        parameters=parameters,
        is_system=False
    )

    db.add(db_preset)
    await db.commit()
    await db.refresh(db_preset)

    return json.dumps({
        "success": True,
        "preset": db_preset.to_dict()
    }, indent=2)


@tool
@register_tool(ToolCategory.PRESET)
async def update_preset(
    preset_id: int,
    name: str = None,
    description: str = None,
    category: str = None,
    runtime: str = None,
    parameters: dict = None,
    db: AsyncSession = None
) -> str:
    """
    Update an existing preset.

    Args:
        preset_id: Preset ID to update
        name: New name (optional)
        description: New description (optional)
        category: New category (optional)
        runtime: New runtime (optional)
        parameters: New parameters (optional)

    Returns:
        JSON string with updated preset
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Get preset
    result = await db.execute(select(ParameterPreset).where(ParameterPreset.id == preset_id))
    preset = result.scalar_one_or_none()

    if not preset:
        return json.dumps({"error": f"Preset {preset_id} not found"})

    # Update fields if provided
    if name is not None:
        # Check if new name conflicts
        result = await db.execute(
            select(ParameterPreset).where(
                ParameterPreset.name == name,
                ParameterPreset.id != preset_id
            )
        )
        if result.scalar_one_or_none():
            return json.dumps({"error": f"Preset '{name}' already exists"})
        preset.name = name

    if description is not None:
        preset.description = description

    if category is not None:
        preset.category = category

    if runtime is not None:
        preset.runtime = runtime

    if parameters is not None:
        if not isinstance(parameters, dict) or not parameters:
            return json.dumps({"error": "Parameters must be a non-empty dictionary"})
        preset.parameters = parameters

    await db.commit()
    await db.refresh(preset)

    return json.dumps({
        "success": True,
        "message": f"Preset {preset_id} updated successfully",
        "preset": preset.to_dict()
    }, indent=2)


@tool
@register_tool(ToolCategory.PRESET)
async def delete_preset(preset_id: int, db: AsyncSession = None) -> str:
    """
    Delete a parameter preset.

    Args:
        preset_id: Preset ID to delete

    Returns:
        JSON string with success message
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    # Get preset
    result = await db.execute(select(ParameterPreset).where(ParameterPreset.id == preset_id))
    preset = result.scalar_one_or_none()

    if not preset:
        return json.dumps({"error": f"Preset {preset_id} not found"})

    preset_name = preset.name
    await db.execute(delete(ParameterPreset).where(ParameterPreset.id == preset_id))
    await db.commit()

    return json.dumps({
        "success": True,
        "message": f"Preset '{preset_name}' (ID: {preset_id}) deleted successfully"
    })


@tool
@register_tool(ToolCategory.PRESET)
async def get_preset_by_id(preset_id: int, db: AsyncSession = None) -> str:
    """
    Get a specific preset by ID.

    Args:
        preset_id: Preset ID to retrieve

    Returns:
        JSON string with preset details
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    preset = await PresetService.get_preset_by_id(db, preset_id)
    if not preset:
        return json.dumps({"error": f"Preset {preset_id} not found"})

    return json.dumps(preset.to_dict(), indent=2)


@tool
@register_tool(ToolCategory.PRESET)
async def get_preset_by_name(name: str, db: AsyncSession = None) -> str:
    """
    Get a specific preset by name.

    Args:
        name: Preset name to retrieve

    Returns:
        JSON string with preset details
    """
    if db is None:
        return json.dumps({"error": "Database session not provided"})

    result = await db.execute(select(ParameterPreset).where(ParameterPreset.name == name))
    preset = result.scalar_one_or_none()

    if not preset:
        return json.dumps({"error": f"Preset '{name}' not found"})

    return json.dumps(preset.to_dict(), indent=2)
