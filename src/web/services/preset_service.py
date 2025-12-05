"""
Preset service - Business logic for parameter preset operations.

Shared between REST API routes and agent tools.
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from web.db.models import ParameterPreset


class PresetService:
	"""Service for parameter preset-related business logic."""

	@staticmethod
	async def list_presets(
		db: AsyncSession,
		category: Optional[str] = None,
		runtime: Optional[str] = None,
		skip: int = 0,
		limit: int = 100
	) -> List[ParameterPreset]:
		"""
		List parameter presets with optional filtering.

		Args:
			db: Database session
			category: Optional category filter
			runtime: Optional runtime filter
			skip: Number of records to skip
			limit: Maximum number of records to return

		Returns:
			List of ParameterPreset objects
		"""
		query = select(ParameterPreset)

		if category:
			query = query.where(ParameterPreset.category == category)

		if runtime:
			query = query.where(ParameterPreset.runtime == runtime)

		query = query.offset(skip).limit(limit)
		result = await db.execute(query)
		return result.scalars().all()

	@staticmethod
	async def get_preset_by_id(db: AsyncSession, preset_id: int) -> Optional[ParameterPreset]:
		"""
		Get preset by ID.

		Args:
			db: Database session
			preset_id: Preset ID

		Returns:
			ParameterPreset object or None if not found
		"""
		result = await db.execute(select(ParameterPreset).where(ParameterPreset.id == preset_id))
		return result.scalar_one_or_none()

	@staticmethod
	async def get_preset_by_name(db: AsyncSession, name: str) -> Optional[ParameterPreset]:
		"""
		Get preset by name.

		Args:
			db: Database session
			name: Preset name

		Returns:
			ParameterPreset object or None if not found
		"""
		result = await db.execute(select(ParameterPreset).where(ParameterPreset.name == name))
		return result.scalar_one_or_none()

	@staticmethod
	async def create_preset(db: AsyncSession, preset: ParameterPreset) -> ParameterPreset:
		"""
		Create a new preset.

		Args:
			db: Database session
			preset: ParameterPreset object to create

		Returns:
			Created preset with ID assigned
		"""
		db.add(preset)
		await db.commit()
		await db.refresh(preset)
		return preset

	@staticmethod
	async def update_preset(db: AsyncSession, preset: ParameterPreset) -> ParameterPreset:
		"""
		Update an existing preset.

		Args:
			db: Database session
			preset: ParameterPreset object with updates

		Returns:
			Updated preset
		"""
		await db.commit()
		await db.refresh(preset)
		return preset

	@staticmethod
	async def delete_preset(db: AsyncSession, preset_id: int) -> bool:
		"""
		Delete a preset.

		Args:
			db: Database session
			preset_id: Preset ID to delete

		Returns:
			True if deleted, False if not found
		"""
		preset = await PresetService.get_preset_by_id(db, preset_id)
		if not preset:
			return False

		await db.delete(preset)
		await db.commit()
		return True
