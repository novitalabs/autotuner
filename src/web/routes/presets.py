"""
API routes for parameter presets.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import List, Optional
import json

from web.db.session import get_db
from web.db.models import ParameterPreset
from web.schemas.preset import (
	PresetCreate,
	PresetUpdate,
	PresetResponse,
	PresetMergeRequest,
	PresetMergeResponse,
	PresetExport,
)
from utils.preset_merger import PresetMerger, MergeStrategy


router = APIRouter(prefix="/api/presets", tags=["presets"])


@router.get("/", response_model=List[PresetResponse])
async def list_presets(category: Optional[str] = None, db: AsyncSession = Depends(get_db)):
	"""List all parameter presets, optionally filtered by category."""
	query = select(ParameterPreset)

	if category:
		query = query.where(ParameterPreset.category == category)

	result = await db.execute(query)
	presets = result.scalars().all()

	return [PresetResponse.from_orm(p) for p in presets]


@router.get("/{preset_id}", response_model=PresetResponse)
async def get_preset(preset_id: int, db: AsyncSession = Depends(get_db)):
	"""Get a specific preset by ID."""
	result = await db.execute(select(ParameterPreset).where(ParameterPreset.id == preset_id))
	preset = result.scalar_one_or_none()

	if not preset:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Preset with id {preset_id} not found")

	return PresetResponse.from_orm(preset)


@router.post("/", response_model=PresetResponse, status_code=status.HTTP_201_CREATED)
async def create_preset(preset: PresetCreate, db: AsyncSession = Depends(get_db)):
	"""Create a new parameter preset."""
	# Check if name already exists
	result = await db.execute(select(ParameterPreset).where(ParameterPreset.name == preset.name))
	existing = result.scalar_one_or_none()

	if existing:
		raise HTTPException(
			status_code=status.HTTP_409_CONFLICT, detail=f"Preset with name '{preset.name}' already exists"
		)

	# Validate parameters
	errors = PresetMerger.validate_parameters(preset.parameters)
	if errors:
		raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"validation_errors": errors})

	# Create new preset
	db_preset = ParameterPreset(
		name=preset.name,
		description=preset.description,
		category=preset.category,
		runtime=preset.runtime,
		parameters=preset.parameters,
		preset_metadata=preset.metadata,
		is_system=False,
	)

	db.add(db_preset)
	await db.commit()
	await db.refresh(db_preset)

	return PresetResponse.from_orm(db_preset)


@router.put("/{preset_id}", response_model=PresetResponse)
async def update_preset(preset_id: int, preset: PresetUpdate, db: AsyncSession = Depends(get_db)):
	"""Update an existing preset (including system presets)."""
	result = await db.execute(select(ParameterPreset).where(ParameterPreset.id == preset_id))
	db_preset = result.scalar_one_or_none()

	if not db_preset:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Preset with id {preset_id} not found")

	# Update fields
	if preset.name is not None:
		# Check if new name conflicts
		result = await db.execute(
			select(ParameterPreset).where(ParameterPreset.name == preset.name, ParameterPreset.id != preset_id)
		)
		if result.scalar_one_or_none():
			raise HTTPException(
				status_code=status.HTTP_409_CONFLICT, detail=f"Preset with name '{preset.name}' already exists"
			)
		db_preset.name = preset.name

	if preset.description is not None:
		db_preset.description = preset.description

	if preset.category is not None:
		db_preset.category = preset.category

	if preset.runtime is not None:
		db_preset.runtime = preset.runtime

	if preset.parameters is not None:
		# Validate parameters
		errors = PresetMerger.validate_parameters(preset.parameters)
		if errors:
			raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"validation_errors": errors})
		db_preset.parameters = preset.parameters

	if preset.metadata is not None:
		db_preset.preset_metadata = preset.metadata

	await db.commit()
	await db.refresh(db_preset)

	return PresetResponse.from_orm(db_preset)


@router.delete("/{preset_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preset(preset_id: int, db: AsyncSession = Depends(get_db)):
	"""Delete a preset (including system presets)."""
	result = await db.execute(select(ParameterPreset).where(ParameterPreset.id == preset_id))
	preset = result.scalar_one_or_none()

	if not preset:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Preset with id {preset_id} not found")

	await db.execute(delete(ParameterPreset).where(ParameterPreset.id == preset_id))
	await db.commit()


@router.post("/import", response_model=PresetResponse, status_code=status.HTTP_201_CREATED)
async def import_preset(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
	"""Import a preset from a JSON file."""
	if not file.filename.endswith('.json'):
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File must be a JSON file")

	try:
		content = await file.read()
		data = json.loads(content)
	except json.JSONDecodeError:
		raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON file")

	# Validate format
	if "version" not in data or "preset" not in data:
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
			detail="Invalid preset format. Must have 'version' and 'preset' fields",
		)

	preset_data = data["preset"]

	if "name" not in preset_data or "parameters" not in preset_data:
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
			detail="Invalid preset format. Must have 'name' and 'parameters' fields",
		)

	# Check if name already exists
	result = await db.execute(select(ParameterPreset).where(ParameterPreset.name == preset_data["name"]))
	existing = result.scalar_one_or_none()

	if existing:
		raise HTTPException(
			status_code=status.HTTP_409_CONFLICT, detail=f"Preset with name '{preset_data['name']}' already exists"
		)

	# Validate parameters
	errors = PresetMerger.validate_parameters(preset_data["parameters"])
	if errors:
		raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"validation_errors": errors})

	# Create preset
	db_preset = ParameterPreset(
		name=preset_data["name"],
		description=preset_data.get("description"),
		category=preset_data.get("category"),
		parameters=preset_data["parameters"],
		preset_metadata=preset_data.get("metadata"),
		is_system=False,
	)

	db.add(db_preset)
	await db.commit()
	await db.refresh(db_preset)

	return PresetResponse.from_orm(db_preset)


@router.get("/{preset_id}/export")
async def export_preset(preset_id: int, db: AsyncSession = Depends(get_db)):
	"""Export a preset as a JSON file."""
	result = await db.execute(select(ParameterPreset).where(ParameterPreset.id == preset_id))
	preset = result.scalar_one_or_none()

	if not preset:
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Preset with id {preset_id} not found")

	export_data = {
		"version": "1.0",
		"preset": {
			"name": preset.name,
			"description": preset.description,
			"category": preset.category,
			"parameters": preset.parameters,
			"metadata": preset.preset_metadata,
		},
	}

	filename = f"preset-{preset.name.lower().replace(' ', '-')}.json"

	return JSONResponse(content=export_data, headers={"Content-Disposition": f"attachment; filename={filename}"})


@router.post("/merge", response_model=PresetMergeResponse)
async def merge_presets(request: PresetMergeRequest, db: AsyncSession = Depends(get_db)):
	"""Merge multiple presets and return the combined parameters."""
	# Fetch all requested presets
	result = await db.execute(select(ParameterPreset).where(ParameterPreset.id.in_(request.preset_ids)))
	presets = result.scalars().all()

	if len(presets) != len(request.preset_ids):
		found_ids = [p.id for p in presets]
		missing_ids = [pid for pid in request.preset_ids if pid not in found_ids]
		raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Presets not found: {missing_ids}")

	# Convert to dict format for merger
	preset_dicts = [{"name": p.name, "parameters": p.parameters} for p in presets]

	# Merge parameters
	try:
		strategy = MergeStrategy(request.merge_strategy)
	except ValueError:
		raise HTTPException(
			status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid merge strategy: {request.merge_strategy}"
		)

	merged_params, conflicts = PresetMerger.merge_parameters(preset_dicts, strategy)

	return PresetMergeResponse(
		parameters=merged_params, applied_presets=[p.name for p in presets], conflicts=conflicts if conflicts else None
	)
