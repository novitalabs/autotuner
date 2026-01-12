"""
Pydantic schemas for parameter presets.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Literal
from datetime import datetime


class PresetBase(BaseModel):
	"""Base schema for parameter presets."""

	name: str = Field(..., min_length=1, max_length=255, description="Unique preset name")
	description: Optional[str] = Field(None, description="Preset description")
	category: Optional[str] = Field(None, description="Preset category (e.g., performance, memory, custom)")
	runtime: Optional[Literal["sglang", "vllm"]] = Field(
		None, description="Target runtime (sglang, vllm, or None for universal)"
	)
	parameters: Dict[str, Any] = Field(..., description="Parameter configuration")
	metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class PresetCreate(PresetBase):
	"""Schema for creating a new preset."""

	pass


class PresetUpdate(BaseModel):
	"""Schema for updating an existing preset."""

	name: Optional[str] = Field(None, min_length=1, max_length=255)
	description: Optional[str] = None
	category: Optional[str] = None
	runtime: Optional[Literal["sglang", "vllm"]] = None
	parameters: Optional[Dict[str, Any]] = None
	metadata: Optional[Dict[str, Any]] = None


class PresetResponse(PresetBase):
	"""Schema for preset response."""

	id: int
	is_system: bool
	created_at: datetime
	updated_at: Optional[datetime]

	class Config:
		from_attributes = True
		# Map the database column name to the schema field
		populate_by_name = True

	@classmethod
	def from_orm(cls, obj):
		"""Custom from_orm to handle metadata field mapping."""
		data = {
			"id": obj.id,
			"name": obj.name,
			"description": obj.description,
			"category": obj.category,
			"runtime": obj.runtime,
			"is_system": obj.is_system,
			"parameters": obj.parameters,
			"metadata": obj.preset_metadata,  # Map the field correctly
			"created_at": obj.created_at,
			"updated_at": obj.updated_at,
		}
		return cls(**data)


class PresetMergeRequest(BaseModel):
	"""Schema for merging multiple presets."""

	preset_ids: List[int] = Field(..., min_items=1, description="List of preset IDs to merge")
	merge_strategy: str = Field(
		default="union", pattern="^(union|intersection|last_wins)$", description="Merge strategy"
	)


class PresetMergeResponse(BaseModel):
	"""Schema for merge result."""

	parameters: Dict[str, Any]
	applied_presets: List[str]
	conflicts: Optional[List[Dict[str, Any]]] = None


class PresetExport(BaseModel):
	"""Schema for preset export format."""

	version: str = Field(default="1.0")
	preset: PresetBase
