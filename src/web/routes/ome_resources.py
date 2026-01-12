"""
API routes for OME resources (ClusterBaseModel, ClusterServingRuntime).
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config.clusterbasemodel_presets as clusterbasemodel_presets
import config.clusterservingruntime_presets as clusterservingruntime_presets
from controllers.ome_controller import OMEController

router = APIRouter(prefix="/api/ome", tags=["ome"])


# Dependency to get OMEController (only if OME mode is available)
def get_ome_controller() -> OMEController:
	"""Get OME controller instance."""
	try:
		return OMEController()
	except Exception as e:
		raise HTTPException(status_code=503, detail=f"OME controller not available: {str(e)}")


# ClusterBaseModel Routes
@router.get("/clusterbasemodels", response_model=Dict[str, Any])
async def list_clusterbasemodels(controller: OMEController = Depends(get_ome_controller)):
	"""List all ClusterBaseModels in the cluster."""
	result = controller.list_clusterbasemodels()
	if result is None:
		raise HTTPException(status_code=500, detail="Failed to list ClusterBaseModels")

	# Extract items from result
	items = result.get("items", [])

	return {
		"count": len(items),
		"items": [
			{
				"name": item["metadata"]["name"],
				"displayName": item["spec"].get("displayName", ""),
				"vendor": item["spec"].get("vendor", ""),
				"version": item["spec"].get("version", ""),
				"disabled": item["spec"].get("disabled", False),
				"storage": item["spec"].get("storage", {}),
			}
			for item in items
		],
	}


@router.get("/clusterbasemodel-presets", response_model=List[Dict[str, Any]])
async def list_clusterbasemodel_presets():
	"""List all available ClusterBaseModel presets."""
	return clusterbasemodel_presets.list_presets()


# ClusterServingRuntime Routes
@router.get("/clusterservingruntimes", response_model=Dict[str, Any])
async def list_clusterservingruntimes(controller: OMEController = Depends(get_ome_controller)):
	"""List all ClusterServingRuntimes in the cluster."""
	result = controller.list_clusterservingruntimes()
	if result is None:
		raise HTTPException(status_code=500, detail="Failed to list ClusterServingRuntimes")

	# Extract items from result
	items = result.get("items", [])

	return {
		"count": len(items),
		"items": [
			{
				"name": item["metadata"]["name"],
				"disabled": item["spec"].get("disabled", False),
				"protocolVersions": item["spec"].get("protocolVersions", []),
				"modelSizeRange": item["spec"].get("modelSizeRange", {}),
				"supportedModelFormats": item["spec"].get("supportedModelFormats", []),
			}
			for item in items
		],
	}


@router.get("/clusterservingruntime-presets", response_model=List[Dict[str, Any]])
async def list_clusterservingruntime_presets():
	"""List all available ClusterServingRuntime presets."""
	return clusterservingruntime_presets.list_presets()


@router.get("/clusterservingruntime-presets/{runtime_type}", response_model=List[Dict[str, Any]])
async def list_clusterservingruntime_presets_by_runtime(runtime_type: str):
	"""List ClusterServingRuntime presets filtered by runtime type."""
	return clusterservingruntime_presets.get_presets_by_runtime(runtime_type)
