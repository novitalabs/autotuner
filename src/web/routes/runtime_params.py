"""
API routes for runtime parameter information.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from pydantic import BaseModel

from utils.runtime_parameters import (
	get_parameters_for_runtime,
	get_commonly_tuned_parameters,
	validate_parameter,
	get_parameter_compatibility,
	SGLANG_PARAMETERS,
	VLLM_PARAMETERS,
	COMMON_PARAMETERS,
)


router = APIRouter(prefix="/api/runtime-params", tags=["runtime-parameters"])


class ParameterListResponse(BaseModel):
	"""Response model for parameter lists."""

	runtime: str
	count: int
	parameters: List[str]


class CommonlyTunedResponse(BaseModel):
	"""Response model for commonly tuned parameters."""

	runtime: str
	parameters: List[str]


class ParameterValidationRequest(BaseModel):
	"""Request model for parameter validation."""

	runtime: str
	parameter: str


class ParameterValidationResponse(BaseModel):
	"""Response model for parameter validation."""

	runtime: str
	parameter: str
	is_valid: bool


class ParameterCompatibilityResponse(BaseModel):
	"""Response model for parameter compatibility."""

	common: List[str]
	sglang_only: List[str]
	vllm_only: List[str]
	stats: Dict[str, int]


@router.get("/", response_model=Dict[str, int])
async def get_parameter_counts():
	"""Get counts of parameters for each runtime."""
	return {
		"sglang_count": len(SGLANG_PARAMETERS),
		"vllm_count": len(VLLM_PARAMETERS),
		"common_count": len(COMMON_PARAMETERS),
	}


@router.get("/compatibility", response_model=ParameterCompatibilityResponse)
async def get_compatibility():
	"""
	Get parameter compatibility information across runtimes.

	Returns:
	    Dictionary with common, sglang-only, and vllm-only parameters
	"""
	compat = get_parameter_compatibility()

	return ParameterCompatibilityResponse(
		common=compat["common"],
		sglang_only=compat["sglang_only"],
		vllm_only=compat["vllm_only"],
		stats={
			"common_count": len(compat["common"]),
			"sglang_only_count": len(compat["sglang_only"]),
			"vllm_only_count": len(compat["vllm_only"]),
			"sglang_total": len(SGLANG_PARAMETERS),
			"vllm_total": len(VLLM_PARAMETERS),
		},
	)


@router.get("/{runtime}", response_model=ParameterListResponse)
async def get_runtime_parameters(
	runtime: str, commonly_tuned_only: bool = Query(False, description="Return only commonly tuned parameters")
):
	"""
	Get all valid parameters for a specific runtime.

	Args:
	    runtime: Runtime name ('sglang' or 'vllm')
	    commonly_tuned_only: If true, return only commonly tuned parameters

	Returns:
	    List of parameter names for the runtime
	"""
	runtime = runtime.lower()
	if runtime not in ["sglang", "vllm"]:
		raise HTTPException(status_code=400, detail=f"Invalid runtime: {runtime}. Must be 'sglang' or 'vllm'")

	try:
		if commonly_tuned_only:
			params = get_commonly_tuned_parameters(runtime)
		else:
			params = sorted(list(get_parameters_for_runtime(runtime)))

		return ParameterListResponse(runtime=runtime, count=len(params), parameters=params)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))


@router.get("/{runtime}/commonly-tuned", response_model=CommonlyTunedResponse)
async def get_commonly_tuned(runtime: str):
	"""
	Get commonly tuned parameters for optimization experiments.

	Args:
	    runtime: Runtime name ('sglang' or 'vllm')

	Returns:
	    List of commonly tuned parameter names
	"""
	runtime = runtime.lower()
	if runtime not in ["sglang", "vllm"]:
		raise HTTPException(status_code=400, detail=f"Invalid runtime: {runtime}. Must be 'sglang' or 'vllm'")

	try:
		params = get_commonly_tuned_parameters(runtime)
		return CommonlyTunedResponse(runtime=runtime, parameters=params)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))


@router.post("/validate", response_model=ParameterValidationResponse)
async def validate_runtime_parameter(request: ParameterValidationRequest):
	"""
	Validate if a parameter is valid for a given runtime.

	Args:
	    request: Parameter validation request

	Returns:
	    Validation result
	"""
	is_valid = validate_parameter(request.runtime, request.parameter)

	return ParameterValidationResponse(runtime=request.runtime, parameter=request.parameter, is_valid=is_valid)
