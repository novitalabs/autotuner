"""
Pydantic schemas for tool authorization and execution.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import datetime


class ToolAuthorizationRequest(BaseModel):
    """Request schema for granting tool authorizations."""

    scopes: List[str] = Field(
        ...,
        description="Authorization scopes to grant (e.g., bash_commands, file_read)"
    )
    expires_at: Optional[datetime] = Field(
        None,
        description="Expiration time for the authorization (null = permanent for session)"
    )


class AuthorizationResponse(BaseModel):
    """Response schema for authorization operations."""

    status: str = Field(..., description="Operation status (granted, revoked, etc.)")
    scopes: List[str] = Field(..., description="Affected authorization scopes")
    tool_results: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Results from executing pending tool calls after authorization was granted"
    )


class ToolExecutionResult(BaseModel):
    """Schema for tool execution results."""

    success: bool = Field(..., description="Whether tool execution succeeded")
    result: str = Field(..., description="Tool output or error message")
    requires_auth: bool = Field(
        default=False,
        description="Whether this tool requires user authorization"
    )
    auth_scope: Optional[str] = Field(
        None,
        description="Required authorization scope if requires_auth is True"
    )
