"""
Pydantic schemas for agent API.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


class ChatSessionCreate(BaseModel):
	"""Schema for creating a new chat session."""

	user_id: Optional[str] = None


class ChatSessionResponse(BaseModel):
	"""Schema for chat session response."""

	id: int
	session_id: str
	user_id: Optional[str]
	context_summary: Optional[str]
	is_active: bool
	created_at: datetime
	updated_at: datetime

	class Config:
		from_attributes = True


class ChatMessageCreate(BaseModel):
	"""Schema for sending a message."""

	content: str = Field(..., min_length=1, max_length=10000)


class ChatMessageResponse(BaseModel):
	"""Schema for chat message response."""

	id: int
	session_id: str
	role: str  # user, assistant, system
	content: str
	tool_calls: Optional[Dict[str, Any]]
	metadata: Optional[Dict[str, Any]]
	token_count: Optional[int]
	created_at: datetime

	class Config:
		from_attributes = True


class AgentEventSubscriptionCreate(BaseModel):
	"""Schema for creating event subscription."""

	task_id: int
	event_types: List[str] = Field(default=["task_completed", "task_failed"])


class AgentEventSubscriptionResponse(BaseModel):
	"""Schema for event subscription response."""

	id: int
	session_id: str
	task_id: int
	event_types: List[str]
	is_active: bool
	created_at: datetime
	expires_at: Optional[datetime]

	class Config:
		from_attributes = True
