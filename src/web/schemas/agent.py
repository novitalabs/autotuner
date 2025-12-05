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
	title: Optional[str]
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
	tool_calls: Optional[List[Dict[str, Any]]] = None
	message_metadata: Optional[Dict[str, Any]] = None
	token_count: Optional[int] = None
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


class MessageSync(BaseModel):
	"""Schema for syncing a message from IndexedDB."""

	role: str
	content: str
	created_at: datetime


class SessionSyncRequest(BaseModel):
	"""Schema for syncing a full session from IndexedDB to backend."""

	session_id: str
	created_at: datetime
	messages: List[MessageSync]


class SessionListItem(BaseModel):
	"""Schema for session list item."""

	session_id: str
	created_at: datetime
	updated_at: datetime
	title: Optional[str]
	last_message_preview: str
	message_count: int


class TitleUpdateRequest(BaseModel):
	"""Schema for updating session title."""

	title: str = Field(..., min_length=1, max_length=100)
