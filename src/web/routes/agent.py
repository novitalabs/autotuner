"""
API routes for agent chat functionality.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
import uuid
import logging

from web.db.session import get_db
from web.db.models import ChatSession, ChatMessage, MessageRole, AgentEventSubscription
from web.schemas.agent import (
	ChatSessionCreate,
	ChatSessionResponse,
	ChatMessageCreate,
	ChatMessageResponse,
	AgentEventSubscriptionCreate,
	AgentEventSubscriptionResponse,
	SessionSyncRequest,
	SessionListItem,
	TitleUpdateRequest,
)
from web.config import get_settings
from web.agent.llm_client import get_llm_client
from web.agent.session_cache import get_session_cache

router = APIRouter(prefix="/api/agent", tags=["agent"])
logger = logging.getLogger(__name__)


@router.get("/status")
async def get_agent_status():
	"""Check if agent is configured and available."""
	settings = get_settings()

	# Check if agent is configured
	is_configured = False
	missing_config = []

	if settings.agent_provider == "local":
		# Local models need base_url
		if not settings.agent_base_url or settings.agent_base_url == "http://localhost:8000/v1":
			missing_config.append("agent_base_url (no local model endpoint configured)")
		else:
			is_configured = True
	elif settings.agent_provider in ["claude", "openai"]:
		# Cloud providers need API key
		if not settings.agent_api_key:
			missing_config.append("agent_api_key (no API key configured)")
		else:
			is_configured = True
	elif settings.agent_provider == "jiekou":
		# Jiekou needs both base_url and API key
		if not settings.agent_base_url or settings.agent_base_url == "http://localhost:8000/v1":
			missing_config.append("agent_base_url (Jiekou API endpoint not configured)")
		if not settings.agent_api_key:
			missing_config.append("agent_api_key (Jiekou API key not configured)")
		if not missing_config:
			is_configured = True
	else:
		missing_config.append(f"agent_provider (invalid provider: {settings.agent_provider})")

	return {
		"available": is_configured,
		"provider": settings.agent_provider,
		"model": settings.agent_model if is_configured else None,
		"missing_config": missing_config if not is_configured else [],
		"message": "Agent is available" if is_configured else "Agent not configured. Please set environment variables: " + ", ".join(missing_config)
	}


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(
	session_data: ChatSessionCreate, db: AsyncSession = Depends(get_db)
):
	"""Create a new chat session."""
	session_id = str(uuid.uuid4())
	chat_session = ChatSession(
		session_id=session_id, user_id=session_data.user_id, is_active=True
	)
	db.add(chat_session)
	await db.commit()
	await db.refresh(chat_session)
	return chat_session


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
	"""Get chat session details."""
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")
	return session


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_messages(
	session_id: str, limit: int = 50, db: AsyncSession = Depends(get_db)
):
	"""Get message history for a session."""
	# Verify session exists
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")

	# Get messages
	result = await db.execute(
		select(ChatMessage)
		.where(ChatMessage.session_id == session_id)
		.order_by(ChatMessage.created_at.desc())
		.limit(limit)
	)
	messages = result.scalars().all()
	return list(reversed(messages))  # Return in chronological order


@router.post("/sessions/sync")
async def sync_session(
	session_data: SessionSyncRequest,
	db: AsyncSession = Depends(get_db)
):
	"""
	Sync full session from frontend IndexedDB to backend.
	Only called once per session (when synced_to_backend=false).
	"""
	session_id = session_data.session_id

	# Check if session already exists
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	existing = result.scalar_one_or_none()

	if existing:
		# Session already synced, ignore
		logger.info(f"Session {session_id} already exists, skipping sync")
		return {"status": "already_exists", "session_id": session_id}

	# Create session
	chat_session = ChatSession(
		session_id=session_id,
		created_at=session_data.created_at,
		is_active=True
	)
	db.add(chat_session)

	# Bulk insert messages
	for msg_data in session_data.messages:
		message = ChatMessage(
			session_id=session_id,
			role=MessageRole(msg_data.role),
			content=msg_data.content,
			created_at=msg_data.created_at
		)
		db.add(message)

	await db.commit()

	# Clean expired sessions from cache
	cache = get_session_cache()
	removed = cache.cleanup_expired()
	if removed > 0:
		logger.info(f"Cleaned {removed} expired sessions from cache")

	return {"status": "synced", "session_id": session_id, "message_count": len(session_data.messages)}


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(
	session_id: str,
	message_data: ChatMessageCreate,
	db: AsyncSession = Depends(get_db),
):
	"""Send a message and get LLM response."""
	# 1. Check memory cache first
	cache = get_session_cache()
	cached_entry = cache.get(session_id)

	if cached_entry:
		# Use cached context (fast path)
		recent_messages = cached_entry.messages
		logger.debug(f"Cache hit for session {session_id}, using {len(recent_messages)} cached messages")
	else:
		# Cache miss: Load from DB
		logger.debug(f"Cache miss for session {session_id}, loading from database")

		# Verify session exists
		result = await db.execute(
			select(ChatSession).where(ChatSession.session_id == session_id)
		)
		session = result.scalar_one_or_none()
		if not session:
			raise HTTPException(status_code=404, detail="Session not found")

		# Get recent messages
		result = await db.execute(
			select(ChatMessage)
			.where(ChatMessage.session_id == session_id)
			.order_by(ChatMessage.created_at.desc())
			.limit(20)  # Last 20 messages for context
		)
		messages_from_db = list(reversed(result.scalars().all()))

		# Convert to cache format
		recent_messages = [
			{"role": msg.role.value, "content": msg.content}
			for msg in messages_from_db
		]

		# Populate cache for next time
		cache.set(session_id, recent_messages)

	# 2. Save user message to database
	user_message = ChatMessage(
		session_id=session_id, role=MessageRole.USER, content=message_data.content
	)
	db.add(user_message)
	await db.commit()
	await db.refresh(user_message)

	try:
		# 3. Build LLM context and get response
		llm_messages = []

		# Add system message
		llm_messages.append({
			"role": "system",
			"content": "You are a helpful AI assistant for the LLM Inference Autotuner. You help users optimize their LLM inference parameters and analyze benchmark results."
		})

		# Add conversation history from cache
		llm_messages.extend(recent_messages)

		# Add current user message
		llm_messages.append({
			"role": "user",
			"content": message_data.content
		})

		# Call LLM
		llm_client = get_llm_client()
		assistant_content = await llm_client.chat(llm_messages)

		# 4. Save assistant message
		assistant_message = ChatMessage(
			session_id=session_id,
			role=MessageRole.ASSISTANT,
			content=assistant_content
		)
		db.add(assistant_message)
		await db.commit()
		await db.refresh(assistant_message)

		# 5. Update cache with new messages
		cached_entry = cache.get(session_id)
		if cached_entry:
			cached_entry.messages.append({"role": "user", "content": message_data.content})
			cached_entry.messages.append({"role": "assistant", "content": assistant_content})
			# Keep only last 20 messages
			cached_entry.messages = cached_entry.messages[-20:]
			logger.debug(f"Updated cache for session {session_id} with new messages")

		return assistant_message

	except Exception as e:
		logger.error(f"Error calling LLM: {str(e)}")
		# Save error message
		error_content = f"Sorry, I encountered an error: {str(e)}"
		assistant_message = ChatMessage(
			session_id=session_id,
			role=MessageRole.ASSISTANT,
			content=error_content
		)
		db.add(assistant_message)
		await db.commit()
		await db.refresh(assistant_message)
		return assistant_message


@router.post(
	"/sessions/{session_id}/subscribe", response_model=AgentEventSubscriptionResponse
)
async def subscribe_to_task(
	session_id: str,
	subscription_data: AgentEventSubscriptionCreate,
	db: AsyncSession = Depends(get_db),
):
	"""Subscribe to task events."""
	# Verify session exists
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")

	# Check if subscription already exists
	result = await db.execute(
		select(AgentEventSubscription)
		.where(AgentEventSubscription.session_id == session_id)
		.where(AgentEventSubscription.task_id == subscription_data.task_id)
		.where(AgentEventSubscription.is_active == True)
	)
	existing_sub = result.scalar_one_or_none()
	if existing_sub:
		raise HTTPException(status_code=400, detail="Subscription already exists")

	# Create subscription
	subscription = AgentEventSubscription(
		session_id=session_id,
		task_id=subscription_data.task_id,
		event_types=subscription_data.event_types,
		is_active=True,
	)
	db.add(subscription)
	await db.commit()
	await db.refresh(subscription)
	return subscription


@router.delete("/sessions/{session_id}/subscribe/{task_id}")
async def unsubscribe_from_task(
	session_id: str, task_id: int, db: AsyncSession = Depends(get_db)
):
	"""Unsubscribe from task events."""
	result = await db.execute(
		select(AgentEventSubscription)
		.where(AgentEventSubscription.session_id == session_id)
		.where(AgentEventSubscription.task_id == task_id)
		.where(AgentEventSubscription.is_active == True)
	)
	subscription = result.scalar_one_or_none()
	if not subscription:
		raise HTTPException(status_code=404, detail="Subscription not found")

	subscription.is_active = False
	await db.commit()
	return {"message": "Unsubscribed successfully"}


@router.get("/sessions", response_model=List[SessionListItem])
async def list_sessions(
	limit: int = 50,
	db: AsyncSession = Depends(get_db)
):
	"""List all sessions, most recent first."""
	result = await db.execute(
		select(ChatSession)
		.where(ChatSession.is_active == True)
		.order_by(ChatSession.updated_at.desc())
		.limit(limit)
	)
	sessions = result.scalars().all()

	# Enrich with last message preview
	session_list = []
	for session in sessions:
		# Get last message
		msg_result = await db.execute(
			select(ChatMessage)
			.where(ChatMessage.session_id == session.session_id)
			.order_by(ChatMessage.created_at.desc())
			.limit(1)
		)
		last_message = msg_result.scalar_one_or_none()

		# Get message count
		count_result = await db.execute(
			select(ChatMessage)
			.where(ChatMessage.session_id == session.session_id)
		)
		message_count = len(count_result.scalars().all())

		session_list.append(SessionListItem(
			session_id=session.session_id,
			created_at=session.created_at,
			updated_at=session.updated_at,
			title=session.title,
			last_message_preview=last_message.content[:100] if last_message else "",
			message_count=message_count
		))

	return session_list


@router.get(
	"/sessions/{session_id}/subscriptions",
	response_model=List[AgentEventSubscriptionResponse],
)
async def get_subscriptions(session_id: str, db: AsyncSession = Depends(get_db)):
	"""Get active subscriptions for a session."""
	result = await db.execute(
		select(AgentEventSubscription)
		.where(AgentEventSubscription.session_id == session_id)
		.where(AgentEventSubscription.is_active == True)
	)
	subscriptions = result.scalars().all()
	return list(subscriptions)


@router.post("/sessions/{session_id}/title/generate")
async def generate_title(
	session_id: str,
	db: AsyncSession = Depends(get_db)
):
	"""Generate title for session based on first user message."""
	# 1. Get session
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")

	# 2. Get first user message
	result = await db.execute(
		select(ChatMessage)
		.where(ChatMessage.session_id == session_id)
		.where(ChatMessage.role == MessageRole.USER)
		.order_by(ChatMessage.created_at.asc())
		.limit(1)
	)
	first_message = result.scalar_one_or_none()
	if not first_message:
		raise HTTPException(status_code=400, detail="No user messages found")

	# 3. Call LLM for title generation
	llm_client = get_llm_client()
	messages = [
		{
			"role": "system",
			"content": "You are a title generator. Generate a concise, descriptive title (6-8 words maximum) for a chat conversation based on the user's first message. Output only the title text, nothing else. Do not use quotes or punctuation at the end."
		},
		{
			"role": "user",
			"content": first_message.content
		}
	]

	try:
		title = await llm_client.chat(messages, temperature=0.3)
		title = title.strip()

		# Remove quotes if LLM added them
		if title.startswith('"') and title.endswith('"'):
			title = title[1:-1]

		# Limit length
		if len(title) > 100:
			title = title[:97] + "..."

		# 4. Save title
		session.title = title
		await db.commit()

		return {"title": title}
	except Exception as e:
		logger.error(f"Failed to generate title: {str(e)}")
		raise HTTPException(status_code=500, detail="Failed to generate title")


@router.patch("/sessions/{session_id}/title", response_model=ChatSessionResponse)
async def update_title(
	session_id: str,
	title_data: TitleUpdateRequest,
	db: AsyncSession = Depends(get_db)
):
	"""Update session title."""
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")

	session.title = title_data.title.strip()
	await db.commit()
	await db.refresh(session)

	return session
