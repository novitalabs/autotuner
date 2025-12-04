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
)
from web.config import get_settings
from web.agent.llm_client import get_llm_client

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


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(
	session_id: str,
	message_data: ChatMessageCreate,
	db: AsyncSession = Depends(get_db),
):
	"""Send a message and get LLM response."""
	# Verify session exists
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")

	# Save user message
	user_message = ChatMessage(
		session_id=session_id, role=MessageRole.USER, content=message_data.content
	)
	db.add(user_message)
	await db.commit()
	await db.refresh(user_message)

	try:
		# Get recent message history for context
		result = await db.execute(
			select(ChatMessage)
			.where(ChatMessage.session_id == session_id)
			.order_by(ChatMessage.created_at.desc())
			.limit(10)  # Last 10 messages for context
		)
		recent_messages = list(reversed(result.scalars().all()))

		# Build messages for LLM
		llm_messages = []

		# Add system message
		llm_messages.append({
			"role": "system",
			"content": "You are a helpful AI assistant for the LLM Inference Autotuner. You help users optimize their LLM inference parameters and analyze benchmark results."
		})

		# Add conversation history
		for msg in recent_messages:
			llm_messages.append({
				"role": msg.role.value,
				"content": msg.content
			})

		# Call LLM
		llm_client = get_llm_client()
		assistant_content = await llm_client.chat(llm_messages)

		# Save assistant message
		assistant_message = ChatMessage(
			session_id=session_id,
			role=MessageRole.ASSISTANT,
			content=assistant_content
		)
		db.add(assistant_message)
		await db.commit()
		await db.refresh(assistant_message)

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
