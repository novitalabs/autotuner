"""
API routes for agent chat functionality.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified
from typing import List, Dict, Any
from datetime import datetime
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
from web.schemas.tools import (
	ToolAuthorizationRequest,
	AuthorizationResponse,
)
from web.config import get_settings
from web.agent.llm_client import get_llm_client
from web.agent.session_cache import get_session_cache
from web.agent.executor import ToolExecutor

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
	"""Send a message and get LLM response with tool support."""
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
		# 3. Build LLM context
		llm_messages = []

		# Add system message with tool usage instructions
		llm_messages.append({
			"role": "system",
			"content": """You are a helpful AI assistant for the LLM Inference Autotuner. You help users optimize their LLM inference parameters and analyze benchmark results.

You have access to various tools for querying tasks, experiments, parameter presets, and external APIs. Use these tools when needed to provide accurate, data-driven responses.

Available tool categories:
- DATABASE: Query tasks, experiments, and parameter presets
- API: Search HuggingFace models and check service health

When a user asks about tasks, experiments, or results, use the appropriate database tools to fetch current data."""
		})

		# Add conversation history from cache
		llm_messages.extend(recent_messages)

		# Add current user message
		llm_messages.append({
			"role": "user",
			"content": message_data.content
		})

		# 4. Get tools for this session
		executor = ToolExecutor(session_id, db)
		available_tools = executor.get_available_tools(include_privileged=False)  # Only safe tools for now

		# 5. Call LLM with tools
		llm_client = get_llm_client()
		print(f"[DEBUG] Calling LLM with {len(available_tools)} tools available")
		llm_response = await llm_client.chat_with_tools(llm_messages, available_tools)
		print(f"[DEBUG] LLM response: content_length={len(llm_response['content']) if llm_response['content'] else 0}, tool_calls={len(llm_response['tool_calls'])}")

		assistant_content = llm_response["content"]
		tool_calls = llm_response["tool_calls"]

		# Filter out 'db' parameter from tool_calls before saving to database
		# The LLM may include 'db' in args, but it's not JSON serializable
		def clean_tool_call_args(tool_call):
			"""Remove db parameter from tool call args for JSON serialization"""
			cleaned_tc = tool_call.copy()
			if "args" in cleaned_tc and isinstance(cleaned_tc["args"], dict):
				cleaned_args = {k: v for k, v in cleaned_tc["args"].items() if k != "db"}
				cleaned_tc["args"] = cleaned_args
			return cleaned_tc

		# 6. Handle tool calls if any
		tool_results = []
		if tool_calls:
			logger.info(f"Processing {len(tool_calls)} tool calls")

			# Execute all tool calls
			tool_results = await executor.execute_tool_calls(tool_calls)

			# Check for authorization errors
			auth_required = []
			for result in tool_results:
				if not result["success"] and result.get("requires_auth") and not result.get("authorized"):
					auth_required.append({
						"tool_name": result["tool_name"],
						"auth_scope": result["auth_scope"]
					})

			# If any tools require authorization, save message with tool_calls metadata
			if auth_required:
				assistant_message = ChatMessage(
					session_id=session_id,
					role=MessageRole.ASSISTANT,
					content=assistant_content if assistant_content else "I need authorization to perform some operations.",
					tool_calls=[{
						"tool_name": tc["name"],
						"args": {k: v for k, v in tc["args"].items() if k != "db"},  # Filter out db parameter
						"id": tc["id"],
						"status": "requires_auth",
						"auth_scope": next((r["auth_scope"] for r in tool_results if r["tool_name"] == tc["name"]), None)
					} for tc in tool_calls],
					message_metadata={
						"auth_required": auth_required
					}
				)
				db.add(assistant_message)
				await db.commit()
				await db.refresh(assistant_message)

				logger.info(f"Saved message with authorization requirement for session {session_id}")
				return assistant_message

			# All tools executed successfully, format results
			tool_output_messages = []
			for result in tool_results:
				tool_output_messages.append({
					"role": "tool",
					"content": result["result"],
					"tool_call_id": result["call_id"]
				})

			# Add tool results to context and ask LLM to synthesize response
			llm_messages.append({
				"role": "assistant",
				"content": assistant_content if assistant_content else ""
			})
			llm_messages.extend(tool_output_messages)

			# Get final response from LLM
			final_response = await llm_client.chat(llm_messages)
			assistant_content = final_response

			# Save assistant message with tool execution metadata
			assistant_message = ChatMessage(
				session_id=session_id,
				role=MessageRole.ASSISTANT,
				content=assistant_content,
				tool_calls=[{
					"tool_name": tc["name"],
					"args": {k: v for k, v in tc["args"].items() if k != "db"},  # Filter out db parameter
					"id": tc["id"],
					"status": "executed",
					"result": next((r["result"] for r in tool_results if r["tool_name"] == tc["name"]), None)
				} for tc in tool_calls]
			)
		else:
			# No tool calls, just save regular assistant message
			assistant_message = ChatMessage(
				session_id=session_id,
				role=MessageRole.ASSISTANT,
				content=assistant_content
			)

		db.add(assistant_message)
		await db.commit()
		await db.refresh(assistant_message)

		# 7. Update cache with new messages
		cached_entry = cache.get(session_id)
		if cached_entry:
			cached_entry.messages.append({"role": "user", "content": message_data.content})
			cached_entry.messages.append({"role": "assistant", "content": assistant_content})
			# Keep only last 20 messages
			cached_entry.messages = cached_entry.messages[-20:]
			logger.debug(f"Updated cache for session {session_id} with new messages")

		return assistant_message

	except Exception as e:
		logger.error(f"Error calling LLM: {str(e)}", exc_info=True)
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
			"content": "You are a title generator. Generate a concise, descriptive title (6-8 words maximum) for a chat conversation based on the user's first message. Output ONLY the title text, nothing else. Do not use quotes or punctuation at the end. Do not add explanations, greetings, or any other text."
		},
		{
			"role": "user",
			"content": f"Generate a short title (6-8 words) for this conversation:\n\n{first_message.content}"
		}
	]

	try:
		title = await llm_client.chat(messages, temperature=0.3)
		title = title.strip()

		# Remove quotes if LLM added them
		if title.startswith('"') and title.endswith('"'):
			title = title[1:-1]

		# Take only first line if LLM generated multiple lines
		if '\n' in title:
			title = title.split('\n')[0].strip()

		# Limit length more aggressively - if too long, take first 50 chars
		if len(title) > 50:
			title = title[:47] + "..."

		# If still empty or too short, use fallback
		if not title or len(title) < 3:
			# Fallback: use first 50 chars of user message
			title = first_message.content[:47] + "..." if len(first_message.content) > 50 else first_message.content

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


# ============================================================================
# Tool Authorization Endpoints
# ============================================================================

@router.post("/sessions/{session_id}/authorize", response_model=AuthorizationResponse)
async def grant_tool_authorization(
	session_id: str,
	auth_data: ToolAuthorizationRequest,
	db: AsyncSession = Depends(get_db)
):
	"""
	Grant authorization for specific tool scopes.

	This allows the agent to execute privileged operations (bash commands, file operations, etc.)
	within this chat session. Authorizations can have an expiration time or be permanent
	for the session duration.
	"""
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")

	# Initialize metadata if needed
	metadata = session.session_metadata or {}
	if "tool_authorizations" not in metadata:
		metadata["tool_authorizations"] = {}

	# Grant authorization for each scope
	now = datetime.utcnow()
	for scope in auth_data.scopes:
		metadata["tool_authorizations"][scope] = {
			"granted": True,
			"granted_at": now.isoformat(),
			"expires_at": auth_data.expires_at.isoformat() if auth_data.expires_at else None
		}
		logger.info(f"Granted authorization for scope '{scope}' in session {session_id}")

	session.session_metadata = metadata
	flag_modified(session, "session_metadata")  # Tell SQLAlchemy the JSON field changed
	await db.commit()

	return AuthorizationResponse(status="granted", scopes=auth_data.scopes)


@router.get("/sessions/{session_id}/authorizations")
async def get_authorizations(
	session_id: str,
	db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
	"""
	Get current authorization grants for a session.

	Returns only active (non-expired) authorizations.
	"""
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")

	authorizations = {}
	if session.session_metadata and "tool_authorizations" in session.session_metadata:
		now = datetime.utcnow()

		# Filter out expired grants
		for scope, grant in session.session_metadata["tool_authorizations"].items():
			if not grant.get("granted"):
				continue

			expires_at = grant.get("expires_at")
			if expires_at is None or datetime.fromisoformat(expires_at) > now:
				authorizations[scope] = grant
			else:
				logger.debug(f"Authorization for scope '{scope}' expired in session {session_id}")

	return {"authorizations": authorizations}


@router.delete("/sessions/{session_id}/authorize/{scope}", response_model=AuthorizationResponse)
async def revoke_authorization(
	session_id: str,
	scope: str,
	db: AsyncSession = Depends(get_db)
):
	"""
	Revoke authorization for a specific scope.

	This immediately removes the user's authorization for the specified operation type.
	"""
	result = await db.execute(
		select(ChatSession).where(ChatSession.session_id == session_id)
	)
	session = result.scalar_one_or_none()
	if not session:
		raise HTTPException(status_code=404, detail="Session not found")

	metadata = session.session_metadata or {}
	revoked = False

	if "tool_authorizations" in metadata and scope in metadata["tool_authorizations"]:
		metadata["tool_authorizations"][scope]["granted"] = False
		session.session_metadata = metadata
		flag_modified(session, "session_metadata")  # Tell SQLAlchemy the JSON field changed
		await db.commit()
		revoked = True
		logger.info(f"Revoked authorization for scope '{scope}' in session {session_id}")

	if not revoked:
		raise HTTPException(
			status_code=404,
			detail=f"No active authorization found for scope '{scope}'"
		)

	return AuthorizationResponse(status="revoked", scopes=[scope])
