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

		# Convert to cache format (include tool_calls for proper context)
		recent_messages = []
		for msg in messages_from_db:
			if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
				# Assistant message with tool calls
				recent_messages.append({
					"role": "assistant",
					"content": msg.content
				})
				# Add tool results as separate ToolMessage entries
				for tool_call in msg.tool_calls:
					if tool_call.get("result"):
						recent_messages.append({
							"role": "tool",
							"content": tool_call["result"],
							"tool_call_id": tool_call.get("id", tool_call.get("tool_name"))
						})
			else:
				# Regular message (user or simple assistant)
				recent_messages.append({
					"role": msg.role.value,
					"content": msg.content
				})

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

You have access to various tools for managing and querying the autotuner system. Use these tools when needed to provide accurate, data-driven responses.

## Tool Calling Guidelines

**CRITICAL RULES:**
1. **Always provide ALL required parameters** - Never omit required parameters like task_id, experiment_id, etc.
2. **Never call tools with empty names** - Always specify the exact tool name
3. **Use correct parameter names** - Check tool signatures carefully
4. **Don't pass 'db' parameter** - The database session is injected automatically

**Common Required Parameters:**
- get_task_by_id(task_id=<int>) - MUST provide task_id as integer
- get_experiment_details(experiment_id=<int>) - MUST provide experiment_id as integer
- list_task_experiments(task_id=<int>) - MUST provide task_id as integer
- get_task_results(task_id=<int>, include_all_experiments=<bool>) - task_id is required
- get_task_logs(task_id=<int>, tail_lines=<int optional>) - task_id is required

## Available Tool Categories (in order of preference)

1. **TASK tools** (High-level task management - USE THESE FIRST):
   - list_tasks() - List all tasks with optional filtering
   - get_task_by_id(task_id) - Get task details by ID [REQUIRES: task_id as int]
   - get_task_by_name(task_name) - Get task details by name [REQUIRES: task_name as str]
   - list_task_experiments(task_id) - List experiments for a task [REQUIRES: task_id as int]
   - get_task_results(task_id) - Get comprehensive task results [REQUIRES: task_id as int]
   - get_task_logs(task_id) - Get task execution logs [REQUIRES: task_id as int]
   - get_experiment_details(experiment_id) - Get experiment details [REQUIRES: experiment_id as int]
   - create_task, start_task, cancel_task, restart_task: Manage task lifecycle
   - delete_task, clear_task_data, update_task_description: Task operations

2. **PRESET tools** (High-level preset management):
   - list_presets, get_preset_by_id, get_preset_by_name: Query presets
   - create_preset, update_preset, delete_preset: Manage parameter presets

3. **DATABASE tools** (Low-level queries - use only if high-level tools don't work):
   - query_records, count_records: Direct database queries
   - Use only when TASK/PRESET tools are insufficient

4. **API tools** (External services):
   - search_huggingface_models, check_service_health: External API calls

## Best Practices

**ALWAYS:**
- Prefer TASK and PRESET tools over DATABASE tools
- Provide all required parameters with correct types
- Extract task IDs from user queries (e.g., "task 10" means task_id=10)
- Use get_task_results() for comprehensive task analysis
- Use list_task_experiments() to see all experiments before getting details

**NEVER:**
- Call tools without required parameters
- Pass 'db' in your tool arguments (it's auto-injected)
- Use empty string "" as tool name
- Use query_records when a specific TASK tool exists

**Example Correct Calls:**
- User asks "分析task 10": Call get_task_results(task_id=10)
- User asks "show experiment 87": Call get_experiment_details(experiment_id=87)
- User asks "list experiments for task 5": Call list_task_experiments(task_id=5)

## Common Scenario Workflows

**Scenario 1: Create new task based on existing task**
User: "参照task 5创建新任务，修改模型为llama-3-8b"
Step 1: get_task_by_id(task_id=5) - Get existing task configuration
Step 2: create_task(
  task_name="new-task-name",
  description="Based on task 5 with modified model",
  model_id="llama-3-8b-instruct",
  model_namespace=<from task 5>,
  base_runtime=<from task 5>,
  parameters=<from task 5>,
  ...other params from task 5
)

**Scenario 2: Analyze task failure through logs**
User: "分析task 8为什么失败了"
Step 1: get_task_by_id(task_id=8) - Check task status and error summary
Step 2: list_task_experiments(task_id=8, status_filter="FAILED") - Get failed experiments
Step 3: get_task_logs(task_id=8, tail_lines=100) - Get recent log entries
Step 4: For each failed experiment: get_experiment_details(experiment_id=X) - Get error details
Analysis: Look for common error patterns in logs (OOM, timeout, connection errors, etc.)

**Scenario 3: Analyze tuning results and parameter impact**
User: "分析task 10的调参结果，哪些参数最重要"
Step 1: get_task_results(task_id=10, include_all_experiments=True) - Get all experiments
Step 2: Analyze returned data:
   - Compare experiments with different parameter values
   - Identify which parameter changes correlate with performance improvements
   - Look at best_experiment parameters vs average experiments
Step 3: list_task_experiments(task_id=10, status_filter="SUCCESS") - Get successful experiments
Step 4: Compare top 3-5 experiments to identify key parameter patterns
Key metrics to analyze: objective_score, latency (p50/p90/p99), throughput, TTFT, TPOT

**Scenario 4: Monitor running task progress**
User: "task 15现在运行到哪了？"
Step 1: get_task_by_id(task_id=15) - Get task status, timing, and progress
Step 2: list_task_experiments(task_id=15) - Check completed experiments
Analysis:
   - Calculate progress: successful_experiments / total_experiments (if total known)
   - Show task status (PENDING/RUNNING/COMPLETED/FAILED)
   - Display elapsed time and estimated completion (if applicable)
   - Show latest experiment results if available

**Scenario 5: SLO violation analysis**
User: "task 10哪些实验违反了SLO约束？"
Step 1: analyze_slo_violations(task_id=10) - Get comprehensive SLO violation analysis
Step 2: Analyze the results:
   - Total violation count and rate
   - Hard fail vs soft penalty breakdown
   - Most frequently violated metrics
   - Parameter patterns in violating experiments
Step 3: Provide recommendations:
   - If ttft/tpot violations are high: Adjust memory fraction or scheduling policy
   - If latency violations: Consider lower concurrency or different tp-size
   - If throughput violations: Increase tp-size or adjust batch scheduling
Alternative: Use get_task_results(task_id=10, include_all_experiments=True) for manual analysis

**Scenario 6: Quick locate experiment failure cause**
User: "experiment 156为什么失败了？"
Step 1: get_experiment_details(experiment_id=156) - Get experiment details and error_message
Step 2: search_experiment_logs(task_id=<from step 1>, experiment_id=156, context_lines=15) - Find log entries
Analysis:
   - Check error_message for key patterns:
     * "OOM" / "out of memory" → Memory issues, try lower mem-fraction-static
     * "timeout" / "timed out" → Benchmark timeout, check model size vs GPU
     * "connection refused" / "failed to connect" → Inference service not ready
     * "CUDA error" → GPU resource issues
   - Review log context for detailed stack traces
   - Compare parameters with successful experiments to identify problematic values

High-level tools provide better error handling, formatted output, and business logic."""
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

		# 5. Multi-turn tool calling loop
		llm_client = get_llm_client()
		max_iterations = 10
		iteration = 0
		assistant_content = ""
		all_tool_calls = []  # Track all tool calls across iterations
		all_tool_results = []  # Track all tool results across iterations
		termination_reason = "natural"  # Track why loop ended

		while iteration < max_iterations:
			iteration += 1
			logger.info(f"Multi-turn iteration {iteration}/{max_iterations} for session {session_id}")

			# 5a. Call LLM with tools
			llm_response = await llm_client.chat_with_tools(llm_messages, available_tools)

			assistant_content = llm_response["content"]
			tool_calls = llm_response["tool_calls"]

			# 5b. Check if LLM wants to stop (no tool calls)
			if not tool_calls:
				logger.info(f"LLM returned no tool calls - natural termination at iteration {iteration}")
				break

			logger.info(f"Processing {len(tool_calls)} tool calls in iteration {iteration}")

			# 5c. Execute all tool calls
			tool_results = await executor.execute_tool_calls(tool_calls)

			# 5d. Check for authorization errors
			auth_required = []
			for result in tool_results:
				if not result["success"] and result.get("requires_auth") and not result.get("authorized"):
					auth_required.append({
						"tool_name": result["tool_name"],
						"auth_scope": result["auth_scope"]
					})

			# If any tools require authorization, stop loop and return auth request
			if auth_required:
				logger.info(f"Authorization required in iteration {iteration}, stopping multi-turn loop")
				termination_reason = "auth_required"

				assistant_message = ChatMessage(
					session_id=session_id,
					role=MessageRole.ASSISTANT,
					content=assistant_content if assistant_content else "I need authorization to perform some operations.",
					tool_calls=[{
						"tool_name": tc["name"],
						"args": {k: v for k, v in tc["args"].items() if k != "db"},
						"id": tc["id"],
						"status": "requires_auth",
						"auth_scope": next((r["auth_scope"] for r in tool_results if r["tool_name"] == tc["name"]), None)
					} for tc in tool_calls],
					message_metadata={
						"auth_required": auth_required,
						"iterations": iteration,
						"termination_reason": termination_reason
					}
				)
				db.add(assistant_message)
				await db.commit()
				await db.refresh(assistant_message)

				logger.info(f"Saved message with authorization requirement for session {session_id}")
				return assistant_message

			# 5e. Check for execution errors
			# NOTE: Jiekou API has issues with ToolMessage in conversation history,
			# especially when tools fail. So we stop here instead of continuing.
			failed_tools = [r for r in tool_results if not r["success"]]
			if failed_tools:
				logger.warning(f"{len(failed_tools)} tools failed in iteration {iteration}, terminating loop (Jiekou API limitation)")
				termination_reason = "tool_execution_error"

				# Build error summary for user
				error_summary = "\n\nSome tool calls failed:\n"
				for failed in failed_tools:
					error_summary += f"- {failed.get('tool_name', 'unknown')}: {failed.get('result', 'Unknown error')}\n"
				assistant_content = (assistant_content or "") + error_summary

				# Break out of loop - don't add ToolMessage to avoid Jiekou API 400 error
				all_tool_calls.extend(tool_calls)
				all_tool_results.extend(tool_results)
				break

			# 5f. Add assistant message with tool calls to context
			llm_messages.append({
				"role": "assistant",
				"content": assistant_content if assistant_content else ""
			})

			# 5g. Add tool results to context as ToolMessage
			for result in tool_results:
				call_id = result.get("call_id") or result.get("tool_name", "unknown")
				llm_messages.append({
					"role": "tool",
					"content": result["result"],
					"tool_call_id": call_id
				})

			# 5h. Track all tool calls and results for database storage
			all_tool_calls.extend(tool_calls)
			all_tool_results.extend(tool_results)

		# Check if max iterations reached
		if iteration >= max_iterations:
			logger.warning(f"Reached max iterations ({max_iterations}) for session {session_id}")
			termination_reason = "max_iterations"
			assistant_content += "\n\n[Note: Reached maximum thinking steps. Providing answer based on information gathered so far.]"

		# 6. Save final assistant message with complete tool execution history
		if all_tool_calls:
			# Find result and status for each tool call
			def find_result_for_call(call_id):
				for r in all_tool_results:
					if r.get("call_id") == call_id:
						return r.get("result"), r.get("success", True)
				# Fallback: match by tool_name if call_id doesn't match
				for r in all_tool_results:
					if r.get("tool_name") == call_id:
						return r.get("result"), r.get("success", True)
				return None, True

			def build_tool_call_entry(tc):
				result, success = find_result_for_call(tc["id"])
				return {
					"tool_name": tc["name"],
					"args": {k: v for k, v in tc["args"].items() if k != "db"},
					"id": tc["id"],
					"status": "executed" if success else "failed",
					"result": result if success else None,
					"error": result if not success else None
				}

			assistant_message = ChatMessage(
				session_id=session_id,
				role=MessageRole.ASSISTANT,
				content=assistant_content,
				tool_calls=[build_tool_call_entry(tc) for tc in all_tool_calls],
				message_metadata={
					"iterations": iteration,
					"termination_reason": termination_reason
				}
			)
		else:
			# No tool calls at all - simple response
			assistant_message = ChatMessage(
				session_id=session_id,
				role=MessageRole.ASSISTANT,
				content=assistant_content
			)

		logger.info(f"Multi-turn conversation completed: {iteration} iterations, {len(all_tool_calls)} total tool calls, termination: {termination_reason}")

		db.add(assistant_message)
		await db.commit()
		await db.refresh(assistant_message)

		# 7. Update cache with new messages (include tool results)
		cached_entry = cache.get(session_id)
		if cached_entry:
			cached_entry.messages.append({"role": "user", "content": message_data.content})
			cached_entry.messages.append({"role": "assistant", "content": assistant_content})

			# Add tool results as ToolMessages if any
			if all_tool_calls:
				for tc in all_tool_calls:
					result = next((r.get("result") for r in all_tool_results if r.get("call_id") == tc["id"]), None)
					if result:
						cached_entry.messages.append({
							"role": "tool",
							"content": result,
							"tool_call_id": tc["id"]
						})

			# Keep only last 20 messages
			cached_entry.messages = cached_entry.messages[-20:]
			logger.debug(f"Updated cache for session {session_id} with new messages")

		return assistant_message

	except Exception as e:
		logger.error(f"Error calling LLM: {str(e)}", exc_info=True)
		# Save error message
		error_message = str(e)
		error_content = f"❌ **Error:** {error_message}"
		assistant_message = ChatMessage(
			session_id=session_id,
			role=MessageRole.ASSISTANT,
			content=error_content
		)
		db.add(assistant_message)
		await db.commit()
		await db.refresh(assistant_message)
		return assistant_message


@router.post("/sessions/{session_id}/messages/stream")
async def send_message_stream(
	session_id: str,
	message_data: ChatMessageCreate,
	db: AsyncSession = Depends(get_db),
):
	"""Send a message and stream LLM response with tool support."""
	from fastapi.responses import StreamingResponse
	import json
	import asyncio

	async def event_stream():
		"""SSE event stream generator."""
		logger.info(f"event_stream started for session {session_id}")
		try:
			# 1. Check memory cache first
			cache = get_session_cache()
			cached_entry = cache.get(session_id)

			if cached_entry:
				recent_messages = cached_entry.messages
				logger.debug(f"Cache hit for session {session_id}")
			else:
				logger.debug(f"Cache miss for session {session_id}, loading from database")
				result = await db.execute(
					select(ChatSession).where(ChatSession.session_id == session_id)
				)
				session = result.scalar_one_or_none()
				if not session:
					yield f"data: {json.dumps({'type': 'error', 'error': 'Session not found'})}\n\n"
					return

				result = await db.execute(
					select(ChatMessage)
					.where(ChatMessage.session_id == session_id)
					.order_by(ChatMessage.created_at.desc())
					.limit(20)
				)
				messages_from_db = list(reversed(result.scalars().all()))

				# Convert to cache format (include tool_calls for proper context)
				recent_messages = []
				for msg in messages_from_db:
					if msg.role == MessageRole.ASSISTANT and msg.tool_calls:
						# Assistant message with tool calls
						recent_messages.append({
							"role": "assistant",
							"content": msg.content
						})
						# Add tool results as separate ToolMessage entries
						for tool_call in msg.tool_calls:
							if tool_call.get("result"):
								recent_messages.append({
									"role": "tool",
									"content": tool_call["result"],
									"tool_call_id": tool_call.get("id", tool_call.get("tool_name"))
								})
					else:
						# Regular message (user or simple assistant)
						recent_messages.append({
							"role": msg.role.value,
							"content": msg.content
						})

				cache.set(session_id, recent_messages)

			# 2. Save user message
			user_message = ChatMessage(
				session_id=session_id, role=MessageRole.USER, content=message_data.content
			)
			db.add(user_message)
			await db.commit()
			await db.refresh(user_message)

			# 3. Build LLM context
			llm_messages = [{
				"role": "system",
				"content": """You are a helpful AI assistant for the LLM Inference Autotuner. You help users optimize their LLM inference parameters and analyze benchmark results.

You have access to various tools for managing and querying the autotuner system. Use these tools when needed to provide accurate, data-driven responses.

## Tool Calling Guidelines

**CRITICAL RULES:**
1. **Always provide ALL required parameters** - Never omit required parameters like task_id, experiment_id, etc.
2. **Never call tools with empty names** - Always specify the exact tool name
3. **Use correct parameter names** - Check tool signatures carefully
4. **Don't pass 'db' parameter** - The database session is injected automatically

**Common Required Parameters:**
- get_task_by_id(task_id=<int>) - MUST provide task_id as integer
- get_experiment_details(experiment_id=<int>) - MUST provide experiment_id as integer
- list_task_experiments(task_id=<int>) - MUST provide task_id as integer
- get_task_results(task_id=<int>, include_all_experiments=<bool>) - task_id is required
- get_task_logs(task_id=<int>, tail_lines=<int optional>) - task_id is required

## Available Tool Categories (in order of preference)

1. **TASK tools** (High-level task management - USE THESE FIRST):
   - list_tasks() - List all tasks with optional filtering
   - get_task_by_id(task_id) - Get task details by ID [REQUIRES: task_id as int]
   - get_task_by_name(task_name) - Get task details by name [REQUIRES: task_name as str]
   - list_task_experiments(task_id) - List experiments for a task [REQUIRES: task_id as int]
   - get_task_results(task_id) - Get comprehensive task results [REQUIRES: task_id as int]
   - get_task_logs(task_id) - Get task execution logs [REQUIRES: task_id as int]
   - get_experiment_details(experiment_id) - Get experiment details [REQUIRES: experiment_id as int]
   - create_task, start_task, cancel_task, restart_task: Manage task lifecycle
   - delete_task, clear_task_data, update_task_description: Task operations

2. **PRESET tools** (High-level preset management):
   - list_presets, get_preset_by_id, get_preset_by_name: Query presets
   - create_preset, update_preset, delete_preset: Manage parameter presets

3. **DATABASE tools** (Low-level queries - use only if high-level tools don't work):
   - query_records, count_records: Direct database queries
   - Use only when TASK/PRESET tools are insufficient

4. **API tools** (External services):
   - search_huggingface_models, check_service_health: External API calls

## Best Practices

**ALWAYS:**
- Prefer TASK and PRESET tools over DATABASE tools
- Provide all required parameters with correct types
- Extract task IDs from user queries (e.g., "task 10" means task_id=10)
- Use get_task_results() for comprehensive task analysis
- Use list_task_experiments() to see all experiments before getting details

**NEVER:**
- Call tools without required parameters
- Pass 'db' in your tool arguments (it's auto-injected)
- Use empty string "" as tool name
- Use query_records when a specific TASK tool exists

**Example Correct Calls:**
- User asks "分析task 10": Call get_task_results(task_id=10)
- User asks "show experiment 87": Call get_experiment_details(experiment_id=87)
- User asks "list experiments for task 5": Call list_task_experiments(task_id=5)

## Common Scenario Workflows

**Scenario 1: Create new task based on existing task**
User: "参照task 5创建新任务，修改模型为llama-3-8b"
Step 1: get_task_by_id(task_id=5) - Get existing task configuration
Step 2: create_task(
  task_name="new-task-name",
  description="Based on task 5 with modified model",
  model_id="llama-3-8b-instruct",
  model_namespace=<from task 5>,
  base_runtime=<from task 5>,
  parameters=<from task 5>,
  ...other params from task 5
)

**Scenario 2: Analyze task failure through logs**
User: "分析task 8为什么失败了"
Step 1: get_task_by_id(task_id=8) - Check task status and error summary
Step 2: list_task_experiments(task_id=8, status_filter="FAILED") - Get failed experiments
Step 3: get_task_logs(task_id=8, tail_lines=100) - Get recent log entries
Step 4: For each failed experiment: get_experiment_details(experiment_id=X) - Get error details
Analysis: Look for common error patterns in logs (OOM, timeout, connection errors, etc.)

**Scenario 3: Analyze tuning results and parameter impact**
User: "分析task 10的调参结果，哪些参数最重要"
Step 1: get_task_results(task_id=10, include_all_experiments=True) - Get all experiments
Step 2: Analyze returned data:
   - Compare experiments with different parameter values
   - Identify which parameter changes correlate with performance improvements
   - Look at best_experiment parameters vs average experiments
Step 3: list_task_experiments(task_id=10, status_filter="SUCCESS") - Get successful experiments
Step 4: Compare top 3-5 experiments to identify key parameter patterns
Key metrics to analyze: objective_score, latency (p50/p90/p99), throughput, TTFT, TPOT

**Scenario 4: Monitor running task progress**
User: "task 15现在运行到哪了？"
Step 1: get_task_by_id(task_id=15) - Get task status, timing, and progress
Step 2: list_task_experiments(task_id=15) - Check completed experiments
Analysis:
   - Calculate progress: successful_experiments / total_experiments (if total known)
   - Show task status (PENDING/RUNNING/COMPLETED/FAILED)
   - Display elapsed time and estimated completion (if applicable)
   - Show latest experiment results if available

**Scenario 5: SLO violation analysis**
User: "task 10哪些实验违反了SLO约束？"
Step 1: analyze_slo_violations(task_id=10) - Get comprehensive SLO violation analysis
Step 2: Analyze the results:
   - Total violation count and rate
   - Hard fail vs soft penalty breakdown
   - Most frequently violated metrics
   - Parameter patterns in violating experiments
Step 3: Provide recommendations:
   - If ttft/tpot violations are high: Adjust memory fraction or scheduling policy
   - If latency violations: Consider lower concurrency or different tp-size
   - If throughput violations: Increase tp-size or adjust batch scheduling
Alternative: Use get_task_results(task_id=10, include_all_experiments=True) for manual analysis

**Scenario 6: Quick locate experiment failure cause**
User: "experiment 156为什么失败了？"
Step 1: get_experiment_details(experiment_id=156) - Get experiment details and error_message
Step 2: search_experiment_logs(task_id=<from step 1>, experiment_id=156, context_lines=15) - Find log entries
Analysis:
   - Check error_message for key patterns:
     * "OOM" / "out of memory" → Memory issues, try lower mem-fraction-static
     * "timeout" / "timed out" → Benchmark timeout, check model size vs GPU
     * "connection refused" / "failed to connect" → Inference service not ready
     * "CUDA error" → GPU resource issues
   - Review log context for detailed stack traces
   - Compare parameters with successful experiments to identify problematic values

High-level tools provide better error handling, formatted output, and business logic."""
			}]
			llm_messages.extend(recent_messages)
			llm_messages.append({"role": "user", "content": message_data.content})

			# 4. Get tools
			executor = ToolExecutor(session_id, db)
			available_tools = executor.get_available_tools(include_privileged=False)

			# 5. Multi-turn tool calling loop with streaming
			llm_client = get_llm_client()
			max_iterations = 10
			iteration = 0
			assistant_content = ""
			all_tool_calls = []  # Track all tool calls across iterations
			all_tool_results = []  # Track all tool results across iterations
			iteration_data = []  # Track content and tool calls for each iteration
			termination_reason = "natural"

			logger.info(f"Starting multi-turn stream with {len(available_tools)} tools available")

			while iteration < max_iterations:
				iteration += 1
				logger.info(f"Multi-turn streaming iteration {iteration}/{max_iterations} for session {session_id}")

				# Send iteration start event
				yield f"data: {json.dumps({'type': 'iteration_start', 'iteration': iteration, 'max_iterations': max_iterations})}\n\n"

				# 5a. Stream LLM response for this iteration
				tool_calls = []
				iteration_content = ""

				async for chunk in llm_client.chat_with_tools_stream(llm_messages, available_tools):
					if chunk["type"] == "content":
						iteration_content += chunk["content"]
						# Send content chunk to frontend
						yield f"data: {json.dumps({'type': 'content', 'content': chunk['content']})}\n\n"
						await asyncio.sleep(0.01)  # Small delay to prevent overwhelming client

					elif chunk["type"] == "done":
						iteration_content = chunk["content"]
						tool_calls = chunk["tool_calls"]

						# Filter out invalid tool calls with empty names (LLM sometimes returns these)
						original_count = len(tool_calls)
						tool_calls = [tc for tc in tool_calls if tc.get("name", "").strip()]
						if original_count != len(tool_calls):
							logger.warning(f"Filtered out {original_count - len(tool_calls)} tool calls with empty names")

				# Accumulate content across iterations
				if iteration_content:
					assistant_content = iteration_content

				# 5b. Check if LLM wants to stop (no tool calls)
				if not tool_calls:
					logger.info(f"LLM returned no tool calls - natural termination at iteration {iteration}")

					# Store final iteration data without tool calls
					iteration_data.append({
						"iteration": iteration,
						"content": iteration_content,
						"tool_calls": []
					})

					# Send iteration complete event
					yield f"data: {json.dumps({'type': 'iteration_complete', 'iteration': iteration, 'tool_calls_count': 0})}\n\n"
					break

				logger.info(f"Processing {len(tool_calls)} tool calls in iteration {iteration}")

				# Send tool calling status
				yield f"data: {json.dumps({'type': 'tool_start', 'tool_calls': tool_calls})}\n\n"

				# 5c. Execute all tool calls
				tool_results = await executor.execute_tool_calls(tool_calls)

				# Send tool results
				yield f"data: {json.dumps({'type': 'tool_results', 'results': tool_results})}\n\n"

				# Store iteration data (before any early exits)
				iteration_data.append({
					"iteration": iteration,
					"content": iteration_content,
					"tool_calls": [{
						"tool_name": tc["name"],
						"args": {k: v for k, v in tc["args"].items() if k != "db"},
						"id": tc["id"]
					} for tc in tool_calls]
				})

				# 5d. Check for authorization errors
				auth_required = []
				for result in tool_results:
					if not result["success"] and result.get("requires_auth") and not result.get("authorized"):
						auth_required.append({
							"tool_name": result["tool_name"],
							"auth_scope": result["auth_scope"]
						})

				# If any tools require authorization, stop loop and return auth request
				if auth_required:
					logger.info(f"Authorization required in iteration {iteration}, stopping multi-turn loop")
					termination_reason = "auth_required"

					assistant_message = ChatMessage(
						session_id=session_id,
						role=MessageRole.ASSISTANT,
						content=assistant_content if assistant_content else "I need authorization to perform some operations.",
						tool_calls=[{
							"tool_name": tc["name"],
							"args": {k: v for k, v in tc["args"].items() if k != "db"},
							"id": tc["id"],
							"status": "requires_auth",
							"auth_scope": next((r["auth_scope"] for r in tool_results if r["tool_name"] == tc["name"]), None)
						} for tc in tool_calls],
						message_metadata={
							"auth_required": auth_required,
							"iterations": iteration,
							"termination_reason": termination_reason,
							"iteration_data": iteration_data
						}
					)
					db.add(assistant_message)
					await db.commit()
					await db.refresh(assistant_message)

					# Send completion
					yield f"data: {json.dumps({'type': 'complete', 'message': {'id': assistant_message.id, 'content': assistant_message.content, 'tool_calls': assistant_message.tool_calls, 'created_at': assistant_message.created_at.isoformat()}})}\n\n"
					return

				# 5e. Check for execution errors
				# Let LLM see all tool results (including failures) so it can handle errors appropriately
				failed_tools = [r for r in tool_results if not r["success"]]
				if failed_tools:
					logger.warning(f"{len(failed_tools)} tools failed in iteration {iteration}, but letting LLM handle the errors")

				# 5f. Add assistant message with tool calls to context
				# Note: For Claude, we don't add tool messages separately (they get skipped)
				# Instead, we add the tool results as a user message summary
				llm_messages.append({
					"role": "assistant",
					"content": iteration_content if iteration_content else "(Called tools)"
				})

				# 5g. Add tool results as a user message (Claude-compatible format)
				# This allows Claude to see what the tools returned without needing tool_use/tool_result pairing
				tool_results_summary = []
				for result in tool_results:
					tool_name = result.get("tool_name", "unknown")
					success = result.get("success", True)
					result_text = result.get("result", "")
					# Truncate very long results
					if len(result_text) > 2000:
						result_text = result_text[:2000] + "... (truncated)"
					status = "SUCCESS" if success else "FAILED"
					tool_results_summary.append(f"[{tool_name}] {status}:\n{result_text}")

				llm_messages.append({
					"role": "user",
					"content": "Tool execution results:\n\n" + "\n\n---\n\n".join(tool_results_summary) + "\n\nPlease continue based on these results. If you have enough information, provide your final answer. If you need more information, call additional tools."
				})

				# 5h. Track all tool calls and results for database storage
				all_tool_calls.extend(tool_calls)
				all_tool_results.extend(tool_results)

				# Send iteration complete event
				yield f"data: {json.dumps({'type': 'iteration_complete', 'iteration': iteration, 'tool_calls_count': len(tool_calls)})}\n\n"

			# Check if max iterations reached
			if iteration >= max_iterations:
				logger.warning(f"Reached max iterations ({max_iterations}) for session {session_id}")
				termination_reason = "max_iterations"
				assistant_content += "\n\n[Note: Reached maximum thinking steps. Providing answer based on information gathered so far.]"

			# 6. Save final assistant message with complete tool execution history
			if all_tool_calls:
				# Find result and status for each tool call
				def find_result_for_call(call_id):
					for r in all_tool_results:
						if r.get("call_id") == call_id:
							return r.get("result"), r.get("success", True)
					# Fallback: match by tool_name if call_id doesn't match
					for r in all_tool_results:
						if r.get("tool_name") == call_id:
							return r.get("result"), r.get("success", True)
					return None, True

				def build_tool_call_entry(tc):
					result, success = find_result_for_call(tc["id"])
					return {
						"tool_name": tc["name"],
						"args": {k: v for k, v in tc["args"].items() if k != "db"},
						"id": tc["id"],
						"status": "executed" if success else "failed",
						"result": result if success else None,
						"error": result if not success else None
					}

				assistant_message = ChatMessage(
					session_id=session_id,
					role=MessageRole.ASSISTANT,
					content=assistant_content,
					tool_calls=[build_tool_call_entry(tc) for tc in all_tool_calls],
					message_metadata={
						"iterations": iteration,
						"termination_reason": termination_reason,
						"iteration_data": iteration_data
					}
				)
			else:
				# No tool calls at all - simple response
				assistant_message = ChatMessage(
					session_id=session_id,
					role=MessageRole.ASSISTANT,
					content=assistant_content
				)

			logger.info(f"Multi-turn streaming completed: {iteration} iterations, {len(all_tool_calls)} total tool calls, termination: {termination_reason}")

			db.add(assistant_message)
			await db.commit()
			await db.refresh(assistant_message)

			# 7. Update cache (include tool results)
			cached_entry = cache.get(session_id)
			if cached_entry:
				cached_entry.messages.append({"role": "user", "content": message_data.content})
				cached_entry.messages.append({"role": "assistant", "content": assistant_content})

				# Add tool results as ToolMessages if any
				if all_tool_calls:
					for tc in all_tool_calls:
						result = next((r.get("result") for r in all_tool_results if r.get("call_id") == tc["id"]), None)
						if result:
							cached_entry.messages.append({
								"role": "tool",
								"content": result,
								"tool_call_id": tc["id"]
							})

				cached_entry.messages = cached_entry.messages[-20:]

			# Send completion (include metadata for multi-iteration display)
			yield f"data: {json.dumps({'type': 'complete', 'message': {'id': assistant_message.id, 'content': assistant_message.content, 'tool_calls': assistant_message.tool_calls, 'metadata': assistant_message.message_metadata, 'created_at': assistant_message.created_at.isoformat()}})}\n\n"

		except Exception as e:
			logger.error(f"Error in stream: {str(e)}", exc_info=True)

			# Save error message to database before yielding error event
			# Also ensure user message is saved (in case error happened early)
			try:
				# Try to save user message if not already saved
				try:
					user_msg_check = await db.execute(
						select(ChatMessage)
						.where(ChatMessage.session_id == session_id)
						.where(ChatMessage.role == MessageRole.USER)
						.where(ChatMessage.content == message_data.content)
						.limit(1)
					)
					existing_user_msg = user_msg_check.scalar_one_or_none()

					if not existing_user_msg:
						# User message wasn't saved yet, save it now
						user_message_recovery = ChatMessage(
							session_id=session_id,
							role=MessageRole.USER,
							content=message_data.content
						)
						db.add(user_message_recovery)
						await db.commit()
						logger.info(f"Saved user message during error recovery for session {session_id}")
				except Exception as user_save_error:
					logger.error(f"Failed to save user message during recovery: {str(user_save_error)}")

				# Save assistant error message
				error_message = str(e)
				error_content = f"❌ **Error:** {error_message}"
				assistant_message = ChatMessage(
					session_id=session_id,
					role=MessageRole.ASSISTANT,
					content=error_content
				)
				db.add(assistant_message)
				await db.commit()
				await db.refresh(assistant_message)
				logger.info(f"Saved error message for session {session_id}")

				# Update cache to include this failed conversation
				# This ensures next message has proper context even after error
				cached_entry = cache.get(session_id)
				if cached_entry:
					cached_entry.messages.append({"role": "user", "content": message_data.content})
					cached_entry.messages.append({"role": "assistant", "content": error_content})
					cached_entry.messages = cached_entry.messages[-20:]
					logger.debug(f"Updated cache after error for session {session_id}")

			except Exception as save_error:
				logger.error(f"Failed to save error message: {str(save_error)}")

			yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

	return StreamingResponse(
		event_stream(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"Connection": "keep-alive",
			"X-Accel-Buffering": "no"  # Disable nginx buffering
		}
	)


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
