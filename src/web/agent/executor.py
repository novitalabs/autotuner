"""
Tool executor with authorization management.

Handles tool execution with authorization checks for privileged operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.tools import BaseTool
import logging
import json

from web.agent.tools import get_tools_registry
from web.db.models import ChatSession

logger = logging.getLogger(__name__)


class AuthorizationError(Exception):
	"""Raised when tool requires authorization but user hasn't granted it."""
	pass


class ToolExecutor:
	"""Executes tools with authorization management."""

	def __init__(self, session_id: str, db: AsyncSession):
		"""
		Initialize executor for a chat session.

		Args:
			session_id: Chat session ID
			db: Database session
		"""
		self.session_id = session_id
		self.db = db
		self.registry = get_tools_registry()

	async def _check_authorization(self, scope: str) -> bool:
		"""
		Check if user has granted authorization for a scope.

		Args:
			scope: Authorization scope to check

		Returns:
			True if authorized, False otherwise
		"""
		from sqlalchemy import select

		# Clear SQLAlchemy cache to ensure we get fresh data
		# This is important when called right after authorization is granted
		self.db.expire_all()

		# Get session with metadata
		result = await self.db.execute(
			select(ChatSession).where(ChatSession.session_id == self.session_id)
		)
		session = result.scalar_one_or_none()

		if not session or not session.session_metadata:
			return False

		# Check tool_authorizations
		authorizations = session.session_metadata.get("tool_authorizations", {})
		if scope not in authorizations:
			return False

		grant = authorizations[scope]
		if not grant.get("granted"):
			return False

		# Check expiration
		expires_at = grant.get("expires_at")
		if expires_at:
			expiry_time = datetime.fromisoformat(expires_at)
			if datetime.utcnow() > expiry_time:
				logger.info(f"Authorization for '{scope}' expired in session {self.session_id}")
				return False

		return True

	async def execute_tool(
		self,
		tool_name: str,
		tool_args: Dict[str, Any]
	) -> Dict[str, Any]:
		"""
		Execute a tool with authorization checks.

		Args:
			tool_name: Name of the tool to execute
			tool_args: Arguments to pass to the tool

		Returns:
			Dict with:
				- 'success': bool
				- 'result': str (tool output or error message)
				- 'requires_auth': bool
				- 'auth_scope': str (if requires_auth)
				- 'authorized': bool (if requires_auth)

		Raises:
			AuthorizationError: If tool requires auth but not granted
		"""
		# Get tool from registry
		tool = self.registry.get_tool(tool_name)
		if not tool:
			return {
				"success": False,
				"result": f"Tool '{tool_name}' not found",
				"requires_auth": False
			}

		# Check if tool requires authorization
		requires_auth = getattr(tool, "_requires_authorization", False)
		auth_scope = getattr(tool, "_authorization_scope", None)

		if requires_auth and auth_scope:
			# Check authorization
			authorized = await self._check_authorization(auth_scope.value)

			if not authorized:
				logger.warning(
					f"Tool '{tool_name}' requires authorization scope '{auth_scope.value}' "
					f"but user has not granted it in session {self.session_id}"
				)
				return {
					"success": False,
					"result": f"Tool requires authorization: {auth_scope.value}",
					"requires_auth": True,
					"auth_scope": auth_scope.value,
					"authorized": False
				}

		# Execute tool
		try:
			# Make a copy of tool_args to avoid mutating the original
			execution_args = tool_args.copy()

			# Inject database session for tools that need it (database, task, preset tools)
			if hasattr(tool, "_tool_category"):
				category = tool._tool_category.value
				if category in ["database", "task", "preset"]:
					execution_args["db"] = self.db

			logger.info(f"Executing tool '{tool_name}' with args (excluding db): {tool_args}")

			# Execute the tool
			if hasattr(tool, "ainvoke"):
				# Async tool
				result = await tool.ainvoke(execution_args)
			else:
				# Sync tool (shouldn't happen with our tools, but handle it)
				result = tool.invoke(execution_args)

			logger.info(f"Tool '{tool_name}' executed successfully")

			return {
				"success": True,
				"result": result,
				"requires_auth": requires_auth,
				"auth_scope": auth_scope.value if auth_scope else None,
				"authorized": True
			}

		except Exception as e:
			logger.error(f"Error executing tool '{tool_name}': {str(e)}", exc_info=True)
			return {
				"success": False,
				"result": f"Tool execution failed: {str(e)}",
				"requires_auth": requires_auth,
				"auth_scope": auth_scope.value if auth_scope else None
			}

	async def execute_tool_calls(
		self,
		tool_calls: List[Dict[str, Any]]
	) -> List[Dict[str, Any]]:
		"""
		Execute multiple tool calls.

		Args:
			tool_calls: List of tool call dicts with 'name', 'args', 'id'

		Returns:
			List of execution results, one per tool call
		"""
		results = []

		for tool_call in tool_calls:
			tool_name = tool_call["name"]
			tool_args = tool_call["args"]
			call_id = tool_call.get("id", "")

			result = await self.execute_tool(tool_name, tool_args)
			result["call_id"] = call_id
			result["tool_name"] = tool_name

			results.append(result)

		return results

	def get_available_tools(self, include_privileged: bool = False) -> List[BaseTool]:
		"""
		Get list of available tools for this session.

		Args:
			include_privileged: Whether to include tools requiring authorization

		Returns:
			List of BaseTool objects with 'db' parameter filtered out from schemas
		"""
		if include_privileged:
			tools = self.registry.get_all_tools()
		else:
			tools = self.registry.get_safe_tools()

		# Filter out 'db' parameter from tool schemas
		# The 'db' parameter is injected at runtime by the executor,
		# but LangChain needs to generate JSON schemas for the LLM API,
		# and AsyncSession cannot be serialized to JSON schema.
		filtered_tools = []
		for tool in tools:
			# Create a copy of the tool with filtered args
			filtered_tool = self._filter_db_parameter(tool)
			filtered_tools.append(filtered_tool)

		return filtered_tools

	def _filter_db_parameter(self, tool: BaseTool) -> BaseTool:
		"""
		Create a copy of the tool with 'db' parameter removed from args schema.

		Args:
			tool: Original tool

		Returns:
			Tool with 'db' parameter filtered out
		"""
		from pydantic import create_model
		from typing import get_type_hints
		import inspect

		# Get the original function (coroutine for async tools, func for sync tools)
		if hasattr(tool, 'coroutine') and tool.coroutine is not None:
			original_func = tool.coroutine
		elif hasattr(tool, 'func') and tool.func is not None:
			original_func = tool.func
		else:
			return tool  # Not a StructuredTool, return as-is

		# Get function signature
		sig = inspect.signature(original_func)

		# Check if 'db' parameter exists
		if 'db' not in sig.parameters:
			return tool  # No 'db' parameter, return as-is

		# Create new args_schema without 'db' parameter
		if hasattr(tool, 'args_schema') and tool.args_schema:
			# Get all fields except 'db'
			original_schema = tool.args_schema
			fields_dict = {}

			for field_name, field_info in original_schema.model_fields.items():
				if field_name != 'db':
					fields_dict[field_name] = (field_info.annotation, field_info)

			# Create new Pydantic model without 'db'
			new_schema = create_model(
				f"{original_schema.__name__}Filtered",
				**fields_dict
			)

			# Create new tool with filtered schema
			from langchain_core.tools import StructuredTool

			filtered_tool = StructuredTool(
				name=tool.name,
				description=tool.description,
				func=original_func if not hasattr(tool, 'coroutine') else None,
				coroutine=original_func if hasattr(tool, 'coroutine') else None,
				args_schema=new_schema
			)

			# Copy metadata attributes
			if hasattr(tool, '_tool_category'):
				filtered_tool._tool_category = tool._tool_category
			if hasattr(tool, '_requires_authorization'):
				filtered_tool._requires_authorization = tool._requires_authorization
			if hasattr(tool, '_authorization_scope'):
				filtered_tool._authorization_scope = tool._authorization_scope

			return filtered_tool

		return tool
