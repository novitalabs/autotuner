"""
LangChain-based LLM client for agent chat.
Supports OpenAI-compatible APIs (Jiekou, local models, OpenAI) and Claude (via native Anthropic SDK).
"""

from typing import List, Dict, Optional, Any, AsyncIterator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from web.config import get_settings
import logging
import json

logger = logging.getLogger(__name__)


class LangChainLLMClient:
	"""LLM client for chat completions. Uses LangChain for OpenAI-compatible APIs, native SDK for Anthropic."""

	def __init__(self):
		self.settings = get_settings()
		self._chat_model: Optional[BaseChatModel] = None
		self._anthropic_client = None  # Native Anthropic client for Claude provider

	def _get_chat_model(self) -> BaseChatModel:
		"""Get or create the appropriate chat model based on provider (OpenAI-compatible only)."""
		if self._chat_model is not None:
			return self._chat_model

		provider = self.settings.agent_provider

		if provider in ["jiekou", "local", "openai"]:
			# OpenAI-compatible API (Jiekou, local vLLM/SGLang, OpenAI)
			self._chat_model = ChatOpenAI(
				model=self.settings.agent_model,
				openai_api_base=self.settings.agent_base_url,
				openai_api_key=self.settings.agent_api_key or "dummy",  # Some providers need a key
				temperature=0.7,
				max_tokens=2000,
				timeout=180.0,  # Increased to 3 minutes for slow LLM responses
			)
		elif provider == "claude":
			# For Claude, we use native Anthropic SDK (not LangChain)
			# Return None here - methods will use _get_anthropic_client() instead
			return None
		else:
			raise ValueError(f"Unsupported provider: {provider}")

		return self._chat_model

	def _get_anthropic_client(self):
		"""Get or create native Anthropic client for Claude provider."""
		if self._anthropic_client is not None:
			return self._anthropic_client

		import anthropic

		# Support custom base_url (e.g., https://api.anthropic.com/)
		# Anthropic SDK expects base_url WITHOUT /v1 - it adds /v1/messages automatically
		base_url = self.settings.agent_base_url
		if base_url:
			# Remove trailing slash and /v1 if present
			base_url = base_url.rstrip('/')
			if base_url.endswith('/v1'):
				base_url = base_url[:-3]

		self._anthropic_client = anthropic.Anthropic(
			api_key=self.settings.agent_api_key,
			base_url=base_url if base_url else None,
			timeout=180.0,
		)
		return self._anthropic_client

	async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
		"""
		Send chat messages and get response.

		Args:
			messages: List of message dicts with 'role' and 'content'
			temperature: Sampling temperature (applied to model if supported)

		Returns:
			Assistant's response content
		"""
		provider = self.settings.agent_provider

		if provider == "claude":
			# Use native Anthropic SDK
			return await self._chat_anthropic(messages, temperature)

		# Convert dict messages to LangChain message objects
		langchain_messages = []
		for msg in messages:
			role = msg["role"]
			content = msg["content"]

			if role == "system":
				langchain_messages.append(SystemMessage(content=content))
			elif role == "user":
				langchain_messages.append(HumanMessage(content=content))
			elif role == "assistant":
				langchain_messages.append(AIMessage(content=content))

		# Get chat model and invoke
		chat_model = self._get_chat_model()

		# For temperature override, create new model instance
		if temperature != 0.7:
			if provider in ["jiekou", "local", "openai"]:
				chat_model = ChatOpenAI(
					model=self.settings.agent_model,
					openai_api_base=self.settings.agent_base_url,
					openai_api_key=self.settings.agent_api_key or "dummy",
					temperature=temperature,
					max_tokens=2000,
					timeout=180.0,
				)

		# Invoke the model asynchronously
		response = await chat_model.ainvoke(langchain_messages)
		return response.content

	async def _chat_anthropic(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
		"""Chat using native Anthropic SDK."""
		import asyncio

		client = self._get_anthropic_client()

		# Extract system message if present
		system_content = None
		api_messages = []
		for msg in messages:
			if msg["role"] == "system":
				system_content = msg["content"]
			else:
				api_messages.append({"role": msg["role"], "content": msg["content"]})

		# Run sync client in thread pool
		def sync_call():
			return client.messages.create(
				model=self.settings.agent_model,
				max_tokens=2000,
				system=system_content if system_content else "",
				messages=api_messages,
				temperature=temperature,
			)

		response = await asyncio.to_thread(sync_call)

		# Extract text from response
		if response.content and len(response.content) > 0:
			return response.content[0].text
		return ""

	async def chat_with_tools(
		self,
		messages: List[Dict[str, str]],
		tools: List[BaseTool],
		temperature: float = 0.7
	) -> Dict[str, Any]:
		"""
		Send chat messages with tool binding and get response.

		Args:
			messages: List of message dicts with 'role' and 'content'
			tools: List of LangChain BaseTool objects available for the LLM
			temperature: Sampling temperature

		Returns:
			Dict with:
				- 'content': Assistant's text response (if any)
				- 'tool_calls': List of tool calls (if any), each with:
					- 'name': Tool name
					- 'args': Tool arguments dict
					- 'id': Unique call ID
				- 'message': Full AIMessage object for context
		"""
		provider = self.settings.agent_provider

		if provider == "claude":
			# Use native Anthropic SDK
			return await self._chat_with_tools_anthropic(messages, tools, temperature)

		# Convert dict messages to LangChain message objects
		langchain_messages = []
		for msg in messages:
			role = msg["role"]
			content = msg.get("content", "")

			if role == "system":
				langchain_messages.append(SystemMessage(content=content))
			elif role == "user":
				langchain_messages.append(HumanMessage(content=content))
			elif role == "assistant":
				langchain_messages.append(AIMessage(content=content))
			elif role == "tool":
				# Tool result messages
				langchain_messages.append(ToolMessage(
					content=content,
					tool_call_id=msg.get("tool_call_id", "")
				))

		# Get chat model
		chat_model = self._get_chat_model()

		# Override temperature if needed
		if temperature != 0.7:
			if provider in ["jiekou", "local", "openai"]:
				chat_model = ChatOpenAI(
					model=self.settings.agent_model,
					openai_api_base=self.settings.agent_base_url,
					openai_api_key=self.settings.agent_api_key or "dummy",
					temperature=temperature,
					max_tokens=2000,
					timeout=180.0,
				)

		# Bind tools to the model
		chat_model_with_tools = chat_model.bind_tools(tools)

		# Invoke the model
		try:
			response = await chat_model_with_tools.ainvoke(langchain_messages)

			# Parse response
			result = {
				"content": response.content if response.content else "",
				"tool_calls": [],
				"message": response
			}

			# Extract tool calls if present
			if hasattr(response, "tool_calls") and response.tool_calls:
				logger.info(f"Raw LLM tool_calls: {response.tool_calls}")
				for tool_call in response.tool_calls:
					parsed_call = {
						"name": tool_call.get("name", ""),
						"args": tool_call.get("args", {}),
						"id": tool_call.get("id", "")
					}
					result["tool_calls"].append(parsed_call)
					logger.info(f"Parsed tool call: name='{parsed_call['name']}', id='{parsed_call['id']}', args={parsed_call['args']}")
				logger.info(f"LLM requested {len(result['tool_calls'])} tool calls")

			return result

		except Exception as e:
			logger.error(f"Error in chat_with_tools: {str(e)}")
			raise

	async def _chat_with_tools_anthropic(
		self,
		messages: List[Dict[str, str]],
		tools: List[BaseTool],
		temperature: float = 0.7
	) -> Dict[str, Any]:
		"""Chat with tools using native Anthropic SDK."""
		import asyncio

		client = self._get_anthropic_client()

		# Convert LangChain tools to Anthropic format
		anthropic_tools = []
		for tool in tools:
			tool_schema = {
				"name": tool.name,
				"description": tool.description,
				"input_schema": {
					"type": "object",
					"properties": {},
					"required": []
				}
			}

			# Extract schema from tool
			if hasattr(tool, "args_schema") and tool.args_schema:
				schema = tool.args_schema.model_json_schema()
				tool_schema["input_schema"]["properties"] = schema.get("properties", {})
				tool_schema["input_schema"]["required"] = schema.get("required", [])

			anthropic_tools.append(tool_schema)

		# Convert messages to Anthropic format
		# NOTE: Anthropic requires tool_result to have corresponding tool_use in previous assistant message
		# Our backend sends separate tool messages without tool_use blocks, so we skip orphan tool results
		system_content = None
		api_messages = []

		for msg in messages:
			role = msg["role"]
			content = msg.get("content", "")

			if role == "system":
				system_content = content
			elif role == "user":
				api_messages.append({"role": "user", "content": content})
			elif role == "assistant":
				# Only add non-empty assistant messages
				if content:
					api_messages.append({"role": "assistant", "content": content})
			elif role == "tool":
				# Skip tool results - they're not properly formatted for Anthropic API
				# The backend will call us fresh each iteration, so we don't need history
				logger.debug(f"Skipping orphan tool result message (tool_call_id={msg.get('tool_call_id', 'unknown')})")

		# Run sync client in thread pool
		def sync_call():
			kwargs = {
				"model": self.settings.agent_model,
				"max_tokens": 2000,
				"messages": api_messages,
				"temperature": temperature,
			}
			if system_content:
				kwargs["system"] = system_content
			if anthropic_tools:
				kwargs["tools"] = anthropic_tools

			return client.messages.create(**kwargs)

		try:
			response = await asyncio.to_thread(sync_call)

			# Parse response
			result = {
				"content": "",
				"tool_calls": [],
				"message": response
			}

			for block in response.content:
				if block.type == "text":
					result["content"] += block.text
				elif block.type == "tool_use":
					parsed_call = {
						"name": block.name,
						"args": block.input,
						"id": block.id
					}
					result["tool_calls"].append(parsed_call)
					logger.info(f"Anthropic tool call: name='{parsed_call['name']}', id='{parsed_call['id']}', args={parsed_call['args']}")

			if result["tool_calls"]:
				logger.info(f"Anthropic requested {len(result['tool_calls'])} tool calls")

			return result

		except Exception as e:
			logger.error(f"Error in _chat_with_tools_anthropic: {str(e)}")
			raise

	async def chat_with_tools_stream(
		self,
		messages: List[Dict[str, str]],
		tools: List[BaseTool],
		temperature: float = 0.7
	) -> AsyncIterator[Dict[str, Any]]:
		"""
		Stream chat messages with tool binding.

		Args:
			messages: List of message dicts with 'role' and 'content'
			tools: List of LangChain BaseTool objects available for the LLM
			temperature: Sampling temperature

		Yields:
			Dict chunks with:
				- 'type': 'content' | 'tool_call' | 'done'
				- 'content': Text chunk (for type='content')
				- 'tool_call': Tool call info (for type='tool_call')
				- 'complete_message': Full message (for type='done')
		"""
		provider = self.settings.agent_provider
		logger.info(f"chat_with_tools_stream called with {len(tools)} tools, {len(messages)} messages, provider={provider}")

		if provider == "claude":
			# Use native Anthropic SDK streaming
			async for chunk in self._chat_with_tools_stream_anthropic(messages, tools, temperature):
				yield chunk
			return

		# Convert dict messages to LangChain message objects
		langchain_messages = []
		for msg in messages:
			role = msg["role"]
			content = msg.get("content", "")

			if role == "system":
				langchain_messages.append(SystemMessage(content=content))
			elif role == "user":
				langchain_messages.append(HumanMessage(content=content))
			elif role == "assistant":
				langchain_messages.append(AIMessage(content=content))
			elif role == "tool":
				langchain_messages.append(ToolMessage(
					content=content,
					tool_call_id=msg.get("tool_call_id", "")
				))

		# Get chat model
		chat_model = self._get_chat_model()

		# Override temperature if needed
		if temperature != 0.7:
			if provider in ["jiekou", "local", "openai"]:
				chat_model = ChatOpenAI(
					model=self.settings.agent_model,
					openai_api_base=self.settings.agent_base_url,
					openai_api_key=self.settings.agent_api_key or "dummy",
					temperature=temperature,
					max_tokens=2000,
					timeout=180.0,
				)

		# Bind tools to the model
		chat_model_with_tools = chat_model.bind_tools(tools)

		# Stream the model
		try:
			full_content = ""
			tool_calls = []
			chunk_count = 0

			async for chunk in chat_model_with_tools.astream(langchain_messages):
				chunk_count += 1
				logger.debug(f"Stream chunk #{chunk_count}: has content={bool(chunk.content)}, has tool_calls={hasattr(chunk, 'tool_calls') and bool(chunk.tool_calls)}")

				# Handle content chunks
				if chunk.content:
					full_content += chunk.content
					yield {
						"type": "content",
						"content": chunk.content
					}

				# Handle tool calls (accumulated in final chunk)
				if hasattr(chunk, "tool_calls") and chunk.tool_calls:
					logger.info(f"Raw LLM tool_calls (stream): {chunk.tool_calls}")
					for tool_call in chunk.tool_calls:
						parsed_call = {
							"name": tool_call.get("name", ""),
							"args": tool_call.get("args", {}),
							"id": tool_call.get("id", "")
						}
						tool_calls.append(parsed_call)
						logger.info(f"Parsed tool call (stream): name='{parsed_call['name']}', id='{parsed_call['id']}', args={parsed_call['args']}")

			# Send complete message at end
			logger.info(f"Stream complete: {chunk_count} chunks, {len(tool_calls)} tool calls")
			if tool_calls:
				logger.info(f"Total tool calls collected in stream: {len(tool_calls)}")
			yield {
				"type": "done",
				"content": full_content,
				"tool_calls": tool_calls
			}

		except Exception as e:
			logger.error(f"Error in chat_with_tools_stream: {str(e)}")
			raise

	async def _chat_with_tools_stream_anthropic(
		self,
		messages: List[Dict[str, str]],
		tools: List[BaseTool],
		temperature: float = 0.7
	) -> AsyncIterator[Dict[str, Any]]:
		"""Stream chat with tools using native Anthropic SDK."""
		import asyncio

		client = self._get_anthropic_client()

		# Convert LangChain tools to Anthropic format
		anthropic_tools = []
		for tool in tools:
			tool_schema = {
				"name": tool.name,
				"description": tool.description,
				"input_schema": {
					"type": "object",
					"properties": {},
					"required": []
				}
			}

			# Extract schema from tool
			if hasattr(tool, "args_schema") and tool.args_schema:
				schema = tool.args_schema.model_json_schema()
				tool_schema["input_schema"]["properties"] = schema.get("properties", {})
				tool_schema["input_schema"]["required"] = schema.get("required", [])

			anthropic_tools.append(tool_schema)

		# Convert messages to Anthropic format
		# NOTE: Anthropic requires tool_result to have corresponding tool_use in previous assistant message
		# Our backend sends separate tool messages without tool_use blocks, so we skip orphan tool results
		system_content = None
		api_messages = []

		for msg in messages:
			role = msg["role"]
			content = msg.get("content", "")

			if role == "system":
				system_content = content
			elif role == "user":
				api_messages.append({"role": "user", "content": content})
			elif role == "assistant":
				# Only add non-empty assistant messages
				if content:
					api_messages.append({"role": "assistant", "content": content})
			elif role == "tool":
				# Skip tool results - they're not properly formatted for Anthropic API
				# The backend will call us fresh each iteration, so we don't need history
				logger.debug(f"Skipping orphan tool result message (tool_call_id={msg.get('tool_call_id', 'unknown')})")

		# Build kwargs
		kwargs = {
			"model": self.settings.agent_model,
			"max_tokens": 2000,
			"messages": api_messages,
			"temperature": temperature,
		}
		if system_content:
			kwargs["system"] = system_content
		if anthropic_tools:
			kwargs["tools"] = anthropic_tools

		# Use streaming
		try:
			full_content = ""
			tool_calls = []

			# Run sync stream in thread pool with queue for async iteration
			import queue
			import threading

			result_queue = queue.Queue()
			error_holder = [None]

			def stream_sync():
				try:
					with client.messages.stream(**kwargs) as stream:
						for text in stream.text_stream:
							result_queue.put(("text", text))
						# Get final message for tool calls
						response = stream.get_final_message()
						result_queue.put(("final", response))
				except Exception as e:
					error_holder[0] = e
				finally:
					result_queue.put(("done", None))

			# Start streaming thread
			thread = threading.Thread(target=stream_sync)
			thread.start()

			# Process queue items
			while True:
				# Wait for item with small timeout to allow async cooperation
				try:
					item = await asyncio.to_thread(result_queue.get, timeout=0.1)
				except:
					await asyncio.sleep(0.01)
					continue

				item_type, item_data = item

				if item_type == "text":
					full_content += item_data
					yield {
						"type": "content",
						"content": item_data
					}
				elif item_type == "final":
					# Extract tool calls from final message
					response = item_data
					for block in response.content:
						if block.type == "tool_use":
							parsed_call = {
								"name": block.name,
								"args": block.input,
								"id": block.id
							}
							tool_calls.append(parsed_call)
							logger.info(f"Anthropic tool call (stream): name='{parsed_call['name']}', id='{parsed_call['id']}'")
				elif item_type == "done":
					break

			thread.join()

			if error_holder[0]:
				raise error_holder[0]

			# Send complete message at end
			logger.info(f"Anthropic stream complete: {len(tool_calls)} tool calls")
			yield {
				"type": "done",
				"content": full_content,
				"tool_calls": tool_calls
			}

		except Exception as e:
			logger.error(f"Error in _chat_with_tools_stream_anthropic: {str(e)}")
			raise


# Singleton instance
_llm_client: Optional[LangChainLLMClient] = None


def get_llm_client() -> LangChainLLMClient:
	"""Get or create LLM client singleton."""
	global _llm_client
	if _llm_client is None:
		_llm_client = LangChainLLMClient()
	return _llm_client
