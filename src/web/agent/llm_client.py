"""
LangChain-based LLM client for agent chat.
Supports OpenAI-compatible APIs (Jiekou, local models, OpenAI) and Claude.
"""

from typing import List, Dict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from web.config import get_settings
import logging

logger = logging.getLogger(__name__)


class LangChainLLMClient:
	"""LangChain-based LLM client for chat completions."""

	def __init__(self):
		self.settings = get_settings()
		self._chat_model: Optional[BaseChatModel] = None

	def _get_chat_model(self) -> BaseChatModel:
		"""Get or create the appropriate chat model based on provider."""
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
				timeout=60.0,
			)
		elif provider == "claude":
			# Anthropic Claude API
			self._chat_model = ChatAnthropic(
				model=self.settings.agent_model,
				anthropic_api_key=self.settings.agent_api_key,
				temperature=0.7,
				max_tokens=2000,
				timeout=60.0,
			)
		else:
			raise ValueError(f"Unsupported provider: {provider}")

		return self._chat_model

	async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7) -> str:
		"""
		Send chat messages and get response using LangChain.

		Args:
			messages: List of message dicts with 'role' and 'content'
			temperature: Sampling temperature (applied to model if supported)

		Returns:
			Assistant's response content
		"""
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
			provider = self.settings.agent_provider
			if provider in ["jiekou", "local", "openai"]:
				chat_model = ChatOpenAI(
					model=self.settings.agent_model,
					openai_api_base=self.settings.agent_base_url,
					openai_api_key=self.settings.agent_api_key or "dummy",
					temperature=temperature,
					max_tokens=2000,
					timeout=60.0,
				)
			elif provider == "claude":
				chat_model = ChatAnthropic(
					model=self.settings.agent_model,
					anthropic_api_key=self.settings.agent_api_key,
					temperature=temperature,
					max_tokens=2000,
					timeout=60.0,
				)

		# Invoke the model asynchronously
		response = await chat_model.ainvoke(langchain_messages)
		return response.content

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
			provider = self.settings.agent_provider
			if provider in ["jiekou", "local", "openai"]:
				chat_model = ChatOpenAI(
					model=self.settings.agent_model,
					openai_api_base=self.settings.agent_base_url,
					openai_api_key=self.settings.agent_api_key or "dummy",
					temperature=temperature,
					max_tokens=2000,
					timeout=60.0,
				)
			elif provider == "claude":
				chat_model = ChatAnthropic(
					model=self.settings.agent_model,
					anthropic_api_key=self.settings.agent_api_key,
					temperature=temperature,
					max_tokens=2000,
					timeout=60.0,
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

	async def chat_with_tools_stream(
		self,
		messages: List[Dict[str, str]],
		tools: List[BaseTool],
		temperature: float = 0.7
	):
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
		logger.info(f"chat_with_tools_stream called with {len(tools)} tools, {len(messages)} messages")

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
			provider = self.settings.agent_provider
			if provider in ["jiekou", "local", "openai"]:
				chat_model = ChatOpenAI(
					model=self.settings.agent_model,
					openai_api_base=self.settings.agent_base_url,
					openai_api_key=self.settings.agent_api_key or "dummy",
					temperature=temperature,
					max_tokens=2000,
					timeout=60.0,
				)
			elif provider == "claude":
				chat_model = ChatAnthropic(
					model=self.settings.agent_model,
					anthropic_api_key=self.settings.agent_api_key,
					temperature=temperature,
					max_tokens=2000,
					timeout=60.0,
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
				logger.info(f"Stream chunk #{chunk_count}: has content={bool(chunk.content)}, has tool_calls={hasattr(chunk, 'tool_calls') and bool(chunk.tool_calls)}")

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


# Singleton instance
_llm_client: Optional[LangChainLLMClient] = None


def get_llm_client() -> LangChainLLMClient:
	"""Get or create LLM client singleton."""
	global _llm_client
	if _llm_client is None:
		_llm_client = LangChainLLMClient()
	return _llm_client
