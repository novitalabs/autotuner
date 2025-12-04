"""
LangChain-based LLM client for agent chat.
Supports OpenAI-compatible APIs (Jiekou, local models, OpenAI) and Claude.
"""

from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from web.config import get_settings


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


# Singleton instance
_llm_client: Optional[LangChainLLMClient] = None


def get_llm_client() -> LangChainLLMClient:
	"""Get or create LLM client singleton."""
	global _llm_client
	if _llm_client is None:
		_llm_client = LangChainLLMClient()
	return _llm_client
