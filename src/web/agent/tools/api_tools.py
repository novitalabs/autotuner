"""
External API tools for agent.

These tools allow the agent to interact with external services like HuggingFace.
All tools are SAFE (no authorization required) as they only query public APIs.
"""

from langchain_core.tools import tool
from web.agent.tools.base import register_tool, ToolCategory
import httpx
import json
import time


@tool
@register_tool(ToolCategory.API)
async def get_huggingface_model_info(model_id: str) -> str:
    """
    Get metadata for a HuggingFace model.

    Args:
        model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")

    Returns:
        JSON string with model metadata including author, downloads, likes, tags, etc.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"https://huggingface.co/api/models/{model_id}"
            )
            response.raise_for_status()
            data = response.json()

            return json.dumps({
                "model_id": data.get("id"),
                "author": data.get("author"),
                "downloads": data.get("downloads"),
                "likes": data.get("likes"),
                "tags": data.get("tags", []),
                "pipeline_tag": data.get("pipeline_tag"),
                "library_name": data.get("library_name"),
                "model_card_exists": data.get("cardData") is not None,
                "last_modified": data.get("lastModified")
            }, indent=2)
    except httpx.HTTPStatusError as e:
        return json.dumps({
            "error": f"HTTP {e.response.status_code}: Model not found or API error",
            "model_id": model_id
        })
    except Exception as e:
        return json.dumps({
            "error": f"Failed to fetch model info: {str(e)}",
            "model_id": model_id
        })


@tool
@register_tool(ToolCategory.API)
async def search_huggingface_models(
    query: str,
    limit: int = 5,
    filter_tag: str = None
) -> str:
    """
    Search for models on HuggingFace.

    Args:
        query: Search query string
        limit: Maximum number of results to return (default 5, max 20)
        filter_tag: Optional tag filter (e.g., "text-generation", "llama", "quantized")

    Returns:
        JSON string with search results
    """
    try:
        limit = min(limit, 20)  # Cap at 20

        params = {
            "search": query,
            "limit": limit
        }
        if filter_tag:
            params["filter"] = filter_tag

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://huggingface.co/api/models",
                params=params
            )
            response.raise_for_status()
            models = response.json()

            return json.dumps([{
                "model_id": m.get("id"),
                "author": m.get("author"),
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "tags": m.get("tags", [])[:5],  # First 5 tags
                "pipeline_tag": m.get("pipeline_tag")
            } for m in models], indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Search failed: {str(e)}",
            "query": query
        })


@tool
@register_tool(ToolCategory.API)
async def check_service_health(url: str) -> str:
    """
    Check health status of an HTTP endpoint.

    Args:
        url: URL to check (e.g., "http://localhost:8000/health")

    Returns:
        JSON string with health check result including status code and response time
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start = time.time()
            response = await client.get(url)
            duration = time.time() - start

            return json.dumps({
                "url": url,
                "status_code": response.status_code,
                "response_time_ms": int(duration * 1000),
                "healthy": 200 <= response.status_code < 300,
                "response_body": response.text[:200] if len(response.text) <= 200 else response.text[:200] + "..."
            }, indent=2)
    except httpx.TimeoutException:
        return json.dumps({
            "url": url,
            "error": "Request timed out after 5 seconds",
            "healthy": False
        })
    except httpx.ConnectError:
        return json.dumps({
            "url": url,
            "error": "Connection refused - service not reachable",
            "healthy": False
        })
    except Exception as e:
        return json.dumps({
            "url": url,
            "error": str(e),
            "healthy": False
        })
