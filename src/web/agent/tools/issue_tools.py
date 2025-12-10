"""
GitHub Issue Management Tools

Tools for reporting bugs/issues to GitHub and querying existing issues.
Uses local cache for faster lookup and offline access.
"""

import json
import httpx
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from .base import register_tool, ToolCategory
from ...config import get_settings


# Cache file location
CACHE_DIR = Path.home() / ".local" / "share" / "inference-autotuner"
CACHE_FILE = CACHE_DIR / "issues_cache.json"

# GitHub API headers
GITHUB_API_VERSION = "2022-11-28"


def _get_github_headers(token: str = "") -> dict:
	"""Get headers for GitHub API requests."""
	headers = {
		"Accept": "application/vnd.github+json",
		"X-GitHub-Api-Version": GITHUB_API_VERSION,
	}
	if token:
		headers["Authorization"] = f"Bearer {token}"
	return headers


def _load_cache() -> dict:
	"""Load issues from local cache."""
	if CACHE_FILE.exists():
		try:
			with open(CACHE_FILE, "r") as f:
				return json.load(f)
		except (json.JSONDecodeError, IOError):
			return {"last_updated": None, "repo": "", "issues": []}
	return {"last_updated": None, "repo": "", "issues": []}


def _save_cache(cache: dict) -> None:
	"""Save issues to local cache."""
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	with open(CACHE_FILE, "w") as f:
		json.dump(cache, f, indent=2)


def _search_cache(query: str, cache: dict) -> list:
	"""Search issues in cache by query string."""
	query_lower = query.lower()
	results = []
	for issue in cache.get("issues", []):
		title = issue.get("title", "").lower()
		body = issue.get("body", "").lower()
		if query_lower in title or query_lower in body:
			results.append(issue)
	return results


@tool
@register_tool(ToolCategory.API)
async def search_known_issues(query: str) -> str:
	"""Search for existing GitHub issues related to a query.

	First searches local cache, then GitHub API if no matches found.
	Use this before creating a new issue to avoid duplicates.

	Args:
		query: Search string to find in issue titles and bodies

	Returns:
		JSON with matching issues or error message
	"""
	settings = get_settings()

	# Search local cache first
	cache = _load_cache()
	cached_results = _search_cache(query, cache)

	if cached_results:
		return json.dumps({
			"success": True,
			"source": "cache",
			"count": len(cached_results),
			"issues": [
				{
					"number": i.get("number"),
					"title": i.get("title"),
					"state": i.get("state"),
					"labels": i.get("labels", []),
					"url": i.get("url"),
					"created_at": i.get("created_at")
				}
				for i in cached_results[:10]  # Limit to 10 results
			],
			"cache_updated": cache.get("last_updated")
		}, indent=2)

	# If no cached results and repo is configured, search GitHub
	if not settings.gh_repo:
		return json.dumps({
			"success": True,
			"source": "cache",
			"count": 0,
			"issues": [],
			"message": "No matching issues in cache. GH_REPO not configured for GitHub search."
		}, indent=2)

	# Search GitHub API
	try:
		# Use proxy if configured
		proxy = settings.https_proxy or settings.http_proxy or None

		async with httpx.AsyncClient(proxy=proxy, timeout=30.0) as client:
			# Search issues using GitHub search API
			search_url = f"https://api.github.com/search/issues?q={query}+repo:{settings.gh_repo}+is:issue"
			response = await client.get(
				search_url,
				headers=_get_github_headers(settings.gh_token)
			)

			if response.status_code == 200:
				data = response.json()
				items = data.get("items", [])

				# Update cache with found issues
				for item in items:
					_update_cache_issue(cache, item)
				cache["last_updated"] = datetime.now(timezone.utc).isoformat()
				_save_cache(cache)

				return json.dumps({
					"success": True,
					"source": "github",
					"count": len(items),
					"issues": [
						{
							"number": i.get("number"),
							"title": i.get("title"),
							"state": i.get("state"),
							"labels": [l.get("name") for l in i.get("labels", [])],
							"url": i.get("html_url"),
							"created_at": i.get("created_at")
						}
						for i in items[:10]
					]
				}, indent=2)
			elif response.status_code == 403:
				return json.dumps({
					"success": False,
					"error": "GitHub API rate limited. Try again later or use cached results.",
					"cached_count": len(cache.get("issues", []))
				}, indent=2)
			else:
				return json.dumps({
					"success": False,
					"error": f"GitHub API error: {response.status_code}",
					"message": response.text[:200]
				}, indent=2)

	except httpx.RequestError as e:
		return json.dumps({
			"success": False,
			"error": f"Network error: {str(e)}",
			"message": "Falling back to cache only"
		}, indent=2)


def _update_cache_issue(cache: dict, github_issue: dict) -> None:
	"""Update or add an issue in the cache."""
	issue_data = {
		"number": github_issue.get("number"),
		"title": github_issue.get("title"),
		"body": github_issue.get("body", ""),
		"state": github_issue.get("state"),
		"labels": [l.get("name") for l in github_issue.get("labels", [])],
		"created_at": github_issue.get("created_at"),
		"updated_at": github_issue.get("updated_at"),
		"url": github_issue.get("html_url")
	}

	# Update existing or add new
	issues = cache.get("issues", [])
	for i, existing in enumerate(issues):
		if existing.get("number") == issue_data["number"]:
			issues[i] = issue_data
			return
	issues.append(issue_data)
	cache["issues"] = issues


@tool
@register_tool(ToolCategory.API)
async def create_issue(title: str, body: str, labels: Optional[str] = None) -> str:
	"""Create a new GitHub issue for reporting bugs or feature requests.

	Before calling this, use search_known_issues() to check for duplicates.
	Requires GH_TOKEN and GH_REPO to be configured.

	Args:
		title: Issue title (concise description of the problem)
		body: Issue body in markdown format (include description, steps to reproduce, expected behavior)
		labels: Comma-separated list of labels (e.g., "bug,autotuner")

	Returns:
		JSON with created issue URL or error message
	"""
	settings = get_settings()

	# Validate configuration
	if not settings.gh_token:
		return json.dumps({
			"success": False,
			"error": "GH_TOKEN not configured",
			"message": "Set GH_TOKEN environment variable with a GitHub personal access token that has 'repo' scope."
		}, indent=2)

	if not settings.gh_repo:
		return json.dumps({
			"success": False,
			"error": "GH_REPO not configured",
			"message": "Set GH_REPO environment variable in format 'owner/repo' (e.g., 'myorg/inference-autotuner')."
		}, indent=2)

	# Parse labels
	label_list = []
	if labels:
		label_list = [l.strip() for l in labels.split(",") if l.strip()]

	# Create issue via GitHub API
	try:
		proxy = settings.https_proxy or settings.http_proxy or None

		async with httpx.AsyncClient(proxy=proxy, timeout=30.0) as client:
			create_url = f"https://api.github.com/repos/{settings.gh_repo}/issues"
			payload = {
				"title": title,
				"body": body,
			}
			if label_list:
				payload["labels"] = label_list

			response = await client.post(
				create_url,
				headers=_get_github_headers(settings.gh_token),
				json=payload
			)

			if response.status_code == 201:
				data = response.json()

				# Update local cache
				cache = _load_cache()
				cache["repo"] = settings.gh_repo
				_update_cache_issue(cache, data)
				cache["last_updated"] = datetime.now(timezone.utc).isoformat()
				_save_cache(cache)

				return json.dumps({
					"success": True,
					"issue_number": data.get("number"),
					"url": data.get("html_url"),
					"title": data.get("title"),
					"state": data.get("state"),
					"message": f"Issue #{data.get('number')} created successfully"
				}, indent=2)
			elif response.status_code == 401:
				return json.dumps({
					"success": False,
					"error": "Authentication failed",
					"message": "GH_TOKEN is invalid or expired. Generate a new token with 'repo' scope."
				}, indent=2)
			elif response.status_code == 403:
				return json.dumps({
					"success": False,
					"error": "Permission denied or rate limited",
					"message": "Check that GH_TOKEN has write access to the repository."
				}, indent=2)
			elif response.status_code == 404:
				return json.dumps({
					"success": False,
					"error": "Repository not found",
					"message": f"GH_REPO '{settings.gh_repo}' does not exist or token lacks access."
				}, indent=2)
			else:
				return json.dumps({
					"success": False,
					"error": f"GitHub API error: {response.status_code}",
					"message": response.text[:300]
				}, indent=2)

	except httpx.RequestError as e:
		return json.dumps({
			"success": False,
			"error": f"Network error: {str(e)}",
			"message": "Could not connect to GitHub API"
		}, indent=2)


@tool
@register_tool(ToolCategory.API)
async def refresh_issues_cache() -> str:
	"""Refresh the local issues cache from GitHub.

	Fetches all open issues from the configured repository
	and updates the local cache for faster future lookups.

	Returns:
		JSON with count of cached issues or error message
	"""
	settings = get_settings()

	if not settings.gh_repo:
		return json.dumps({
			"success": False,
			"error": "GH_REPO not configured",
			"message": "Set GH_REPO environment variable to refresh cache."
		}, indent=2)

	try:
		proxy = settings.https_proxy or settings.http_proxy or None

		async with httpx.AsyncClient(proxy=proxy, timeout=60.0) as client:
			# Fetch open issues (paginated)
			all_issues = []
			page = 1
			per_page = 100

			while True:
				list_url = f"https://api.github.com/repos/{settings.gh_repo}/issues"
				params = {
					"state": "all",  # Get both open and closed
					"per_page": per_page,
					"page": page
				}

				response = await client.get(
					list_url,
					params=params,
					headers=_get_github_headers(settings.gh_token)
				)

				if response.status_code != 200:
					if response.status_code == 403:
						# Rate limited, save what we have
						break
					return json.dumps({
						"success": False,
						"error": f"GitHub API error: {response.status_code}",
						"message": response.text[:200]
					}, indent=2)

				issues = response.json()
				if not issues:
					break

				# Filter out pull requests (they also appear in issues endpoint)
				issues = [i for i in issues if "pull_request" not in i]
				all_issues.extend(issues)

				if len(issues) < per_page:
					break
				page += 1

				# Safety limit
				if page > 10:
					break

			# Build new cache
			cache = {
				"last_updated": datetime.now(timezone.utc).isoformat(),
				"repo": settings.gh_repo,
				"issues": []
			}

			for issue in all_issues:
				_update_cache_issue(cache, issue)

			_save_cache(cache)

			open_count = sum(1 for i in cache["issues"] if i.get("state") == "open")
			closed_count = len(cache["issues"]) - open_count

			return json.dumps({
				"success": True,
				"total_issues": len(cache["issues"]),
				"open_issues": open_count,
				"closed_issues": closed_count,
				"last_updated": cache["last_updated"],
				"repo": settings.gh_repo,
				"message": f"Cache refreshed with {len(cache['issues'])} issues"
			}, indent=2)

	except httpx.RequestError as e:
		return json.dumps({
			"success": False,
			"error": f"Network error: {str(e)}",
			"message": "Could not connect to GitHub API"
		}, indent=2)


@tool
@register_tool(ToolCategory.API)
async def get_issue_by_number(issue_number: int) -> str:
	"""Get details of a specific GitHub issue by its number.

	Args:
		issue_number: The issue number to retrieve

	Returns:
		JSON with full issue details or error message
	"""
	settings = get_settings()

	# Check local cache first
	cache = _load_cache()
	for issue in cache.get("issues", []):
		if issue.get("number") == issue_number:
			return json.dumps({
				"success": True,
				"source": "cache",
				"issue": issue
			}, indent=2)

	# Not in cache, fetch from GitHub
	if not settings.gh_repo:
		return json.dumps({
			"success": False,
			"error": "Issue not in cache and GH_REPO not configured",
			"message": "Configure GH_REPO to fetch from GitHub"
		}, indent=2)

	try:
		proxy = settings.https_proxy or settings.http_proxy or None

		async with httpx.AsyncClient(proxy=proxy, timeout=30.0) as client:
			issue_url = f"https://api.github.com/repos/{settings.gh_repo}/issues/{issue_number}"

			response = await client.get(
				issue_url,
				headers=_get_github_headers(settings.gh_token)
			)

			if response.status_code == 200:
				data = response.json()

				# Update cache
				_update_cache_issue(cache, data)
				cache["last_updated"] = datetime.now(timezone.utc).isoformat()
				_save_cache(cache)

				return json.dumps({
					"success": True,
					"source": "github",
					"issue": {
						"number": data.get("number"),
						"title": data.get("title"),
						"body": data.get("body"),
						"state": data.get("state"),
						"labels": [l.get("name") for l in data.get("labels", [])],
						"created_at": data.get("created_at"),
						"updated_at": data.get("updated_at"),
						"url": data.get("html_url"),
						"author": data.get("user", {}).get("login")
					}
				}, indent=2)
			elif response.status_code == 404:
				return json.dumps({
					"success": False,
					"error": f"Issue #{issue_number} not found",
					"message": f"No issue with number {issue_number} exists in {settings.gh_repo}"
				}, indent=2)
			else:
				return json.dumps({
					"success": False,
					"error": f"GitHub API error: {response.status_code}",
					"message": response.text[:200]
				}, indent=2)

	except httpx.RequestError as e:
		return json.dumps({
			"success": False,
			"error": f"Network error: {str(e)}",
			"message": "Could not connect to GitHub API"
		}, indent=2)
