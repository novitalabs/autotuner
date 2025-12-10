"""
HuggingFace CLI Tools for Agent

Tools for managing HuggingFace models using the hf CLI.
Includes download, cache management, and repo operations.
"""

import json
import subprocess
import re
from typing import Optional
from langchain_core.tools import tool
from .base import register_tool, ToolCategory


def _run_hf_command(args: list, timeout: int = 300) -> dict:
    """Run an hf CLI command and return structured result."""
    try:
        result = subprocess.run(
            ["hf"] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "returncode": -1
        }
    except FileNotFoundError:
        return {
            "success": False,
            "stdout": "",
            "stderr": "hf CLI not found. Install with: pip install huggingface_hub[cli]",
            "returncode": -1
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1
        }


@tool
@register_tool(ToolCategory.API)
def hf_cache_scan() -> str:
    """Scan the HuggingFace cache directory to see downloaded models.

    Returns:
        JSON with list of cached models, their sizes, and last access times.
    """
    result = _run_hf_command(["cache", "scan"], timeout=60)

    if not result["success"]:
        return json.dumps({
            "success": False,
            "error": result["stderr"]
        }, indent=2)

    # Parse the cache scan output
    output = result["stdout"]
    lines = output.strip().split("\n")

    # Parse summary line at the end
    summary = {}
    models = []

    for line in lines:
        # Parse model entries (they have REPO ID column)
        if line.startswith("REPO ID") or line.startswith("-"):
            continue

        # Look for summary line
        if "repos" in line.lower() and "revisions" in line.lower():
            # Summary line like: "Done in 0.1s. Scanned 5 repo(s) for a total of 15.2G."
            match = re.search(r"(\d+)\s*repo", line)
            if match:
                summary["total_repos"] = int(match.group(1))
            match = re.search(r"total of\s*([\d.]+[KMGT]?[B]?)", line, re.IGNORECASE)
            if match:
                summary["total_size"] = match.group(1)
            continue

        # Try to parse model line
        # Format: MODEL_ID  REVISIONS  SIZE  LAST_ACCESSED  REFS  LOCAL_PATH
        parts = line.split()
        if len(parts) >= 3:
            # Check if first part looks like a model ID (contains /)
            if "/" in parts[0]:
                models.append({
                    "model_id": parts[0],
                    "size": parts[2] if len(parts) > 2 else "unknown"
                })

    return json.dumps({
        "success": True,
        "summary": summary,
        "models": models,
        "raw_output": output[:2000] if len(output) > 2000 else output
    }, indent=2)


@tool
@register_tool(ToolCategory.API)
def hf_download_model(
    repo_id: str,
    local_dir: Optional[str] = None,
    include_pattern: Optional[str] = None,
    exclude_pattern: Optional[str] = None,
    revision: Optional[str] = None
) -> str:
    """Download a model from HuggingFace Hub.

    This downloads model files to the HuggingFace cache or a specified local directory.
    For large models, this may take significant time.

    Args:
        repo_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        local_dir: Optional local directory to download to (instead of HF cache)
        include_pattern: Glob pattern for files to include (e.g., "*.safetensors")
        exclude_pattern: Glob pattern for files to exclude (e.g., "*.bin")
        revision: Git revision (branch, tag, or commit hash)

    Returns:
        JSON with download result and local path
    """
    args = ["download", repo_id, "--quiet"]

    if local_dir:
        args.extend(["--local-dir", local_dir])

    if include_pattern:
        args.extend(["--include", include_pattern])

    if exclude_pattern:
        args.extend(["--exclude", exclude_pattern])

    if revision:
        args.extend(["--revision", revision])

    # Long timeout for large model downloads (30 minutes)
    result = _run_hf_command(args, timeout=1800)

    if not result["success"]:
        return json.dumps({
            "success": False,
            "repo_id": repo_id,
            "error": result["stderr"]
        }, indent=2)

    # The stdout contains the local path
    local_path = result["stdout"].strip()

    return json.dumps({
        "success": True,
        "repo_id": repo_id,
        "local_path": local_path,
        "message": f"Successfully downloaded {repo_id}"
    }, indent=2)


@tool
@register_tool(ToolCategory.API)
def hf_repo_info(repo_id: str) -> str:
    """Get information about a HuggingFace repository.

    Uses hf repo-files to list files in the repository.

    Args:
        repo_id: HuggingFace model/dataset ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")

    Returns:
        JSON with repository file list
    """
    args = ["repo-files", repo_id, "list"]
    result = _run_hf_command(args, timeout=30)

    if not result["success"]:
        return json.dumps({
            "success": False,
            "repo_id": repo_id,
            "error": result["stderr"]
        }, indent=2)

    # Parse file list
    files = [f.strip() for f in result["stdout"].strip().split("\n") if f.strip()]

    # Categorize files
    config_files = [f for f in files if f.endswith((".json", ".yaml", ".yml"))]
    model_files = [f for f in files if f.endswith((".safetensors", ".bin", ".pt", ".pth", ".gguf"))]
    tokenizer_files = [f for f in files if "tokenizer" in f.lower() or f.endswith(".model")]
    other_files = [f for f in files if f not in config_files + model_files + tokenizer_files]

    return json.dumps({
        "success": True,
        "repo_id": repo_id,
        "total_files": len(files),
        "config_files": config_files,
        "model_files": model_files,
        "tokenizer_files": tokenizer_files,
        "other_files": other_files[:20],  # Limit other files
        "message": f"Found {len(files)} files in {repo_id}"
    }, indent=2)


@tool
@register_tool(ToolCategory.API)
def hf_env_info() -> str:
    """Get HuggingFace environment information.

    Shows cache directory location, token status, and library versions.

    Returns:
        JSON with environment details
    """
    result = _run_hf_command(["env"], timeout=10)

    if not result["success"]:
        return json.dumps({
            "success": False,
            "error": result["stderr"]
        }, indent=2)

    # Parse environment info
    env_info = {}
    for line in result["stdout"].strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            env_info[key.strip()] = value.strip()

    return json.dumps({
        "success": True,
        "environment": env_info
    }, indent=2)


@tool
@register_tool(ToolCategory.API)
def hf_auth_status() -> str:
    """Check HuggingFace authentication status.

    Returns:
        JSON with login status and username if logged in
    """
    result = _run_hf_command(["auth", "whoami"], timeout=10)

    if not result["success"]:
        # Check if it's an auth error or other error
        if "not logged in" in result["stderr"].lower() or result["returncode"] == 1:
            return json.dumps({
                "success": True,
                "logged_in": False,
                "message": "Not logged in to HuggingFace"
            }, indent=2)
        return json.dumps({
            "success": False,
            "error": result["stderr"]
        }, indent=2)

    username = result["stdout"].strip()
    return json.dumps({
        "success": True,
        "logged_in": True,
        "username": username,
        "message": f"Logged in as {username}"
    }, indent=2)
