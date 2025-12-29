"""
Worker local configuration management.

Stores worker-specific configuration in a local file on the worker machine.
This persists across worker restarts and is independent of Redis/Manager.

Config file location: ~/.config/autotuner/worker.json
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Default config file location
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "autotuner"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "worker.json"


class WorkerLocalConfig(BaseModel):
    """Local worker configuration stored on the worker machine."""

    # Worker identification
    alias: Optional[str] = Field(None, description="Human-readable worker name")

    # Deployment settings
    deployment_mode: str = Field("docker", description="Deployment mode: docker, local, or ome")

    # Worker behavior
    max_parallel: int = Field(5, description="Maximum parallel jobs")

    # Optional metadata
    description: Optional[str] = Field(None, description="Worker description")
    tags: list[str] = Field(default_factory=list, description="Worker tags for filtering")

    # Manager connection (for remote workers)
    manager_ssh: Optional[str] = Field(None, description="SSH command to connect to manager")
    redis_host: str = Field("localhost", description="Redis host")
    redis_port: int = Field(6379, description="Redis port")


def get_config_path() -> Path:
    """Get the config file path, respecting AUTOTUNER_CONFIG_DIR env var."""
    config_dir = os.environ.get("AUTOTUNER_CONFIG_DIR")
    if config_dir:
        return Path(config_dir) / "worker.json"
    return DEFAULT_CONFIG_FILE


def load_worker_config() -> WorkerLocalConfig:
    """Load worker configuration from local file.

    Returns:
        WorkerLocalConfig with values from file or defaults
    """
    config_path = get_config_path()

    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            config = WorkerLocalConfig.model_validate(data)
            logger.info(f"Loaded worker config from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}, using defaults")

    return WorkerLocalConfig()


def save_worker_config(config: WorkerLocalConfig) -> bool:
    """Save worker configuration to local file.

    Args:
        config: Configuration to save

    Returns:
        True if saved successfully
    """
    config_path = get_config_path()

    try:
        # Ensure directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(config.model_dump(), f, indent=2)

        logger.info(f"Saved worker config to {config_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        return False


def update_worker_config(**kwargs) -> WorkerLocalConfig:
    """Update specific fields in worker configuration.

    Args:
        **kwargs: Fields to update

    Returns:
        Updated configuration
    """
    config = load_worker_config()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config field: {key}")

    save_worker_config(config)
    return config


def get_worker_alias() -> Optional[str]:
    """Get worker alias from config file or environment.

    Priority:
    1. WORKER_ALIAS environment variable (for backward compatibility)
    2. Config file alias
    """
    # Environment variable takes precedence for backward compatibility
    env_alias = os.environ.get("WORKER_ALIAS")
    if env_alias:
        return env_alias

    config = load_worker_config()
    return config.alias


def get_deployment_mode() -> str:
    """Get deployment mode from config file or environment.

    Priority:
    1. DEPLOYMENT_MODE environment variable
    2. Config file deployment_mode
    3. Default: "docker"
    """
    env_mode = os.environ.get("DEPLOYMENT_MODE")
    if env_mode:
        return env_mode

    config = load_worker_config()
    return config.deployment_mode
