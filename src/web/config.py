"""
Application configuration settings.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import os


def get_default_database_url() -> str:
	"""Get default database URL in user's home directory."""
	return f"sqlite+aiosqlite:///{Path.home()}/.local/share/inference-autotuner/autotuner.db"


class Settings(BaseSettings):
	"""Application settings."""

	model_config = SettingsConfigDict(
		env_file=str(Path(__file__).parent.parent.parent / ".env"),
		env_file_encoding='utf-8',
		case_sensitive=False,
		extra='ignore'  # Ignore frontend-only variables like VITE_*
	)

	# Application
	app_name: str = "LLM Inference Autotuner API"
	app_version: str = "0.1.0"
	debug: bool = True

	# Database
	# Store database in user's home directory by default
	# Use environment variable DATABASE_URL to override
	database_url: str = Field(default_factory=get_default_database_url)

	# Redis (for ARQ)
	redis_host: str = "localhost"
	redis_port: int = 6379
	redis_db: int = 0

	# CORS
	cors_origins: list = ["*"]  # Allow all origins in development

	# Autotuner
	docker_model_path: str = "/mnt/data/models"
	deployment_mode: str = "docker"

	# Proxy settings (optional)
	http_proxy: str = Field(default="", description="HTTP proxy URL (e.g., http://proxy.example.com:8080)")
	https_proxy: str = Field(default="", description="HTTPS proxy URL")
	no_proxy: str = Field(default="localhost,127.0.0.1", description="Comma-separated list of hosts to bypass proxy")

	# HuggingFace Token (optional, required for gated models)
	hf_token: str = Field(default="", description="HuggingFace access token for downloading gated models")

	# Timezone settings
	timezone: str = Field(default="UTC", description="Timezone for displaying timestamps (e.g., 'UTC', 'Asia/Shanghai', 'America/New_York')")


@lru_cache()
def get_settings() -> Settings:
	"""Get cached settings instance."""
	return Settings()
