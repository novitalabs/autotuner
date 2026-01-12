"""
HTTP readiness polling utilities for model servers.

Consolidates the repeated "wait until service is ready" logic
from Docker and Local controllers.
"""

import time
import requests
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


def wait_http_ready(
	base_url: str,
	timeout: int = 300,
	poll_interval: int = 2,
	health_path: str = "/health",
	models_path: str = "/v1/models",
	extra_check: Optional[Callable[[], Optional[str]]] = None,
) -> bool:
	"""
	Wait for HTTP service to become ready by polling endpoints.

	Checks both /health and /v1/models endpoints (common for LLM servers).
	Optionally runs an extra check function for platform-specific validation.

	Args:
	    base_url: Base URL of service (e.g., "http://localhost:8000")
	    timeout: Maximum time to wait in seconds
	    poll_interval: Time between polls in seconds
	    health_path: Health check endpoint path
	    models_path: Models list endpoint path
	    extra_check: Optional function that returns None if OK, error string if not
	                (e.g., to check container/process status)

	Returns:
	    True if service became ready, False if timeout or extra_check failed

	Example:
	    >>> def check_container():
	    ...     if container.status != 'running':
	    ...         return f"Container died: {container.status}"
	    ...     return None
	    >>>
	    >>> ready = wait_http_ready(
	    ...     "http://localhost:8000",
	    ...     timeout=120,
	    ...     extra_check=check_container
	    ... )
	"""
	start_time = time.time()
	last_error = None

	logger.info("Waiting for service at %s to become ready (timeout: %ds)", base_url, timeout)

	while time.time() - start_time < timeout:
		elapsed = int(time.time() - start_time)

		# Run platform-specific check if provided
		if extra_check:
			error = extra_check()
			if error:
				logger.error("Extra check failed: %s", error)
				return False

		# Try health endpoint
		try:
			response = requests.get(f"{base_url}{health_path}", timeout=5)

			if response.status_code == 200:
				logger.info("Health check passed after %ds", elapsed)

				# Also verify models endpoint
				try:
					models_response = requests.get(f"{base_url}{models_path}", timeout=5)
					if models_response.status_code == 200:
						logger.info("Service fully ready after %ds", elapsed)
						return True
					else:
						last_error = f"Models endpoint returned {models_response.status_code}"

				except requests.RequestException as e:
					last_error = f"Models endpoint error: {e}"

			else:
				last_error = f"Health check returned {response.status_code}"

		except requests.RequestException as e:
			last_error = f"Connection error: {e}"

		# Log progress periodically
		if elapsed > 0 and elapsed % 30 == 0:
			logger.info("Still waiting... (%ds elapsed, last error: %s)", elapsed, last_error)

		time.sleep(poll_interval)

	# Timeout
	logger.error("Service did not become ready within %ds (last error: %s)", timeout, last_error)
	return False


def check_http_endpoint(url: str, expected_status: int = 200, timeout: int = 5) -> tuple[bool, Optional[str]]:
	"""
	Check if an HTTP endpoint is responding correctly.

	Args:
	    url: Full URL to check
	    expected_status: Expected HTTP status code
	    timeout: Request timeout in seconds

	Returns:
	    (is_ok, error_message) tuple:
	    - is_ok: True if endpoint returned expected status
	    - error_message: None if OK, error description if not

	Example:
	    >>> is_ok, error = check_http_endpoint("http://localhost:8000/health")
	    >>> if not is_ok:
	    ...     print(f"Health check failed: {error}")
	"""
	try:
		response = requests.get(url, timeout=timeout)

		if response.status_code == expected_status:
			return True, None
		else:
			return False, f"Status {response.status_code} (expected {expected_status})"

	except requests.Timeout:
		return False, f"Timeout after {timeout}s"
	except requests.ConnectionError as e:
		return False, f"Connection error: {e}"
	except requests.RequestException as e:
		return False, f"Request error: {e}"


def wait_for_condition(
	condition_func: Callable[[], bool], timeout: int = 60, poll_interval: int = 1, description: str = "condition"
) -> bool:
	"""
	Generic wait-for-condition with timeout.

	Args:
	    condition_func: Function that returns True when ready
	    timeout: Maximum time to wait
	    poll_interval: Time between checks
	    description: Description for logging

	Returns:
	    True if condition met, False if timeout

	Example:
	    >>> def is_file_ready():
	    ...     return Path("/tmp/marker").exists()
	    >>>
	    >>> ready = wait_for_condition(is_file_ready, timeout=30, description="marker file")
	"""
	start_time = time.time()

	logger.info("Waiting for %s (timeout: %ds)", description, timeout)

	while time.time() - start_time < timeout:
		if condition_func():
			elapsed = int(time.time() - start_time)
			logger.info("%s met after %ds", description.capitalize(), elapsed)
			return True

		time.sleep(poll_interval)

	logger.error("%s not met within %ds", description.capitalize(), timeout)
	return False
