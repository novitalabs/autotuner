"""Docker container management API endpoints."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime
import docker
from docker.errors import NotFound, APIError
import asyncio

router = APIRouter(tags=["docker"])


class ContainerInfo(BaseModel):
	"""Docker container information model."""

	id: str
	short_id: str
	name: str
	image: str
	status: str
	state: str
	created: str
	started_at: Optional[str] = None
	finished_at: Optional[str] = None
	ports: dict
	labels: dict
	command: Optional[str] = None


class ContainerStats(BaseModel):
	"""Docker container resource statistics."""

	cpu_percent: float
	memory_usage: str
	memory_limit: str
	memory_percent: float
	network_rx: str
	network_tx: str
	block_read: str
	block_write: str


class ContainerLogs(BaseModel):
	"""Docker container logs."""

	logs: str
	lines: int


def get_docker_client():
	"""Get Docker client instance."""
	try:
		client = docker.from_env()
		# Test connection
		client.ping()
		return client
	except Exception as e:
		raise HTTPException(status_code=503, detail=f"Unable to connect to Docker daemon: {str(e)}")


def format_bytes(bytes_value: int) -> str:
	"""Format bytes to human-readable string."""
	for unit in ["B", "KB", "MB", "GB", "TB"]:
		if bytes_value < 1024.0:
			return f"{bytes_value:.2f} {unit}"
		bytes_value /= 1024.0
	return f"{bytes_value:.2f} PB"


async def stream_container_logs(container_id: str, follow: bool = True):
	"""
	Stream Docker container logs in real-time.

	Args:
	    container_id: Container ID or name
	    follow: Whether to follow new log lines

	Yields:
	    Server-Sent Events formatted log lines
	"""
	client = None
	try:
		client = get_docker_client()
		container = client.containers.get(container_id)

		# Stream logs from Docker
		log_stream = container.logs(stream=True, follow=follow, timestamps=False, tail=500)  # Start with last 500 lines

		for log_line in log_stream:
			try:
				# Decode and send each log line
				line = log_line.decode("utf-8", errors="replace").rstrip()
				if line:  # Only send non-empty lines
					yield f"data: {line}\n\n"
					await asyncio.sleep(0.01)  # Small delay to prevent overwhelming
			except Exception as e:
				yield f"data: Error decoding log line: {str(e)}\n\n"
				break

	except NotFound:
		yield f"data: Container {container_id} not found\n\n"
	except APIError as e:
		yield f"data: Docker API error: {str(e)}\n\n"
	except Exception as e:
		yield f"data: Error streaming logs: {str(e)}\n\n"
	finally:
		if client:
			client.close()


def parse_container_info(container) -> ContainerInfo:
	"""Parse Docker container object to ContainerInfo model."""
	# Reload to get fresh data
	container.reload()

	# Get container attributes
	attrs = container.attrs
	state = attrs.get("State", {})
	config = attrs.get("Config", {})
	network_settings = attrs.get("NetworkSettings", {})

	# Parse ports
	ports = {}
	port_bindings = network_settings.get("Ports", {})
	if port_bindings:
		for container_port, host_bindings in port_bindings.items():
			if host_bindings:
				for binding in host_bindings:
					host_port = binding.get("HostPort", "")
					if host_port:
						ports[container_port] = host_port

	# Get command
	cmd = config.get("Cmd", [])
	command = " ".join(cmd) if cmd else None

	return ContainerInfo(
		id=container.id,
		short_id=container.short_id,
		name=container.name,
		image=attrs.get("Config", {}).get("Image", ""),
		status=container.status,
		state=state.get("Status", "unknown"),
		created=attrs.get("Created", ""),
		started_at=state.get("StartedAt"),
		finished_at=state.get("FinishedAt"),
		ports=ports,
		labels=config.get("Labels", {}),
		command=command,
	)


@router.get("/containers", response_model=List[ContainerInfo])
async def list_containers(all: bool = True):
	"""
	List all Docker containers.

	Args:
	    all: If True, include stopped containers. Default True.

	Returns:
	    List of container information.
	"""
	client = get_docker_client()

	try:
		containers = client.containers.list(all=all)
		return [parse_container_info(container) for container in containers]
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	finally:
		client.close()


@router.get("/containers/{container_id}", response_model=ContainerInfo)
async def get_container(container_id: str):
	"""
	Get detailed information about a specific container.

	Args:
	    container_id: Container ID or name.

	Returns:
	    Container information.
	"""
	client = get_docker_client()

	try:
		container = client.containers.get(container_id)
		return parse_container_info(container)
	except NotFound:
		raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	finally:
		client.close()


@router.get("/containers/{container_id}/logs")
async def get_container_logs(
	container_id: str,
	tail: int = 1000,
	timestamps: bool = False,
	follow: bool = False,
	since: Optional[str] = None,
):
	"""
	Get logs from a specific container.

	Args:
	    container_id: Container ID or name.
	    tail: Number of lines to return from the end. Default 1000.
	    timestamps: Include timestamps in logs. Default False.
	    follow: If True, streams logs in real-time (Server-Sent Events). Default False.
	    since: Show logs since timestamp (ISO 8601 format).

	Returns:
	    Container logs (static) or streaming response.
	"""
	# If follow mode, return streaming response (Server-Sent Events)
	if follow:
		return StreamingResponse(
			stream_container_logs(container_id, follow=True),
			media_type="text/event-stream",
			headers={
				"Cache-Control": "no-cache",
				"Connection": "keep-alive",
				"X-Accel-Buffering": "no",  # Disable nginx buffering
			},
		)

	# Otherwise return static logs
	client = get_docker_client()

	try:
		container = client.containers.get(container_id)

		# Build logs kwargs
		logs_kwargs = {"tail": tail, "timestamps": timestamps}
		if since:
			logs_kwargs["since"] = since

		# Get logs
		logs = container.logs(**logs_kwargs).decode("utf-8", errors="replace")
		lines = len(logs.split("\n"))

		return ContainerLogs(logs=logs, lines=lines)
	except NotFound:
		raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	finally:
		client.close()


@router.get("/containers/{container_id}/stats", response_model=ContainerStats)
async def get_container_stats(container_id: str):
	"""
	Get resource usage statistics for a specific container.

	Args:
	    container_id: Container ID or name.

	Returns:
	    Container resource statistics.
	"""
	client = get_docker_client()

	try:
		container = client.containers.get(container_id)

		# Get stats (stream=False for single snapshot)
		stats = container.stats(stream=False)

		# Calculate CPU percentage
		cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
		system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
		num_cpus = stats["cpu_stats"]["online_cpus"]

		cpu_percent = 0.0
		if system_delta > 0 and cpu_delta > 0:
			cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0

		# Memory stats
		memory_usage = stats["memory_stats"]["usage"]
		memory_limit = stats["memory_stats"]["limit"]
		memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0

		# Network stats
		networks = stats.get("networks", {})
		network_rx = sum(net["rx_bytes"] for net in networks.values())
		network_tx = sum(net["tx_bytes"] for net in networks.values())

		# Block I/O stats
		blkio_stats = stats.get("blkio_stats", {}).get("io_service_bytes_recursive", [])
		block_read = sum(entry["value"] for entry in blkio_stats if entry.get("op") == "read")
		block_write = sum(entry["value"] for entry in blkio_stats if entry.get("op") == "write")

		return ContainerStats(
			cpu_percent=round(cpu_percent, 2),
			memory_usage=format_bytes(memory_usage),
			memory_limit=format_bytes(memory_limit),
			memory_percent=round(memory_percent, 2),
			network_rx=format_bytes(network_rx),
			network_tx=format_bytes(network_tx),
			block_read=format_bytes(block_read),
			block_write=format_bytes(block_write),
		)
	except NotFound:
		raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	except KeyError as e:
		raise HTTPException(status_code=500, detail=f"Failed to parse stats: missing key {str(e)}")
	finally:
		client.close()


@router.post("/containers/{container_id}/start")
async def start_container(container_id: str):
	"""
	Start a stopped container.

	Args:
	    container_id: Container ID or name.

	Returns:
	    Success message.
	"""
	client = get_docker_client()

	try:
		container = client.containers.get(container_id)
		container.start()
		return {"message": f"Container {container_id} started successfully"}
	except NotFound:
		raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	finally:
		client.close()


@router.post("/containers/{container_id}/stop")
async def stop_container(container_id: str, timeout: int = 10):
	"""
	Stop a running container.

	Args:
	    container_id: Container ID or name.
	    timeout: Timeout in seconds before killing. Default 10.

	Returns:
	    Success message.
	"""
	client = get_docker_client()

	try:
		container = client.containers.get(container_id)
		container.stop(timeout=timeout)
		return {"message": f"Container {container_id} stopped successfully"}
	except NotFound:
		raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	finally:
		client.close()


@router.post("/containers/{container_id}/restart")
async def restart_container(container_id: str, timeout: int = 10):
	"""
	Restart a container.

	Args:
	    container_id: Container ID or name.
	    timeout: Timeout in seconds before killing. Default 10.

	Returns:
	    Success message.
	"""
	client = get_docker_client()

	try:
		container = client.containers.get(container_id)
		container.restart(timeout=timeout)
		return {"message": f"Container {container_id} restarted successfully"}
	except NotFound:
		raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	finally:
		client.close()


@router.delete("/containers/{container_id}")
async def remove_container(container_id: str, force: bool = False):
	"""
	Remove a container.

	Args:
	    container_id: Container ID or name.
	    force: Force removal of running container. Default False.

	Returns:
	    Success message.
	"""
	client = get_docker_client()

	try:
		container = client.containers.get(container_id)
		container.remove(force=force)
		return {"message": f"Container {container_id} removed successfully"}
	except NotFound:
		raise HTTPException(status_code=404, detail=f"Container {container_id} not found")
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	finally:
		client.close()


@router.get("/info")
async def get_docker_info():
	"""
	Get Docker daemon information.

	Returns:
	    Docker daemon info including version, containers count, images count, etc.
	"""
	client = get_docker_client()

	try:
		info = client.info()
		version = client.version()

		return {
			"version": version.get("Version", "unknown"),
			"api_version": version.get("ApiVersion", "unknown"),
			"containers": info.get("Containers", 0),
			"containers_running": info.get("ContainersRunning", 0),
			"containers_paused": info.get("ContainersPaused", 0),
			"containers_stopped": info.get("ContainersStopped", 0),
			"images": info.get("Images", 0),
			"driver": info.get("Driver", "unknown"),
			"memory_total": format_bytes(info.get("MemTotal", 0)),
			"cpus": info.get("NCPU", 0),
			"operating_system": info.get("OperatingSystem", "unknown"),
			"architecture": info.get("Architecture", "unknown"),
		}
	except APIError as e:
		raise HTTPException(status_code=500, detail=f"Docker API error: {str(e)}")
	finally:
		client.close()
