"""
ARQ Worker management tools for agent.

These tools allow monitoring and controlling the ARQ background worker.
Worker restart requires authorization as it affects running tasks.
"""

from langchain_core.tools import tool
from web.agent.tools.base import register_tool, ToolCategory, AuthorizationScope
import json
import subprocess
import asyncio
import os
from pathlib import Path


# ============================================================================
# DISTRIBUTED WORKER TOOLS - For monitoring cluster-wide workers
# ============================================================================

@tool
@register_tool(ToolCategory.SYSTEM)
async def list_distributed_workers() -> str:
    """
    List all distributed ARQ workers registered in the cluster.

    Returns information about all workers including:
    - Worker ID and hostname
    - GPU count and model
    - Current status (online/busy/offline)
    - Deployment mode (docker/ome)
    - Current job count

    This is useful for monitoring cluster capacity and worker health.

    Returns:
        JSON string with list of all registered workers
    """
    try:
        from web.workers.registry import get_worker_registry, worker_info_to_response

        registry = await get_worker_registry()
        workers = await registry.get_all_workers(include_offline=True)

        worker_list = []
        online_count = 0
        busy_count = 0
        offline_count = 0
        total_gpus = 0

        for worker in workers:
            response = worker_info_to_response(worker)
            status = response.status.value if hasattr(response.status, 'value') else response.status

            if status == "online":
                online_count += 1
            elif status == "busy":
                busy_count += 1
            elif status == "offline":
                offline_count += 1

            total_gpus += response.gpu_count

            worker_list.append({
                "worker_id": response.worker_id,
                "hostname": response.hostname,
                "alias": response.alias,
                "gpu_count": response.gpu_count,
                "gpu_model": response.gpu_model,
                "deployment_mode": response.deployment_mode,
                "status": status,
                "current_jobs": response.current_jobs,
                "max_parallel": response.max_parallel,
                "seconds_since_heartbeat": round(response.seconds_since_heartbeat, 1),
            })

        return json.dumps({
            "success": True,
            "total_workers": len(worker_list),
            "online_count": online_count,
            "busy_count": busy_count,
            "offline_count": offline_count,
            "total_gpus": total_gpus,
            "workers": worker_list
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list distributed workers: {str(e)}"
        })


@tool
@register_tool(ToolCategory.SYSTEM)
async def get_distributed_worker_status(worker_id: str) -> str:
    """
    Get detailed status of a specific distributed worker.

    Returns comprehensive information including:
    - Worker configuration and capabilities
    - GPU status with real-time metrics (utilization, memory, temperature)
    - Per-node GPU breakdown for cluster (OME) mode workers
    - Current job information

    Args:
        worker_id: Worker ID to get status for (e.g., "bj-7.67-worker")

    Returns:
        JSON string with detailed worker status including GPU metrics
    """
    try:
        from web.workers.registry import get_worker_registry, worker_info_to_response

        registry = await get_worker_registry()
        worker = await registry.get_worker(worker_id)

        if not worker:
            return json.dumps({
                "success": False,
                "error": f"Worker '{worker_id}' not found"
            })

        response = worker_info_to_response(worker)

        # Format GPU information
        gpus_info = []
        if worker.gpus:
            # Group by node for OME mode
            if worker.deployment_mode == "ome":
                node_map = {}
                for gpu in worker.gpus:
                    gpu_dict = gpu if isinstance(gpu, dict) else gpu.model_dump() if hasattr(gpu, 'model_dump') else gpu
                    node_name = gpu_dict.get('node_name', 'unknown')
                    if node_name not in node_map:
                        node_map[node_name] = []
                    node_map[node_name].append(gpu_dict)

                for node_name, node_gpus in sorted(node_map.items()):
                    has_metrics = any(g.get('utilization_percent') is not None for g in node_gpus)
                    gpus_info.append({
                        "node_name": node_name,
                        "gpu_count": len(node_gpus),
                        "has_metrics": has_metrics,
                        "gpus": [
                            {
                                "index": g.get('index'),
                                "name": g.get('name'),
                                "memory_total_gb": g.get('memory_total_gb'),
                                "memory_used_gb": g.get('memory_used_gb'),
                                "utilization_percent": g.get('utilization_percent'),
                                "temperature_c": g.get('temperature_c'),
                            }
                            for g in node_gpus
                        ]
                    })
            else:
                # Non-OME mode: flat list
                for gpu in worker.gpus:
                    gpu_dict = gpu if isinstance(gpu, dict) else gpu.model_dump() if hasattr(gpu, 'model_dump') else gpu
                    gpus_info.append({
                        "index": gpu_dict.get('index'),
                        "name": gpu_dict.get('name'),
                        "memory_total_gb": gpu_dict.get('memory_total_gb'),
                        "memory_used_gb": gpu_dict.get('memory_used_gb'),
                        "utilization_percent": gpu_dict.get('utilization_percent'),
                        "temperature_c": gpu_dict.get('temperature_c'),
                    })

        return json.dumps({
            "success": True,
            "worker": {
                "worker_id": response.worker_id,
                "hostname": response.hostname,
                "alias": response.alias,
                "ip_address": response.ip_address,
                "deployment_mode": response.deployment_mode,
                "status": response.status.value if hasattr(response.status, 'value') else response.status,
                "current_jobs": response.current_jobs,
                "max_parallel": response.max_parallel,
                "gpu_count": response.gpu_count,
                "gpu_model": response.gpu_model,
                "registered_at": response.registered_at.isoformat() if response.registered_at else None,
                "last_heartbeat": response.last_heartbeat.isoformat() if response.last_heartbeat else None,
                "seconds_since_heartbeat": round(response.seconds_since_heartbeat, 1),
            },
            "gpu_status": gpus_info
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to get worker status: {str(e)}"
        })


@tool
@register_tool(ToolCategory.SYSTEM)
async def rename_distributed_worker(worker_id: str, alias: str = None) -> str:
    """
    Set or clear an alias for a distributed worker.

    Aliases make it easier to identify workers in the dashboard.
    Pass empty string or null to clear the alias.

    Args:
        worker_id: Worker ID to rename
        alias: New alias for the worker, or empty/null to clear

    Returns:
        JSON string with updated worker information
    """
    try:
        from web.workers.registry import get_worker_registry

        registry = await get_worker_registry()

        # Clear alias if empty string
        if alias == "":
            alias = None

        worker = await registry.set_worker_alias(worker_id, alias)

        if not worker:
            return json.dumps({
                "success": False,
                "error": f"Worker '{worker_id}' not found"
            })

        return json.dumps({
            "success": True,
            "message": f"Worker alias {'set to ' + repr(alias) if alias else 'cleared'}",
            "worker": {
                "worker_id": worker.worker_id,
                "hostname": worker.hostname,
                "alias": worker.alias,
            }
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to rename worker: {str(e)}"
        })


@tool
@register_tool(ToolCategory.SYSTEM)
async def get_cluster_gpu_summary() -> str:
    """
    Get a summary of GPU resources across all distributed workers.

    Provides cluster-wide view of:
    - Total GPU count and models
    - Available capacity (workers with free slots)
    - GPU utilization statistics (for workers reporting metrics)

    Useful for understanding cluster capacity before starting new tasks.

    Returns:
        JSON string with cluster GPU summary
    """
    try:
        from web.workers.registry import get_worker_registry, worker_info_to_response

        registry = await get_worker_registry()
        workers = await registry.get_all_workers(include_offline=False)

        total_gpus = 0
        total_capacity = 0
        used_capacity = 0
        gpu_models = {}
        nodes_with_metrics = 0
        total_utilization = 0
        total_memory_used = 0
        total_memory_total = 0

        for worker in workers:
            total_gpus += worker.gpu_count
            total_capacity += worker.max_parallel
            used_capacity += worker.current_jobs

            # Count GPU models
            if worker.gpu_model:
                model = worker.gpu_model.replace('NVIDIA ', '').replace('GeForce ', '')
                gpu_models[model] = gpu_models.get(model, 0) + worker.gpu_count

            # Aggregate GPU metrics
            if worker.gpus:
                for gpu in worker.gpus:
                    gpu_dict = gpu if isinstance(gpu, dict) else gpu.model_dump() if hasattr(gpu, 'model_dump') else gpu
                    util = gpu_dict.get('utilization_percent')
                    mem_used = gpu_dict.get('memory_used_gb')
                    mem_total = gpu_dict.get('memory_total_gb')

                    if util is not None:
                        nodes_with_metrics += 1
                        total_utilization += util
                    if mem_used is not None and mem_total is not None:
                        total_memory_used += mem_used
                        total_memory_total += mem_total

        avg_utilization = total_utilization / nodes_with_metrics if nodes_with_metrics > 0 else None
        memory_usage_percent = (total_memory_used / total_memory_total * 100) if total_memory_total > 0 else None

        return json.dumps({
            "success": True,
            "cluster_summary": {
                "total_workers": len(workers),
                "total_gpus": total_gpus,
                "gpu_models": gpu_models,
                "job_capacity": {
                    "total_slots": total_capacity,
                    "used_slots": used_capacity,
                    "available_slots": total_capacity - used_capacity,
                },
                "gpu_metrics": {
                    "gpus_with_metrics": nodes_with_metrics,
                    "average_utilization_percent": round(avg_utilization, 1) if avg_utilization else None,
                    "total_memory_used_gb": round(total_memory_used, 1) if total_memory_used else None,
                    "total_memory_total_gb": round(total_memory_total, 1) if total_memory_total else None,
                    "memory_usage_percent": round(memory_usage_percent, 1) if memory_usage_percent else None,
                }
            }
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to get cluster GPU summary: {str(e)}"
        })


# ============================================================================
# REMOTE WORKER DEPLOYMENT TOOLS
# ============================================================================

import re


def _parse_ssh_command(ssh_command: str) -> dict:
    """Parse SSH command to extract host, port, user.

    Args:
        ssh_command: SSH command like "ssh -p 18022 root@192.168.1.100"

    Returns:
        Dict with user, host, port or None if invalid
    """
    # Extract port if specified with -p
    port_match = re.search(r'-p\s+(\d+)', ssh_command)
    port = port_match.group(1) if port_match else "22"

    # Extract user@host
    user_host = re.search(r'(\w+)@([\w\.\-]+)', ssh_command)
    if user_host:
        return {"user": user_host.group(1), "host": user_host.group(2), "port": port}
    return None


def _run_ssh(ssh_cmd: str, command: str, timeout: int = 60) -> tuple:
    """Execute command on remote host via SSH.

    Args:
        ssh_cmd: Base SSH command (e.g., "ssh -p 18022 root@host")
        command: Command to execute on remote host
        timeout: Command timeout in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    # Escape double quotes in command
    escaped_cmd = command.replace('"', '\\"')
    full_cmd = f'{ssh_cmd} "{escaped_cmd}"'
    result = subprocess.run(
        full_cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result.returncode, result.stdout, result.stderr


def _build_rsync_command(ssh_command: str, local_path: str, remote_path: str) -> str:
    """Build rsync command with proper SSH options.

    Args:
        ssh_command: SSH command (e.g., "ssh -p 18022 root@host")
        local_path: Local source path
        remote_path: Remote destination path

    Returns:
        Full rsync command string
    """
    ssh_info = _parse_ssh_command(ssh_command)
    if not ssh_info:
        return None

    # Build SSH options for rsync
    ssh_opts = f"ssh -p {ssh_info['port']} -o StrictHostKeyChecking=no"

    # Build rsync command - exclude venv, cache, logs, etc.
    excludes = [
        "env/", ".venv/", "__pycache__/", "*.pyc", ".git/",
        "logs/", "*.log", ".env", ".env.*", "node_modules/",
        "frontend/node_modules/", "frontend/dist/", ".pytest_cache/",
        "autotuner.db", "*.db", ".mypy_cache/"
    ]
    exclude_args = " ".join([f"--exclude='{e}'" for e in excludes])

    remote_dest = f"{ssh_info['user']}@{ssh_info['host']}:{remote_path}"

    return f"rsync -avz --delete {exclude_args} -e \"{ssh_opts}\" {local_path}/ {remote_dest}/"


async def _auto_install_worker(ssh_command: str, project_path: str, local_project: str) -> dict:
    """Auto-install the worker on a remote machine.

    Args:
        ssh_command: SSH command for remote connection
        project_path: Target installation path on remote
        local_project: Local project path to sync from

    Returns:
        Dict with success status and optional error message
    """
    try:
        # Step 1: Install system dependencies
        print(f"[Deploy] Installing system dependencies on remote...")
        install_deps_cmd = """
        if command -v apt-get &>/dev/null; then
            apt-get update -qq && apt-get install -y -qq python3 python3-pip python3-venv netcat-openbsd tar >/dev/null 2>&1
        elif command -v dnf &>/dev/null; then
            dnf install -y -q python3 python3-pip tar nc >/dev/null 2>&1
        elif command -v yum &>/dev/null; then
            yum install -y -q python3 python3-pip tar nc >/dev/null 2>&1
        fi
        echo done
        """
        ret, out, err = _run_ssh(ssh_command, install_deps_cmd, timeout=120)
        if "done" not in out:
            return {"success": False, "error": f"Failed to install system dependencies: {err}"}

        # Step 2: Create project directory
        print(f"[Deploy] Creating project directory at {project_path}...")
        ret, out, err = _run_ssh(ssh_command, f"mkdir -p {project_path} && echo ok", timeout=10)
        if "ok" not in out:
            return {"success": False, "error": f"Failed to create directory: {err}"}

        # Step 3: Sync project files
        print(f"[Deploy] Syncing project files to remote...")

        # Check if rsync is available locally
        rsync_available = subprocess.run(
            ["which", "rsync"],
            capture_output=True
        ).returncode == 0

        if rsync_available:
            # Use rsync
            rsync_cmd = _build_rsync_command(ssh_command, local_project, project_path)
            if not rsync_cmd:
                return {"success": False, "error": "Failed to build rsync command"}

            result = subprocess.run(
                rsync_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                return {"success": False, "error": f"Failed to sync files: {result.stderr[:500]}"}
        else:
            # Fallback: use tar + ssh
            print(f"[Deploy] rsync not available, using tar + ssh...")
            ssh_info = _parse_ssh_command(ssh_command)
            if not ssh_info:
                return {"success": False, "error": "Failed to parse SSH command"}

            # Excludes for tar
            excludes = [
                "--exclude=env", "--exclude=.venv", "--exclude=__pycache__",
                "--exclude=.git", "--exclude=logs", "--exclude=*.log",
                "--exclude=.env", "--exclude=.env.*", "--exclude=node_modules",
                "--exclude=frontend/node_modules", "--exclude=frontend/dist",
                "--exclude=.pytest_cache", "--exclude=*.db", "--exclude=.mypy_cache"
            ]
            exclude_str = " ".join(excludes)

            # Create tar and pipe to remote
            ssh_opts = f"-p {ssh_info['port']} -o StrictHostKeyChecking=no"
            tar_cmd = f"cd {local_project} && tar czf - {exclude_str} . | ssh {ssh_opts} {ssh_info['user']}@{ssh_info['host']} 'cd {project_path} && tar xzf -'"

            result = subprocess.run(
                tar_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                return {"success": False, "error": f"Failed to sync files via tar: {result.stderr[:500]}"}

        # Step 4: Make scripts executable
        ret, out, err = _run_ssh(
            ssh_command,
            f"chmod +x {project_path}/scripts/*.sh 2>/dev/null; echo ok",
            timeout=10
        )

        print(f"[Deploy] Project files synced successfully")
        return {"success": True, "message": "Project installed"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Installation timed out"}
    except Exception as e:
        return {"success": False, "error": f"Installation failed: {str(e)}"}


async def _setup_venv_and_deps(ssh_command: str, project_path: str) -> dict:
    """Create virtual environment and install dependencies.

    Args:
        ssh_command: SSH command for remote connection
        project_path: Project path on remote

    Returns:
        Dict with success status and optional error message
    """
    try:
        # Find Python 3.9+ - use simple detection
        print(f"[Deploy] Setting up Python virtual environment...")

        # Try python3 first, check version
        ret, out, err = _run_ssh(ssh_command, "python3 -c 'import sys; print(sys.version_info.minor)'", timeout=10)
        if ret == 0:
            try:
                minor = int(out.strip())
                if minor >= 9:
                    python_cmd = "python3"
                else:
                    return {"success": False, "error": f"Python 3.{minor} found, but Python 3.9+ required"}
            except ValueError:
                return {"success": False, "error": f"Failed to parse Python version: {out}"}
        else:
            return {"success": False, "error": "Python 3 not found on remote machine"}

        # Create virtual environment
        venv_cmd = f"cd {project_path} && {python_cmd} -m venv env && echo ok"
        ret, out, err = _run_ssh(ssh_command, venv_cmd, timeout=60)
        if "ok" not in out:
            return {"success": False, "error": f"Failed to create venv: {err}"}

        # Install dependencies
        print(f"[Deploy] Installing Python dependencies...")
        pip_cmd = f"cd {project_path} && ./env/bin/pip install --upgrade pip -q && ./env/bin/pip install -r requirements.txt -q && echo ok"
        ret, out, err = _run_ssh(ssh_command, pip_cmd, timeout=300)
        if "ok" not in out:
            return {"success": False, "error": f"Failed to install dependencies: {err[:500]}"}

        print(f"[Deploy] Virtual environment ready")
        return {"success": True, "message": "Virtual environment created"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Setup timed out"}
    except Exception as e:
        return {"success": False, "error": f"Setup failed: {str(e)}"}


@tool
@register_tool(
    ToolCategory.SYSTEM,
    requires_auth=True,
    auth_scope=AuthorizationScope.ARQ_CONTROL
)
async def deploy_worker(
    ssh_command: str,
    name: str = None,
    mode: str = "docker",
    auto_install: bool = True,
    manager_ssh: str = None
) -> str:
    """
    Deploy an ARQ worker to a remote machine via SSH.

    This tool will:
    1. Test SSH connectivity to the remote machine
    2. Auto-install the project if not present (install deps, sync files)
    3. Create worker configuration (.env.worker) with Redis connection
    4. Start the worker using start_remote_worker.sh
    5. Wait for worker to register with the manager

    Prerequisites:
    - SSH key-based authentication configured (no password prompt)
    - Either: REDIS_HOST accessible from remote, OR manager_ssh for SSH tunnel
    - Root/sudo access on remote machine (for auto-install)

    Args:
        ssh_command: SSH connection command (e.g., "ssh -p 18022 root@192.168.1.100")
        name: Worker alias/name for identification in dashboard (optional)
        mode: Deployment mode - "docker" or "ome" (default: "docker")
        auto_install: Automatically install project if not found (default: True)
        manager_ssh: SSH command for worker to connect back to manager for Redis tunnel
                    (e.g., "ssh -p 33773 user@manager-ip"). Required if Redis not directly accessible.

    Returns:
        JSON string with deployment status and worker info
    """
    PROJECT_PATH = "/opt/inference-autotuner"
    LOCAL_PROJECT = str(Path(__file__).parent.parent.parent.parent.parent)

    try:
        from web.config import get_settings
        from web.workers.registry import get_worker_registry

        # 1. Parse SSH command
        ssh_info = _parse_ssh_command(ssh_command)
        if not ssh_info:
            return json.dumps({
                "success": False,
                "error": "Invalid SSH command format. Expected: ssh [-p port] user@host"
            })

        # 2. Test SSH connectivity
        try:
            ret, out, err = _run_ssh(ssh_command, "echo ok", timeout=10)
            if ret != 0:
                return json.dumps({
                    "success": False,
                    "error": f"SSH connection failed: {err.strip() or 'Connection refused'}"
                })
        except subprocess.TimeoutExpired:
            return json.dumps({
                "success": False,
                "error": "SSH connection timed out"
            })

        # 3. Get Redis config from local settings
        settings = get_settings()
        redis_host = settings.redis_host
        redis_port = settings.redis_port

        # 4. Check Redis is externally accessible (skip if using SSH tunnel)
        if redis_host in ("localhost", "127.0.0.1") and not manager_ssh:
            return json.dumps({
                "success": False,
                "error": (
                    "REDIS_HOST is set to localhost. Remote workers cannot connect directly. "
                    "Either set REDIS_HOST to Manager's external IP, or provide manager_ssh "
                    "parameter for SSH tunnel (e.g., 'ssh -p 33773 user@manager-ip')."
                )
            })

        # 5. Check if project exists on remote
        ret, out, _ = _run_ssh(ssh_command, f"test -d {PROJECT_PATH}/src && echo exists", timeout=10)
        project_exists = "exists" in out

        # 6. Auto-install if project doesn't exist
        if not project_exists:
            if not auto_install:
                return json.dumps({
                    "success": False,
                    "error": f"Project not found at {PROJECT_PATH} on remote machine. Set auto_install=True to install."
                })

            # Run auto-installation
            install_result = await _auto_install_worker(ssh_command, PROJECT_PATH, LOCAL_PROJECT)
            if not install_result["success"]:
                return json.dumps(install_result)

        # 7. Check if virtual environment exists (may have been created by auto-install)
        ret, out, _ = _run_ssh(ssh_command, f"test -f {PROJECT_PATH}/env/bin/activate && echo exists", timeout=10)
        if "exists" not in out:
            if auto_install:
                # Create venv and install deps
                install_result = await _setup_venv_and_deps(ssh_command, PROJECT_PATH)
                if not install_result["success"]:
                    return json.dumps(install_result)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Virtual environment not found at {PROJECT_PATH}/env. Set auto_install=True to create."
                })

        # 8. Create .env.worker on remote
        worker_config_lines = [
            f"REDIS_HOST={redis_host}",
            f"REDIS_PORT={redis_port}",
            f'WORKER_ALIAS="{name or ""}"',
            f"DEPLOYMENT_MODE={mode}",
        ]
        if manager_ssh:
            # Quote the MANAGER_SSH value since it may contain special characters
            worker_config_lines.append(f'MANAGER_SSH="{manager_ssh}"')
        worker_config = "\n".join(worker_config_lines) + "\n"
        # Use heredoc to write config file
        write_cmd = f"cat > {PROJECT_PATH}/.env.worker << 'ENVEOF'\n{worker_config}ENVEOF"
        ret, out, err = _run_ssh(ssh_command, write_cmd, timeout=10)
        if ret != 0:
            return json.dumps({
                "success": False,
                "error": f"Failed to create .env.worker: {err}"
            })

        # 9. Stop any existing worker
        _run_ssh(ssh_command, f"cd {PROJECT_PATH} && pkill -f 'arq.*autotuner_worker' || true", timeout=10)
        await asyncio.sleep(2)

        # 10. Start worker on remote
        start_cmd = f"cd {PROJECT_PATH} && nohup ./scripts/start_remote_worker.sh > /tmp/worker_deploy.log 2>&1 &"
        ret, out, err = _run_ssh(ssh_command, start_cmd, timeout=30)

        # Give worker time to start
        await asyncio.sleep(5)

        # 11. Check if worker process is running on remote
        ret, out, _ = _run_ssh(ssh_command, "pgrep -f 'arq.*autotuner_worker'", timeout=10)
        if ret != 0:
            # Worker not running, get logs
            _, log_out, _ = _run_ssh(ssh_command, f"tail -30 {PROJECT_PATH}/logs/worker.log 2>/dev/null || cat /tmp/worker_deploy.log", timeout=10)
            return json.dumps({
                "success": False,
                "error": "Worker process failed to start",
                "remote_logs": log_out.strip()[-500:] if log_out else "No logs available"
            })

        # 12. Wait for worker registration (up to 30s)
        registry = await get_worker_registry()
        for _ in range(10):
            await asyncio.sleep(3)
            workers = await registry.get_all_workers(include_offline=True)
            for w in workers:
                # Match by alias or hostname
                hostname_match = ssh_info["host"] in (w.hostname or "")
                alias_match = name and w.alias == name
                if alias_match or hostname_match:
                    return json.dumps({
                        "success": True,
                        "message": "Worker deployed and registered successfully",
                        "worker": {
                            "worker_id": w.worker_id,
                            "hostname": w.hostname,
                            "alias": w.alias,
                            "gpu_count": w.gpu_count,
                            "gpu_model": w.gpu_model,
                            "deployment_mode": w.deployment_mode,
                            "status": w.status.value if hasattr(w.status, 'value') else w.status,
                        }
                    }, indent=2)

        # Worker started but not registered yet
        return json.dumps({
            "success": True,
            "message": "Worker started but not yet registered. It may take a moment to connect.",
            "check_command": f"{ssh_command} 'tail -50 {PROJECT_PATH}/logs/worker.log'"
        }, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "Command timed out during deployment"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to deploy worker: {str(e)}"
        })


# ============================================================================
# LOCAL ARQ WORKER TOOLS - For local worker process management
# ============================================================================


@tool
@register_tool(ToolCategory.SYSTEM)
async def get_arq_worker_status() -> str:
    """
    Check the status of ARQ background workers.

    Returns information about running worker processes including:
    - Number of worker processes
    - Process IDs
    - CPU/memory usage

    This is a read-only operation that doesn't require authorization.

    Returns:
        JSON string with worker status information
    """
    try:
        # Find ARQ worker processes
        result = subprocess.run(
            ["pgrep", "-af", "arq.*autotuner_worker"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0 or not result.stdout.strip():
            return json.dumps({
                "success": True,
                "status": "stopped",
                "workers": [],
                "message": "No ARQ worker processes found"
            }, indent=2)

        # Parse process info
        workers = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split(None, 1)
            if len(parts) >= 2:
                pid = parts[0]
                cmd = parts[1]

                # Get process details using ps
                ps_result = subprocess.run(
                    ["ps", "-p", pid, "-o", "pid,pcpu,pmem,etime", "--no-headers"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                if ps_result.returncode == 0 and ps_result.stdout.strip():
                    ps_parts = ps_result.stdout.strip().split()
                    if len(ps_parts) >= 4:
                        workers.append({
                            "pid": int(pid),
                            "cpu_percent": float(ps_parts[1]),
                            "mem_percent": float(ps_parts[2]),
                            "elapsed_time": ps_parts[3],
                            "command": cmd[:100]  # Truncate long commands
                        })

        return json.dumps({
            "success": True,
            "status": "running" if workers else "stopped",
            "worker_count": len(workers),
            "workers": workers
        }, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "Timeout while checking worker status"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to check worker status: {str(e)}"
        })


@tool
@register_tool(
    ToolCategory.SYSTEM,
    requires_auth=True,
    auth_scope=AuthorizationScope.ARQ_CONTROL
)
async def restart_arq_worker() -> str:
    """
    Restart the ARQ background worker process.

    This will:
    1. Gracefully terminate existing worker processes
    2. Start new worker processes
    3. Verify workers are running

    Use this when:
    - Worker code has been updated and needs reloading
    - Worker appears stuck or unresponsive
    - After configuration changes

    WARNING: This will interrupt any currently running tasks.
    Tasks in progress will be marked as failed and need to be restarted.

    Returns:
        JSON string with restart status
    """
    project_root = Path(__file__).parent.parent.parent.parent.parent

    try:
        # Step 1: Find and kill existing workers
        # First try without sudo (for same-user processes)
        kill_result = subprocess.run(
            ["pkill", "-f", "arq.*autotuner_worker"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Check if any workers are still running (might be owned by different user)
        check_result = subprocess.run(
            ["pgrep", "-f", "arq.*autotuner_worker"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if check_result.returncode == 0 and check_result.stdout.strip():
            # Workers still running, try with sudo using SIGKILL
            # Use sudo with NOPASSWD configured for this specific command
            pids = check_result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    subprocess.run(
                        ["sudo", "-n", "kill", "-9", pid.strip()],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

        # Wait for processes to terminate
        await asyncio.sleep(2)

        # Step 2: Start new worker
        worker_script = project_root / "scripts" / "start_worker.sh"
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "worker.log"

        # Start worker in background
        env = os.environ.copy()
        process = subprocess.Popen(
            ["bash", str(worker_script)],
            stdout=open(log_file, "a"),
            stderr=subprocess.STDOUT,
            cwd=str(project_root),
            env=env,
            start_new_session=True  # Detach from parent
        )

        # Wait for worker to start
        await asyncio.sleep(3)

        # Step 3: Verify worker is running
        verify_result = subprocess.run(
            ["pgrep", "-f", "arq.*autotuner_worker"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if verify_result.returncode == 0 and verify_result.stdout.strip():
            pids = verify_result.stdout.strip().split('\n')
            return json.dumps({
                "success": True,
                "message": f"ARQ worker restarted successfully",
                "worker_pids": [int(p) for p in pids if p.strip()],
                "log_file": str(log_file)
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "error": "Worker process started but not running. Check logs.",
                "log_file": str(log_file)
            }, indent=2)

    except subprocess.TimeoutExpired:
        return json.dumps({
            "success": False,
            "error": "Timeout during worker restart"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to restart worker: {str(e)}"
        })


@tool
@register_tool(ToolCategory.SYSTEM)
async def list_arq_jobs() -> str:
    """
    List queued and running ARQ jobs.

    Shows information about:
    - Jobs currently being processed
    - Jobs waiting in queue
    - Recently completed jobs

    This is a read-only operation that doesn't require authorization.

    Returns:
        JSON string with job queue information
    """
    try:
        import redis.asyncio as redis
        from web.config import get_settings

        settings = get_settings()

        # Connect to Redis
        r = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True
        )

        # Get ARQ queue info
        # ARQ uses specific key patterns
        queued = await r.llen("arq:queue")
        in_progress_keys = await r.keys("arq:in-progress:*")

        # Get job details from in-progress
        in_progress_jobs = []
        for key in in_progress_keys[:10]:  # Limit to first 10
            job_id = key.split(":")[-1]
            job_data = await r.get(f"arq:job:{job_id}")
            if job_data:
                try:
                    import json as json_module
                    data = json_module.loads(job_data)
                    in_progress_jobs.append({
                        "job_id": job_id,
                        "function": data.get("function", "unknown"),
                        "enqueue_time": data.get("enqueue_time")
                    })
                except:
                    in_progress_jobs.append({"job_id": job_id})

        # Get recent results
        result_keys = await r.keys("arq:result:*")
        recent_results = []
        for key in result_keys[:5]:  # Last 5 results
            job_id = key.split(":")[-1]
            result_data = await r.get(key)
            if result_data:
                recent_results.append({
                    "job_id": job_id,
                    "has_result": True
                })

        await r.close()

        return json.dumps({
            "success": True,
            "queue_length": queued,
            "in_progress_count": len(in_progress_keys),
            "in_progress_jobs": in_progress_jobs,
            "recent_results_count": len(result_keys),
            "recent_results": recent_results
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list ARQ jobs: {str(e)}"
        })
