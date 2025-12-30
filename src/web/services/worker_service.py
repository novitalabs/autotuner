"""
Worker deployment and management service.

This module provides shared logic for:
- Deploying workers to remote machines
- Restoring offline workers
- Managing worker slots

Used by both agent tools and API endpoints.
"""

import asyncio
import subprocess
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from web.db.models import WorkerSlot, WorkerSlotStatus

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuration for worker deployment."""
    ssh_command: str
    name: Optional[str] = None
    controller_type: str = "docker"
    manager_ssh: Optional[str] = None
    ssh_forward_tunnel: Optional[str] = None
    ssh_reverse_tunnel: Optional[str] = None
    auto_install: bool = True
    project_path: str = "/opt/inference-autotuner"


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool
    message: str
    worker_id: Optional[str] = None
    slot_id: Optional[int] = None
    error: Optional[str] = None
    logs: Optional[str] = None
    worker_info: Optional[Dict[str, Any]] = None


def parse_ssh_command(ssh_command: str) -> Optional[Dict[str, str]]:
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


def run_ssh(ssh_cmd: str, command: str, timeout: int = 60) -> Tuple[int, str, str]:
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


def build_rsync_command(ssh_command: str, local_path: str, remote_path: str) -> Optional[str]:
    """Build rsync command with proper SSH options.

    Args:
        ssh_command: SSH command (e.g., "ssh -p 18022 root@host")
        local_path: Local source path
        remote_path: Remote destination path

    Returns:
        Full rsync command string or None if invalid
    """
    ssh_info = parse_ssh_command(ssh_command)
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


async def auto_install_worker(ssh_command: str, project_path: str, local_project: str) -> Dict[str, Any]:
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
        logger.info(f"[Deploy] Installing system dependencies on remote...")
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
        ret, out, err = run_ssh(ssh_command, install_deps_cmd, timeout=120)
        if "done" not in out:
            return {"success": False, "error": f"Failed to install system dependencies: {err}"}

        # Step 2: Create project directory
        logger.info(f"[Deploy] Creating project directory at {project_path}...")
        ret, out, err = run_ssh(ssh_command, f"mkdir -p {project_path} && echo ok", timeout=10)
        if "ok" not in out:
            return {"success": False, "error": f"Failed to create directory: {err}"}

        # Step 3: Sync project files
        logger.info(f"[Deploy] Syncing project files to remote...")

        # Check if rsync is available locally
        rsync_available = subprocess.run(
            ["which", "rsync"],
            capture_output=True
        ).returncode == 0

        if rsync_available:
            # Use rsync
            rsync_cmd = build_rsync_command(ssh_command, local_project, project_path)
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
            logger.info(f"[Deploy] rsync not available, using tar + ssh...")
            ssh_info = parse_ssh_command(ssh_command)
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
        run_ssh(
            ssh_command,
            f"chmod +x {project_path}/scripts/*.sh 2>/dev/null; echo ok",
            timeout=10
        )

        logger.info(f"[Deploy] Project files synced successfully")
        return {"success": True, "message": "Project installed"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Installation timed out"}
    except Exception as e:
        return {"success": False, "error": f"Installation failed: {str(e)}"}


async def setup_venv_and_deps(ssh_command: str, project_path: str) -> Dict[str, Any]:
    """Create virtual environment and install dependencies.

    Args:
        ssh_command: SSH command for remote connection
        project_path: Project path on remote

    Returns:
        Dict with success status and optional error message
    """
    try:
        # Find Python 3.9+
        logger.info(f"[Deploy] Setting up Python virtual environment...")

        # Try python3 first, check version
        ret, out, err = run_ssh(ssh_command, "python3 -c 'import sys; print(sys.version_info.minor)'", timeout=10)
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
        ret, out, err = run_ssh(ssh_command, venv_cmd, timeout=60)
        if "ok" not in out:
            return {"success": False, "error": f"Failed to create venv: {err}"}

        # Install dependencies
        logger.info(f"[Deploy] Installing Python dependencies...")
        pip_cmd = f"cd {project_path} && ./env/bin/pip install --upgrade pip -q && ./env/bin/pip install -r requirements.txt -q && echo ok"
        ret, out, err = run_ssh(ssh_command, pip_cmd, timeout=300)
        if "ok" not in out:
            return {"success": False, "error": f"Failed to install dependencies: {err[:500]}"}

        logger.info(f"[Deploy] Virtual environment ready")
        return {"success": True, "message": "Virtual environment created"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Setup timed out"}
    except Exception as e:
        return {"success": False, "error": f"Setup failed: {str(e)}"}


class WorkerService:
    """Service for worker deployment and management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_slot_by_id(self, slot_id: int) -> Optional[WorkerSlot]:
        """Get worker slot by ID."""
        result = await self.db.execute(
            select(WorkerSlot).where(WorkerSlot.id == slot_id)
        )
        return result.scalar_one_or_none()

    async def get_slot_by_ssh(self, ssh_command: str) -> Optional[WorkerSlot]:
        """Get worker slot by SSH command."""
        result = await self.db.execute(
            select(WorkerSlot).where(WorkerSlot.ssh_command == ssh_command)
        )
        return result.scalar_one_or_none()

    async def get_slot_by_name(self, name: str) -> Optional[WorkerSlot]:
        """Get worker slot by name."""
        result = await self.db.execute(
            select(WorkerSlot).where(WorkerSlot.name == name)
        )
        return result.scalar_one_or_none()

    async def get_all_slots(self) -> List[WorkerSlot]:
        """Get all worker slots."""
        result = await self.db.execute(select(WorkerSlot).order_by(WorkerSlot.created_at.desc()))
        return list(result.scalars().all())

    async def create_or_update_slot(
        self,
        config: DeploymentConfig,
        worker_id: Optional[str] = None,
        worker_info: Optional[Dict[str, Any]] = None,
    ) -> WorkerSlot:
        """Create or update a worker slot from deployment config.

        Args:
            config: Deployment configuration
            worker_id: Worker ID from Redis registration
            worker_info: Worker info dict with hostname, gpu_count, etc.

        Returns:
            Created or updated WorkerSlot
        """
        # Check for existing slot with same SSH command
        slot = await self.get_slot_by_ssh(config.ssh_command)

        if slot:
            # Update existing slot
            if config.name:
                slot.name = config.name
            slot.controller_type = config.controller_type
            slot.manager_ssh = config.manager_ssh
            slot.ssh_forward_tunnel = config.ssh_forward_tunnel
            slot.ssh_reverse_tunnel = config.ssh_reverse_tunnel
            slot.project_path = config.project_path
            if worker_id:
                slot.worker_id = worker_id
            if worker_info:
                slot.hostname = worker_info.get("hostname")
                slot.gpu_count = worker_info.get("gpu_count")
                slot.gpu_model = worker_info.get("gpu_model")
                slot.current_status = WorkerSlotStatus.ONLINE
                slot.last_seen_at = datetime.utcnow()
                slot.last_error = None
        else:
            # Count existing slots for default name
            all_slots = await self.get_all_slots()
            default_name = config.name or f"worker-{len(all_slots) + 1}"

            # Create new slot
            slot = WorkerSlot(
                worker_id=worker_id,
                name=default_name,
                controller_type=config.controller_type,
                ssh_command=config.ssh_command,
                ssh_forward_tunnel=config.ssh_forward_tunnel,
                ssh_reverse_tunnel=config.ssh_reverse_tunnel,
                project_path=config.project_path,
                manager_ssh=config.manager_ssh,
                hostname=worker_info.get("hostname") if worker_info else None,
                gpu_count=worker_info.get("gpu_count") if worker_info else None,
                gpu_model=worker_info.get("gpu_model") if worker_info else None,
                current_status=WorkerSlotStatus.ONLINE if worker_id else WorkerSlotStatus.UNKNOWN,
                last_seen_at=datetime.utcnow() if worker_id else None,
            )
            self.db.add(slot)

        await self.db.commit()
        await self.db.refresh(slot)
        return slot

    async def update_slot_status(
        self,
        slot: WorkerSlot,
        status: WorkerSlotStatus,
        error: Optional[str] = None,
        worker_id: Optional[str] = None,
        worker_info: Optional[Dict[str, Any]] = None,
    ):
        """Update slot status and optionally worker info.

        Args:
            slot: WorkerSlot to update
            status: New status
            error: Error message (if failed)
            worker_id: Worker ID to update
            worker_info: Worker info dict to update
        """
        slot.current_status = status
        slot.last_error = error

        if worker_id:
            slot.worker_id = worker_id

        if worker_info:
            slot.hostname = worker_info.get("hostname", slot.hostname)
            slot.gpu_count = worker_info.get("gpu_count", slot.gpu_count)
            slot.gpu_model = worker_info.get("gpu_model", slot.gpu_model)

        if status == WorkerSlotStatus.ONLINE:
            slot.last_seen_at = datetime.utcnow()
            slot.last_error = None

        await self.db.commit()

    async def delete_slot(self, slot_id: int) -> bool:
        """Delete a worker slot.

        Args:
            slot_id: Slot ID to delete

        Returns:
            True if deleted, False if not found
        """
        slot = await self.get_slot_by_id(slot_id)
        if not slot:
            return False

        await self.db.delete(slot)
        await self.db.commit()
        return True

    async def start_remote_worker(
        self,
        slot: WorkerSlot,
        auto_install: bool = True
    ) -> DeploymentResult:
        """Start a worker on a remote machine using slot configuration.

        Args:
            slot: WorkerSlot with deployment configuration
            auto_install: Whether to auto-install if project not found

        Returns:
            DeploymentResult with success status and worker info
        """
        try:
            from web.config import get_settings
            from web.workers.registry import get_worker_registry

            # Get local project path for sync
            local_project = str(Path(__file__).parent.parent.parent.parent)

            ssh_info = parse_ssh_command(slot.ssh_command)
            if not ssh_info:
                return DeploymentResult(
                    success=False,
                    message="Invalid SSH command format",
                    error="Invalid SSH command format. Expected: ssh [-p port] user@host"
                )

            # Test SSH connectivity
            logger.info(f"[Deploy] Testing SSH connection to {ssh_info['host']}...")
            try:
                ret, out, err = run_ssh(slot.ssh_command, "echo ok", timeout=10)
                if ret != 0:
                    await self.update_slot_status(
                        slot, WorkerSlotStatus.OFFLINE,
                        error=f"SSH connection failed: {err.strip() or 'Connection refused'}"
                    )
                    return DeploymentResult(
                        success=False,
                        message="SSH connection failed",
                        error=f"SSH connection failed: {err.strip() or 'Connection refused'}"
                    )
            except subprocess.TimeoutExpired:
                await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error="SSH connection timed out")
                return DeploymentResult(
                    success=False,
                    message="SSH connection timed out",
                    error="SSH connection timed out"
                )

            # Get Redis config from local settings
            settings = get_settings()
            redis_host = settings.redis_host
            redis_port = settings.redis_port

            # Check Redis is externally accessible (skip if using SSH tunnel)
            if redis_host in ("localhost", "127.0.0.1") and not slot.manager_ssh:
                return DeploymentResult(
                    success=False,
                    message="Redis not accessible",
                    error=(
                        "REDIS_HOST is set to localhost. Remote workers cannot connect directly. "
                        "Either set REDIS_HOST to Manager's external IP, or configure manager_ssh "
                        "for SSH tunnel."
                    )
                )

            # Check if project exists on remote
            ret, out, _ = run_ssh(slot.ssh_command, f"test -d {slot.project_path}/src && echo exists", timeout=10)
            project_exists = "exists" in out

            # Auto-install if project doesn't exist
            if not project_exists:
                if not auto_install:
                    return DeploymentResult(
                        success=False,
                        message="Project not found",
                        error=f"Project not found at {slot.project_path} on remote machine."
                    )

                logger.info(f"[Deploy] Installing project on remote...")
                install_result = await auto_install_worker(slot.ssh_command, slot.project_path, local_project)
                if not install_result["success"]:
                    await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=install_result.get("error"))
                    return DeploymentResult(
                        success=False,
                        message="Installation failed",
                        error=install_result.get("error")
                    )

            # Check if virtual environment exists
            ret, out, _ = run_ssh(slot.ssh_command, f"test -f {slot.project_path}/env/bin/activate && echo exists", timeout=10)
            if "exists" not in out:
                if auto_install:
                    # Create venv and install deps
                    install_result = await setup_venv_and_deps(slot.ssh_command, slot.project_path)
                    if not install_result["success"]:
                        await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=install_result.get("error"))
                        return DeploymentResult(
                            success=False,
                            message="Setup failed",
                            error=install_result.get("error")
                        )
                else:
                    return DeploymentResult(
                        success=False,
                        message="Virtual environment not found",
                        error=f"Virtual environment not found at {slot.project_path}/env"
                    )

            # Create .env.worker on remote
            logger.info(f"[Deploy] Creating worker configuration...")
            worker_config_lines = [
                f"REDIS_HOST={redis_host}",
                f"REDIS_PORT={redis_port}",
                f'WORKER_ALIAS="{slot.name or ""}"',
                f"DEPLOYMENT_MODE={slot.controller_type}",
            ]
            if slot.manager_ssh:
                worker_config_lines.append(f'MANAGER_SSH="{slot.manager_ssh}"')
            worker_config = "\n".join(worker_config_lines) + "\n"

            write_cmd = f"cat > {slot.project_path}/.env.worker << 'ENVEOF'\n{worker_config}ENVEOF"
            ret, out, err = run_ssh(slot.ssh_command, write_cmd, timeout=10)
            if ret != 0:
                return DeploymentResult(
                    success=False,
                    message="Failed to create configuration",
                    error=f"Failed to create .env.worker: {err}"
                )

            # Stop any existing worker
            logger.info(f"[Deploy] Stopping any existing worker...")
            run_ssh(slot.ssh_command, f"cd {slot.project_path} && pkill -f 'arq.*autotuner_worker' || true", timeout=10)
            await asyncio.sleep(2)

            # Start worker on remote
            logger.info(f"[Deploy] Starting worker...")
            start_cmd = f"cd {slot.project_path} && nohup ./scripts/start_remote_worker.sh > /tmp/worker_deploy.log 2>&1 &"
            run_ssh(slot.ssh_command, start_cmd, timeout=30)

            # Give worker time to start
            await asyncio.sleep(5)

            # Check if worker process is running on remote
            ret, out, _ = run_ssh(slot.ssh_command, "pgrep -f 'arq.*autotuner_worker'", timeout=10)
            if ret != 0:
                # Worker not running, get logs
                _, log_out, _ = run_ssh(
                    slot.ssh_command,
                    f"tail -30 {slot.project_path}/logs/worker.log 2>/dev/null || cat /tmp/worker_deploy.log",
                    timeout=10
                )
                await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error="Worker process failed to start")
                return DeploymentResult(
                    success=False,
                    message="Worker process failed to start",
                    error="Worker process failed to start",
                    logs=log_out.strip()[-500:] if log_out else None
                )

            # Wait for worker registration (up to 30s)
            logger.info(f"[Deploy] Waiting for worker registration...")
            registry = await get_worker_registry()
            for _ in range(10):
                await asyncio.sleep(3)
                workers = await registry.get_all_workers(include_offline=True)
                for w in workers:
                    # Match by alias or hostname
                    hostname_match = ssh_info["host"] in (w.hostname or "")
                    alias_match = slot.name and w.alias == slot.name
                    if alias_match or hostname_match:
                        # Update slot with worker info
                        worker_info = {
                            "hostname": w.hostname,
                            "gpu_count": w.gpu_count,
                            "gpu_model": w.gpu_model,
                        }
                        await self.update_slot_status(
                            slot,
                            WorkerSlotStatus.ONLINE,
                            worker_id=w.worker_id,
                            worker_info=worker_info
                        )

                        return DeploymentResult(
                            success=True,
                            message="Worker deployed and registered successfully",
                            worker_id=w.worker_id,
                            slot_id=slot.id,
                            worker_info={
                                "worker_id": w.worker_id,
                                "hostname": w.hostname,
                                "alias": w.alias,
                                "gpu_count": w.gpu_count,
                                "gpu_model": w.gpu_model,
                                "deployment_mode": w.deployment_mode,
                                "status": w.status.value if hasattr(w.status, 'value') else w.status,
                            }
                        )

            # Worker started but not registered yet
            return DeploymentResult(
                success=True,
                message="Worker started but not yet registered. It may take a moment to connect.",
                slot_id=slot.id,
            )

        except subprocess.TimeoutExpired:
            await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error="Command timed out")
            return DeploymentResult(
                success=False,
                message="Deployment timed out",
                error="Command timed out during deployment"
            )
        except Exception as e:
            await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=str(e))
            return DeploymentResult(
                success=False,
                message="Deployment failed",
                error=f"Failed to deploy worker: {str(e)}"
            )

    async def deploy_worker(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy a new worker and create/update its slot.

        Args:
            config: Deployment configuration

        Returns:
            DeploymentResult with success status and worker info
        """
        # Create or update slot first
        slot = await self.create_or_update_slot(config)

        # Start the worker
        result = await self.start_remote_worker(slot, auto_install=config.auto_install)
        result.slot_id = slot.id

        return result

    async def restore_worker(self, slot_id: int, auto_install: bool = False) -> DeploymentResult:
        """Restore an offline worker by its slot ID.

        Args:
            slot_id: Worker slot ID to restore
            auto_install: Whether to auto-install if project not found

        Returns:
            DeploymentResult with success status and worker info
        """
        slot = await self.get_slot_by_id(slot_id)
        if not slot:
            return DeploymentResult(
                success=False,
                message="Worker slot not found",
                error=f"Worker slot {slot_id} not found"
            )

        return await self.start_remote_worker(slot, auto_install=auto_install)
