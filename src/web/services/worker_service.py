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
    force_sync: bool = False  # Force sync code even if project exists
    project_path: str = "/opt/inference-autotuner"


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    success: bool
    message: str
    worker_id: Optional[str] = None
    slot_id: Optional[int] = None
    error: Optional[str] = None
    suggestion: Optional[str] = None  # Fix suggestion for errors
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


async def setup_ssh_key_for_manager(ssh_command: str, manager_ssh: str) -> Dict[str, Any]:
    """Setup SSH key authentication from remote worker to manager.

    This allows the worker to establish an SSH tunnel back to the manager for Redis access.

    Args:
        ssh_command: SSH command to reach the remote worker
        manager_ssh: SSH command the worker uses to reach the manager (e.g., "ssh -p 33773 claude@manager")

    Returns:
        Dict with success status and optional error message
    """
    try:
        # Parse manager SSH to get user and host
        manager_info = parse_ssh_command(manager_ssh)
        if not manager_info:
            return {"success": False, "error": "Invalid manager_ssh format"}

        # Step 1: Check if remote has SSH key, generate if not
        logger.info(f"[SSH Key] Checking SSH key on remote worker...")
        ret, out, err = run_ssh(
            ssh_command,
            "test -f ~/.ssh/id_ed25519.pub && cat ~/.ssh/id_ed25519.pub || "
            "(ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519 -q && cat ~/.ssh/id_ed25519.pub)",
            timeout=30
        )
        if ret != 0 or not out.strip():
            return {"success": False, "error": f"Failed to get/generate SSH key: {err}"}

        remote_pubkey = out.strip()
        if not remote_pubkey.startswith("ssh-"):
            return {"success": False, "error": f"Invalid public key format: {remote_pubkey[:50]}"}

        # Step 2: Check if key already in manager's authorized_keys
        local_auth_keys = Path.home() / ".ssh" / "authorized_keys"
        key_exists = False
        if local_auth_keys.exists():
            existing_keys = local_auth_keys.read_text()
            # Check by key content (ignore comment at end)
            key_parts = remote_pubkey.split()
            if len(key_parts) >= 2:
                key_content = f"{key_parts[0]} {key_parts[1]}"
                key_exists = key_content in existing_keys

        if key_exists:
            logger.info(f"[SSH Key] Remote key already in authorized_keys")
            return {"success": True, "message": "SSH key already configured"}

        # Step 3: Add key to local authorized_keys
        logger.info(f"[SSH Key] Adding remote key to manager's authorized_keys...")
        local_auth_keys.parent.mkdir(parents=True, exist_ok=True)
        with open(local_auth_keys, "a") as f:
            f.write(f"\n# Auto-added for worker SSH tunnel\n{remote_pubkey}\n")

        # Step 4: Verify SSH connectivity from remote to manager
        logger.info(f"[SSH Key] Testing SSH connection from worker to manager...")
        test_cmd = f"ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p {manager_info['port']} {manager_info['user']}@{manager_info['host']} 'echo connected'"
        ret, out, err = run_ssh(ssh_command, test_cmd, timeout=20)

        if "connected" in out:
            logger.info(f"[SSH Key] SSH key setup successful")
            return {"success": True, "message": "SSH key configured and verified"}
        else:
            return {
                "success": False,
                "error": f"SSH key added but connection test failed: {err.strip()}"
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "SSH key setup timed out"}
    except Exception as e:
        return {"success": False, "error": f"SSH key setup failed: {str(e)}"}


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


async def sync_venv_offline(ssh_command: str, project_path: str, local_project: str) -> Dict[str, Any]:
    """Sync virtual environment to remote machine without requiring network access.

    This creates a tarball of the local venv's site-packages and transfers it to the remote,
    avoiding the need for pip install on the remote machine.

    Args:
        ssh_command: SSH command for remote connection
        project_path: Project path on remote
        local_project: Local project path

    Returns:
        Dict with success status and optional error message
    """
    try:
        local_venv = Path(local_project) / "env"
        if not local_venv.exists():
            return {"success": False, "error": "Local venv not found"}

        ssh_info = parse_ssh_command(ssh_command)
        if not ssh_info:
            return {"success": False, "error": "Invalid SSH command format"}

        # Step 1: Create venv on remote (just the structure, no packages)
        logger.info(f"[Deploy] Creating Python virtual environment on remote...")
        ret, out, err = run_ssh(ssh_command, "python3 -c 'import sys; print(sys.version_info.minor)'", timeout=10)
        if ret != 0:
            return {"success": False, "error": "Python 3 not found on remote machine"}

        venv_cmd = f"cd {project_path} && python3 -m venv env && echo ok"
        ret, out, err = run_ssh(ssh_command, venv_cmd, timeout=60)
        if "ok" not in out:
            return {"success": False, "error": f"Failed to create venv: {err}"}

        # Step 2: Sync site-packages using rsync
        logger.info(f"[Deploy] Syncing Python packages to remote (offline mode)...")

        local_site_packages = local_venv / "lib"
        remote_site_packages = f"{project_path}/env/lib"

        ssh_opts = f"ssh -p {ssh_info['port']} -o StrictHostKeyChecking=no"
        remote_dest = f"{ssh_info['user']}@{ssh_info['host']}:{remote_site_packages}/"

        # Use rsync if available
        rsync_available = subprocess.run(["which", "rsync"], capture_output=True).returncode == 0

        if rsync_available:
            rsync_cmd = f'rsync -avz --delete -e "{ssh_opts}" {local_site_packages}/ {remote_dest}'
            result = subprocess.run(rsync_cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return {"success": False, "error": f"Failed to sync packages: {result.stderr[:300]}"}
        else:
            # Fallback: tar + ssh
            tar_cmd = f"cd {local_venv} && tar czf - lib | ssh {ssh_opts} {ssh_info['user']}@{ssh_info['host']} 'cd {project_path}/env && tar xzf -'"
            result = subprocess.run(tar_cmd, shell=True, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return {"success": False, "error": f"Failed to sync packages via tar: {result.stderr[:300]}"}

        # Step 3: Also sync bin scripts that reference python
        logger.info(f"[Deploy] Fixing venv bin scripts...")
        fix_bin_cmd = f"""
cd {project_path}/env/bin
for f in *; do
    if [ -f "$f" ] && head -1 "$f" 2>/dev/null | grep -q python; then
        sed -i '1s|.*|#!/usr/bin/env python3|' "$f" 2>/dev/null || true
    fi
done
# Ensure activate script exists
test -f activate && echo ok
"""
        ret, out, err = run_ssh(ssh_command, fix_bin_cmd, timeout=30)
        if "ok" not in out:
            logger.warning(f"[Deploy] Warning: bin scripts fix may have issues: {err}")

        logger.info(f"[Deploy] Offline venv sync completed")
        return {"success": True, "message": "Virtual environment synced (offline mode)"}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Offline sync timed out"}
    except Exception as e:
        return {"success": False, "error": f"Offline sync failed: {str(e)}"}


async def setup_reverse_tunnel(
    ssh_command: str,
    reverse_tunnel_spec: str,
    remote_project_path: str
) -> Dict[str, Any]:
    """Setup reverse SSH tunnel from Manager to Remote Worker.

    This creates a tunnel where the Manager maintains the SSH connection,
    forwarding a port on the remote to Manager's Redis.

    Args:
        ssh_command: SSH command to reach remote (e.g., "ssh -p 18022 root@host")
        reverse_tunnel_spec: Tunnel spec like "6380:localhost:6379" (remote_port:manager_host:manager_port)
        remote_project_path: Remote project path for storing tunnel PID

    Returns:
        Dict with success status, tunnel_pid, and remote_redis_port
    """
    try:
        # Parse reverse tunnel spec: remote_port:manager_host:manager_port
        parts = reverse_tunnel_spec.split(":")
        if len(parts) != 3:
            return {"success": False, "error": f"Invalid reverse tunnel spec: {reverse_tunnel_spec}. Expected: remote_port:manager_host:manager_port"}

        remote_port = parts[0]
        manager_host = parts[1]
        manager_port = parts[2]

        logger.info(f"[Deploy] Setting up reverse tunnel: remote:{remote_port} -> {manager_host}:{manager_port}")

        # Kill any existing reverse tunnel to this remote
        subprocess.run(
            f"pkill -f 'ssh.*-R.*{remote_port}:.*{ssh_command.split()[-1]}' 2>/dev/null || true",
            shell=True,
            timeout=5
        )
        await asyncio.sleep(1)

        # Build SSH command with reverse tunnel
        # Extract port and host from ssh_command
        ssh_info = parse_ssh_command(ssh_command)
        if not ssh_info:
            return {"success": False, "error": "Invalid SSH command format"}

        # Build tunnel command
        tunnel_cmd = [
            "ssh",
            "-p", ssh_info["port"],
            "-N",  # No command execution
            "-R", f"{remote_port}:{manager_host}:{manager_port}",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=60",
            "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            f"{ssh_info['user']}@{ssh_info['host']}"
        ]

        # Start tunnel in background
        log_dir = Path(__file__).parent.parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        tunnel_log = log_dir / f"reverse_tunnel_{ssh_info['host']}.log"

        process = subprocess.Popen(
            tunnel_cmd,
            stdout=open(tunnel_log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )

        # Wait for tunnel to establish
        await asyncio.sleep(3)

        # Check if tunnel is still running
        if process.poll() is not None:
            # Process exited, read log for error
            log_content = tunnel_log.read_text() if tunnel_log.exists() else "No log"
            return {
                "success": False,
                "error": f"Reverse tunnel failed to start: {log_content[:200]}"
            }

        # Verify tunnel is working by checking port on remote
        ret, out, _ = run_ssh(ssh_command, f"nc -z localhost {remote_port} && echo connected", timeout=10)
        if "connected" not in out:
            process.terminate()
            return {
                "success": False,
                "error": f"Reverse tunnel started but port {remote_port} not accessible on remote"
            }

        # Save tunnel PID locally for cleanup
        pid_file = log_dir / f"reverse_tunnel_{ssh_info['host']}.pid"
        pid_file.write_text(str(process.pid))

        logger.info(f"[Deploy] Reverse tunnel established (PID: {process.pid})")
        return {
            "success": True,
            "tunnel_pid": process.pid,
            "remote_redis_port": remote_port,
            "message": f"Reverse tunnel active: remote:{remote_port} -> {manager_host}:{manager_port}"
        }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Reverse tunnel setup timed out"}
    except Exception as e:
        return {"success": False, "error": f"Reverse tunnel setup failed: {str(e)}"}


def get_deployment_error_suggestion(error: str) -> str:
    """Get user-friendly suggestion for fixing deployment errors.

    Args:
        error: Error message from deployment

    Returns:
        Suggestion string for fixing the error
    """
    error_lower = error.lower()

    if "permission denied" in error_lower and "publickey" in error_lower:
        return (
            "SSH key authentication failed. The remote worker cannot connect to the manager. "
            "Enable auto_install=true to automatically configure SSH keys, or manually run: "
            "ssh-copy-id -p <port> <user>@<manager-host> from the remote machine."
        )

    if "redis" in error_lower and "localhost" in error_lower:
        return (
            "Remote workers cannot connect to Redis on localhost. Options: "
            "1) Set REDIS_HOST to the manager's external IP in .env, or "
            "2) Configure manager_ssh for SSH tunnel."
        )

    if "network is unreachable" in error_lower or "could not resolve" in error_lower:
        return (
            "Remote machine has no network access. Use auto_install with offline mode "
            "to sync the venv without requiring pip. Check the machine's network configuration."
        )

    if "connection refused" in error_lower or "connection timed out" in error_lower:
        return (
            "Cannot establish SSH connection. Verify: "
            "1) SSH service is running on the remote machine, "
            "2) Port is correct and not blocked by firewall, "
            "3) SSH tunnel (if any) is active."
        )

    if "no module named" in error_lower:
        module = error.split("No module named")[-1].strip().strip("'\"")
        return (
            f"Missing Python module '{module}'. Use auto_install=true to install dependencies, "
            "or if network is unavailable, the offline sync will transfer packages from the manager."
        )

    if "project not found" in error_lower:
        return (
            "Project not installed on remote machine. Use auto_install=true to automatically "
            "sync project files, or manually clone/copy the project to the specified path."
        )

    if "venv" in error_lower or "virtual environment" in error_lower:
        return (
            "Virtual environment issue. Use auto_install=true to create and configure the venv. "
            "If network is unavailable, packages will be synced from the manager."
        )

    if "timed out" in error_lower:
        return (
            "Operation timed out. This may indicate: "
            "1) Slow network connection, "
            "2) Remote machine is overloaded, "
            "3) Large file transfer in progress. Try again or increase timeout."
        )

    # Default suggestion
    return "Check the worker logs for more details. Try restore with auto_install=true."


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
        auto_install: bool = True,
        force_sync: bool = False
    ) -> DeploymentResult:
        """Start a worker on a remote machine using slot configuration.

        Args:
            slot: WorkerSlot with deployment configuration
            auto_install: Whether to auto-install if project not found
            force_sync: Force sync code even if project exists (useful for updates)

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
                    error_msg = f"SSH connection failed: {err.strip() or 'Connection refused'}"
                    await self.update_slot_status(
                        slot, WorkerSlotStatus.OFFLINE,
                        error=error_msg
                    )
                    return DeploymentResult(
                        success=False,
                        message="SSH connection failed",
                        error=error_msg,
                        suggestion=get_deployment_error_suggestion(error_msg)
                    )
            except subprocess.TimeoutExpired:
                error_msg = "SSH connection timed out"
                await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=error_msg)
                return DeploymentResult(
                    success=False,
                    message="SSH connection timed out",
                    error=error_msg,
                    suggestion=get_deployment_error_suggestion(error_msg)
                )

            # Get Redis config from local settings
            settings = get_settings()
            redis_host = settings.redis_host
            redis_port = settings.redis_port

            # Check Redis is externally accessible (skip if using SSH tunnel)
            has_tunnel = slot.manager_ssh or slot.ssh_reverse_tunnel
            if redis_host in ("localhost", "127.0.0.1") and not has_tunnel:
                error_msg = (
                    "REDIS_HOST is set to localhost. Remote workers cannot connect directly. "
                    "Either set REDIS_HOST to Manager's external IP, or configure ssh_reverse_tunnel "
                    "(recommended) or manager_ssh for SSH tunnel."
                )
                return DeploymentResult(
                    success=False,
                    message="Redis not accessible",
                    error=error_msg,
                    suggestion=get_deployment_error_suggestion(error_msg)
                )

            # Setup reverse tunnel if configured (Manager -> Remote)
            remote_redis_port = redis_port  # Default to manager's redis port
            if slot.ssh_reverse_tunnel:
                logger.info(f"[Deploy] Setting up reverse SSH tunnel...")
                tunnel_result = await setup_reverse_tunnel(
                    slot.ssh_command,
                    slot.ssh_reverse_tunnel,
                    slot.project_path
                )
                if not tunnel_result["success"]:
                    error_msg = tunnel_result.get("error", "Reverse tunnel setup failed")
                    await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=error_msg)
                    return DeploymentResult(
                        success=False,
                        message="Reverse tunnel setup failed",
                        error=error_msg,
                        suggestion="Check SSH connectivity and ensure the remote port is available."
                    )
                remote_redis_port = tunnel_result["remote_redis_port"]
                logger.info(f"[Deploy] {tunnel_result.get('message')}")

            # Auto-setup SSH key for manager tunnel if needed
            if slot.manager_ssh and auto_install:
                logger.info(f"[Deploy] Setting up SSH key for manager tunnel...")
                key_result = await setup_ssh_key_for_manager(slot.ssh_command, slot.manager_ssh)
                if not key_result["success"]:
                    logger.warning(f"[Deploy] SSH key setup warning: {key_result.get('error')}")
                    # Don't fail deployment, just warn - user may have already configured keys
                else:
                    logger.info(f"[Deploy] {key_result.get('message')}")

            # Check if project exists on remote
            ret, out, _ = run_ssh(slot.ssh_command, f"test -d {slot.project_path}/src && echo exists", timeout=10)
            project_exists = "exists" in out

            # Sync code if project doesn't exist OR force_sync is requested
            need_sync = not project_exists or force_sync

            if need_sync:
                if not auto_install and not project_exists:
                    error_msg = f"Project not found at {slot.project_path} on remote machine."
                    return DeploymentResult(
                        success=False,
                        message="Project not found",
                        error=error_msg,
                        suggestion=get_deployment_error_suggestion(error_msg)
                    )

                if auto_install or force_sync:
                    action = "Syncing" if project_exists else "Installing"
                    logger.info(f"[Deploy] {action} project on remote...")
                    install_result = await auto_install_worker(slot.ssh_command, slot.project_path, local_project)
                    if not install_result["success"]:
                        error_msg = install_result.get("error", "Installation failed")
                        await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=error_msg)
                        return DeploymentResult(
                            success=False,
                            message=f"{action} failed",
                            error=error_msg,
                            suggestion=get_deployment_error_suggestion(error_msg)
                        )

            # Check if virtual environment exists
            ret, out, _ = run_ssh(slot.ssh_command, f"test -f {slot.project_path}/env/bin/activate && echo exists", timeout=10)
            if "exists" not in out:
                if auto_install:
                    # Try online install first (pip install)
                    logger.info(f"[Deploy] Setting up virtual environment...")
                    install_result = await setup_venv_and_deps(slot.ssh_command, slot.project_path)
                    if not install_result["success"]:
                        error_msg = install_result.get("error", "")
                        # Check if it's a network issue - try offline sync
                        if any(kw in error_msg.lower() for kw in ["network", "unreachable", "timed out", "connection"]):
                            logger.info(f"[Deploy] Network issue detected, trying offline venv sync...")
                            offline_result = await sync_venv_offline(
                                slot.ssh_command, slot.project_path, local_project
                            )
                            if not offline_result["success"]:
                                combined_error = f"Online install failed: {error_msg}\nOffline sync also failed: {offline_result.get('error')}"
                                await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=combined_error)
                                return DeploymentResult(
                                    success=False,
                                    message="Setup failed",
                                    error=combined_error,
                                    suggestion=get_deployment_error_suggestion(combined_error)
                                )
                            logger.info(f"[Deploy] Offline venv sync successful")
                        else:
                            await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=error_msg)
                            return DeploymentResult(
                                success=False,
                                message="Setup failed",
                                error=error_msg,
                                suggestion=get_deployment_error_suggestion(error_msg)
                            )
                else:
                    error_msg = f"Virtual environment not found at {slot.project_path}/env"
                    return DeploymentResult(
                        success=False,
                        message="Virtual environment not found",
                        error=error_msg,
                        suggestion=get_deployment_error_suggestion(error_msg)
                    )

            # Create .env.worker on remote
            logger.info(f"[Deploy] Creating worker configuration...")

            # Determine Redis connection settings based on tunnel type
            if slot.ssh_reverse_tunnel:
                # Reverse tunnel: Worker connects to localhost on the tunneled port
                worker_redis_host = "localhost"
                worker_redis_port = remote_redis_port
                use_reverse_tunnel = "true"
            elif slot.manager_ssh:
                # Forward tunnel: Worker will create tunnel and connect to it
                worker_redis_host = redis_host
                worker_redis_port = redis_port
                use_reverse_tunnel = "false"
            else:
                # Direct connection: Worker connects to manager's Redis directly
                worker_redis_host = redis_host
                worker_redis_port = redis_port
                use_reverse_tunnel = "false"

            worker_config_lines = [
                f"REDIS_HOST={worker_redis_host}",
                f"REDIS_PORT={worker_redis_port}",
                f'WORKER_ALIAS="{slot.name or ""}"',
                f"DEPLOYMENT_MODE={slot.controller_type}",
                f"USE_REVERSE_TUNNEL={use_reverse_tunnel}",
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
                log_text = log_out.strip()[-500:] if log_out else ""
                error_msg = "Worker process failed to start"
                await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=error_msg)
                return DeploymentResult(
                    success=False,
                    message="Worker process failed to start",
                    error=error_msg,
                    suggestion=get_deployment_error_suggestion(log_text or error_msg),
                    logs=log_text
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
            error_msg = "Command timed out during deployment"
            await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error="Command timed out")
            return DeploymentResult(
                success=False,
                message="Deployment timed out",
                error=error_msg,
                suggestion=get_deployment_error_suggestion(error_msg)
            )
        except Exception as e:
            error_msg = str(e)
            await self.update_slot_status(slot, WorkerSlotStatus.OFFLINE, error=error_msg)
            return DeploymentResult(
                success=False,
                message="Deployment failed",
                error=f"Failed to deploy worker: {error_msg}",
                suggestion=get_deployment_error_suggestion(error_msg)
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
        result = await self.start_remote_worker(
            slot,
            auto_install=config.auto_install,
            force_sync=config.force_sync
        )
        result.slot_id = slot.id

        return result

    async def restore_worker(
        self,
        slot_id: int,
        auto_install: bool = False,
        force_sync: bool = False
    ) -> DeploymentResult:
        """Restore an offline worker by its slot ID.

        Args:
            slot_id: Worker slot ID to restore
            auto_install: Whether to auto-install if project not found
            force_sync: Force sync code even if project exists (useful for updates)

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

        return await self.start_remote_worker(
            slot,
            auto_install=auto_install,
            force_sync=force_sync
        )
