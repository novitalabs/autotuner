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
    6. Save deployment configuration to database for future restores

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
    try:
        from web.services.worker_service import WorkerService, DeploymentConfig
        from web.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            service = WorkerService(db)

            config = DeploymentConfig(
                ssh_command=ssh_command,
                name=name,
                controller_type=mode,
                manager_ssh=manager_ssh,
                auto_install=auto_install,
            )

            result = await service.deploy_worker(config)

            response = {
                "success": result.success,
                "message": result.message,
                "slot_id": result.slot_id,
            }

            if result.worker_id:
                response["worker_id"] = result.worker_id
            if result.worker_info:
                response["worker"] = result.worker_info
            if result.error:
                response["error"] = result.error
            if result.logs:
                response["remote_logs"] = result.logs

            return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to deploy worker: {str(e)}"
        })


@tool
@register_tool(
    ToolCategory.SYSTEM,
    requires_auth=True,
    auth_scope=AuthorizationScope.ARQ_CONTROL
)
async def restore_worker(slot_id: int, auto_install: bool = False) -> str:
    """
    Restore an offline worker by its slot ID.

    This tool connects to a previously deployed worker machine via SSH
    and restarts the worker process. Use this when a worker goes offline
    due to machine restart, network issues, or process crash.

    The slot configuration (SSH command, project path, etc.) is retrieved
    from the database, so you don't need to provide deployment details again.

    Args:
        slot_id: Worker slot ID (can be found using list_worker_slots)
        auto_install: Re-install/update project files if needed (default: False)

    Returns:
        JSON string with restore status and worker info
    """
    try:
        from web.services.worker_service import WorkerService
        from web.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            service = WorkerService(db)
            result = await service.restore_worker(slot_id, auto_install=auto_install)

            response = {
                "success": result.success,
                "message": result.message,
                "slot_id": slot_id,
            }

            if result.worker_id:
                response["worker_id"] = result.worker_id
            if result.worker_info:
                response["worker"] = result.worker_info
            if result.error:
                response["error"] = result.error
            if result.logs:
                response["remote_logs"] = result.logs

            return json.dumps(response, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to restore worker: {str(e)}"
        })


@tool
@register_tool(ToolCategory.SYSTEM)
async def list_worker_slots() -> str:
    """
    List all persistent worker slot configurations.

    Shows all workers that have been deployed through this system,
    including:
    - Online workers currently running
    - Offline workers that can be restored
    - Unknown workers that never successfully connected

    Each slot includes deployment configuration (SSH command, project path)
    and cached hardware info (GPU count, model).

    Returns:
        JSON string with list of worker slots
    """
    try:
        from web.services.worker_service import WorkerService
        from web.db.session import AsyncSessionLocal

        async with AsyncSessionLocal() as db:
            service = WorkerService(db)
            slots = await service.get_all_slots()

            slot_list = []
            online_count = 0
            offline_count = 0
            unknown_count = 0

            for slot in slots:
                status = slot.current_status.value if hasattr(slot.current_status, 'value') else str(slot.current_status)

                if status == "online":
                    online_count += 1
                elif status == "offline":
                    offline_count += 1
                else:
                    unknown_count += 1

                slot_list.append({
                    "id": slot.id,
                    "name": slot.name,
                    "worker_id": slot.worker_id,
                    "ssh_command": slot.ssh_command,
                    "controller_type": slot.controller_type,
                    "status": status,
                    "hostname": slot.hostname,
                    "gpu_count": slot.gpu_count,
                    "gpu_model": slot.gpu_model,
                    "last_seen_at": slot.last_seen_at.isoformat() if slot.last_seen_at else None,
                    "last_error": slot.last_error,
                })

            return json.dumps({
                "success": True,
                "total_slots": len(slot_list),
                "online_count": online_count,
                "offline_count": offline_count,
                "unknown_count": unknown_count,
                "slots": slot_list
            }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"Failed to list worker slots: {str(e)}"
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
