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
        kill_result = subprocess.run(
            ["pkill", "-f", "arq.*autotuner_worker"],
            capture_output=True,
            text=True,
            timeout=10
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
