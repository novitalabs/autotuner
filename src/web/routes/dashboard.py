"""
Dashboard API endpoints.
"""

import subprocess
import psutil
import os
import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta

from ..db.session import get_db
from ..db.models import Task, Experiment, TaskStatus, ExperimentStatus
from ..config import get_settings

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])
logger = logging.getLogger(__name__)

settings = get_settings()


@router.get("/gpu-status")
async def get_gpu_status() -> Dict[str, Any]:
	"""Get GPU status from nvidia-smi."""
	try:
		# Run nvidia-smi to get GPU info
		result = subprocess.run(
			[
				"nvidia-smi",
				"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
				"--format=csv,noheader,nounits"
			],
			capture_output=True,
			text=True,
			timeout=5
		)

		if result.returncode != 0:
			return {"available": False, "error": "nvidia-smi failed"}

		gpus = []
		for line in result.stdout.strip().split("\n"):
			if not line:
				continue

			parts = [p.strip() for p in line.split(",")]
			if len(parts) >= 7:
				gpus.append({
					"index": int(parts[0]),
					"name": parts[1],
					"memory_total_mb": int(parts[2]),
					"memory_used_mb": int(parts[3]),
					"memory_free_mb": int(parts[4]),
					"utilization_percent": int(parts[5]),
					"temperature_c": int(parts[6]),
					"memory_usage_percent": round(int(parts[3]) / int(parts[2]) * 100, 1)
				})

		return {
			"available": True,
			"gpus": gpus,
			"timestamp": datetime.now().isoformat()
		}

	except FileNotFoundError:
		return {"available": False, "error": "nvidia-smi not found"}
	except Exception as e:
		return {"available": False, "error": str(e)}


@router.get("/worker-status")
async def get_worker_status() -> Dict[str, Any]:
	"""Get ARQ worker process status."""
	try:
		# Check if worker process is running
		worker_running = False
		worker_pid = None
		worker_cpu = 0.0
		worker_memory_mb = 0.0
		worker_uptime = None

		# Look for ARQ worker process
		for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
			try:
				cmdline = proc.info['cmdline']
				if cmdline and 'arq' in ' '.join(cmdline) and 'autotuner_worker' in ' '.join(cmdline):
					worker_running = True
					worker_pid = proc.info['pid']
					worker_cpu = proc.info['cpu_percent']
					worker_memory_mb = round(proc.info['memory_info'].rss / 1024 / 1024, 1)

					# Calculate uptime
					create_time = datetime.fromtimestamp(proc.info['create_time'])
					uptime_seconds = (datetime.now() - create_time).total_seconds()
					worker_uptime = int(uptime_seconds)
					break
			except (psutil.NoSuchProcess, psutil.AccessDenied):
				continue

		# Check Redis connection
		redis_available = False
		try:
			import redis
			r = redis.Redis(
				host=settings.redis_host,
				port=settings.redis_port,
				db=settings.redis_db,
				socket_connect_timeout=2
			)
			redis_available = r.ping()
		except Exception:
			pass

		return {
			"worker_running": worker_running,
			"worker_pid": worker_pid,
			"worker_cpu_percent": worker_cpu,
			"worker_memory_mb": worker_memory_mb,
			"worker_uptime_seconds": worker_uptime,
			"redis_available": redis_available,
			"timestamp": datetime.now().isoformat()
		}

	except Exception as e:
		return {
			"error": str(e),
			"timestamp": datetime.now().isoformat()
		}


@router.get("/db-statistics")
async def get_db_statistics(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
	"""Get database statistics summary."""
	# Total tasks by status
	result = await db.execute(
		select(Task.status, func.count(Task.id))
		.group_by(Task.status)
	)
	tasks_by_status = {status.value: count for status, count in result.all()}

	# Total experiments by status
	result = await db.execute(
		select(Experiment.status, func.count(Experiment.id))
		.group_by(Experiment.status)
	)
	experiments_by_status = {status.value: count for status, count in result.all()}

	# Total counts
	result = await db.execute(select(func.count(Task.id)))
	total_tasks = result.scalar()

	result = await db.execute(select(func.count(Experiment.id)))
	total_experiments = result.scalar()

	# Recent activity (last 24 hours)
	yesterday = datetime.now() - timedelta(hours=24)
	result = await db.execute(
		select(func.count(Task.id))
		.where(Task.created_at >= yesterday)
	)
	tasks_last_24h = result.scalar()

	result = await db.execute(
		select(func.count(Experiment.id))
		.where(Experiment.created_at >= yesterday)
	)
	experiments_last_24h = result.scalar()

	# Average experiment duration
	result = await db.execute(
		select(func.avg(Experiment.elapsed_time))
		.where(Experiment.status == ExperimentStatus.SUCCESS)
		.where(Experiment.elapsed_time.isnot(None))
	)
	avg_experiment_duration = result.scalar()

	# Running tasks
	result = await db.execute(
		select(Task)
		.where(Task.status == TaskStatus.RUNNING)
		.order_by(Task.started_at.desc())
	)
	running_tasks = result.scalars().all()

	running_tasks_info = []
	for task in running_tasks:
		# Count experiments for this task
		result = await db.execute(
			select(func.count(Experiment.id))
			.where(Experiment.task_id == task.id)
			.where(Experiment.status == ExperimentStatus.SUCCESS)
		)
		completed_count = result.scalar()

		running_tasks_info.append({
			"id": task.id,
			"name": task.task_name,
			"started_at": task.started_at.isoformat() if task.started_at else None,
			"max_iterations": task.optimization_config.get("max_iterations", 0) if task.optimization_config else 0,
			"completed_experiments": completed_count
		})

	return {
		"total_tasks": total_tasks,
		"total_experiments": total_experiments,
		"tasks_by_status": tasks_by_status,
		"experiments_by_status": experiments_by_status,
		"tasks_last_24h": tasks_last_24h,
		"experiments_last_24h": experiments_last_24h,
		"avg_experiment_duration_seconds": round(avg_experiment_duration, 1) if avg_experiment_duration else None,
		"running_tasks": running_tasks_info,
		"timestamp": datetime.now().isoformat()
	}


@router.get("/cluster-gpu-status")
async def get_cluster_gpu_status() -> Dict[str, Any]:
	"""Get GPU status across all nodes in the Kubernetes cluster with detailed metrics."""
	try:
		# Get current node hostname to use local nvidia-smi for it
		local_hostname = subprocess.run(
			["hostname"],
			capture_output=True,
			text=True,
			timeout=2
		).stdout.strip()

		# Query Kubernetes nodes with GPU resources
		result = subprocess.run(
			[
				"kubectl",
				"get",
				"nodes",
				"-o",
				"json"
			],
			capture_output=True,
			text=True,
			timeout=10
		)

		if result.returncode != 0:
			return {"available": False, "error": "kubectl command failed", "mode": "cluster"}

		import json
		nodes_data = json.loads(result.stdout)

		cluster_gpus = []
		node_summaries = []
		total_gpus = 0
		total_allocatable_gpus = 0

		for node in nodes_data.get("items", []):
			node_name = node["metadata"]["name"]
			capacity = node["status"].get("capacity", {})
			allocatable = node["status"].get("allocatable", {})

			# Check for NVIDIA GPUs
			gpu_capacity = 0
			gpu_allocatable = 0

			for key in capacity:
				if "nvidia.com/gpu" in key:
					gpu_capacity = int(capacity[key])
					total_gpus += gpu_capacity
					break

			for key in allocatable:
				if "nvidia.com/gpu" in key:
					gpu_allocatable = int(allocatable[key])
					total_allocatable_gpus += gpu_allocatable
					break

			if gpu_capacity > 0:
				# Get actual GPU metrics from the node
				node_gpus = []
				gpu_metrics = {}

				# If this is the local node, use direct nvidia-smi
				if node_name == local_hostname:
					try:
						nvidia_result = subprocess.run(
							[
								"nvidia-smi",
								"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
								"--format=csv,noheader,nounits"
							],
							capture_output=True,
							text=True,
							timeout=5
						)

						if nvidia_result.returncode == 0 and nvidia_result.stdout:
							for line in nvidia_result.stdout.strip().split("\n"):
								if not line:
									continue
								parts = [p.strip() for p in line.split(",")]
								if len(parts) >= 7:
									gpu_idx = int(parts[0])
									gpu_metrics[gpu_idx] = {
										"name": parts[1],
										"memory_total_mb": int(parts[2]),
										"memory_used_mb": int(parts[3]),
										"memory_free_mb": int(parts[4]),
										"utilization_percent": int(parts[5]),
										"temperature_c": int(parts[6]),
										"memory_usage_percent": round(int(parts[3]) / int(parts[2]) * 100, 1)
									}
					except Exception as e:
						logger.error(f"Error getting local GPU metrics: {e}")
				else:
					# For remote nodes, try to find a pod with GPU access (has nvidia.com/gpu resource request)
					try:
						pod_result = subprocess.run(
							[
								"kubectl",
								"get",
								"pods",
								"--all-namespaces",
								"--field-selector",
								f"spec.nodeName={node_name}",
								"-o",
								"json"
							],
							capture_output=True,
							text=True,
							timeout=5
						)

						if pod_result.returncode == 0:
							pods_data = json.loads(pod_result.stdout)

							# Find a pod that has GPU access (has nvidia.com/gpu in resources)
							target_pod = None
							target_namespace = None

							for pod in pods_data.get("items", []):
								pod_name = pod["metadata"]["name"]
								pod_namespace = pod["metadata"]["namespace"]

								# Check if pod has GPU resources
								containers = pod["spec"].get("containers", [])
								for container in containers:
									resources = container.get("resources", {})
									limits = resources.get("limits", {})
									requests = resources.get("requests", {})

									if any("nvidia.com/gpu" in key for key in list(limits.keys()) + list(requests.keys())):
										target_pod = pod_name
										target_namespace = pod_namespace
										break

								if target_pod:
									break

							# If we found a pod with GPU access, exec nvidia-smi in it
							if target_pod:
								nvidia_result = subprocess.run(
									[
										"kubectl",
										"exec",
										"-n",
										target_namespace,
										target_pod,
										"--",
										"nvidia-smi",
										"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
										"--format=csv,noheader,nounits"
									],
									capture_output=True,
									text=True,
									timeout=5
								)

								if nvidia_result.returncode == 0 and nvidia_result.stdout:
									for line in nvidia_result.stdout.strip().split("\n"):
										if not line:
											continue
										parts = [p.strip() for p in line.split(",")]
										if len(parts) >= 7:
											gpu_idx = int(parts[0])
											gpu_metrics[gpu_idx] = {
												"name": parts[1],
												"memory_total_mb": int(parts[2]),
												"memory_used_mb": int(parts[3]),
												"memory_free_mb": int(parts[4]),
												"utilization_percent": int(parts[5]),
												"temperature_c": int(parts[6]),
												"memory_usage_percent": round(int(parts[3]) / int(parts[2]) * 100, 1)
											}
					except Exception as e:
						logger.warning(f"Could not get GPU metrics for remote node {node_name}: {e}")

				# Get node labels for GPU type
				labels = node["metadata"].get("labels", {})
				gpu_type = labels.get("nvidia.com/gpu.product", "Unknown")

				# Create GPU entries for this node
				node_gpu_list = []
				for i in range(gpu_capacity):
					gpu_entry = {
						"index": i,
						"node_name": node_name,
						"name": gpu_metrics.get(i, {}).get("name", gpu_type),
						"capacity": 1,
						"allocatable": 1 if i < gpu_allocatable else 0,
						"is_local": node_name == local_hostname
					}

					# Add detailed metrics if available
					if i in gpu_metrics:
						gpu_entry.update({
							"memory_total_mb": gpu_metrics[i]["memory_total_mb"],
							"memory_used_mb": gpu_metrics[i]["memory_used_mb"],
							"memory_free_mb": gpu_metrics[i]["memory_free_mb"],
							"memory_usage_percent": gpu_metrics[i]["memory_usage_percent"],
							"utilization_percent": gpu_metrics[i]["utilization_percent"],
							"temperature_c": gpu_metrics[i]["temperature_c"],
							"has_metrics": True
						})
					else:
						gpu_entry["has_metrics"] = False

					node_gpus.append(gpu_entry)
					node_gpu_list.append(gpu_entry)

				cluster_gpus.extend(node_gpus)

				# Create node summary
				gpus_with_metrics = [g for g in node_gpu_list if g.get("has_metrics")]
				if gpus_with_metrics:
					avg_util = sum(g.get("utilization_percent", 0) for g in gpus_with_metrics) / len(gpus_with_metrics)
					avg_mem = sum(g.get("memory_usage_percent", 0) for g in gpus_with_metrics) / len(gpus_with_metrics)
					total_mem = sum(g.get("memory_total_mb", 0) for g in gpus_with_metrics)
					used_mem = sum(g.get("memory_used_mb", 0) for g in gpus_with_metrics)
				else:
					avg_util = 0
					avg_mem = 0
					total_mem = 0
					used_mem = 0

				# Get GPU model from first GPU with metrics or label
				gpu_model = gpu_type
				if gpus_with_metrics:
					gpu_model = gpus_with_metrics[0].get("name", gpu_type)

				node_summaries.append({
					"node_name": node_name,
					"gpu_count": gpu_capacity,
					"allocatable_gpus": gpu_allocatable,
					"gpu_model": gpu_model,
					"gpus": node_gpu_list,
					"avg_utilization": round(avg_util, 1),
					"avg_memory_usage": round(avg_mem, 1),
					"total_memory_mb": total_mem,
					"used_memory_mb": used_mem,
					"is_local": node_name == local_hostname
				})

		return {
			"available": True,
			"mode": "cluster",
			"nodes": cluster_gpus,
			"node_summaries": node_summaries,
			"total_gpus": total_gpus,
			"total_allocatable_gpus": total_allocatable_gpus,
			"local_hostname": local_hostname,
			"timestamp": datetime.now().isoformat()
		}

	except FileNotFoundError:
		return {"available": False, "error": "kubectl not found", "mode": "cluster"}
	except Exception as e:
		return {"available": False, "error": str(e), "mode": "cluster"}


@router.get("/experiment-timeline")
async def get_experiment_timeline(
	hours: int = 24,
	db: AsyncSession = Depends(get_db)
) -> List[Dict[str, Any]]:
	"""
	Get experiment timeline data for visualization.

	Args:
		hours: Number of hours to look back (default 24)
	"""
	cutoff_time = datetime.now() - timedelta(hours=hours)

	result = await db.execute(
		select(Experiment)
		.where(Experiment.created_at >= cutoff_time)
		.order_by(Experiment.created_at)
	)
	experiments = result.scalars().all()

	timeline = []
	for exp in experiments:
		timeline.append({
			"id": exp.id,
			"task_id": exp.task_id,
			"experiment_id": exp.experiment_id,
			"status": exp.status.value,
			"created_at": exp.created_at.isoformat() if exp.created_at else None,
			"started_at": exp.started_at.isoformat() if exp.started_at else None,
			"completed_at": exp.completed_at.isoformat() if exp.completed_at else None,
			"elapsed_time": exp.elapsed_time,
			"objective_score": exp.objective_score
		})

	return timeline
