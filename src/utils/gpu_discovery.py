"""
GPU Discovery Utility

Provides functions to discover and select idle GPUs across the Kubernetes cluster.
"""

import subprocess
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .gpu_types import ClusterGPUInfo


@dataclass
class NodeGPUSummary:
    """Summary of GPU resources on a node."""
    node_name: str
    total_gpus: int
    allocatable_gpus: int
    gpus_with_metrics: List[ClusterGPUInfo]
    avg_utilization: float
    avg_memory_usage: float
    idle_gpu_count: int  # GPUs with low utilization


def get_cluster_gpu_status() -> List[ClusterGPUInfo]:
    """
    Query cluster-wide GPU status using kubectl and nvidia-smi.

    Returns:
        List of GPUInfo objects for all GPUs in the cluster
    """
    try:
        # Get local hostname
        local_hostname = subprocess.run(
            ["hostname"],
            capture_output=True,
            text=True,
            timeout=2
        ).stdout.strip()

        # Query Kubernetes nodes
        result = subprocess.run(
            ["kubectl", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print("Warning: kubectl command failed")
            return []

        nodes_data = json.loads(result.stdout)
        all_gpus = []

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
                    break

            for key in allocatable:
                if "nvidia.com/gpu" in key:
                    gpu_allocatable = int(allocatable[key])
                    break

            if gpu_capacity == 0:
                continue

            # Get GPU metrics
            gpu_metrics = {}

            # For local node, use direct nvidia-smi
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
                    print(f"Error getting GPU metrics for local node {node_name}: {e}")
            else:
                # For remote nodes, find pods with GPU access
                try:
                    pod_result = subprocess.run(
                        [
                            "kubectl", "get", "pods", "--all-namespaces",
                            "--field-selector", f"spec.nodeName={node_name}",
                            "-o", "json"
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )

                    if pod_result.returncode == 0:
                        pods_data = json.loads(pod_result.stdout)

                        # Find a pod with GPU access
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

                        # If found, exec nvidia-smi
                        if target_pod:
                            nvidia_result = subprocess.run(
                                [
                                    "kubectl", "exec", "-n", target_namespace, target_pod,
                                    "--", "nvidia-smi",
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
                    print(f"Warning: Could not get GPU metrics for remote node {node_name}: {e}")

            # Get GPU type from labels
            labels = node["metadata"].get("labels", {})
            gpu_type = labels.get("nvidia.com/gpu.product", "Unknown")

            # Create GPUInfo objects for each GPU
            for i in range(gpu_capacity):
                is_allocatable = i < gpu_allocatable
                has_metrics = i in gpu_metrics

                if has_metrics:
                    metrics = gpu_metrics[i]
                    gpu_info = ClusterGPUInfo(
                        node_name=node_name,
                        gpu_index=i,
                        gpu_model=metrics["name"],
                        memory_total_mb=metrics["memory_total_mb"],
                        memory_used_mb=metrics["memory_used_mb"],
                        memory_free_mb=metrics["memory_free_mb"],
                        utilization_gpu=metrics["utilization_percent"],
                        has_metrics=True,
                        allocatable=is_allocatable
                    )
                else:
                    # GPU without metrics - create placeholder
                    gpu_info = ClusterGPUInfo(
                        node_name=node_name,
                        gpu_index=i,
                        gpu_model=gpu_type,
                        memory_total_mb=0,
                        memory_used_mb=0,
                        memory_free_mb=0,
                        utilization_gpu=0,
                        has_metrics=False,
                        allocatable=is_allocatable
                    )

                all_gpus.append(gpu_info)

        return all_gpus

    except Exception as e:
        print(f"Error querying cluster GPU status: {e}")
        return []


def get_node_gpu_summaries() -> Dict[str, NodeGPUSummary]:
    """
    Get GPU summaries grouped by node.

    Returns:
        Dictionary mapping node name to NodeGPUSummary
    """
    all_gpus = get_cluster_gpu_status()
    summaries = {}

    # Group by node
    nodes = {}
    for gpu in all_gpus:
        if gpu.node_name not in nodes:
            nodes[gpu.node_name] = []
        nodes[gpu.node_name].append(gpu)

    # Create summaries
    for node_name, gpus in nodes.items():
        gpus_with_metrics = [g for g in gpus if g.has_metrics]
        allocatable_gpus = sum(1 for g in gpus if g.allocatable)

        if gpus_with_metrics:
            avg_utilization = sum(g.utilization_gpu for g in gpus_with_metrics) / len(gpus_with_metrics)
            # Calculate memory usage percentage
            avg_memory_usage = sum(
                (g.memory_used_mb / g.memory_total_mb * 100) if g.memory_total_mb > 0 else 0
                for g in gpus_with_metrics
            ) / len(gpus_with_metrics)
            # Define "idle" as allocatable (not reserved by K8s) AND < 30% utilization AND < 50% memory
            idle_count = sum(1 for g in gpus_with_metrics
                           if g.allocatable and g.utilization_gpu < 30 and (g.memory_used_mb / g.memory_total_mb * 100 if g.memory_total_mb > 0 else 0) < 50)
        else:
            # No metrics available - assume GPUs are idle if allocatable
            # This happens when no GPU pods are running on the node
            avg_utilization = 0.0
            avg_memory_usage = 0.0
            idle_count = allocatable_gpus  # All allocatable GPUs are considered idle

        summary = NodeGPUSummary(
            node_name=node_name,
            total_gpus=len(gpus),
            allocatable_gpus=allocatable_gpus,
            gpus_with_metrics=gpus_with_metrics,
            avg_utilization=avg_utilization,
            avg_memory_usage=avg_memory_usage,
            idle_gpu_count=idle_count
        )
        summaries[node_name] = summary

    return summaries


def find_best_node_for_deployment(
    required_gpus: int = 1,
    utilization_threshold: float = 30.0,
    memory_threshold: float = 50.0
) -> Optional[str]:
    """
    Find the best node for deploying a new inference service.

    Selection criteria (in order):
    1. Must have enough allocatable GPUs
    2. Prefer nodes with idle GPUs (low utilization and memory usage)
    3. Among idle nodes, prefer the one with most idle GPUs
    4. If no idle nodes, prefer node with lowest average utilization

    Args:
        required_gpus: Number of GPUs required for deployment
        utilization_threshold: GPU utilization % threshold for "idle" (default: 30%)
        memory_threshold: Memory usage % threshold for "idle" (default: 50%)

    Returns:
        Node name to deploy to, or None if no suitable node found
    """
    summaries = get_node_gpu_summaries()

    if not summaries:
        print("Warning: No GPU nodes found in cluster")
        return None

    # Filter nodes with enough allocatable GPUs
    suitable_nodes = [
        (name, summary) for name, summary in summaries.items()
        if summary.allocatable_gpus >= required_gpus
    ]

    if not suitable_nodes:
        print(f"Warning: No nodes with {required_gpus} allocatable GPU(s)")
        return None

    # Sort by selection criteria
    def node_priority(item: Tuple[str, NodeGPUSummary]) -> Tuple[int, int, float]:
        name, summary = item
        # Return tuple: (has_idle_gpus, idle_count, -avg_utilization)
        # Python sorts tuples lexicographically, so this prioritizes:
        # 1. Nodes with idle GPUs (higher priority)
        # 2. More idle GPUs (higher count)
        # 3. Lower utilization (negated for ascending order)
        has_idle = 1 if summary.idle_gpu_count >= required_gpus else 0
        return (-has_idle, -summary.idle_gpu_count, summary.avg_utilization)

    # Sort and select best node
    suitable_nodes.sort(key=node_priority)
    best_node_name, best_summary = suitable_nodes[0]

    print(f"Selected node '{best_node_name}' for deployment:")
    print(f"  - Allocatable GPUs: {best_summary.allocatable_gpus}/{best_summary.total_gpus}")
    print(f"  - Idle GPUs: {best_summary.idle_gpu_count}")
    print(f"  - Avg Utilization: {best_summary.avg_utilization:.1f}%")
    print(f"  - Avg Memory Usage: {best_summary.avg_memory_usage:.1f}%")

    return best_node_name
