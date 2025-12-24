"""
GPU type definitions for different contexts.

This module consolidates GPU information types that were previously
scattered across gpu_discovery.py and gpu_monitor.py, preventing
import confusion and enabling better code reuse.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class LocalGPUInfo:
    """
    GPU information for local machine (nvidia-smi output).
    
    Used by gpu_monitor.py for single-machine GPU allocation.
    """
    index: int
    uuid: str
    name: str
    memory_total_mb: int
    memory_free_mb: int
    memory_used_mb: int
    utilization_gpu: int
    utilization_memory: float
    temperature: Optional[int] = None
    power_draw: Optional[float] = None
    power_limit: Optional[float] = None
    compute_mode: Optional[str] = None
    processes: List[dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def score(self) -> float:
        """
        Selection score for GPU allocation.
        
        Prefers GPUs with:
        - More free memory
        - Lower utilization
        - Fewer running processes
        """
        memory_score = self.memory_free_mb / max(self.memory_total_mb, 1)
        utilization_score = (100 - self.utilization_gpu) / 100
        process_penalty = len(self.processes) * 0.1
        
        return memory_score * 0.6 + utilization_score * 0.3 - process_penalty


@dataclass
class ClusterGPUInfo:
    """
    GPU information for Kubernetes cluster nodes.
    
    Used by gpu_discovery.py for cluster-wide GPU discovery and allocation.
    """
    node_name: str
    gpu_index: int
    gpu_model: str
    memory_total_mb: int
    memory_free_mb: int
    memory_used_mb: int
    utilization_gpu: int
    has_metrics: bool
    allocatable: bool = True  # Whether GPU is available for allocation
    pod_name: Optional[str] = None  # Current pod using this GPU (if any)
    namespace: Optional[str] = None

    @property
    def index(self) -> int:
        return self.gpu_index

    @property
    def name(self) -> str:
        return self.gpu_model

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def score(self) -> float:
        """
        Selection score for GPU allocation.
        
        Similar to LocalGPUInfo but considers cluster-specific factors.
        """
        if not self.allocatable or not self.has_metrics:
            return -1.0  # Not suitable for allocation

        memory_score = self.memory_free_mb / max(self.memory_total_mb, 1)
        utilization_score = (100 - self.utilization_gpu) / 100
        
        return memory_score * 0.7 + utilization_score * 0.3


# Type aliases for backward compatibility
# These allow existing code to continue using "GPUInfo" while we migrate
# to more specific type names
GPUInfo = LocalGPUInfo  # Default to local for backward compatibility


def is_gpu_available(gpu: LocalGPUInfo, min_memory_mb: int = 0) -> bool:
    """
    Check if a GPU is available for allocation.
    
    Args:
        gpu: GPU information
        min_memory_mb: Minimum free memory required (MB)
        
    Returns:
        True if GPU has sufficient free memory and is not in exclusive mode
    """
    if gpu.memory_free_mb < min_memory_mb:
        return False
        
    # Check if GPU is in exclusive compute mode (only one process allowed)
    if gpu.compute_mode in ["Exclusive_Process", "Exclusive_Thread"]:
        # Only available if no processes are running
        return len(gpu.processes) == 0
        
    return True


def is_cluster_gpu_available(gpu: ClusterGPUInfo, min_memory_mb: int = 0) -> bool:
    """
    Check if a cluster GPU is available for allocation.
    
    Args:
        gpu: Cluster GPU information
        min_memory_mb: Minimum free memory required (MB)
        
    Returns:
        True if GPU is allocatable, has metrics, and meets memory requirement
    """
    return (
        gpu.allocatable
        and gpu.has_metrics
        and gpu.memory_free_mb >= min_memory_mb
        and gpu.pod_name is None  # Not currently allocated
    )
