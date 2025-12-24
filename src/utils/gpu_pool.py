"""
GPU Resource Pool for parallel experiment execution.

This module provides a GPU resource pool that manages allocation and deallocation
of GPU resources across concurrent experiments. It ensures:
- No GPU allocation conflicts
- Fair FIFO allocation order
- Automatic cleanup on failure
- Integration with existing gpu_monitor infrastructure
"""

import asyncio
import logging
from typing import List, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from utils.gpu_monitor import get_gpu_monitor
from utils.gpu_scheduler import estimate_gpu_requirements

logger = logging.getLogger(__name__)


@dataclass
class GPUAllocation:
    """Represents an allocated GPU resource."""
    gpu_indices: List[int]
    allocated_at: datetime
    experiment_id: Optional[int] = None
    params: Optional[dict] = None

    def __hash__(self):
        """Make GPUAllocation hashable for use in sets."""
        return hash((tuple(self.gpu_indices), self.allocated_at, self.experiment_id))


class GPUResourcePool:
    """
    GPU resource pool for managing concurrent experiment execution.

    Features:
    - FIFO queue for fair allocation
    - Atomic acquire/release operations
    - Integration with gpu_monitor for availability checking
    - Automatic cleanup via context manager

    Example:
        async with GPUResourcePool(max_parallel=3) as pool:
            allocation = await pool.acquire(required_gpus=2, experiment_id=1)
            try:
                # Run experiment with allocation.gpu_indices
                pass
            finally:
                await pool.release(allocation)
    """

    def __init__(self, max_parallel: int = 1):
        """
        Initialize GPU resource pool.

        Args:
            max_parallel: Maximum number of concurrent experiments
        """
        self.max_parallel = max_parallel
        self._lock = asyncio.Lock()
        self._wait_queue: asyncio.Queue = asyncio.Queue()
        self._allocations: Set[GPUAllocation] = set()
        self._in_use_gpus: Set[int] = set()
        self._gpu_monitor = get_gpu_monitor()
        self._initialized = False

        logger.info(f"GPUResourcePool initialized with max_parallel={max_parallel}")

    async def __aenter__(self):
        """Async context manager entry."""
        self._initialized = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup all allocations."""
        async with self._lock:
            if self._allocations:
                logger.warning(
                    f"GPUResourcePool cleanup: {len(self._allocations)} "
                    f"allocations still active"
                )
                # Force release all allocations
                for allocation in list(self._allocations):
                    await self._release_internal(allocation)
        return False

    async def acquire(
        self,
        required_gpus: int,
        min_memory_mb: int = 8000,
        experiment_id: Optional[int] = None,
        params: Optional[dict] = None,
        timeout: Optional[float] = None
    ) -> GPUAllocation:
        """
        Acquire GPU resources for an experiment.

        This method:
        1. Waits for available slot if at max_parallel capacity
        2. Selects optimal GPUs using availability scoring
        3. Returns GPUAllocation object

        Args:
            required_gpus: Number of GPUs needed
            min_memory_mb: Minimum free memory per GPU
            experiment_id: Optional experiment ID for tracking
            params: Optional parameters for tracking
            timeout: Optional timeout in seconds

        Returns:
            GPUAllocation object with selected GPU indices

        Raises:
            asyncio.TimeoutError: If timeout expires
            RuntimeError: If insufficient GPUs available
        """
        start_time = asyncio.get_event_loop().time()

        logger.info(
            f"Requesting {required_gpus} GPUs (min_memory={min_memory_mb}MB) "
            f"for experiment {experiment_id}"
        )

        try:
            if timeout:
                return await asyncio.wait_for(
                    self._acquire_internal(
                        required_gpus, min_memory_mb, experiment_id, params
                    ),
                    timeout=timeout
                )
            else:
                return await self._acquire_internal(
                    required_gpus, min_memory_mb, experiment_id, params
                )
        except asyncio.TimeoutError:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.error(
                f"GPU acquisition timeout after {elapsed:.1f}s "
                f"(requested {required_gpus} GPUs)"
            )
            raise

    async def _acquire_internal(
        self,
        required_gpus: int,
        min_memory_mb: int,
        experiment_id: Optional[int],
        params: Optional[dict]
    ) -> GPUAllocation:
        """Internal acquisition logic with lock."""

        # Wait for slot if at capacity
        while True:
            async with self._lock:
                if len(self._allocations) < self.max_parallel:
                    break

            # Wait a bit before checking again
            logger.debug(
                f"At capacity ({len(self._allocations)}/{self.max_parallel}), "
                f"waiting for slot..."
            )
            await asyncio.sleep(1.0)

        # Select GPUs
        async with self._lock:
            gpu_indices = await self._select_gpus(required_gpus, min_memory_mb)

            # Create allocation
            allocation = GPUAllocation(
                gpu_indices=gpu_indices,
                allocated_at=datetime.now(),
                experiment_id=experiment_id,
                params=params
            )

            # Mark as allocated
            self._allocations.add(allocation)
            self._in_use_gpus.update(gpu_indices)

            logger.info(
                f"✓ Allocated GPUs {gpu_indices} to experiment {experiment_id} "
                f"({len(self._allocations)}/{self.max_parallel} slots used)"
            )

            return allocation

    async def _select_gpus(
        self,
        required_gpus: int,
        min_memory_mb: int
    ) -> List[int]:
        """
        Select optimal GPUs for allocation.

        Uses availability scoring from gpu_monitor and excludes already allocated GPUs.

        Args:
            required_gpus: Number of GPUs needed
            min_memory_mb: Minimum free memory per GPU

        Returns:
            List of GPU indices

        Raises:
            RuntimeError: If insufficient GPUs available
        """
        if not self._gpu_monitor.is_available():
            logger.warning(
                "GPU monitoring not available, returning sequential GPU indices"
            )
            return list(range(required_gpus))

        # Query GPU status
        snapshot = self._gpu_monitor.query_gpus(use_cache=False)
        if not snapshot or not snapshot.gpus:
            raise RuntimeError("Failed to query GPU status")

        # Filter available GPUs (excluding already allocated ones)
        available_gpus = []
        for gpu in snapshot.gpus:
            if gpu.index in self._in_use_gpus:
                logger.debug(f"GPU {gpu.index} already allocated, skipping")
                continue

            if gpu.memory_free_mb < min_memory_mb:
                logger.debug(
                    f"GPU {gpu.index} has insufficient memory "
                    f"({gpu.memory_free_mb}MB < {min_memory_mb}MB)"
                )
                continue

            available_gpus.append(gpu)

        # Check if enough GPUs available
        if len(available_gpus) < required_gpus:
            raise RuntimeError(
                f"Insufficient GPUs available: need {required_gpus}, "
                f"found {len(available_gpus)} "
                f"(min_memory={min_memory_mb}MB, "
                f"already_allocated={len(self._in_use_gpus)})"
            )

        # Sort by availability score (higher is better)
        available_gpus.sort(key=lambda g: g.score, reverse=True)

        # Select top N GPUs
        selected = available_gpus[:required_gpus]
        gpu_indices = [gpu.index for gpu in selected]

        # Validate memory balance for multi-GPU configurations
        if required_gpus > 1:
            from utils.gpu_selection import validate_memory_balance

            memory_amounts = [gpu.memory_free_mb for gpu in selected]
            is_balanced, msg = validate_memory_balance(memory_amounts, min_ratio=0.8)

            if not is_balanced:
                logger.error(f"GPU memory imbalance detected: {msg}")
                raise RuntimeError(
                    f"GPU memory capacity is unbalanced. {msg}. "
                    f"For multi-GPU (TP > 1) configurations, all GPUs must have similar available memory."
                )

            logger.info(f"GPU memory balance validated: {msg}")

        logger.debug(
            f"Selected GPUs {gpu_indices} "
            f"(scores: {[f'{g.score:.3f}' for g in selected]})"
        )

        return gpu_indices

    async def release(self, allocation: GPUAllocation) -> None:
        """
        Release GPU resources.

        Args:
            allocation: GPUAllocation object from acquire()
        """
        async with self._lock:
            await self._release_internal(allocation)

    async def _release_internal(self, allocation: GPUAllocation) -> None:
        """Internal release logic (assumes lock is held)."""
        if allocation in self._allocations:
            self._allocations.remove(allocation)
            self._in_use_gpus.difference_update(allocation.gpu_indices)

            duration = (datetime.now() - allocation.allocated_at).total_seconds()
            logger.info(
                f"✓ Released GPUs {allocation.gpu_indices} "
                f"from experiment {allocation.experiment_id} "
                f"(held for {duration:.1f}s, "
                f"{len(self._allocations)}/{self.max_parallel} slots used)"
            )
        else:
            logger.warning(
                f"Attempted to release unknown allocation: {allocation}"
            )

    def get_status(self) -> dict:
        """
        Get current pool status.

        Returns:
            Dictionary with pool statistics
        """
        return {
            "max_parallel": self.max_parallel,
            "active_allocations": len(self._allocations),
            "in_use_gpus": sorted(list(self._in_use_gpus)),
            "allocations": [
                {
                    "gpu_indices": alloc.gpu_indices,
                    "experiment_id": alloc.experiment_id,
                    "allocated_at": alloc.allocated_at.isoformat(),
                    "duration_seconds": (
                        datetime.now() - alloc.allocated_at
                    ).total_seconds(),
                }
                for alloc in self._allocations
            ]
        }


async def estimate_and_acquire(
    pool: GPUResourcePool,
    task_config: dict,
    experiment_id: Optional[int] = None,
    params: Optional[dict] = None,
    timeout: Optional[float] = None
) -> GPUAllocation:
    """
    Helper function to estimate GPU requirements and acquire resources.

    This combines estimate_gpu_requirements() from gpu_scheduler with
    the resource pool acquisition.

    Args:
        pool: GPUResourcePool instance
        task_config: Task configuration dictionary
        experiment_id: Optional experiment ID for tracking
        params: Optional parameters for tracking
        timeout: Optional timeout in seconds

    Returns:
        GPUAllocation object

    Example:
        async with GPUResourcePool(max_parallel=3) as pool:
            allocation = await estimate_and_acquire(
                pool, task_config, experiment_id=1
            )
            try:
                # Run experiment
                pass
            finally:
                await pool.release(allocation)
    """
    required_gpus, estimated_memory_mb = estimate_gpu_requirements(task_config)

    logger.info(
        f"Estimated requirements: {required_gpus} GPUs, "
        f"{estimated_memory_mb}MB per GPU"
    )

    return await pool.acquire(
        required_gpus=required_gpus,
        min_memory_mb=estimated_memory_mb,
        experiment_id=experiment_id,
        params=params,
        timeout=timeout
    )
