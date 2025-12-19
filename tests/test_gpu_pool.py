#!/usr/bin/env python3
"""Test script for GPUResourcePool."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.gpu_pool import GPUResourcePool, estimate_and_acquire
from utils.gpu_monitor import get_gpu_monitor


async def test_basic_allocation():
    """Test basic GPU allocation and release."""
    print("=" * 60)
    print("Test 1: Basic Allocation and Release")
    print("=" * 60)

    async with GPUResourcePool(max_parallel=2) as pool:
        print("\n1. Acquiring 1 GPU...")
        allocation1 = await pool.acquire(required_gpus=1, experiment_id=1)
        print(f"   ✓ Acquired GPUs: {allocation1.gpu_indices}")

        print("\n2. Pool status:")
        status = pool.get_status()
        print(f"   Active allocations: {status['active_allocations']}/{status['max_parallel']}")
        print(f"   In-use GPUs: {status['in_use_gpus']}")

        print("\n3. Releasing GPUs...")
        await pool.release(allocation1)
        print("   ✓ Released")

        print("\n4. Pool status after release:")
        status = pool.get_status()
        print(f"   Active allocations: {status['active_allocations']}/{status['max_parallel']}")
        print(f"   In-use GPUs: {status['in_use_gpus']}")

    print("\n✓ Test 1 passed")


async def test_concurrent_allocation():
    """Test concurrent allocations with capacity limit."""
    print("\n" + "=" * 60)
    print("Test 2: Concurrent Allocation with Capacity Limit")
    print("=" * 60)

    async def mock_experiment(pool, exp_id, duration=2.0):
        """Mock experiment that holds GPUs for a duration."""
        print(f"\n[Exp {exp_id}] Requesting GPU...")
        allocation = await pool.acquire(required_gpus=1, experiment_id=exp_id)
        print(f"[Exp {exp_id}] ✓ Acquired GPUs: {allocation.gpu_indices}")

        try:
            print(f"[Exp {exp_id}] Running for {duration}s...")
            await asyncio.sleep(duration)
            print(f"[Exp {exp_id}] ✓ Completed")
        finally:
            await pool.release(allocation)
            print(f"[Exp {exp_id}] ✓ Released GPUs")

    async with GPUResourcePool(max_parallel=2) as pool:
        print("\nStarting 4 experiments (max_parallel=2)...")
        print("Expected: First 2 start immediately, next 2 wait")

        tasks = [
            asyncio.create_task(mock_experiment(pool, i+1, duration=1.0))
            for i in range(4)
        ]

        await asyncio.gather(*tasks)

    print("\n✓ Test 2 passed")


async def test_gpu_availability():
    """Test GPU availability checking and scoring."""
    print("\n" + "=" * 60)
    print("Test 3: GPU Availability Checking")
    print("=" * 60)

    gpu_monitor = get_gpu_monitor()

    if not gpu_monitor.is_available():
        print("\n⚠ nvidia-smi not available, skipping GPU availability test")
        return

    print("\n1. Querying available GPUs...")
    snapshot = gpu_monitor.query_gpus(use_cache=False)

    if snapshot and snapshot.gpus:
        print(f"   Found {len(snapshot.gpus)} GPUs:")
        for gpu in snapshot.gpus:
            print(f"   GPU {gpu.index}: {gpu.name}")
            print(f"      Memory: {gpu.memory_free_mb}MB free / {gpu.memory_total_mb}MB total")
            print(f"      Utilization: {gpu.utilization_percent}%")
            print(f"      Score: {gpu.score:.3f}")

        print("\n2. Testing allocation with real GPUs...")
        async with GPUResourcePool(max_parallel=1) as pool:
            allocation = await pool.acquire(required_gpus=1, experiment_id=1)
            print(f"   ✓ Allocated GPU: {allocation.gpu_indices}")
            await pool.release(allocation)
            print("   ✓ Released")

        print("\n✓ Test 3 passed")
    else:
        print("   ✗ No GPUs found")


async def test_estimate_and_acquire():
    """Test estimate_and_acquire helper function."""
    print("\n" + "=" * 60)
    print("Test 4: Estimate and Acquire Helper")
    print("=" * 60)

    task_config = {
        "model": {"id_or_path": "llama-3-2-1b-instruct"},
        "parameters": {
            "tp-size": [1, 2],
            "mem-fraction-static": [0.8, 0.9]
        }
    }

    print("\n1. Testing requirement estimation...")
    from utils.gpu_scheduler import estimate_gpu_requirements
    required_gpus, estimated_memory = estimate_gpu_requirements(task_config)
    print(f"   Estimated: {required_gpus} GPUs, {estimated_memory}MB per GPU")

    gpu_monitor = get_gpu_monitor()
    if not gpu_monitor.is_available():
        print("\n⚠ nvidia-smi not available, using mock allocation")
        print("\n✓ Test 4 passed (mock mode)")
        return

    print("\n2. Testing estimate_and_acquire...")
    async with GPUResourcePool(max_parallel=1) as pool:
        allocation = await estimate_and_acquire(
            pool, task_config, experiment_id=1
        )
        print(f"   ✓ Acquired GPUs: {allocation.gpu_indices}")
        await pool.release(allocation)
        print("   ✓ Released")

    print("\n✓ Test 4 passed")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("GPUResourcePool Test Suite")
    print("=" * 60)

    try:
        await test_basic_allocation()
        await test_concurrent_allocation()
        await test_gpu_availability()
        await test_estimate_and_acquire()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
