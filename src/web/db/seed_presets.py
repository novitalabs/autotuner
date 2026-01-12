"""
System preset seeding for parameter presets.
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from web.db.models import ParameterPreset


SYSTEM_PRESETS = [
	{
		"name": "vLLM Performance Tuning",
		"description": "Comprehensive vLLM parameters covering parallelism, memory, scheduling, and optimization",
		"category": "performance",
		"runtime": "vllm",
		"is_system": True,
		"parameters": {
			# === Parallelism ===
			"tensor-parallel-size": [1, 2, 4],
			"pipeline-parallel-size": [1],
			"data-parallel-size": [1],
			# === Memory & KV Cache ===
			"gpu-memory-utilization": [0.85, 0.9, 0.95],
			"kv-cache-dtype": ["auto", "fp8_e5m2"],
			"block-size": [16, 32],
			"swap-space": [4],  # CPU swap space in GiB
			"cpu-offload-gb": [0],
			# === Scheduling & Batching ===
			"max-num-batched-tokens": [2048, 4096, 8192],
			"max-num-seqs": [64, 128, 256],
			"enable-chunked-prefill": [True],
			"max-num-partial-prefills": [1],
			"scheduling-policy": ["fcfs"],
			# === Prefix Caching ===
			"enable-prefix-caching": [True, False],
			# === Optimization ===
			"enforce-eager": [False],  # Disable CUDA graphs
			"disable-sliding-window": [False],
			# === Quantization ===
			"quantization": [None],
			"dtype": ["auto"],
		},
		"metadata": {
			"author": "system",
			"tags": ["vllm", "performance", "comprehensive"],
			"recommended_for": "General vLLM performance tuning with latency/throughput trade-offs",
		},
	},
	{
		"name": "SGLang Performance Tuning",
		"description": "Comprehensive SGLang parameters covering parallelism, memory, scheduling, compute, and batch optimization",
		"category": "performance",
		"runtime": "sglang",
		"is_system": True,
		"parameters": {
			# === Parallelism ===
			"tp-size": [1, 2, 4],
			"dp-size": [1],
			"ep-size": [1],  # Expert Parallel for MoE
			# === Memory & Throughput ===
			"mem-fraction-static": [0.85, 0.88, 0.92],
			"max-prefill-tokens": [8192, 16384],
			"max-running-requests": [64, 128, 256],
			"cuda-graph-max-bs": [128, 256],
			"page-size": [1],
			"schedule-conservativeness": [0.8, 1.0],
			# === Scheduling Policy ===
			"schedule-policy": ["fcfs", "lpm"],
			"chunked-prefill-size": [2048, 4096, 8192],
			"disable-radix-cache": [False],
			"radix-eviction-policy": ["lru"],
			# === Compute Optimization ===
			"attention-backend": ["flashinfer", "triton"],
			"sampling-backend": ["flashinfer"],
			"disable-cuda-graph": [False],
			"enable-torch-compile": [False],
			"enable-mixed-chunk": [True, False],
			"enable-dp-attention": [False],
			"triton-attention-num-kv-splits": [8],
			# === MoE Optimization ===
			"moe-runner-backend": ["auto"],
			"moe-a2a-backend": ["none"],
			"disable-shared-experts-fusion": [False],
			# === Batch Optimization ===
			"num-continuous-decode-steps": [1, 2],
			"stream-interval": [1],
			# === KV Cache ===
			"kv-cache-dtype": ["auto", "fp8_e5m2"],
			# === Quantization ===
			"quantization": [None],
			"enable-fp32-lm-head": [False],
		},
		"metadata": {
			"author": "system",
			"tags": ["sglang", "performance", "comprehensive"],
			"recommended_for": "General SGLang performance tuning with latency/throughput trade-offs",
		},
	},
]


async def seed_system_presets(db: AsyncSession):
	"""Seed database with system presets if they don't exist."""
	for preset_data in SYSTEM_PRESETS:
		# Check if preset already exists
		result = await db.execute(select(ParameterPreset).where(ParameterPreset.name == preset_data["name"]))
		existing = result.scalar_one_or_none()

		if not existing:
			# Create preset with correct field name
			preset_dict = preset_data.copy()
			preset_dict["preset_metadata"] = preset_dict.pop("metadata")
			preset = ParameterPreset(**preset_dict)
			db.add(preset)
			print(f"  ✅ Seeded system preset: {preset_data['name']}")
		else:
			print(f"  ⏭️  System preset already exists: {preset_data['name']}")

	await db.commit()
	print(f"✅ System presets seeding complete ({len(SYSTEM_PRESETS)} presets)")
