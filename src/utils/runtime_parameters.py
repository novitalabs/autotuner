"""
Runtime Parameter Registry

This module contains comprehensive lists of valid command-line parameters
for different inference runtimes (SGLang, vLLM).

These parameters were extracted from the respective runtime help outputs.
"""

from typing import Dict, List, Set


# SGLang Server Parameters (v0.5.2)
# Extracted from: python3 -m sglang.launch_server --help
SGLANG_PARAMETERS: Set[str] = {
	"allow-auto-truncate",
	"api-key",
	"attention-backend",
	"base-gpu-id",
	"bucket-e2e-request-latency",
	"bucket-inter-token-latency",
	"bucket-time-to-first-token",
	"chat-template",
	"chunked-prefill-size",
	"collect-tokens-histogram",
	"completion-template",
	"constrained-json-whitespace-pattern",
	"context-length",
	"cpu-offload-gb",
	"crash-dump-folder",
	"cuda-graph-bs",
	"cuda-graph-max-bs",
	"custom-weight-loader",
	"data-parallel-size",
	"debug-tensor-dump-inject",
	"debug-tensor-dump-input-file",
	"debug-tensor-dump-output-folder",
	"debug-tensor-dump-prefill-only",
	"decode-attention-backend",
	"decode-log-interval",
	"deepep-config",
	"deepep-mode",
	"delete-ckpt-after-loading",
	"device",
	"disable-chunked-prefix-cache",
	"disable-cuda-graph",
	"disable-cuda-graph-padding",
	"disable-custom-all-reduce",
	"disable-fast-image-processor",
	"disable-flashinfer-cutlass-moe-fp4-allgather",
	"disable-hybrid-swa-memory",
	"disable-outlines-disk-cache",
	"disable-overlap-schedule",
	"disable-radix-cache",
	"disable-shared-experts-fusion",
	"disaggregation-bootstrap-port",
	"disaggregation-decode-dp",
	"disaggregation-decode-tp",
	"disaggregation-ib-device",
	"disaggregation-mode",
	"disaggregation-prefill-pp",
	"disaggregation-transfer-backend",
	"dist-init-addr",
	"dist-timeout",
	"download-dir",
	"ds-channel-config-path",
	"ds-heavy-channel-num",
	"ds-heavy-channel-type",
	"ds-heavy-token-num",
	"ds-sparse-decode-threshold",
	"dtype",
	"enable-cache-report",
	"enable-cudagraph-gc",
	"enable-custom-logit-processor",
	"enable-deepep-moe",
	"enable-double-sparsity",
	"enable-dp-attention",
	"enable-dp-lm-head",
	"enable-eplb",
	"enable-ep-moe",
	"enable-expert-distribution-metrics",
	"enable-flashinfer-allreduce-fusion",
	"enable-flashinfer-cutlass-moe",
	"enable-flashinfer-mxfp4-moe",
	"enable-flashinfer-trtllm-moe",
	"enable-hierarchical-cache",
	"enable-lmcache",
	"enable-lora",
	"enable-memory-saver",
	"enable-metrics",
	"enable-metrics-for-all-schedulers",
	"enable-mixed-chunk",
	"enable-mscclpp",
	"enable-multimodal",
	"enable-nan-detection",
	"enable-nccl-nvls",
	"enable-p2p-check",
	"enable-pdmux",
	"enable-profile-cuda-graph",
	"enable-request-time-stats-logging",
	"enable-return-hidden-states",
	"enable-symm-mem",
	"enable-tokenizer-batch-encode",
	"enable-torch-compile",
	"enable-triton-kernel-moe",
	"enable-two-batch-overlap",
	"ep-dispatch-algorithm",
	"eplb-algorithm",
	"eplb-min-rebalancing-utilization-threshold",
	"eplb-rebalance-layers-per-chunk",
	"eplb-rebalance-num-iterations",
	"ep-num-redundant-experts",
	"expert-distribution-recorder-buffer-size",
	"expert-distribution-recorder-mode",
	"expert-parallel-size",
	"file-storage-path",
	"flashinfer-mla-disable-ragged",
	"flashinfer-mxfp4-moe-precision",
	"gc-warning-threshold-secs",
	"generation-tokens-buckets",
	"gpu-id-step",
	"grammar-backend",
	"hicache-io-backend",
	"hicache-mem-layout",
	"hicache-ratio",
	"hicache-size",
	"hicache-storage-backend",
	"hicache-storage-backend-extra-config",
	"hicache-storage-prefetch-policy",
	"hicache-write-policy",
	"host",
	"hybrid-kvcache-ratio",
	"init-expert-location",
	"is-embedding",
	"json-model-override-args",
	"kv-cache-dtype",
	"kv-events-config",
	"load-balance-method",
	"load-format",
	"log-level",
	"log-level-http",
	"log-requests",
	"log-requests-level",
	"lora-backend",
	"lora-paths",
	"lora-target-modules",
	"mamba-ssm-dtype",
	"max-loaded-loras",
	"max-lora-rank",
	"max-loras-per-batch",
	"max-mamba-cache-size",
	"max-micro-batch-size",
	"max-prefill-tokens",
	"max-queued-requests",
	"max-running-requests",
	"max-total-tokens",
	"mem-fraction-static",
	"mm-attention-backend",
	"model-impl",
	"model-loader-extra-config",
	"model-path",
	"moe-a2a-backend",
	"moe-dense-tp-size",
	"moe-runner-backend",
	"nccl-port",
	"nnodes",
	"node-rank",
	"numa-node",
	"num-continuous-decode-steps",
	"num-reserved-decode-tokens",
	"offload-group-size",
	"offload-mode",
	"offload-num-in-group",
	"offload-prefetch-step",
	"page-size",
	"pipeline-parallel-size",
	"port",
	"preferred-sampling-params",
	"prefill-attention-backend",
	"prefill-round-robin-balance",
	"prompt-tokens-buckets",
	"quantization",
	"quantization-param-path",
	"random-seed",
	"reasoning-parser",
	"revision",
	"sampling-backend",
	"schedule-conservativeness",
	"schedule-policy",
	"scheduler-recv-interval",
	"served-model-name",
	"show-time-cost",
	"skip-server-warmup",
	"skip-tokenizer-init",
	"sleep-on-idle",
	"sm-group-num",
	"speculative-accept-threshold-acc",
	"speculative-accept-threshold-single",
	"speculative-algorithm",
	"speculative-attention-mode",
	"speculative-draft-model-path",
	"speculative-draft-model-revision",
	"speculative-eagle-topk",
	"speculative-num-draft-tokens",
	"speculative-num-steps",
	"speculative-token-map",
	"stream-interval",
	"stream-output",
	"swa-full-tokens-ratio",
	"tbo-token-distribution-threshold",
	"tensor-parallel-size",
	"tokenizer-mode",
	"tokenizer-path",
	"tokenizer-worker-num",
	"tool-call-parser",
	"tool-server",
	"torchao-config",
	"torch-compile-max-bs",
	"triton-attention-num-kv-splits",
	"triton-attention-reduce-in-fp32",
	"trust-remote-code",
	"warmups",
	"watchdog-timeout",
	"weight-loader-disable-mmap",
	"weight-version",
}


# vLLM Server Parameters (v0.10.0)
# Extracted from: vllm/vllm-openai --help
VLLM_PARAMETERS: Set[str] = {
	"additional-config",
	"allow-credentials",
	"allowed-headers",
	"allowed-local-media-path",
	"allowed-methods",
	"allowed-origins",
	"api-key",
	"api-server-count",
	"async-scheduling",
	"block-size",
	"calculate-kv-scales",
	"chat-template",
	"chat-template-content-format",
	"code-revision",
	"collect-detailed-traces",
	"compilation-config",
	"config",
	"config-format",
	"cpu-offload-gb",
	"cuda-graph-sizes",
	"data-parallel-address",
	"data-parallel-backend",
	"data-parallel-hybrid-lb",
	"data-parallel-rank",
	"data-parallel-rpc-port",
	"data-parallel-size",
	"data-parallel-size-local",
	"data-parallel-start-rank",
	"default-mm-loras",
	"disable-async-output-proc",
	"disable-cascade-attn",
	"disable-chunked-mm-input",
	"disable-custom-all-reduce",
	"disable-fastapi-docs",
	"disable-frontend-multiprocessing",
	"disable-hybrid-kv-cache-manager",
	"disable-log-requests",
	"disable-log-stats",
	"disable-mm-preprocessor-cache",
	"disable-sliding-window",
	"disable-uvicorn-access-log",
	"distributed-executor-backend",
	"download-dir",
	"dtype",
	"enable-auto-tool-choice",
	"enable-chunked-prefill",
	"enable-eplb",
	"enable-expert-parallel",
	"enable-force-include-usage",
	"enable-lora",
	"enable-lora-bias",
	"enable-multimodal-encoder-data-parallel",
	"enable-prefix-caching",
	"enable-prompt-adapter",
	"enable-prompt-embeds",
	"enable-prompt-tokens-details",
	"enable-request-id-headers",
	"enable-server-load-tracking",
	"enable-sleep-mode",
	"enable-ssl-refresh",
	"enable-tokenizer-info-endpoint",
	"enforce-eager",
	"eplb-log-balancedness",
	"eplb-step-interval",
	"eplb-window-size",
	"fully-sharded-loras",
	"generation-config",
	"gpu-memory-utilization",
	"guided-decoding-backend",
	"guided-decoding-disable-additional-properties",
	"guided-decoding-disable-any-whitespace",
	"guided-decoding-disable-fallback",
	"headless",
	"hf-config-path",
	"hf-overrides",
	"hf-token",
	"host",
	"ignore-patterns",
	"interleave-mm-strings",
	"kv-cache-dtype",
	"kv-events-config",
	"kv-transfer-config",
	"limit-mm-per-prompt",
	"load-format",
	"log-config-file",
	"logits-processor-pattern",
	"logprobs-mode",
	"long-prefill-token-threshold",
	"lora-dtype",
	"lora-extra-vocab-size",
	"lora-modules",
	"max-cpu-loras",
	"max-log-len",
	"max-logprobs",
	"max-long-partial-prefills",
	"max-lora-rank",
	"max-loras",
	"max-model-len",
	"max-num-batched-tokens",
	"max-num-partial-prefills",
	"max-num-seqs",
	"max-parallel-loading-workers",
	"max-seq-len-to-capture",
	"media-io-kwargs",
	"middleware",
	"mm-processor-kwargs",
	"model",
	"model-impl",
	"model-loader-extra-config",
	"multi-step-stream-outputs",
	"num-gpu-blocks-override",
	"num-lookahead-slots",
	"num-redundant-experts",
	"num-scheduler-steps",
	"otlp-traces-endpoint",
	"override-attention-dtype",
	"override-generation-config",
	"override-neuron-config",
	"override-pooler-config",
	"pipeline-parallel-size",
	"port",
	"preemption-mode",
	"prefix-caching-hash-algo",
	"pt-load-map-location",
	"quantization",
	"ray-workers-use-nsight",
	"reasoning-parser",
	"response-role",
	"return-tokens-as-token-ids",
	"revision",
	"root-path",
	"rope-scaling",
	"rope-theta",
	"scheduler-cls",
	"scheduler-delay-factor",
	"scheduling-policy",
	"seed",
	"served-model-name",
	"show-hidden-metrics-for-version",
	"skip-tokenizer-init",
	"speculative-config",
	"ssl-ca-certs",
	"ssl-certfile",
	"ssl-cert-reqs",
	"ssl-keyfile",
	"swap-space",
	"task",
	"tensor-parallel-size",
	"tokenizer",
	"tokenizer-mode",
	"tokenizer-revision",
	"tool-call-parser",
	"tool-parser-plugin",
	"trust-remote-code",
	"use-tqdm-on-load",
	"uvicorn-log-level",
	"worker-cls",
	"worker-extension-cls",
}


# Common parameters between SGLang and vLLM
COMMON_PARAMETERS: Set[str] = SGLANG_PARAMETERS & VLLM_PARAMETERS

# Parameters unique to each runtime
SGLANG_ONLY_PARAMETERS: Set[str] = SGLANG_PARAMETERS - VLLM_PARAMETERS
VLLM_ONLY_PARAMETERS: Set[str] = VLLM_PARAMETERS - SGLANG_PARAMETERS


# Commonly tuned parameters for optimization (subset of all parameters)
# These are the parameters most frequently used in autotuning experiments
COMMONLY_TUNED_SGLANG: List[str] = [
	# Parallelism
	"tensor-parallel-size",
	"pipeline-parallel-size",
	"data-parallel-size",
	# Memory Management
	"mem-fraction-static",
	"max-total-tokens",
	"page-size",
	"cpu-offload-gb",
	"hybrid-kvcache-ratio",
	# Scheduling & Request Management
	"schedule-policy",
	"schedule-conservativeness",
	"max-running-requests",
	"max-queued-requests",
	"max-prefill-tokens",
	# Chunked Prefill
	"chunked-prefill-size",
	"enable-mixed-chunk",
	"max-micro-batch-size",
	# CUDA Graphs
	"cuda-graph-max-bs",
	"cuda-graph-bs",
	"disable-cuda-graph",
	"disable-cuda-graph-padding",
	# KV Cache
	"kv-cache-dtype",
	"disable-radix-cache",
	"enable-hierarchical-cache",
	"hicache-ratio",
	# Data Types & Quantization
	"dtype",
	"quantization",
	"quantization-param-path",
	# Attention Backends
	"attention-backend",
	"prefill-attention-backend",
	"decode-attention-backend",
	# Model Loading
	"load-format",
	"context-length",
	# LoRA
	"enable-lora",
	"max-loras-per-batch",
	"max-lora-rank",
	# Speculative Decoding
	"speculative-algorithm",
	"speculative-num-steps",
	"speculative-num-draft-tokens",
	# MOE
	"enable-ep-moe",
	"enable-deepep-moe",
	"moe-runner-backend",
	"expert-parallel-size",
	# Performance Optimizations
	"enable-torch-compile",
	"disable-custom-all-reduce",
	"enable-flashinfer-cutlass-moe",
	"disable-overlap-schedule",
	"enable-two-batch-overlap",
	"sampling-backend",
	"grammar-backend",
]

COMMONLY_TUNED_VLLM: List[str] = [
	# Parallelism
	"tensor-parallel-size",
	"pipeline-parallel-size",
	"data-parallel-size",
	# Memory Management
	"gpu-memory-utilization",
	"swap-space",
	"block-size",
	"cpu-offload-gb",
	"num-gpu-blocks-override",
	# Scheduling & Request Management
	"scheduling-policy",
	"max-num-seqs",
	"max-num-batched-tokens",
	"max-model-len",
	"scheduler-delay-factor",
	"preemption-mode",
	"num-scheduler-steps",
	# Chunked Prefill
	"enable-chunked-prefill",
	"max-num-partial-prefills",
	"max-long-partial-prefills",
	"long-prefill-token-threshold",
	# CUDA Graphs
	"cuda-graph-sizes",
	"enforce-eager",
	"max-seq-len-to-capture",
	# KV Cache
	"kv-cache-dtype",
	"enable-prefix-caching",
	"prefix-caching-hash-algo",
	"calculate-kv-scales",
	"disable-hybrid-kv-cache-manager",
	# Data Types & Quantization
	"dtype",
	"quantization",
	"override-attention-dtype",
	# Model Loading
	"load-format",
	"max-parallel-loading-workers",
	# LoRA
	"enable-lora",
	"enable-lora-bias",
	"max-loras",
	"max-lora-rank",
	"fully-sharded-loras",
	# Speculative Decoding
	"speculative-config",
	"num-lookahead-slots",
	# Expert Parallel (MOE)
	"enable-expert-parallel",
	"enable-eplb",
	"num-redundant-experts",
	# Performance Optimizations
	"disable-custom-all-reduce",
	"multi-step-stream-outputs",
	"async-scheduling",
	"distributed-executor-backend",
	"guided-decoding-backend",
	# Sliding Window & Cascade Attention
	"disable-sliding-window",
	"disable-cascade-attn",
	# Data Parallel
	"data-parallel-hybrid-lb",
	"data-parallel-backend",
]


def get_parameters_for_runtime(runtime: str) -> Set[str]:
	"""
	Get all valid parameters for a given runtime.

	Args:
	    runtime: Runtime name ('sglang' or 'vllm')

	Returns:
	    Set of valid parameter names for the runtime

	Raises:
	    ValueError: If runtime is not recognized
	"""
	runtime = runtime.lower()
	if runtime == "sglang":
		return SGLANG_PARAMETERS.copy()
	elif runtime == "vllm":
		return VLLM_PARAMETERS.copy()
	else:
		raise ValueError(f"Unknown runtime: {runtime}. Must be 'sglang' or 'vllm'")


def get_commonly_tuned_parameters(runtime: str) -> List[str]:
	"""
	Get commonly tuned parameters for optimization.

	Args:
	    runtime: Runtime name ('sglang' or 'vllm')

	Returns:
	    List of commonly tuned parameter names

	Raises:
	    ValueError: If runtime is not recognized
	"""
	runtime = runtime.lower()
	if runtime == "sglang":
		return COMMONLY_TUNED_SGLANG.copy()
	elif runtime == "vllm":
		return COMMONLY_TUNED_VLLM.copy()
	else:
		raise ValueError(f"Unknown runtime: {runtime}. Must be 'sglang' or 'vllm'")


def validate_parameter(runtime: str, parameter: str) -> bool:
	"""
	Check if a parameter is valid for a given runtime.

	Args:
	    runtime: Runtime name ('sglang' or 'vllm')
	    parameter: Parameter name (with or without '--' prefix)

	Returns:
	    True if parameter is valid for the runtime
	"""
	# Remove -- prefix if present
	param_name = parameter.lstrip("-")

	try:
		valid_params = get_parameters_for_runtime(runtime)
		return param_name in valid_params
	except ValueError:
		return False


def get_parameter_compatibility() -> Dict[str, List[str]]:
	"""
	Get parameter compatibility information.

	Returns:
	    Dictionary with 'common', 'sglang_only', and 'vllm_only' parameter lists
	"""
	return {
		"common": sorted(list(COMMON_PARAMETERS)),
		"sglang_only": sorted(list(SGLANG_ONLY_PARAMETERS)),
		"vllm_only": sorted(list(VLLM_ONLY_PARAMETERS)),
	}


# For backward compatibility with existing code that uses underscores
# Convert parameter names to use underscores instead of hyphens
def normalize_parameter_name(param: str, to_cli_format: bool = True) -> str:
	"""
	Normalize parameter name between CLI format (hyphens) and Python format (underscores).

	Args:
	    param: Parameter name
	    to_cli_format: If True, convert underscores to hyphens. If False, convert hyphens to underscores.

	Returns:
	    Normalized parameter name
	"""
	param = param.lstrip("-")
	if to_cli_format:
		return param.replace("_", "-")
	else:
		return param.replace("-", "_")
