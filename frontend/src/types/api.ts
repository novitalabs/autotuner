// API Response Types based on backend schemas

// Quantization Configuration Types
export interface QuantizationConfig {
	presets?: string[];
	gemm_dtype?: string | string[];
	kvcache_dtype?: string | string[];
	attention_dtype?: string | string[];
	moe_dtype?: string | string[];
}

// Parallel Configuration Types
export interface ParallelConfig {
	presets?: string[];
	tp?: number | number[];
	pp?: number | number[];
	dp?: number | number[];
	cp?: number | number[];
	dcp?: number | number[];
	enable_expert_parallel?: boolean | boolean[];
	moe_tp?: number | number[];
	moe_ep?: number | number[];
	moe_cluster?: number | number[];
	moe_dense_tp?: number | number[];
}

// SLO Configuration Types
export interface SLOMetricConfig {
	threshold: number;
	weight?: number;
	hard_fail?: boolean;
	fail_ratio?: number;
}

export interface SLOLatencyConfig {
	p50?: SLOMetricConfig;
	p90?: SLOMetricConfig;
	p99?: SLOMetricConfig;
}

export interface SLOConfig {
	latency?: SLOLatencyConfig;
	ttft?: SLOMetricConfig;
	tpot?: SLOMetricConfig;
	steepness?: number;
}

export interface Task {
	id: number;
	task_name: string;
	description: string | null;
	status: TaskStatus;
	model: Record<string, any>;  // API returns as "model", not "model_config"
	base_runtime: string;
	runtime_image_tag: string | null;
	parameters: Record<string, any>;
	optimization: Record<string, any>;  // API returns as "optimization", not "optimization_config"
	benchmark: Record<string, any>;  // API returns as "benchmark", not "benchmark_config"
	slo?: SLOConfig;  // Optional SLO configuration
	quant_config?: QuantizationConfig;  // Optional quantization configuration
	parallel_config?: ParallelConfig;  // Optional parallel execution configuration
	deployment_mode: string;
	gpu_type?: string;  // Optional GPU type filter for worker selection
	total_experiments: number;
	successful_experiments: number;
	best_experiment_id: number | null;
	created_at: string;
	started_at: string | null;
	completed_at: string | null;
	elapsed_time: number | null;
}

export enum TaskStatus {
	PENDING = "pending",
	RUNNING = "running",
	COMPLETED = "completed",
	FAILED = "failed",
	CANCELLED = "cancelled"
}

export interface Experiment {
	id: number;
	task_id: number;
	experiment_id: number;
	parameters: Record<string, any>;
	status: ExperimentStatus;
	error_message: string | null;
	metrics: Record<string, any> | null;
	objective_score: number | null;
	slo_violation?: boolean;  // Flag for hard SLO violations
	gpu_info?: {
		model: string;
		count: number;
		device_ids?: string[];
		world_size?: number;
		gpu_info?: {
			count: number;
			indices: number[];
			allocation_method: string;
			details?: Array<{
				index: number;
				name: string;
				memory_free_mb: number;
				utilization_percent: number;
				availability_score: number;
			}>;
		};
	};
	service_name: string | null;
	service_url: string | null;
	created_at: string;
	started_at: string | null;
	completed_at: string | null;
	elapsed_time: number | null;
}

export enum ExperimentStatus {
	PENDING = "pending",
	DEPLOYING = "deploying",
	BENCHMARKING = "benchmarking",
	SUCCESS = "success",
	FAILED = "failed"
}

export interface HealthResponse {
	status: string;
	database?: string;
	redis?: string;
}

export interface SystemInfoResponse {
	app_name: string;
	version: string;
	deployment_mode: string;
	available_runtimes: string[];
	timezone?: string;
}

// Request Types
export interface TaskCreate {
	task_name: string;
	description?: string;
	model: Record<string, any>;
	base_runtime: string;
	runtime_image_tag?: string;
	parameters: Record<string, any>;
	optimization: Record<string, any>;
	benchmark: Record<string, any>;
	slo?: SLOConfig;  // Optional SLO configuration
	quant_config?: QuantizationConfig;  // Optional quantization configuration
	deployment_mode?: string;
	gpu_type?: string;  // Optional GPU type filter for worker selection
}

// Docker Types
export interface ContainerInfo {
	id: string;
	short_id: string;
	name: string;
	image: string;
	status: string;
	state: string;
	created: string;
	started_at: string | null;
	finished_at: string | null;
	ports: Record<string, string>;
	labels: Record<string, string>;
	command: string | null;
}

export interface ContainerStats {
	cpu_percent: number;
	memory_usage: string;
	memory_limit: string;
	memory_percent: number;
	network_rx: string;
	network_tx: string;
	block_read: string;
	block_write: string;
}

export interface ContainerLogs {
	logs: string;
	lines: number;
}

export interface DockerInfo {
	version: string;
	api_version: string;
	containers: number;
	containers_running: number;
	containers_paused: number;
	containers_stopped: number;
	images: number;
	driver: string;
	memory_total: string;
	cpus: number;
	operating_system: string;
	architecture: string;
}
