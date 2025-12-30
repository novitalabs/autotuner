// Dashboard API client

import axios from 'axios';
import type {
	GPUStatus,
	WorkerStatus,
	DBStatistics,
	ExperimentTimelineItem,
} from '../types/dashboard';

// Use environment variable or default to relative path (proxy)
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export interface ClusterGPUInfo {
	index: number;
	node_name: string;
	name: string;
	capacity: number;
	allocatable: number;
	has_metrics?: boolean;
	memory_total_mb?: number;
	memory_used_mb?: number;
	memory_free_mb?: number;
	memory_usage_percent?: number;
	utilization_percent?: number;
	temperature_c?: number;
	is_local?: boolean;
}

export interface ClusterNodeSummary {
	node_name: string;
	gpu_count: number;
	allocatable_gpus: number;
	gpu_model: string;
	gpus: ClusterGPUInfo[];
	avg_utilization: number;
	avg_memory_usage: number;
	total_memory_mb: number;
	used_memory_mb: number;
	is_local?: boolean;
}

export interface ClusterGPUStatus {
	available: boolean;
	mode: 'cluster';
	nodes?: ClusterGPUInfo[];
	node_summaries?: ClusterNodeSummary[];
	total_gpus?: number;
	total_allocatable_gpus?: number;
	local_hostname?: string;
	error?: string;
	timestamp: string;
}

export interface GPUHistoryEntry {
	timestamp: string;
	gpus: Array<{
		index: number;
		utilization: number | null;
		memory_used: number | null;
		temperature: number | null;
	}>;
}

export interface WorkerGPUHistory {
	worker_id: string;
	history: GPUHistoryEntry[];
}

export interface WorkerGPUInfo {
	index: number;
	name: string;
	memory_total_gb: number;
	memory_free_gb: number | null;
	memory_used_gb: number | null;
	utilization_percent: number | null;
	temperature_c: number | null;
	node_name: string | null;
}

export interface DistributedWorker {
	worker_id: string;
	hostname: string;
	alias: string | null;
	ip_address: string | null;
	gpu_count: number;
	gpu_model: string | null;
	gpu_memory_gb: number | null;
	gpus: WorkerGPUInfo[] | null;
	deployment_mode: string;
	max_parallel: number;
	current_jobs: number;
	status: 'online' | 'busy' | 'offline';
	registered_at: string;
	last_heartbeat: string;
	seconds_since_heartbeat: number;
}

export interface DistributedWorkersStatus {
	workers: DistributedWorker[];
	total_count: number;
	online_count: number;
	busy_count: number;
	offline_count: number;
}

// Worker Slot Types (persistent deployment configurations)
export interface WorkerSlot {
	id: number;
	worker_id: string | null;
	name: string;
	controller_type: string;
	ssh_command: string;
	ssh_forward_tunnel: string | null;
	ssh_reverse_tunnel: string | null;
	project_path: string;
	manager_ssh: string | null;
	current_status: 'online' | 'offline' | 'unknown';
	last_seen_at: string | null;
	last_error: string | null;
	hostname: string | null;
	gpu_count: number | null;
	gpu_model: string | null;
	created_at: string | null;
	updated_at: string | null;
}

export interface WorkerSlotsStatus {
	slots: WorkerSlot[];
	total_count: number;
	online_count: number;
	offline_count: number;
	unknown_count: number;
}

export interface RestoreWorkerResponse {
	success: boolean;
	message: string;
	worker_id: string | null;
	slot_id: number;
	error: string | null;
	logs: string | null;
	worker_info: Record<string, unknown> | null;
}

export const dashboardApi = {
	async getGPUStatus(): Promise<GPUStatus> {
		const response = await axios.get<GPUStatus>(`${API_BASE_URL}/dashboard/gpu-status`);
		return response.data;
	},

	async getClusterGPUStatus(): Promise<ClusterGPUStatus> {
		const response = await axios.get<ClusterGPUStatus>(`${API_BASE_URL}/dashboard/cluster-gpu-status`);
		return response.data;
	},

	async getWorkerStatus(): Promise<WorkerStatus> {
		const response = await axios.get<WorkerStatus>(`${API_BASE_URL}/dashboard/worker-status`);
		return response.data;
	},

	async getDistributedWorkers(): Promise<DistributedWorkersStatus> {
		const response = await axios.get<DistributedWorkersStatus>(`${API_BASE_URL}/workers`);
		return response.data;
	},

	async renameWorker(workerId: string, alias: string | null): Promise<DistributedWorker> {
		const response = await axios.patch<DistributedWorker>(
			`${API_BASE_URL}/workers/${encodeURIComponent(workerId)}/alias`,
			{ alias }
		);
		return response.data;
	},

	async getWorkerGPUHistory(workerId: string): Promise<WorkerGPUHistory> {
		const response = await axios.get<WorkerGPUHistory>(
			`${API_BASE_URL}/workers/${encodeURIComponent(workerId)}/gpu-history`
		);
		return response.data;
	},

	async getDBStatistics(): Promise<DBStatistics> {
		const response = await axios.get<DBStatistics>(`${API_BASE_URL}/dashboard/db-statistics`);
		return response.data;
	},

	async getExperimentTimeline(hours: number = 24): Promise<ExperimentTimelineItem[]> {
		const response = await axios.get<ExperimentTimelineItem[]>(
			`${API_BASE_URL}/dashboard/experiment-timeline`,
			{ params: { hours } }
		);
		return response.data;
	},

	// Worker Slot APIs (persistent deployment configurations)
	async getWorkerSlots(): Promise<WorkerSlotsStatus> {
		const response = await axios.get<WorkerSlotsStatus>(`${API_BASE_URL}/workers/slots`);
		return response.data;
	},

	async restoreWorker(slotId: number, autoInstall: boolean = false): Promise<RestoreWorkerResponse> {
		const response = await axios.post<RestoreWorkerResponse>(
			`${API_BASE_URL}/workers/slots/${slotId}/restore`,
			{ auto_install: autoInstall }
		);
		return response.data;
	},

	async deleteWorkerSlot(slotId: number): Promise<void> {
		await axios.delete(`${API_BASE_URL}/workers/slots/${slotId}`);
	},
};
