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

export interface ClusterGPUStatus {
	available: boolean;
	mode: 'cluster';
	nodes?: Array<{
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
	}>;
	total_gpus?: number;
	total_allocatable_gpus?: number;
	error?: string;
	timestamp: string;
}

export interface DistributedWorker {
	worker_id: string;
	hostname: string;
	ip_address: string | null;
	gpu_count: number;
	gpu_model: string | null;
	gpu_memory_gb: number | null;
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

export const dashboardApi = {
	async getGPUStatus(): Promise<GPUStatus> {
		const response = await axios.get<GPUStatus>(`${API_BASE_URL}/dashboard/gpu-status`);
		return response.data;
	},

	async getClusterGPUStatus(): Promise<ClusterGPUStatus> {
		const response = await axios.get<ClusterGPUStatus>(`${API_BASE_URL}/api/dashboard/cluster-gpu-status`);
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
};
