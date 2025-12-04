import axios, { AxiosInstance, AxiosError } from "axios";
import toast from "react-hot-toast";
import type {
	Task,
	TaskCreate,
	Experiment,
	HealthResponse,
	SystemInfoResponse,
	ContainerInfo,
	ContainerStats,
	ContainerLogs,
	DockerInfo
} from "@/types/api";

class ApiClient {
	private client: AxiosInstance;

	constructor() {
		this.client = axios.create({
			baseURL: import.meta.env.VITE_API_URL || "/api",
			headers: {
				"Content-Type": "application/json"
			}
		});

		// Response interceptor to handle errors
		this.client.interceptors.response.use(
			(response) => response,
			(error: AxiosError) => {
				// Handle HTTP errors (4xx, 5xx)
				if (error.response) {
					const status = error.response.status;
					const data = error.response.data as any;

					// Extract error message
					const message =
						data?.detail || data?.message || error.message || "An error occurred";

					// Show error toast based on status code
					if (status >= 400 && status < 500) {
						// Client errors (4xx)
						toast.error(`${status}: ${message}`, {
							duration: 4000,
							position: "bottom-right"
						});
					} else if (status >= 500) {
						// Server errors (5xx)
						toast.error(`Server Error: ${message}`, {
							duration: 5000,
							position: "bottom-right"
						});
					}
				} else if (error.request) {
					// Network error (no response received)
					toast.error("Network error: Unable to reach the server", {
						duration: 4000,
						position: "bottom-right"
					});
				}

				return Promise.reject(error);
			}
		);
	}

	// Health & System
	async getHealth(): Promise<HealthResponse> {
		const { data } = await this.client.get("/health");
		return data;
	}

	async getSystemInfo(): Promise<SystemInfoResponse> {
		const { data } = await this.client.get("/system/info");
		return data;
	}

	// Tasks
	async getTasks(): Promise<Task[]> {
		const { data } = await this.client.get("/tasks/");
		return data;
	}

	async getTask(id: number): Promise<Task> {
		const { data } = await this.client.get(`/tasks/${id}`);
		return data;
	}

	async getTaskByName(name: string): Promise<Task> {
		const { data } = await this.client.get(`/tasks/name/${name}`);
		return data;
	}

	async createTask(task: TaskCreate): Promise<Task> {
		const { data } = await this.client.post("/tasks/", task);
		return data;
	}

	async updateTask(id: number, task: TaskCreate): Promise<Task> {
		const { data} = await this.client.put(`/tasks/${id}`, task);
		return data;
	}

	async startTask(id: number): Promise<{ status: string; message: string; job_id: string }> {
		const { data } = await this.client.post(`/tasks/${id}/start`);
		return data;
	}

	async cancelTask(id: number): Promise<{ status: string; message: string }> {
		const { data } = await this.client.post(`/tasks/${id}/cancel`);
		return data;
	}

	async restartTask(id: number): Promise<Task> {
		const { data} = await this.client.post(`/tasks/${id}/restart`);
		return data;
	}

	async clearTask(id: number): Promise<Task> {
		const { data } = await this.client.post(`/tasks/${id}/clear`);
		return data;
	}

	async patchTask(id: number, updates: { description?: string }): Promise<Task> {
		const { data } = await this.client.patch(`/tasks/${id}`, updates);
		return data;
	}

	async deleteTask(id: number): Promise<void> {
		await this.client.delete(`/tasks/${id}`);
	}

	// Experiments
	async getExperiments(): Promise<Experiment[]> {
		const { data } = await this.client.get("/experiments/");
		return data;
	}

	async getExperiment(id: number): Promise<Experiment> {
		const { data } = await this.client.get(`/experiments/${id}`);
		return data;
	}

	async getExperimentsByTask(taskId: number): Promise<Experiment[]> {
		const { data } = await this.client.get(`/experiments/task/${taskId}`);
		return data;
	}

	// Task Logs
	async getTaskLogs(taskId: number): Promise<{ logs: string }> {
		const { data } = await this.client.get(`/tasks/${taskId}/logs`);
		return data;
	}

	async clearTaskLogs(taskId: number): Promise<void> {
		await this.client.delete(`/tasks/${taskId}/logs`);
	}

	// Docker
	async getContainers(all: boolean = true): Promise<ContainerInfo[]> {
		const { data } = await this.client.get("/docker/containers", {
			params: { all }
		});
		return data;
	}

	async getContainer(containerId: string): Promise<ContainerInfo> {
		const { data } = await this.client.get(`/docker/containers/${containerId}`);
		return data;
	}

	async getContainerLogs(
		containerId: string,
		tail: number = 1000,
		timestamps: boolean = false
	): Promise<ContainerLogs> {
		const { data } = await this.client.get(`/docker/containers/${containerId}/logs`, {
			params: { tail, timestamps }
		});
		return data;
	}

	async getContainerStats(containerId: string): Promise<ContainerStats> {
		const { data } = await this.client.get(`/docker/containers/${containerId}/stats`);
		return data;
	}

	async startContainer(containerId: string): Promise<{ message: string }> {
		const { data } = await this.client.post(`/docker/containers/${containerId}/start`);
		return data;
	}

	async stopContainer(containerId: string, timeout: number = 10): Promise<{ message: string }> {
		const { data } = await this.client.post(`/docker/containers/${containerId}/stop`, null, {
			params: { timeout }
		});
		return data;
	}

	async restartContainer(containerId: string, timeout: number = 10): Promise<{ message: string }> {
		const { data } = await this.client.post(`/docker/containers/${containerId}/restart`, null, {
			params: { timeout }
		});
		return data;
	}

	async removeContainer(containerId: string, force: boolean = false): Promise<{ message: string }> {
		const { data } = await this.client.delete(`/docker/containers/${containerId}`, {
			params: { force }
		});
		return data;
	}

	async getDockerInfo(): Promise<DockerInfo> {
		const { data } = await this.client.get("/docker/info");
		return data;
	}

	// Generic methods for other API calls
	async get<T = any>(url: string, config?: any): Promise<T> {
		const { data } = await this.client.get(url, config);
		return data;
	}

	async post<T = any>(url: string, body?: any, config?: any): Promise<T> {
		const { data } = await this.client.post(url, body, config);
		return data;
	}

	async put<T = any>(url: string, body?: any, config?: any): Promise<T> {
		const { data } = await this.client.put(url, body, config);
		return data;
	}

	async delete<T = any>(url: string, config?: any): Promise<T> {
		const { data } = await this.client.delete(url, config);
		return data;
	}

	async patch<T = any>(url: string, body?: any, config?: any): Promise<T> {
		const { data } = await this.client.patch(url, body, config);
		return data;
	}
}

export const apiClient = new ApiClient();
export default apiClient;
