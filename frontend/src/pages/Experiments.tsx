import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiClient } from "@/services/api";
import type { Experiment, Task } from "@/types/api";
import ExperimentLogViewer from "@/components/ExperimentLogViewer";
import GPUMetricsChart from "@/components/GPUMetricsChart";
import { useEscapeKey } from "@/hooks/useEscapeKey";
import { useTaskWebSocket } from "@/hooks/useTaskWebSocket";

export default function Experiments() {
	const [selectedTaskId, setSelectedTaskId] = useState<number | "all">("all");
	const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
	const [logViewerExperiment, setLogViewerExperiment] = useState<Experiment | null>(null);

	// Handle Escape key to close modals
	useEscapeKey(() => {
		if (selectedExperiment) {
			setSelectedExperiment(null);
		} else if (logViewerExperiment) {
			setLogViewerExperiment(null);
		}
	}, !!(selectedExperiment || logViewerExperiment));

	// Fetch all tasks for the filter dropdown
	const { data: tasks = [] } = useQuery({
		queryKey: ["tasks"],
		queryFn: () => apiClient.getTasks()
	});

	// Find running task for WebSocket connection
	const runningTask = tasks.find((task: Task) => task.status === "running");

	// Connect to WebSocket for the running task (if any)
	// This will automatically update experiments when events are received
	useTaskWebSocket(
		runningTask?.id || null,
		!!runningTask
	);

	// Fetch experiments based on filter
	const {
		data: experiments = [],
		isLoading,
		error
	} = useQuery({
		queryKey: ["experiments", selectedTaskId],
		queryFn: () => {
			if (selectedTaskId === "all") {
				return apiClient.getExperiments();
			}
			return apiClient.getExperimentsByTask(selectedTaskId);
		}
	});

	const getStatusColor = (status: string) => {
		switch (status) {
			case "success":
				return "bg-green-100 text-green-800";
			case "failed":
				return "bg-red-100 text-red-800";
			case "pending":
				return "bg-gray-100 text-gray-800";
			case "deploying":
			case "benchmarking":
				return "bg-blue-100 text-blue-800";
			default:
				return "bg-gray-100 text-gray-800";
		}
	};

	const formatDuration = (seconds: number | null) => {
		if (!seconds) return "N/A";
		const mins = Math.floor(seconds / 60);
		const secs = Math.floor(seconds % 60);
		return `${mins}m ${secs}s`;
	};

	const formatScore = (score: number | null) => {
		if (score === null) return "N/A";
		return score.toFixed(4);
	};

	return (
		<div className="px-4 py-6 sm:px-0">
			<div className="sm:flex sm:items-center">
				<div className="sm:flex-auto">
					<h1 className="text-2xl font-bold text-gray-900">Experiments</h1>
					<p className="mt-2 text-sm text-gray-700">
						View and compare experiment results from all autotuning tasks.
					</p>
				</div>
			</div>

			{/* Filter Section */}
			<div className="mt-6 flex items-center gap-4">
				<label htmlFor="task-filter" className="text-sm font-medium text-gray-700">
					Filter by Task:
				</label>
				<select
					id="task-filter"
					className="rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-emerald-500 sm:text-sm"
					value={selectedTaskId}
					onChange={(e) =>
						setSelectedTaskId(e.target.value === "all" ? "all" : Number(e.target.value))
					}
				>
					<option value="all">All Tasks</option>
					{tasks.map((task: Task) => (
						<option key={task.id} value={task.id}>
							{task.task_name}
						</option>
					))}
				</select>
				<span className="text-sm text-gray-500">
					{experiments.length} experiment{experiments.length !== 1 ? "s" : ""}
				</span>
			</div>

			{/* Experiments Table */}
			<div className="mt-6 flow-root">
				{isLoading ? (
					<div className="text-center py-12">
						<div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent motion-reduce:animate-[spin_1.5s_linear_infinite]"></div>
						<p className="mt-2 text-sm text-gray-600">Loading experiments...</p>
					</div>
				) : error ? (
					<div className="rounded-md bg-red-50 p-4">
						<p className="text-sm text-red-800">
							Error loading experiments: {(error as Error).message}
						</p>
					</div>
				) : experiments.length === 0 ? (
					<div className="text-center py-12 border-2 border-dashed border-gray-300 rounded-lg">
						<svg
							className="mx-auto h-12 w-12 text-gray-400"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth={2}
								d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
							/>
						</svg>
						<h3 className="mt-2 text-sm font-medium text-gray-900">No experiments</h3>
						<p className="mt-1 text-sm text-gray-500">
							{selectedTaskId === "all"
								? "No experiments have been run yet."
								: "This task has no experiments yet."}
						</p>
					</div>
				) : (
					<div className="overflow-x-auto">
						<table className="min-w-full divide-y divide-gray-300">
							<thead>
								<tr>
									<th className="py-3.5 pl-4 pr-3 text-left text-sm font-semibold text-gray-900">
										ID
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Task
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Status
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										GPUs
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Parameters
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Objective Score
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Duration
									</th>
									<th className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900">
										Created
									</th>
									<th className="relative py-3.5 pl-3 pr-4">
										<span className="sr-only">Actions</span>
									</th>
								</tr>
							</thead>
							<tbody className="divide-y divide-gray-200 bg-white">
								{experiments.map((experiment: Experiment) => {
									const task = tasks.find(
										(t: Task) => t.id === experiment.task_id
									);
									return (
										<tr key={experiment.id} className="hover:bg-gray-50">
											<td className="whitespace-nowrap py-4 pl-4 pr-3 text-sm font-medium text-gray-900">
												{experiment.experiment_id}
											</td>
											<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
												{task?.task_name || `Task ${experiment.task_id}`}
											</td>
											<td className="whitespace-nowrap px-3 py-4 text-sm">
												<div className="flex items-center gap-2">
													<span
														className={`inline-flex rounded-full px-2 text-xs font-semibold leading-5 ${getStatusColor(experiment.status)}`}
													>
														{experiment.status}
													</span>
													{experiment.slo_violation && (
														<span
															className="inline-flex items-center rounded-full bg-red-100 px-2 py-0.5 text-xs font-semibold text-red-800"
															title="Hard SLO violation detected"
														>
															<svg className="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
																<path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
															</svg>
															SLO
														</span>
													)}
												</div>
											</td>
											<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
												{experiment.gpu_info ? (
													<div className="flex items-center gap-1">
														<svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
															<path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z" />
														</svg>
														<span className="font-medium">{experiment.gpu_info.count}</span>
														<span className="text-gray-500 text-xs">
															{experiment.gpu_info.model ? `(${experiment.gpu_info.model.split(' ')[0]})` : ''}
														</span>
														{experiment.metrics?.gpu_monitoring && (
															<span
																className="ml-1 inline-flex items-center text-xs text-blue-600"
																title="GPU monitoring data available"
															>
																📊
															</span>
														)}
													</div>
												) : (
													<span className="text-gray-400">N/A</span>
												)}
											</td>
											<td className="px-3 py-4 text-sm text-gray-700">
												<div className="max-w-xs truncate">
													{Object.entries(
														experiment.parameters || {}
													).map(([key, value]) => (
														<span key={key} className="mr-2">
															{key}={String(value)}
														</span>
													))}
												</div>
											</td>
											<td className="whitespace-nowrap px-3 py-4 text-sm font-mono text-gray-900">
												{formatScore(experiment.objective_score)}
											</td>
											<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
												{formatDuration(experiment.elapsed_time)}
											</td>
											<td className="whitespace-nowrap px-3 py-4 text-sm text-gray-700">
												{new Date(
													experiment.created_at
												).toLocaleDateString()}
											</td>
											<td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm">
												<div className="flex items-center justify-end gap-2">
													<button
														onClick={() => setLogViewerExperiment(experiment)}
														className="p-1.5 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
														title="View Logs"
													>
														<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
															<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
														</svg>
													</button>
													<button
														onClick={() =>
															setSelectedExperiment(experiment)
														}
														className="text-blue-600 hover:text-blue-900"
													>
														View Details
													</button>
												</div>
											</td>
										</tr>
									);
								})}
							</tbody>
						</table>
					</div>
				)}
			</div>

			{/* Details Modal */}
			{selectedExperiment && (
				<div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
					<div className="bg-white rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
						<div className="px-6 py-4 border-b border-gray-200">
							<div className="flex items-center justify-between">
								<h2 className="text-xl font-bold text-gray-900">
									Experiment {selectedExperiment.experiment_id} Details
								</h2>
								<button
									onClick={() => setSelectedExperiment(null)}
									className="text-gray-400 hover:text-gray-500"
								>
									<svg
										className="h-6 w-6"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M6 18L18 6M6 6l12 12"
										/>
									</svg>
								</button>
							</div>
						</div>

						<div className="px-6 py-4 space-y-6">
							{/* Status and Basic Info */}
							<div>
								<h3 className="text-sm font-medium text-gray-900 mb-3">Status</h3>
								<div className="grid grid-cols-2 gap-4">
									<div>
										<span className="text-sm text-gray-500">Status:</span>
										<span
											className={`ml-2 inline-flex rounded-full px-2 text-xs font-semibold ${getStatusColor(selectedExperiment.status)}`}
										>
											{selectedExperiment.status}
										</span>
									</div>
									<div>
										<span className="text-sm text-gray-500">Duration:</span>
										<span className="ml-2 text-sm text-gray-900">
											{formatDuration(selectedExperiment.elapsed_time)}
										</span>
									</div>
									<div>
										<span className="text-sm text-gray-500">
											Objective Score:
										</span>
										<span className="ml-2 text-sm font-mono text-gray-900">
											{formatScore(selectedExperiment.objective_score)}
										</span>
									</div>
									<div>
										<span className="text-sm text-gray-500">Service URL:</span>
										<span className="ml-2 text-sm text-gray-900">
											{selectedExperiment.service_url || "N/A"}
										</span>
									</div>
								</div>
							</div>

							{/* Parameters */}
							<div>
								<h3 className="text-sm font-medium text-gray-900 mb-3">
									Parameters
								</h3>
								<div className="bg-gray-50 rounded-lg p-4">
									<pre className="text-sm text-gray-900 overflow-x-auto">
										{JSON.stringify(selectedExperiment.parameters, null, 2)}
									</pre>
								</div>
							</div>

							{/* Metrics */}
							{selectedExperiment.metrics && (
								<div>
									<h3 className="text-sm font-medium text-gray-900 mb-3">
										Metrics
									</h3>
									<div className="bg-gray-50 rounded-lg p-4">
										<pre className="text-sm text-gray-900 overflow-x-auto">
											{JSON.stringify(selectedExperiment.metrics, null, 2)}
										</pre>
									</div>
								</div>
							)}

							{/* GPU Metrics */}
							{selectedExperiment.metrics?.gpu_monitoring && (
								<div>
									<h3 className="text-sm font-medium text-gray-900 mb-3">
										GPU Monitoring
									</h3>
									<GPUMetricsChart gpuMonitoring={selectedExperiment.metrics.gpu_monitoring} />
								</div>
							)}

							{/* Error Message */}
							{selectedExperiment.error_message && (
								<div>
									<h3 className="text-sm font-medium text-gray-900 mb-3">
										Error
									</h3>
									<div className="bg-red-50 rounded-lg p-4">
										<p className="text-sm text-red-900">
											{selectedExperiment.error_message}
										</p>
									</div>
								</div>
							)}

							{/* Timestamps */}
							<div>
								<h3 className="text-sm font-medium text-gray-900 mb-3">Timeline</h3>
								<div className="space-y-2 text-sm">
									<div>
										<span className="text-gray-500">Created:</span>
										<span className="ml-2 text-gray-900">
											{new Date(
												selectedExperiment.created_at
											).toLocaleString()}
										</span>
									</div>
									{selectedExperiment.started_at && (
										<div>
											<span className="text-gray-500">Started:</span>
											<span className="ml-2 text-gray-900">
												{new Date(
													selectedExperiment.started_at
												).toLocaleString()}
											</span>
										</div>
									)}
									{selectedExperiment.completed_at && (
										<div>
											<span className="text-gray-500">Completed:</span>
											<span className="ml-2 text-gray-900">
												{new Date(
													selectedExperiment.completed_at
												).toLocaleString()}
											</span>
										</div>
									)}
								</div>
							</div>
						</div>

						<div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
							<button
								onClick={() => setSelectedExperiment(null)}
								className="w-full sm:w-auto px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
							>
								Close
							</button>
						</div>
					</div>
				</div>
			)}

			{/* Log Viewer Modal */}
			{logViewerExperiment && (
				<ExperimentLogViewer
					taskId={logViewerExperiment.task_id}
					experimentId={logViewerExperiment.experiment_id}
					onClose={() => setLogViewerExperiment(null)}
				/>
			)}
		</div>
	);
}
