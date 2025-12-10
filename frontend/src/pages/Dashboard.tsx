import { useQuery } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { dashboardApi } from '../services/dashboardApi';
import { useTimezone } from '../contexts/TimezoneContext';
import ExperimentLogViewer from '../components/ExperimentLogViewer';
import {
	CpuChipIcon,
	ServerIcon,
	CircleStackIcon,
	ClockIcon,
} from '@heroicons/react/24/outline';

function formatBytes(mb: number): string {
	if (mb >= 1024) {
		return `${(mb / 1024).toFixed(1)} GB`;
	}
	return `${mb.toFixed(0)} MB`;
}

function formatUptime(seconds: number | null): string {
	if (!seconds) return 'N/A';
	const hours = Math.floor(seconds / 3600);
	const minutes = Math.floor((seconds % 3600) / 60);
	return `${hours}h ${minutes}m`;
}

// Store GPU utilization history (last 12 data points = 1 minute at 5s intervals)
const MAX_HISTORY_LENGTH = 12;
type GPUHistory = Record<number, number[]>;

export default function Dashboard() {
	// Track GPU utilization history
	const [gpuHistory, setGpuHistory] = useState<GPUHistory>({});
	// Track which GPU view mode (local or cluster) - persist in localStorage
	const [gpuViewMode, setGpuViewMode] = useState<'local' | 'cluster'>(() => {
		const saved = localStorage.getItem('gpuViewMode');
		return (saved === 'local' || saved === 'cluster') ? saved : 'cluster';
	});

	// Save GPU view mode to localStorage whenever it changes
	useEffect(() => {
		localStorage.setItem('gpuViewMode', gpuViewMode);
	}, [gpuViewMode]);

	// Track selected experiment for log viewer
	const [selectedExperiment, setSelectedExperiment] = useState<{ taskId: number; experimentId: number } | null>(null);

	// Get timezone formatting functions
	const { formatTime, timezoneOffsetMs } = useTimezone();

	// Fetch dashboard data with auto-refresh
	const { data: gpuStatus, isLoading: gpuLoading } = useQuery({
		queryKey: ['gpuStatus'],
		queryFn: dashboardApi.getGPUStatus,
		refetchInterval: 5000, // Refresh every 5 seconds
		enabled: gpuViewMode === 'local',
	});

	// Fetch cluster GPU status
	const { data: clusterGpuStatus, isLoading: clusterGpuLoading } = useQuery({
		queryKey: ['clusterGpuStatus'],
		queryFn: dashboardApi.getClusterGPUStatus,
		refetchInterval: 10000, // Refresh every 10 seconds
		enabled: gpuViewMode === 'cluster',
	});

	// Update GPU history when new data arrives
	useEffect(() => {
		if (gpuStatus?.gpus) {
			setGpuHistory((prev) => {
				const updated = { ...prev };
				gpuStatus.gpus!.forEach((gpu) => {
					const history = prev[gpu.index] || [];
					const newHistory = [...history, gpu.utilization_percent].slice(-MAX_HISTORY_LENGTH);
					updated[gpu.index] = newHistory;
				});
				return updated;
			});
		}
	}, [gpuStatus]);

	const { data: workerStatus, isLoading: workerLoading } = useQuery({
		queryKey: ['workerStatus'],
		queryFn: dashboardApi.getWorkerStatus,
		refetchInterval: 5000,
	});

	const { data: dbStats, isLoading: dbStatsLoading } = useQuery({
		queryKey: ['dbStatistics'],
		queryFn: dashboardApi.getDBStatistics,
		refetchInterval: 10000, // Refresh every 10 seconds
	});

	const { data: timeline, isLoading: timelineLoading } = useQuery({
		queryKey: ['experimentTimeline', 24],
		queryFn: () => dashboardApi.getExperimentTimeline(24),
		refetchInterval: 30000, // Refresh every 30 seconds
	});

	// GPU Status Card
	const renderGPUCard = () => (
		<div className="bg-white shadow rounded-lg p-4">
			<div className="flex items-center justify-between mb-3">
				<h3 className="text-base font-medium text-gray-900 flex items-center">
					<CpuChipIcon className="h-5 w-5 mr-2 text-blue-500" />
					GPU Status {gpuViewMode === 'cluster' && '(Cluster-wide)'}
				</h3>
				<div className="flex items-center gap-2">
					{/* Mode toggle buttons */}
					<button
						onClick={() => setGpuViewMode('local')}
						className={`px-2 py-0.5 text-xs rounded ${gpuViewMode === 'local' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-600'}`}
					>
						Local
					</button>
					<button
						onClick={() => setGpuViewMode('cluster')}
						className={`px-2 py-0.5 text-xs rounded ${gpuViewMode === 'cluster' ? 'bg-blue-100 text-blue-800' : 'bg-gray-100 text-gray-600'}`}
					>
						Cluster
					</button>
					{gpuViewMode === 'local' && gpuStatus && (
						<span className={`px-2 py-0.5 text-xs rounded-full ${gpuStatus.available ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
							{gpuStatus.available ? 'Available' : 'Unavailable'}
						</span>
					)}
					{gpuViewMode === 'cluster' && clusterGpuStatus && (
						<span className={`px-2 py-0.5 text-xs rounded-full ${clusterGpuStatus.available ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
							{clusterGpuStatus.available ? `${clusterGpuStatus.total_allocatable_gpus}/${clusterGpuStatus.total_gpus} Available` : 'Unavailable'}
						</span>
					)}
				</div>
			</div>

			{/* Local GPU View */}
			{gpuViewMode === 'local' && (
				<>
					{gpuLoading && <p className="text-gray-500 text-sm">Loading...</p>}
					{gpuStatus?.error && <p className="text-red-500 text-sm">Error: {gpuStatus.error}</p>}

					{gpuStatus?.gpus && (
						<div className="space-y-2">
							{gpuStatus.gpus.map((gpu) => {
								const history = gpuHistory[gpu.index] || [];
								return (
									<div key={gpu.index} className="border rounded p-2">
										<div className="flex justify-between items-center">
											<div className="flex items-center gap-2">
												<strong className="text-sm font-medium">GPU {gpu.index}: {gpu.name}</strong>
												<span className="text-xs text-gray-500">{gpu.temperature_c}°C</span>
											</div>
											<div className="flex items-center gap-2">
												<span className={`px-1.5 py-0.5 text-xs rounded ${gpu.utilization_percent > 80 ? 'bg-red-100 text-red-800' : gpu.utilization_percent > 50 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}`}>
													{gpu.utilization_percent}%
												</span>
												{/* Tiny utilization history sparkline */}
												{history.length > 1 && (
													<div className="flex items-end gap-px" style={{ height: '16px' }}>
														{history.map((util, idx) => (
															<div
																key={idx}
																className={`${
																	util > 80
																		? 'bg-red-400'
																		: util > 50
																		? 'bg-yellow-400'
																		: 'bg-green-400'
																}`}
																style={{
																	width: '2px',
																	height: `${Math.max(util * 0.16, 2)}px`,
																	minHeight: '2px'
																}}
																title={`${util}%`}
															></div>
														))}
													</div>
												)}
											</div>
										</div>

										{/* Memory Usage Bar */}
										<div className="mt-1">
										<div className="flex justify-between text-xs text-gray-600 mb-1">
											<span>Memory</span>
											<span>{formatBytes(gpu.memory_used_mb)} / {formatBytes(gpu.memory_total_mb)}</span>
										</div>
										<div className="w-full bg-gray-200 rounded-full h-2">
											<div
												className={`h-2 rounded-full ${gpu.memory_usage_percent > 90 ? 'bg-red-600' : gpu.memory_usage_percent > 70 ? 'bg-yellow-500' : 'bg-blue-600'}`}
												style={{ width: `${gpu.memory_usage_percent}%` }}
											></div>
										</div>
										<p className="text-xs text-gray-500 mt-0.5">{gpu.memory_usage_percent.toFixed(1)}% used</p>
									</div>
								</div>
								);
							})}
						</div>
					)}
				</>
			)}

			{/* Cluster GPU View */}
			{gpuViewMode === 'cluster' && (
				<>
					{clusterGpuLoading && <p className="text-gray-500 text-sm">Loading cluster GPUs...</p>}
					{clusterGpuStatus?.error && <p className="text-red-500 text-sm">Error: {clusterGpuStatus.error}</p>}

					{clusterGpuStatus?.nodes && (
						<div className="space-y-3">
							{/* Group GPUs by node */}
							{Object.entries(
								clusterGpuStatus.nodes.reduce((acc: Record<string, any[]>, gpu: any) => {
									if (!acc[gpu.node_name]) {
										acc[gpu.node_name] = [];
									}
									acc[gpu.node_name].push(gpu);
									return acc;
								}, {})
							).map(([nodeName, gpus]: [string, any[]]) => (
								<div key={nodeName} className="border rounded p-2 bg-gray-50">
									<div className="flex items-center justify-between mb-2">
										<h4 className="text-sm font-semibold text-gray-700 flex items-center">
											<ServerIcon className="h-4 w-4 mr-1 text-gray-600" />
											{nodeName}
										</h4>
										<span className="text-xs text-gray-600">
											{gpus.filter(g => g.allocatable).length}/{gpus.length} available
										</span>
									</div>
									<div className="grid grid-cols-4 gap-2">
										{gpus.map((gpu, idx) => (
											<div
												key={idx}
												className={`p-2 rounded border ${
													gpu.allocatable
														? 'bg-white border-green-300'
														: 'bg-gray-100 border-red-300'
												}`}
												title={`GPU ${gpu.index}: ${gpu.name}${gpu.has_metrics ? `\nMemory: ${gpu.memory_used_mb}MB / ${gpu.memory_total_mb}MB\nUtil: ${gpu.utilization_percent}%\nTemp: ${gpu.temperature_c}°C` : ''}`}
											>
												<div className="flex justify-between items-center mb-1">
													<span className="text-xs font-medium">GPU {gpu.index}</span>
													{gpu.has_metrics && (
														<span className="text-xs text-gray-500">{gpu.temperature_c}°C</span>
													)}
												</div>
												{gpu.has_metrics ? (
													<>
														{/* Memory usage bar */}
														<div className="mb-1">
															<div className="w-full bg-gray-200 rounded-full h-1.5">
																<div
																	className={`h-1.5 rounded-full ${gpu.memory_usage_percent > 90 ? 'bg-red-600' : gpu.memory_usage_percent > 70 ? 'bg-yellow-500' : 'bg-blue-600'}`}
																	style={{ width: `${gpu.memory_usage_percent}%` }}
																></div>
															</div>
															<div className="flex justify-between text-[10px] text-gray-600 mt-0.5">
																<span>Mem</span>
																<span>{gpu.memory_usage_percent.toFixed(0)}%</span>
															</div>
														</div>
														{/* Utilization badge */}
														<div className="flex justify-between items-center">
															<span className="text-[10px] text-gray-500">Util</span>
															<span className={`px-1 py-0.5 text-[10px] rounded ${gpu.utilization_percent > 80 ? 'bg-red-100 text-red-800' : gpu.utilization_percent > 50 ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'}`}>
																{gpu.utilization_percent}%
															</span>
														</div>
													</>
												) : (
													<div className="text-[10px] text-gray-500 text-center py-1">
														No metrics
													</div>
												)}
											</div>
										))}
									</div>
								</div>
							))}
						</div>
					)}
				</>
			)}
		</div>
	);

	// Worker Status Card
	const renderWorkerCard = () => (
		<div className="bg-white shadow rounded-lg p-6">
			<div className="flex items-center justify-between mb-4">
				<h3 className="text-lg font-medium text-gray-900 flex items-center">
					<ServerIcon className="h-6 w-6 mr-2 text-purple-500" />
					ARQ Worker
				</h3>
				{workerStatus && (
					<span className={`px-2 py-1 text-xs rounded-full ${workerStatus.worker_running ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
						{workerStatus.worker_running ? 'Running' : 'Stopped'}
					</span>
				)}
			</div>

			{workerLoading && <p className="text-gray-500">Loading...</p>}
			{workerStatus?.error && <p className="text-red-500">Error: {workerStatus.error}</p>}

			{workerStatus && (
				<div className="space-y-3">
					<div className="flex justify-between">
						<span className="text-gray-600">Process ID:</span>
						<span className="font-medium">{workerStatus.worker_pid || 'N/A'}</span>
					</div>
					<div className="flex justify-between">
						<span className="text-gray-600">CPU Usage:</span>
						<span className="font-medium">{workerStatus.worker_cpu_percent.toFixed(1)}%</span>
					</div>
					<div className="flex justify-between">
						<span className="text-gray-600">Memory:</span>
						<span className="font-medium">{workerStatus.worker_memory_mb.toFixed(1)} MB</span>
					</div>
					<div className="flex justify-between">
						<span className="text-gray-600">Uptime:</span>
						<span className="font-medium">{formatUptime(workerStatus.worker_uptime_seconds)}</span>
					</div>
					<div className="flex justify-between">
						<span className="text-gray-600">Redis:</span>
						<span className={`font-medium ${workerStatus.redis_available ? 'text-green-600' : 'text-red-600'}`}>
							{workerStatus.redis_available ? 'Connected' : 'Disconnected'}
						</span>
					</div>
				</div>
			)}
		</div>
	);

	// Database Statistics Card
	const renderDBStatsCard = () => (
		<div className="bg-white shadow rounded-lg p-6">
			<div className="flex items-center mb-4">
				<CircleStackIcon className="h-6 w-6 mr-2 text-green-500" />
				<h3 className="text-lg font-medium text-gray-900">Database Statistics</h3>
			</div>

			{dbStatsLoading && <p className="text-gray-500">Loading...</p>}

			{dbStats && (
				<div className="space-y-4">
					{/* Total counts */}
					<div className="grid grid-cols-2 gap-4">
						<div className="text-center p-3 bg-blue-50 rounded">
							<p className="text-2xl font-bold text-blue-600">{dbStats.total_tasks}</p>
							<p className="text-sm text-gray-600">Total Tasks</p>
						</div>
						<div className="text-center p-3 bg-purple-50 rounded">
							<p className="text-2xl font-bold text-purple-600">{dbStats.total_experiments}</p>
							<p className="text-sm text-gray-600">Total Experiments</p>
						</div>
					</div>

					{/* 24h activity */}
					<div className="grid grid-cols-2 gap-4">
						<div className="text-center p-3 bg-green-50 rounded">
							<p className="text-xl font-bold text-green-600">{dbStats.tasks_last_24h}</p>
							<p className="text-xs text-gray-600">Tasks (24h)</p>
						</div>
						<div className="text-center p-3 bg-yellow-50 rounded">
							<p className="text-xl font-bold text-yellow-600">{dbStats.experiments_last_24h}</p>
							<p className="text-xs text-gray-600">Experiments (24h)</p>
						</div>
					</div>

					{/* Average duration */}
					{dbStats.avg_experiment_duration_seconds && (
						<div className="text-center p-3 bg-gray-50 rounded">
							<p className="text-xl font-bold text-gray-700">
								{Math.round(dbStats.avg_experiment_duration_seconds / 60)} min
							</p>
							<p className="text-xs text-gray-600">Avg Experiment Duration</p>
						</div>
					)}

					{/* Status breakdown */}
					<div>
						<p className="text-sm font-medium text-gray-700 mb-2">Task Status:</p>
						<div className="space-y-1">
							{Object.entries(dbStats.tasks_by_status).map(([status, count]) => (
								<div key={status} className="flex justify-between text-sm">
									<span className="text-gray-600 capitalize">{status}:</span>
									<span className="font-medium">{count}</span>
								</div>
							))}
						</div>
					</div>
				</div>
			)}
		</div>
	);

	// Running Tasks Card
	const renderRunningTasksCard = () => (
		<div className="bg-white shadow rounded-lg p-6">
			<div className="flex items-center mb-4">
				<ClockIcon className="h-6 w-6 mr-2 text-orange-500" />
				<h3 className="text-lg font-medium text-gray-900">Running Tasks</h3>
			</div>

			{dbStatsLoading && <p className="text-gray-500">Loading...</p>}

			{dbStats?.running_tasks && dbStats.running_tasks.length === 0 && (
				<p className="text-gray-500 text-sm">No tasks currently running</p>
			)}

			{dbStats?.running_tasks && dbStats.running_tasks.length > 0 && (
				<div className="space-y-3">
					{dbStats.running_tasks.map((task) => (
						<div key={task.id} className="border rounded p-3">
							<div className="flex justify-between items-start mb-2">
								<div>
									<p className="font-medium">Task {task.id}: {task.name}</p>
									{task.started_at && (
										<p className="text-xs text-gray-500">
											Started: {new Date(task.started_at).toLocaleString()}
										</p>
									)}
								</div>
							</div>
							<div className="mt-2">
								<div className="flex justify-between text-sm text-gray-600 mb-1">
									<span>Progress</span>
									<span>{task.completed_experiments} / {task.max_iterations}</span>
								</div>
								<div className="w-full bg-gray-200 rounded-full h-2">
									<div
										className="bg-blue-600 h-2 rounded-full"
										style={{
											width: `${(task.completed_experiments / task.max_iterations) * 100}%`,
										}}
									></div>
								</div>
							</div>
						</div>
					))}
				</div>
			)}
		</div>
	);

	// Experiment Timeline Chart (Gantt-style)
	const renderTimelineChart = () => {
		if (timelineLoading || !timeline) {
			return (
				<div className="bg-white shadow rounded-lg p-6">
					<h3 className="text-lg font-medium text-gray-900 mb-4">Experiment Timeline (24h)</h3>
					<p className="text-gray-500">Loading...</p>
				</div>
			);
		}

		// Filter experiments with valid start times
		// Include both completed experiments and currently running/deploying ones
		const validExperiments = timeline.filter((exp) => exp.started_at);

		if (validExperiments.length === 0) {
			return (
				<div className="bg-white shadow rounded-lg p-6">
					<h3 className="text-lg font-medium text-gray-900 mb-4">Experiment Timeline (24h)</h3>
					<p className="text-gray-500">No experiments in the last 24 hours</p>
				</div>
			);
		}

		// Limit display to most recent 20 experiments for readability
		const displayExperiments = validExperiments
			.sort((a, b) => new Date(b.started_at!).getTime() - new Date(a.started_at!).getTime())
			.slice(0, 20)
			.reverse(); // Reverse to show oldest at top

		// Find time range from the 20 displayed experiments
		// Apply timezone offset to convert UTC to configured timezone for visual alignment
		const startTimes = displayExperiments.map((e) => new Date(e.started_at!).getTime() + timezoneOffsetMs);
		let minTime = Math.min(...startTimes);
		let maxTime = Date.now(); // Use current time in configured timezone as max

		// Ensure at least 1 hour duration
		const MIN_DURATION_MS = 3600000; // 1 hour
		const actualRange = maxTime - minTime;
		if (actualRange < MIN_DURATION_MS) {
			// Expand minTime backward to reach 1 hour
			minTime = maxTime - MIN_DURATION_MS;
		}

		const timeRange = maxTime - minTime;

		return (
			<div className="bg-white shadow rounded-lg p-6">
				<div className="flex justify-between items-center mb-4">
					<h3 className="text-lg font-medium text-gray-900">Experiment Timeline (24h)</h3>
					<span className="text-sm text-gray-500">
						Showing {displayExperiments.length} most recent experiments
					</span>
				</div>

				{/* Timeline visualization */}
				<div className="relative" style={{ minHeight: '400px' }}>
					{/* Time axis with scale markers */}
					<div className="relative mb-2 ml-24">
						{/* Generate time scale markers */}
						{(() => {
							const timeRangeMs = maxTime - minTime;

							// Determine appropriate interval based on time range
							let intervalMs: number;
							if (timeRangeMs <= 3600000) {
								// <= 1 hour: 10-minute intervals
								intervalMs = 600000;
							} else if (timeRangeMs <= 7200000) {
								// <= 2 hours: 15-minute intervals
								intervalMs = 900000;
							} else if (timeRangeMs <= 14400000) {
								// <= 4 hours: 30-minute intervals
								intervalMs = 1800000;
							} else if (timeRangeMs <= 28800000) {
								// <= 8 hours: 1-hour intervals
								intervalMs = 3600000;
							} else {
								// > 8 hours: 2-hour intervals
								intervalMs = 7200000;
							}

							// Round minTime down to nearest interval boundary
							const roundedMinTime = Math.floor(minTime / intervalMs) * intervalMs;

							// Generate time markers starting from rounded time
							const markers: number[] = [];
							let currentTime = roundedMinTime;

							// Start from first marker that's >= minTime
							while (currentTime < minTime) {
								currentTime += intervalMs;
							}

							// Add markers up to maxTime
							while (currentTime <= maxTime) {
								markers.push(currentTime);
								currentTime += intervalMs;
							}

							return (
								<>
									{/* Timeline container */}
									<div className="relative h-6 border-b border-gray-300">
										{markers.map((time, idx) => {
											const leftPercent = ((time - minTime) / timeRangeMs) * 100;
											return (
												<div
													key={idx}
													className="absolute"
													style={{ left: `${leftPercent}%` }}
												>
													{/* Tick mark */}
													<div className="absolute bottom-0 w-px h-2 bg-gray-400" style={{ left: '-0.5px' }}></div>
													{/* Time label */}
													<div className="absolute top-1 text-xs text-gray-600 whitespace-nowrap" style={{ transform: 'translateX(-50%)' }}>
														{formatTime(new Date(time))}
													</div>
												</div>
											);
										})}
									</div>
									{/* Vertical gridlines for better alignment */}
									<div className="absolute top-6 left-0 right-0 bottom-0 pointer-events-none">
										{markers.map((time, idx) => {
											const leftPercent = ((time - minTime) / timeRangeMs) * 100;
											return (
												<div
													key={idx}
													className="absolute top-0 bottom-0 w-px bg-gray-200"
													style={{ left: `${leftPercent}%` }}
												></div>
											);
										})}
									</div>
								</>
							);
						})()}
					</div>

					{/* Experiment bars */}
					<div className="space-y-1">
						{displayExperiments.map((exp) => {
							const startTime = new Date(exp.started_at!).getTime() + timezoneOffsetMs;
							// Use completed_at if available, otherwise use current time (for running experiments)
							const endTime = exp.completed_at
								? new Date(exp.completed_at).getTime() + timezoneOffsetMs
								: Date.now();
							const duration = endTime - startTime;
							const leftPercent = ((startTime - minTime) / timeRange) * 100;
							const widthPercent = (duration / timeRange) * 100;

							// Color based on status
							const isRunning = !exp.completed_at && (exp.status === 'deploying' || exp.status === 'benchmarking' || exp.status === 'pending');
							const statusColor = isRunning
								? 'bg-blue-500 animate-pulse'
								: exp.status === 'success'
								? 'bg-green-500'
								: exp.status === 'failed'
								? 'bg-red-500'
								: 'bg-yellow-500';

							// Running experiments have no right border radius
							const borderRadius = isRunning ? 'rounded-l' : 'rounded';

							return (
								<div key={exp.id} className="flex items-center group">
									{/* Experiment label */}
									<div className="w-24 text-xs text-gray-600 font-medium text-right pr-2">
										Task {exp.task_id} Exp {exp.experiment_id}
									</div>

									{/* Timeline bar container */}
									<div className="flex-1 relative h-8 bg-gray-100 rounded">
										{/* Experiment bar */}
										<div
											className={`absolute h-full ${statusColor} ${borderRadius} cursor-pointer hover:opacity-80 transition-opacity`}
											style={{
												left: `${leftPercent}%`,
												width: `${widthPercent}%`,
											}}
											onClick={() => setSelectedExperiment({ taskId: exp.task_id, experimentId: exp.experiment_id })}
											title={`Experiment ${exp.experiment_id}\nDuration: ${Math.round(
												duration / 1000
											)}s\nStatus: ${exp.status}\nScore: ${
												exp.objective_score !== null && exp.objective_score !== undefined
													? Math.abs(exp.objective_score) >= 100
														? exp.objective_score.toFixed(1)
														: exp.objective_score.toFixed(4)
													: 'N/A'
											}\n\nClick to view logs`}
										>
											{/* Duration label (only show if wide enough) */}
											{widthPercent > 5 && (
												<div className="absolute inset-0 flex items-center justify-center text-xs text-white font-medium">
													{Math.round(duration / 1000)}s
												</div>
											)}
										</div>
									</div>
								</div>
							);
						})}
					</div>
				</div>

				{/* Legend */}
				<div className="flex gap-4 mt-4 text-sm">
					<div className="flex items-center gap-2">
						<div className="w-4 h-4 bg-green-500 rounded"></div>
						<span className="text-gray-600">Success</span>
					</div>
					<div className="flex items-center gap-2">
						<div className="w-4 h-4 bg-red-500 rounded"></div>
						<span className="text-gray-600">Failed</span>
					</div>
					<div className="flex items-center gap-2">
						<div className="w-4 h-4 bg-blue-500 rounded"></div>
						<span className="text-gray-600">Running</span>
					</div>
					<div className="flex items-center gap-2">
						<div className="w-4 h-4 bg-yellow-500 rounded"></div>
						<span className="text-gray-600">Other</span>
					</div>
				</div>

				{/* Statistics summary */}
				<div className="grid grid-cols-3 gap-4 mt-4">
					<div className="text-center p-3 bg-green-50 rounded">
						<p className="text-xl font-bold text-green-600">
							{timeline.filter((e) => e.status === 'success').length}
						</p>
						<p className="text-xs text-gray-600">Successful</p>
					</div>
					<div className="text-center p-3 bg-red-50 rounded">
						<p className="text-xl font-bold text-red-600">
							{timeline.filter((e) => e.status === 'failed').length}
						</p>
						<p className="text-xs text-gray-600">Failed</p>
					</div>
					<div className="text-center p-3 bg-gray-50 rounded">
						<p className="text-xl font-bold text-gray-700">{timeline.length}</p>
						<p className="text-xs text-gray-600">Total</p>
					</div>
				</div>
			</div>
		);
	};

	return (
		<div className="px-4 py-6 sm:px-6 lg:px-8">
			<div className="mb-6">
				<h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
				<p className="text-gray-600">System overview and real-time status</p>
			</div>

			{/* Grid layout */}
			{/* First row: Worker Status, DB Statistics, Running Tasks */}
			<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
				{renderWorkerCard()}
				{renderDBStatsCard()}
				{renderRunningTasksCard()}
			</div>

			{/* Second row: GPU Status and Timeline */}
			<div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
				<div className="lg:col-span-1">
					{renderGPUCard()}
				</div>
				<div className="lg:col-span-2">
					{renderTimelineChart()}
				</div>
			</div>

			{/* Experiment Log Viewer Modal */}
			{selectedExperiment && (
				<ExperimentLogViewer
					taskId={selectedExperiment.taskId}
					experimentId={selectedExperiment.experimentId}
					onClose={() => setSelectedExperiment(null)}
				/>
			)}
		</div>
	);
}
