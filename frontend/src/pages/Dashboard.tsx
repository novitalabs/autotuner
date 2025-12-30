import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useEffect } from 'react';
import { dashboardApi } from '../services/dashboardApi';
import type { DistributedWorker, WorkerGPUHistory, WorkerSlot } from '../services/dashboardApi';
import { useTimezone } from '../contexts/TimezoneContext';
import ExperimentLogViewer from '../components/ExperimentLogViewer';
import {
	CpuChipIcon,
	ServerIcon,
	CircleStackIcon,
	ClockIcon,
	ServerStackIcon,
	PencilIcon,
	CheckIcon,
	XMarkIcon,
	ArrowPathIcon,
	TrashIcon,
} from '@heroicons/react/24/outline';

function formatUptime(seconds: number | null): string {
	if (!seconds) return 'N/A';
	const hours = Math.floor(seconds / 3600);
	const minutes = Math.floor((seconds % 3600) / 60);
	return `${hours}h ${minutes}m`;
}

export default function Dashboard() {
	// Track selected experiment for log viewer
	const [selectedExperiment, setSelectedExperiment] = useState<{ taskId: number; experimentId: number } | null>(null);

	// Track worker being edited for alias rename
	const [editingWorkerId, setEditingWorkerId] = useState<string | null>(null);
	const [editingAlias, setEditingAlias] = useState<string>('');

	// Track GPU history per worker
	const [gpuHistories, setGpuHistories] = useState<Record<string, WorkerGPUHistory>>({});

	// Query client for cache invalidation
	const queryClient = useQueryClient();

	// Mutation for renaming worker
	const renameWorkerMutation = useMutation({
		mutationFn: ({ workerId, alias }: { workerId: string; alias: string | null }) =>
			dashboardApi.renameWorker(workerId, alias),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ['distributedWorkers'] });
			setEditingWorkerId(null);
			setEditingAlias('');
		},
	});

	// Track restoring worker slot
	const [restoringSlotId, setRestoringSlotId] = useState<number | null>(null);

	// Mutation for restoring worker
	const restoreWorkerMutation = useMutation({
		mutationFn: (slotId: number) => dashboardApi.restoreWorker(slotId, false),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ['workerSlots'] });
			queryClient.invalidateQueries({ queryKey: ['distributedWorkers'] });
			setRestoringSlotId(null);
		},
		onError: () => {
			setRestoringSlotId(null);
		},
	});

	// Mutation for deleting worker slot
	const deleteSlotMutation = useMutation({
		mutationFn: (slotId: number) => dashboardApi.deleteWorkerSlot(slotId),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ['workerSlots'] });
		},
	});

	// Get timezone formatting functions
	const { formatTime, timezoneOffsetMs } = useTimezone();

	// Fetch dashboard data with auto-refresh
	const { data: workerStatus, isLoading: workerLoading } = useQuery({
		queryKey: ['workerStatus'],
		queryFn: dashboardApi.getWorkerStatus,
		refetchInterval: 5000,
	});

	const { data: distributedWorkers, isLoading: distributedWorkersLoading } = useQuery({
		queryKey: ['distributedWorkers'],
		queryFn: dashboardApi.getDistributedWorkers,
		refetchInterval: 5000,
	});

	// Fetch worker slots (persistent deployment configurations)
	const { data: workerSlots, isLoading: workerSlotsLoading } = useQuery({
		queryKey: ['workerSlots'],
		queryFn: dashboardApi.getWorkerSlots,
		refetchInterval: 10000, // Less frequent since slots don't change often
	});

	// Fetch GPU history for all workers when worker list updates
	useEffect(() => {
		if (distributedWorkers?.workers) {
			distributedWorkers.workers.forEach(async (worker) => {
				try {
					const history = await dashboardApi.getWorkerGPUHistory(worker.worker_id);
					setGpuHistories((prev) => ({ ...prev, [worker.worker_id]: history }));
				} catch (e) {
					// Ignore errors - history may not be available yet
				}
			});
		}
	}, [distributedWorkers]);

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

	// Worker Status Card
	const renderWorkerCard = () => {
		// Helper to format seconds since heartbeat
		const formatHeartbeat = (seconds: number) => {
			if (seconds < 60) return `${Math.round(seconds)}s ago`;
			if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
			return `${Math.round(seconds / 3600)}h ago`;
		};

		// Status badge color
		const getStatusColor = (status: string) => {
			switch (status) {
				case 'online': return 'bg-green-100 text-green-800';
				case 'busy': return 'bg-yellow-100 text-yellow-800';
				case 'offline': return 'bg-red-100 text-red-800';
				default: return 'bg-gray-100 text-gray-800';
			}
		};

		return (
			<div className="bg-white shadow rounded-lg p-6">
				<div className="flex items-center justify-between mb-4">
					<h3 className="text-lg font-medium text-gray-900 flex items-center">
						<ServerStackIcon className="h-6 w-6 mr-2 text-purple-500" />
						Workers
					</h3>
					<div className="flex items-center gap-2">
						{distributedWorkers && (
							<>
								<span className="px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
									{distributedWorkers.online_count} online
								</span>
								{distributedWorkers.busy_count > 0 && (
									<span className="px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800">
										{distributedWorkers.busy_count} busy
									</span>
								)}
							</>
						)}
						{workerSlots && workerSlots.offline_count + workerSlots.unknown_count > 0 && (
							<span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-800">
								{workerSlots.offline_count + workerSlots.unknown_count} offline
							</span>
						)}
					</div>
				</div>

				{(workerLoading || distributedWorkersLoading || workerSlotsLoading) && <p className="text-gray-500">Loading...</p>}
				{workerStatus?.error && <p className="text-red-500">Error: {workerStatus.error}</p>}

				{/* Local Worker Process Info */}
				{workerStatus && (
					<div className="mb-4 p-3 bg-gray-50 rounded border">
						<div className="flex items-center justify-between mb-2">
							<span className="text-sm font-medium text-gray-700">Local Process</span>
							<span className={`px-2 py-0.5 text-xs rounded-full ${workerStatus.worker_running ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
								{workerStatus.worker_running ? 'Running' : 'Stopped'}
							</span>
						</div>
						<div className="grid grid-cols-2 gap-2 text-sm">
							<div className="flex justify-between">
								<span className="text-gray-500">PID:</span>
								<span>{workerStatus.worker_pid || 'N/A'}</span>
							</div>
							<div className="flex justify-between">
								<span className="text-gray-500">CPU:</span>
								<span>{workerStatus.worker_cpu_percent.toFixed(1)}%</span>
							</div>
							<div className="flex justify-between">
								<span className="text-gray-500">Memory:</span>
								<span>{workerStatus.worker_memory_mb.toFixed(0)} MB</span>
							</div>
							<div className="flex justify-between">
								<span className="text-gray-500">Uptime:</span>
								<span>{formatUptime(workerStatus.worker_uptime_seconds)}</span>
							</div>
						</div>
						<div className="mt-2 flex justify-between text-sm">
							<span className="text-gray-500">Redis:</span>
							<span className={workerStatus.redis_available ? 'text-green-600' : 'text-red-600'}>
								{workerStatus.redis_available ? 'Connected' : 'Disconnected'}
							</span>
						</div>
					</div>
				)}

				{/* Distributed Workers */}
				{distributedWorkers && distributedWorkers.workers.length > 0 && (
					<div className="space-y-2">
						<div className="text-sm font-medium text-gray-700 mb-2">
							Registered Workers ({distributedWorkers.total_count})
						</div>
						{distributedWorkers.workers.map((worker: DistributedWorker) => (
							<div key={worker.worker_id} className="p-3 border rounded bg-white hover:bg-gray-50">
								<div className="flex items-center justify-between mb-2">
									<div className="flex items-center gap-2 flex-1 min-w-0">
										<ServerIcon className="h-4 w-4 text-gray-400 flex-shrink-0" />
										{editingWorkerId === worker.worker_id ? (
											<div className="flex items-center gap-1 flex-1">
												<input
													type="text"
													value={editingAlias}
													onChange={(e) => setEditingAlias(e.target.value)}
													placeholder={worker.hostname}
													className="text-sm border rounded px-1 py-0.5 w-24"
													autoFocus
													onKeyDown={(e) => {
														if (e.key === 'Enter') {
															renameWorkerMutation.mutate({
																workerId: worker.worker_id,
																alias: editingAlias.trim() || null
															});
														} else if (e.key === 'Escape') {
															setEditingWorkerId(null);
															setEditingAlias('');
														}
													}}
												/>
												<button
													onClick={() => renameWorkerMutation.mutate({
														workerId: worker.worker_id,
														alias: editingAlias.trim() || null
													})}
													className="p-0.5 hover:bg-green-100 rounded"
													title="Save"
												>
													<CheckIcon className="h-3.5 w-3.5 text-green-600" />
												</button>
												<button
													onClick={() => {
														setEditingWorkerId(null);
														setEditingAlias('');
													}}
													className="p-0.5 hover:bg-red-100 rounded"
													title="Cancel"
												>
													<XMarkIcon className="h-3.5 w-3.5 text-red-600" />
												</button>
											</div>
										) : (
											<>
												<span className="font-medium text-sm truncate" title={worker.worker_id}>
													{worker.alias || worker.hostname}
												</span>
												{worker.alias && (
													<span className="text-xs text-gray-400 truncate" title={worker.hostname}>
														({worker.hostname})
													</span>
												)}
												<button
													onClick={() => {
														setEditingWorkerId(worker.worker_id);
														setEditingAlias(worker.alias || '');
													}}
													className="p-0.5 hover:bg-gray-200 rounded flex-shrink-0"
													title="Rename worker"
												>
													<PencilIcon className="h-3 w-3 text-gray-400" />
												</button>
											</>
										)}
									</div>
									<span className={`px-2 py-0.5 text-xs rounded-full flex-shrink-0 ${getStatusColor(worker.status)}`}>
										{worker.status}
									</span>
								</div>
								<div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-gray-600">
									<div className="flex items-center gap-1">
										<CpuChipIcon className="h-3 w-3" />
										<span>{worker.gpu_count} GPU{worker.gpu_count !== 1 ? 's' : ''}</span>
									</div>
									<div>
										{worker.gpu_model ? (
											<span className="truncate" title={worker.gpu_model}>
												{worker.gpu_model.replace('NVIDIA ', '').replace('GeForce ', '')}
											</span>
										) : (
											<span className="text-gray-400">No GPU</span>
										)}
									</div>
									<div>
										<span className="text-gray-500">Jobs:</span> {worker.current_jobs}/{worker.max_parallel}
									</div>
									<div>
										<span className="text-gray-500">Mode:</span> {worker.deployment_mode}
									</div>
									<div className="col-span-2 text-gray-400">
										<span>Heartbeat: {formatHeartbeat(worker.seconds_since_heartbeat)}</span>
									</div>
								</div>
								{/* GPU Status with History Sparkline */}
								{worker.gpus && worker.gpus.length > 0 && (
									<div className="mt-2 pt-2 border-t">
										<div className="text-xs text-gray-500 mb-1">GPU Status</div>
										{/* For OME mode workers, group GPUs by node */}
										{worker.deployment_mode === 'ome' ? (
											(() => {
												// Group GPUs by node_name
												const nodeMap: Record<string, typeof worker.gpus> = {};
												worker.gpus.forEach((gpu) => {
													const nodeName = gpu.node_name || 'unknown';
													if (!nodeMap[nodeName]) nodeMap[nodeName] = [];
													nodeMap[nodeName].push(gpu);
												});
												const nodes = Object.entries(nodeMap).sort(([a], [b]) => a.localeCompare(b));

												return (
													<div className="space-y-2">
														{nodes.map(([nodeName, nodeGpus]) => {
															const hasMetrics = nodeGpus.some(g => g.memory_used_gb !== null);
															const avgUtil = hasMetrics
																? nodeGpus.reduce((sum, g) => sum + (g.utilization_percent || 0), 0) / nodeGpus.length
																: 0;
															const totalMem = nodeGpus.reduce((sum, g) => sum + (g.memory_total_gb || 0), 0);
															const usedMem = nodeGpus.reduce((sum, g) => sum + (g.memory_used_gb || 0), 0);
															const memPercent = totalMem > 0 ? (usedMem / totalMem) * 100 : 0;

															return (
																<div key={nodeName} className={`p-2 rounded ${hasMetrics ? 'bg-blue-50 border border-blue-100' : 'bg-gray-50'}`}>
																	<div className="flex items-center justify-between mb-1">
																		<span className="text-xs font-medium text-gray-700 truncate" title={nodeName}>
																			{nodeName.replace('host-', '')}
																		</span>
																		<span className="text-xs text-gray-500">{nodeGpus.length} GPUs</span>
																	</div>
																	{hasMetrics ? (
																		<>
																			<div className="flex items-center gap-2 text-xs mb-1">
																				<span className="text-gray-500">Util:</span>
																				<span className={`font-medium ${avgUtil > 80 ? 'text-red-600' : avgUtil > 50 ? 'text-yellow-600' : 'text-green-600'}`}>
																					{avgUtil.toFixed(0)}%
																				</span>
																				<span className="text-gray-300">|</span>
																				<span className="text-gray-500">Mem:</span>
																				<span className={`${memPercent > 90 ? 'text-red-600' : memPercent > 70 ? 'text-yellow-600' : 'text-blue-600'}`}>
																					{usedMem.toFixed(0)}/{totalMem.toFixed(0)}G ({memPercent.toFixed(0)}%)
																				</span>
																			</div>
																			{/* Per-GPU mini display */}
																			<div className="grid grid-cols-4 gap-0.5">
																				{nodeGpus.map((gpu, idx) => (
																					<div
																						key={idx}
																						className={`text-center text-[10px] py-0.5 rounded ${
																							(gpu.utilization_percent || 0) > 80 ? 'bg-red-200 text-red-800' :
																							(gpu.utilization_percent || 0) > 50 ? 'bg-yellow-200 text-yellow-800' :
																							'bg-green-200 text-green-800'
																						}`}
																						title={`GPU ${gpu.index}: ${gpu.utilization_percent?.toFixed(0) || 0}% util, ${gpu.memory_used_gb?.toFixed(1) || 0}/${gpu.memory_total_gb?.toFixed(0) || 0}G, ${gpu.temperature_c || 'N/A'}°C`}
																					>
																						{gpu.utilization_percent?.toFixed(0) || 0}%
																					</div>
																				))}
																			</div>
																		</>
																	) : (
																		<div className="text-xs text-gray-400">No metrics (remote node)</div>
																	)}
																</div>
															);
														})}
													</div>
												);
											})()
										) : (
											/* Non-OME mode: original GPU display */
											<div className="grid gap-1">
												{worker.gpus.map((gpu) => {
													// Get history for this GPU from worker's history
													const workerHistory = gpuHistories[worker.worker_id]?.history || [];
													const gpuUtilHistory = workerHistory
														.map(h => h.gpus.find(g => g.index === gpu.index)?.utilization)
														.filter((v): v is number => v !== null && v !== undefined);
													const memoryPercent = gpu.memory_used_gb !== null && gpu.memory_total_gb !== null
														? (gpu.memory_used_gb / gpu.memory_total_gb) * 100
														: 0;

													return (
														<div key={gpu.index} className="flex items-center gap-2 text-xs">
															<span className="text-gray-600 w-12 flex-shrink-0">GPU {gpu.index}</span>
															{/* Utilization History Sparkline */}
															{gpuUtilHistory.length > 1 && (
																<div className="flex items-end gap-px flex-shrink-0" style={{ height: '14px' }} title={`Last ${gpuUtilHistory.length} heartbeats`}>
																	{gpuUtilHistory.map((util, idx) => (
																		<div
																			key={idx}
																			className={`${
																				util > 80 ? 'bg-red-400' : util > 50 ? 'bg-yellow-400' : 'bg-green-400'
																			}`}
																			style={{
																				width: '3px',
																				height: `${Math.max(util * 0.14, 1)}px`,
																				minHeight: '1px'
																			}}
																			title={`${util.toFixed(0)}%`}
																		></div>
																	))}
																</div>
															)}
															{gpu.utilization_percent !== null && (
																<span className={`w-8 text-right flex-shrink-0 ${gpu.utilization_percent > 80 ? 'text-red-600' : gpu.utilization_percent > 50 ? 'text-yellow-600' : 'text-green-600'}`}>
																	{gpu.utilization_percent.toFixed(0)}%
																</span>
															)}
															{/* Memory bar with text overlay */}
															{gpu.memory_used_gb !== null && gpu.memory_total_gb !== null && (
																<div className="flex-1 relative h-4 bg-gray-200 rounded overflow-hidden min-w-[80px]">
																	<div
																		className={`absolute inset-y-0 left-0 ${memoryPercent > 90 ? 'bg-red-300' : memoryPercent > 70 ? 'bg-yellow-300' : 'bg-blue-300'}`}
																		style={{ width: `${memoryPercent}%` }}
																	></div>
																	<span className="absolute inset-0 flex items-center justify-center text-[10px] text-gray-700 font-medium">
																		{gpu.memory_used_gb.toFixed(1)}/{gpu.memory_total_gb.toFixed(0)}G
																	</span>
																</div>
															)}
															{gpu.temperature_c !== null && (
																<span className={`w-8 text-right flex-shrink-0 ${gpu.temperature_c > 80 ? 'text-red-600' : gpu.temperature_c > 60 ? 'text-yellow-600' : 'text-gray-500'}`}>
																	{gpu.temperature_c}°C
																</span>
															)}
														</div>
													);
												})}
											</div>
										)}
									</div>
								)}
							</div>
						))}
					</div>
				)}

				{distributedWorkers && distributedWorkers.workers.length === 0 && !workerSlots?.slots?.length && (
					<div className="text-sm text-gray-500 text-center py-4">
						No workers registered yet
					</div>
				)}

				{/* Offline Worker Slots (not currently running) */}
				{workerSlots && workerSlots.slots.length > 0 && (() => {
					// Get worker_ids of currently running workers
					const runningWorkerIds = new Set(
						(distributedWorkers?.workers || []).map(w => w.worker_id)
					);
					// Filter slots that are offline/unknown or don't have running workers
					const offlineSlots = workerSlots.slots.filter(
						slot => slot.current_status !== 'online' || !runningWorkerIds.has(slot.worker_id || '')
					);

					if (offlineSlots.length === 0) return null;

					return (
						<div className="mt-4 pt-4 border-t">
							<div className="text-sm font-medium text-gray-700 mb-2">
								Offline Workers ({offlineSlots.length})
							</div>
							<div className="space-y-2">
								{offlineSlots.map((slot: WorkerSlot) => (
									<div key={slot.id} className="p-3 border rounded bg-gray-50">
										<div className="flex items-center justify-between mb-2">
											<div className="flex items-center gap-2 flex-1 min-w-0">
												<ServerIcon className="h-4 w-4 text-gray-400 flex-shrink-0" />
												<span className="font-medium text-sm truncate" title={slot.ssh_command}>
													{slot.name}
												</span>
												{slot.hostname && (
													<span className="text-xs text-gray-400 truncate">
														({slot.hostname})
													</span>
												)}
											</div>
											<div className="flex items-center gap-2">
												<span className={`px-2 py-0.5 text-xs rounded-full flex-shrink-0 ${
													slot.current_status === 'offline' ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
												}`}>
													{slot.current_status}
												</span>
												<button
													onClick={() => {
														setRestoringSlotId(slot.id);
														restoreWorkerMutation.mutate(slot.id);
													}}
													disabled={restoringSlotId === slot.id}
													className={`px-2 py-1 text-xs rounded flex items-center gap-1 ${
														restoringSlotId === slot.id
															? 'bg-gray-200 text-gray-500 cursor-wait'
															: 'bg-blue-100 text-blue-700 hover:bg-blue-200'
													}`}
													title="Restore worker"
												>
													<ArrowPathIcon className={`h-3 w-3 ${restoringSlotId === slot.id ? 'animate-spin' : ''}`} />
													{restoringSlotId === slot.id ? 'Restoring...' : 'Restore'}
												</button>
												<button
													onClick={() => {
														if (confirm(`Delete worker slot "${slot.name}"?`)) {
															deleteSlotMutation.mutate(slot.id);
														}
													}}
													className="p-1 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded"
													title="Delete slot"
												>
													<TrashIcon className="h-3.5 w-3.5" />
												</button>
											</div>
										</div>
										<div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-gray-600">
											<div className="flex items-center gap-1">
												<CpuChipIcon className="h-3 w-3" />
												<span>{slot.gpu_count ?? '?'} GPU{(slot.gpu_count ?? 0) !== 1 ? 's' : ''}</span>
											</div>
											<div>
												{slot.gpu_model ? (
													<span className="truncate" title={slot.gpu_model}>
														{slot.gpu_model.replace('NVIDIA ', '').replace('GeForce ', '')}
													</span>
												) : (
													<span className="text-gray-400">Unknown GPU</span>
												)}
											</div>
											<div>
												<span className="text-gray-500">Mode:</span> {slot.controller_type}
											</div>
											<div className="truncate" title={slot.ssh_command}>
												<span className="text-gray-500">SSH:</span> {slot.ssh_command.replace('ssh ', '')}
											</div>
											{slot.last_error && (
												<div className="col-span-2 text-red-500 truncate" title={slot.last_error}>
													Error: {slot.last_error}
												</div>
											)}
											{slot.last_seen_at && (
												<div className="col-span-2 text-gray-400">
													Last seen: {new Date(slot.last_seen_at).toLocaleString()}
												</div>
											)}
										</div>
									</div>
								))}
							</div>
						</div>
					);
				})()}
			</div>
		);
	};

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
										className="bg-emerald-500 h-2 rounded-full"
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
			{/* First row: Worker Status, DB Statistics */}
			<div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
				{renderWorkerCard()}
				{renderDBStatsCard()}
			</div>

			{/* Second row: Running Tasks (left) + Timeline (right) */}
			<div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
				<div className="lg:col-span-1">
					{renderRunningTasksCard()}
				</div>
				<div className="lg:col-span-3">
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
