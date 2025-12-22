import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/services/api";
import type { ContainerInfo } from "@/types/api";
import toast from "react-hot-toast";

export default function Containers() {
	const queryClient = useQueryClient();
	const [showAll, setShowAll] = useState(false);
	const [selectedContainer, setSelectedContainer] = useState<string | null>(null);
	const [showLogs, setShowLogs] = useState(false);
	const [showStats, setShowStats] = useState(false);

	// Streaming log state
	const [isStreaming, setIsStreaming] = useState(false);
	const [streamLogs, setStreamLogs] = useState<string[]>([]);
	const [initialLoadDone, setInitialLoadDone] = useState(false);
	const eventSourceRef = useRef<EventSource | null>(null);
	const logEndRef = useRef<HTMLDivElement | null>(null);
	const [autoScroll, setAutoScroll] = useState(true);

	// Fetch Docker info first to check availability
	const { data: dockerInfo, error: dockerInfoError, isLoading: dockerInfoLoading } = useQuery({
		queryKey: ["dockerInfo"],
		queryFn: () => apiClient.getDockerInfo(),
		refetchInterval: (query) => {
			// Stop polling if Docker is unavailable (503 error)
			if (query.state.error && (query.state.error as any)?.response?.status === 503) {
				return false;
			}
			return 10000;
		},
		retry: false // Don't retry if Docker is unavailable
	});

	// Check if Docker daemon is unavailable (503 error)
	const isDockerUnavailable = dockerInfoError && (dockerInfoError as any)?.response?.status === 503;

	// Fetch containers - only when Docker is available
	const {
		data: containers,
		isLoading,
		error
	} = useQuery({
		queryKey: ["containers", showAll],
		queryFn: () => apiClient.getContainers(showAll),
		refetchInterval: 3000, // Auto-refresh every 3 seconds
		enabled: !isDockerUnavailable && !dockerInfoLoading // Only fetch when Docker is available
	});

	// Fetch container logs (static mode)
	const {
		data: logs,
		isLoading: logsLoading
	} = useQuery({
		queryKey: ["containerLogs", selectedContainer],
		queryFn: () => apiClient.getContainerLogs(selectedContainer!, 500),
		enabled: !!selectedContainer && showLogs && !isStreaming,
		refetchInterval: isStreaming ? false : 2000
	});

	// Fetch container stats
	const { data: stats } = useQuery({
		queryKey: ["containerStats", selectedContainer],
		queryFn: () => apiClient.getContainerStats(selectedContainer!),
		enabled: !!selectedContainer && showStats,
		refetchInterval: 2000
	});

	// Start container mutation
	const startMutation = useMutation({
		mutationFn: (containerId: string) => apiClient.startContainer(containerId),
		onSuccess: (_data, containerId) => {
			toast.success(`Container started successfully`);
			queryClient.invalidateQueries({ queryKey: ["containers"] });
			queryClient.invalidateQueries({ queryKey: ["containerStats", containerId] });
		}
	});

	// Stop container mutation
	const stopMutation = useMutation({
		mutationFn: (containerId: string) => apiClient.stopContainer(containerId),
		onSuccess: (_data, containerId) => {
			toast.success(`Container stopped successfully`);
			queryClient.invalidateQueries({ queryKey: ["containers"] });
			queryClient.invalidateQueries({ queryKey: ["containerStats", containerId] });
		}
	});

	// Restart container mutation
	const restartMutation = useMutation({
		mutationFn: (containerId: string) => apiClient.restartContainer(containerId),
		onSuccess: (_data, containerId) => {
			toast.success(`Container restarted successfully`);
			queryClient.invalidateQueries({ queryKey: ["containers"] });
			queryClient.invalidateQueries({ queryKey: ["containerStats", containerId] });
		}
	});

	// Remove container mutation
	const removeMutation = useMutation({
		mutationFn: ({ containerId, force }: { containerId: string; force: boolean }) =>
			apiClient.removeContainer(containerId, force),
		onSuccess: (_data, { containerId }) => {
			toast.success(`Container removed successfully`);
			queryClient.invalidateQueries({ queryKey: ["containers"] });
			if (selectedContainer === containerId) {
				setSelectedContainer(null);
				setShowLogs(false);
				setShowStats(false);
			}
		}
	});

	const getStatusColor = (status: string) => {
		switch (status.toLowerCase()) {
			case "running":
				return "text-green-600 bg-green-50";
			case "exited":
				return "text-gray-600 bg-gray-50";
			case "paused":
				return "text-yellow-600 bg-yellow-50";
			case "restarting":
				return "text-blue-600 bg-blue-50";
			case "dead":
				return "text-red-600 bg-red-50";
			default:
				return "text-gray-600 bg-gray-50";
		}
	};

	const formatDate = (dateStr: string | null) => {
		if (!dateStr) return "N/A";
		try {
			return new Date(dateStr).toLocaleString();
		} catch {
			return dateStr;
		}
	};

	const handleViewDetails = (container: ContainerInfo) => {
		setSelectedContainer(container.id);
		setShowLogs(true);
		setShowStats(true);
		// Reset streaming state for new container
		setInitialLoadDone(false);
		setIsStreaming(false);
		setStreamLogs([]);
	};

	const handleCloseDetails = () => {
		// Stop streaming if active
		if (eventSourceRef.current) {
			eventSourceRef.current.close();
			eventSourceRef.current = null;
		}
		setSelectedContainer(null);
		setShowLogs(false);
		setShowStats(false);
		setIsStreaming(false);
		setStreamLogs([]);
		setInitialLoadDone(false);
	};

	const toggleStreaming = () => {
		if (isStreaming) {
			// Stop streaming
			if (eventSourceRef.current) {
				eventSourceRef.current.close();
				eventSourceRef.current = null;
			}
			setIsStreaming(false);
		} else {
			// Start streaming - preserve existing logs if any
			if (logs?.logs && streamLogs.length === 0) {
				const existingLogs = logs.logs.split("\n").filter(Boolean);
				setStreamLogs(existingLogs);
			}
			setIsStreaming(true);

			const apiUrl = import.meta.env.VITE_API_URL || "/api";
			const eventSource = new EventSource(
				`${apiUrl}/docker/containers/${selectedContainer}/logs?follow=true`
			);

			eventSource.onmessage = (event) => {
				const logLine = event.data;
				setStreamLogs((prev) => [...prev, logLine]);
			};

			eventSource.onerror = (error) => {
				console.error("EventSource error:", error);
				eventSource.close();
				setIsStreaming(false);
			};

			eventSourceRef.current = eventSource;
		}
	};

	// Auto-start streaming after initial logs are loaded
	useEffect(() => {
		if (!initialLoadDone && logs && !logsLoading && selectedContainer && showLogs) {
			// Initialize streamLogs with existing logs
			const existingLogs = logs.logs ? logs.logs.split("\n").filter(Boolean) : [];
			setStreamLogs(existingLogs);
			setInitialLoadDone(true);

			// Auto-start streaming
			setIsStreaming(true);

			const apiUrl = import.meta.env.VITE_API_URL || "/api";
			const eventSource = new EventSource(
				`${apiUrl}/docker/containers/${selectedContainer}/logs?follow=true`
			);

			eventSource.onmessage = (event) => {
				const logLine = event.data;
				setStreamLogs((prev) => [...prev, logLine]);
			};

			eventSource.onerror = (error) => {
				console.error("EventSource error:", error);
				eventSource.close();
				setIsStreaming(false);
			};

			eventSourceRef.current = eventSource;
		}
	}, [logs, logsLoading, initialLoadDone, selectedContainer, showLogs]);

	// Cleanup on unmount or container change
	useEffect(() => {
		return () => {
			if (eventSourceRef.current) {
				eventSourceRef.current.close();
			}
		};
	}, []);

	// Auto-scroll to bottom when new logs arrive
	useEffect(() => {
		if (autoScroll && logEndRef.current) {
			logEndRef.current.scrollIntoView({ behavior: "smooth" });
		}
	}, [streamLogs, autoScroll]);

	const handleAction = (containerId: string, action: string) => {
		switch (action) {
			case "start":
				startMutation.mutate(containerId);
				break;
			case "stop":
				stopMutation.mutate(containerId);
				break;
			case "restart":
				restartMutation.mutate(containerId);
				break;
			case "remove":
				if (confirm("Are you sure you want to remove this container?")) {
					const force = confirm("Force remove? (Required if container is running)");
					removeMutation.mutate({ containerId, force });
				}
				break;
		}
	};

	const selectedContainerData = containers?.find((c) => c.id === selectedContainer);

	return (
		<div className="p-6">
			<div className="mb-6">
				<h1 className="text-3xl font-bold text-gray-900">Docker Containers</h1>
				<p className="text-gray-600 mt-2">Manage and monitor Docker containers</p>
			</div>

			{/* Docker Unavailable State */}
			{isDockerUnavailable && (
				<div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
					<svg
						className="w-16 h-16 mx-auto mb-4 text-gray-400"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={1.5}
							d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
						/>
					</svg>
					<h2 className="text-xl font-semibold text-gray-700 mb-2">Docker Not Available</h2>
					<p className="text-gray-500 mb-4">
						Unable to connect to the Docker daemon. Please ensure Docker is installed and running.
					</p>
					<div className="bg-gray-100 rounded p-4 text-left max-w-md mx-auto">
						<p className="text-sm text-gray-600 font-mono">
							# Check Docker status<br />
							sudo systemctl status docker<br /><br />
							# Start Docker<br />
							sudo systemctl start docker
						</p>
					</div>
				</div>
			)}

			{/* Docker Info Summary */}
			{!isDockerUnavailable && dockerInfo && (
				<div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
					<div className="bg-white p-4 rounded-lg shadow">
						<div className="text-sm text-gray-600">Total Containers</div>
						<div className="text-2xl font-bold text-gray-900">{dockerInfo.containers}</div>
					</div>
					<div className="bg-white p-4 rounded-lg shadow">
						<div className="text-sm text-gray-600">Running</div>
						<div className="text-2xl font-bold text-green-600">
							{dockerInfo.containers_running}
						</div>
					</div>
					<div className="bg-white p-4 rounded-lg shadow">
						<div className="text-sm text-gray-600">Stopped</div>
						<div className="text-2xl font-bold text-gray-600">
							{dockerInfo.containers_stopped}
						</div>
					</div>
					<div className="bg-white p-4 rounded-lg shadow">
						<div className="text-sm text-gray-600">Images</div>
						<div className="text-2xl font-bold text-blue-600">{dockerInfo.images}</div>
					</div>
				</div>
			)}

			{/* Controls */}
			{!isDockerUnavailable && (
			<div className="mb-4 flex items-center justify-between">
				<label className="flex items-center gap-2">
					<input
						type="checkbox"
						checked={showAll}
						onChange={(e) => setShowAll(e.target.checked)}
						className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
					/>
					<span className="text-sm text-gray-700">Show all containers (including stopped)</span>
				</label>
			</div>
			)}

			{/* Loading State */}
			{!isDockerUnavailable && isLoading && (
				<div className="bg-white rounded-lg shadow p-8 text-center">
					<div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
					<p className="text-gray-600">Loading containers...</p>
				</div>
			)}

			{/* Error State */}
			{!isDockerUnavailable && error && (
				<div className="bg-red-50 border border-red-200 rounded-lg p-4">
					<p className="text-red-800">Failed to load containers. Make sure Docker is running.</p>
				</div>
			)}

			{/* Containers List */}
			{!isDockerUnavailable && !isLoading && !error && (
				<div className="grid grid-cols-1 gap-4">
					{containers?.length === 0 ? (
						<div className="bg-white rounded-lg shadow p-8 text-center">
							<p className="text-gray-600">No containers found</p>
						</div>
					) : (
						containers?.map((container) => (
							<div
								key={container.id}
								className="bg-white rounded-lg shadow hover:shadow-md transition-shadow"
							>
								<div className="p-4">
									<div className="flex items-start justify-between">
										<div className="flex-1">
											<div className="flex items-center gap-3 mb-2">
												<h3 className="text-lg font-semibold text-gray-900">
													{container.name}
												</h3>
												<span
													className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(
														container.status
													)}`}
												>
													{container.status}
												</span>
											</div>
											<div className="space-y-1 text-sm text-gray-600">
												<div>
													<span className="font-medium">Image:</span> {container.image}
												</div>
												<div>
													<span className="font-medium">ID:</span> {container.short_id}
												</div>
												{Object.keys(container.ports).length > 0 && (
													<div>
														<span className="font-medium">Ports:</span>{" "}
														{Object.entries(container.ports)
															.map(([cPort, hPort]) => `${hPort} → ${cPort}`)
															.join(", ")}
													</div>
												)}
												{container.command && (
													<div>
														<span className="font-medium">Command:</span>{" "}
														<code className="text-xs bg-gray-100 px-1 py-0.5 rounded">
															{container.command.length > 100
																? container.command.substring(0, 100) + "..."
																: container.command}
														</code>
													</div>
												)}
												<div>
													<span className="font-medium">Created:</span>{" "}
													{formatDate(container.created)}
												</div>
											</div>
										</div>

										<div className="flex gap-2 ml-4">
											{container.status === "running" ? (
												<>
													<button
														onClick={() => handleAction(container.id, "stop")}
														disabled={stopMutation.isPending}
														className="px-3 py-1.5 text-sm bg-yellow-600 text-white rounded hover:bg-yellow-700 disabled:opacity-50"
													>
														Stop
													</button>
													<button
														onClick={() => handleAction(container.id, "restart")}
														disabled={restartMutation.isPending}
														className="px-3 py-1.5 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
													>
														Restart
													</button>
												</>
											) : (
												<button
													onClick={() => handleAction(container.id, "start")}
													disabled={startMutation.isPending}
													className="px-3 py-1.5 text-sm bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
												>
													Start
												</button>
											)}
											<button
												onClick={() => handleViewDetails(container)}
												className="px-3 py-1.5 text-sm bg-gray-600 text-white rounded hover:bg-gray-700"
											>
												Details
											</button>
											<button
												onClick={() => handleAction(container.id, "remove")}
												disabled={removeMutation.isPending}
												className="px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
											>
												Remove
											</button>
										</div>
									</div>
								</div>
							</div>
						))
					)}
				</div>
			)}

			{/* Container Details Modal */}
			{selectedContainer && selectedContainerData && (
				<div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
					<div className="bg-white rounded-lg shadow-xl max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
						{/* Modal Header */}
						<div className="p-4 border-b border-gray-200 flex items-center justify-between">
							<div>
								<h2 className="text-xl font-bold text-gray-900">
									{selectedContainerData.name}
								</h2>
								<p className="text-sm text-gray-600">ID: {selectedContainerData.short_id}</p>
							</div>
							<button
								onClick={handleCloseDetails}
								className="text-gray-500 hover:text-gray-700 text-2xl leading-none"
							>
								×
							</button>
						</div>

						{/* Modal Body */}
						<div className="flex-1 overflow-auto p-4 space-y-4">
							{/* Stats */}
							{showStats && stats && (
								<div>
									<h3 className="text-lg font-semibold text-gray-900 mb-2">
										Resource Usage
									</h3>
									<div className="grid grid-cols-2 md:grid-cols-4 gap-4">
										<div className="bg-gray-50 p-3 rounded">
											<div className="text-xs text-gray-600">CPU</div>
											<div className="text-lg font-bold text-gray-900">
												{stats.cpu_percent.toFixed(2)}%
											</div>
										</div>
										<div className="bg-gray-50 p-3 rounded">
											<div className="text-xs text-gray-600">Memory</div>
											<div className="text-lg font-bold text-gray-900">
												{stats.memory_percent.toFixed(2)}%
											</div>
											<div className="text-xs text-gray-600">
												{stats.memory_usage} / {stats.memory_limit}
											</div>
										</div>
										<div className="bg-gray-50 p-3 rounded">
											<div className="text-xs text-gray-600">Network RX</div>
											<div className="text-lg font-bold text-gray-900">
												{stats.network_rx}
											</div>
										</div>
										<div className="bg-gray-50 p-3 rounded">
											<div className="text-xs text-gray-600">Network TX</div>
											<div className="text-lg font-bold text-gray-900">
												{stats.network_tx}
											</div>
										</div>
									</div>
								</div>
							)}

							{/* Container Details */}
							<div>
								<h3 className="text-lg font-semibold text-gray-900 mb-2">
									Container Information
								</h3>
								<div className="bg-gray-50 p-4 rounded space-y-2 text-sm">
									<div>
										<span className="font-medium text-gray-700">Status:</span>{" "}
										<span className={getStatusColor(selectedContainerData.status)}>
											{selectedContainerData.status}
										</span>
									</div>
									<div>
										<span className="font-medium text-gray-700">State:</span>{" "}
										{selectedContainerData.state}
									</div>
									<div>
										<span className="font-medium text-gray-700">Image:</span>{" "}
										{selectedContainerData.image}
									</div>
									<div>
										<span className="font-medium text-gray-700">Created:</span>{" "}
										{formatDate(selectedContainerData.created)}
									</div>
									{selectedContainerData.started_at && (
										<div>
											<span className="font-medium text-gray-700">Started:</span>{" "}
											{formatDate(selectedContainerData.started_at)}
										</div>
									)}
									{selectedContainerData.finished_at && (
										<div>
											<span className="font-medium text-gray-700">Finished:</span>{" "}
											{formatDate(selectedContainerData.finished_at)}
										</div>
									)}
									{Object.keys(selectedContainerData.labels).length > 0 && (
										<div>
											<span className="font-medium text-gray-700">Labels:</span>
											<div className="mt-1 space-y-1">
												{Object.entries(selectedContainerData.labels).map(
													([key, value]) => (
														<div key={key} className="ml-4 text-xs">
															<span className="text-gray-600">{key}:</span> {value}
														</div>
													)
												)}
											</div>
										</div>
									)}
								</div>
							</div>

							{/* Logs */}
							{showLogs && (
								<div>
									<div className="flex items-center justify-between mb-2">
										<h3 className="text-lg font-semibold text-gray-900">
											Container Logs
											{isStreaming && (
												<span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">
													<span className="w-2 h-2 bg-green-500 rounded-full mr-1 animate-pulse"></span>
													Live
												</span>
											)}
										</h3>
										<div className="flex items-center gap-2">
											<label className="flex items-center text-sm text-gray-600">
												<input
													type="checkbox"
													checked={autoScroll}
													onChange={(e) => setAutoScroll(e.target.checked)}
													className="mr-1 rounded"
												/>
												Auto-scroll
											</label>
											<button
												onClick={toggleStreaming}
												className={`px-3 py-1 text-sm rounded ${
													isStreaming
														? "bg-yellow-600 hover:bg-yellow-700 text-white"
														: "bg-blue-600 hover:bg-blue-700 text-white"
												}`}
											>
												{isStreaming ? "Stop Streaming" : "Start Streaming"}
											</button>
										</div>
									</div>
									<div className="bg-gray-900 text-gray-100 p-4 rounded font-mono text-xs overflow-auto max-h-96">
										{logsLoading && !isStreaming ? (
											<div className="text-gray-400">Loading logs...</div>
										) : (
											<>
												<pre className="whitespace-pre-wrap">
													{isStreaming
														? streamLogs.join("\n") || "Waiting for logs..."
														: logs?.logs || "No logs available"}
												</pre>
												<div ref={logEndRef} />
											</>
										)}
									</div>
								</div>
							)}
						</div>

						{/* Modal Footer */}
						<div className="p-4 border-t border-gray-200 flex justify-end gap-2">
							<button
								onClick={handleCloseDetails}
								className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
							>
								Close
							</button>
						</div>
					</div>
				</div>
			)}
		</div>
	);
}
