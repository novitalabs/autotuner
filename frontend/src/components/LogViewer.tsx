import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "@/services/api";
import { useEscapeKey } from "@/hooks/useEscapeKey";

interface LogViewerProps {
	taskId: number;
	taskName: string;
	onClose: () => void;
}

export default function LogViewer({ taskId, taskName, onClose }: LogViewerProps) {
	// Handle Escape key to close modal
	useEscapeKey(onClose);
	const [autoScroll, setAutoScroll] = useState(true);
	const [isStreaming, setIsStreaming] = useState(false); // Will be set to true after initial load
	const [streamLogs, setStreamLogs] = useState<string[]>([]);
	const [initialLoadDone, setInitialLoadDone] = useState(false);
	const logContainerRef = useRef<HTMLDivElement>(null);
	const eventSourceRef = useRef<EventSource | null>(null);
	const queryClient = useQueryClient();

	// Fetch static logs (initially enabled to load existing logs)
	const {
		data: logData,
		isLoading,
		error
	} = useQuery({
		queryKey: ["taskLogs", taskId],
		queryFn: () => apiClient.getTaskLogs(taskId),
		enabled: !isStreaming,
		refetchInterval: isStreaming ? false : 2000 // Refresh every 2s when not streaming
	});

	// Clear logs mutation
	const clearLogsMutation = useMutation({
		mutationFn: () => apiClient.clearTaskLogs(taskId),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ["taskLogs", taskId] });
			setStreamLogs([]);
		}
	});

	// Auto-scroll effect
	useEffect(() => {
		if (autoScroll && logContainerRef.current) {
			logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
		}
	}, [streamLogs, logData, autoScroll]);

	// Handle streaming
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
			if (logData?.logs && streamLogs.length === 0) {
				// If we have static logs but empty streamLogs, initialize from static logs
				const existingLogs = logData.logs.split('\n').filter(Boolean);
				setStreamLogs(existingLogs);
			}
			setIsStreaming(true);

			const apiUrl = import.meta.env.VITE_API_URL || "/api";
			const eventSource = new EventSource(`${apiUrl}/tasks/${taskId}/logs?follow=true`);

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

	// Cleanup on unmount
	useEffect(() => {
		return () => {
			if (eventSourceRef.current) {
				eventSourceRef.current.close();
			}
		};
	}, []);

	// Auto-start streaming on mount after initial logs are loaded
	useEffect(() => {
		if (!initialLoadDone && logData && !isLoading) {
			// Initialize streamLogs with existing logs
			const existingLogs = logData.logs ? logData.logs.split('\n').filter(Boolean) : [];
			setStreamLogs(existingLogs);
			setInitialLoadDone(true);

			// Start streaming
			setIsStreaming(true);

			const apiUrl = import.meta.env.VITE_API_URL || "/api";
			const eventSource = new EventSource(`${apiUrl}/tasks/${taskId}/logs?follow=true`);

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
	}, [logData, isLoading, initialLoadDone, taskId]); // Dependencies to track initial load

	const displayLogs = isStreaming
		? streamLogs.join("\n")
		: logData?.logs || "No logs available yet.";

	const copyToClipboard = () => {
		navigator.clipboard.writeText(displayLogs);
	};

	const downloadLogs = () => {
		const blob = new Blob([displayLogs], { type: "text/plain" });
		const url = URL.createObjectURL(blob);
		const a = document.createElement("a");
		a.href = url;
		a.download = `task_${taskId}_${taskName}.log`;
		document.body.appendChild(a);
		a.click();
		document.body.removeChild(a);
		URL.revokeObjectURL(url);
	};

	return (
		<div className="fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center p-4 z-50">
			<div className="bg-white rounded-lg shadow-xl w-full max-w-6xl h-[90vh] flex flex-col">
				{/* Header */}
				<div className="px-6 py-4 border-b border-gray-200 flex items-center justify-between">
					<div>
						<h2 className="text-xl font-bold text-gray-900">Task Logs</h2>
						<p className="text-sm text-gray-500 mt-1">
							{taskName} (Task #{taskId})
						</p>
					</div>
					<button onClick={onClose} className="text-gray-400 hover:text-gray-500">
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

				{/* Toolbar */}
				<div className="px-6 py-3 border-b border-gray-200 flex items-center justify-between bg-gray-50">
					<div className="flex items-center gap-2">
						<button
							onClick={toggleStreaming}
							className={`inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-md ${
								isStreaming
									? "bg-red-100 text-red-700 hover:bg-red-200"
									: "bg-blue-100 text-blue-700 hover:bg-blue-200"
							}`}
						>
							{isStreaming ? (
								<>
									<svg
										className="w-4 h-4 mr-1.5"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
										/>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
										/>
									</svg>
									Stop Streaming
								</>
							) : (
								<>
									<svg
										className="w-4 h-4 mr-1.5"
										fill="none"
										viewBox="0 0 24 24"
										stroke="currentColor"
									>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"
										/>
										<path
											strokeLinecap="round"
											strokeLinejoin="round"
											strokeWidth={2}
											d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
										/>
									</svg>
									Start Streaming
								</>
							)}
						</button>

						<label className="inline-flex items-center">
							<input
								type="checkbox"
								checked={autoScroll}
								onChange={(e) => setAutoScroll(e.target.checked)}
								className="rounded border-gray-300 text-blue-600 focus:ring-emerald-500"
							/>
							<span className="ml-2 text-sm text-gray-700">Auto-scroll</span>
						</label>

						{isStreaming && (
							<span className="inline-flex items-center text-sm text-green-600">
								<span className="animate-pulse mr-1.5">●</span>
								Live
							</span>
						)}
					</div>

					<div className="flex items-center gap-2">
						<button
							onClick={copyToClipboard}
							className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
						>
							<svg
								className="w-4 h-4 mr-1.5"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth={2}
									d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
								/>
							</svg>
							Copy
						</button>

						<button
							onClick={downloadLogs}
							className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
						>
							<svg
								className="w-4 h-4 mr-1.5"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth={2}
									d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
								/>
							</svg>
							Download
						</button>

						<button
							onClick={() => clearLogsMutation.mutate()}
							disabled={clearLogsMutation.isPending}
							className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-red-700 bg-white border border-red-300 rounded-md hover:bg-red-50 disabled:opacity-50"
						>
							<svg
								className="w-4 h-4 mr-1.5"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<path
									strokeLinecap="round"
									strokeLinejoin="round"
									strokeWidth={2}
									d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
								/>
							</svg>
							Clear
						</button>
					</div>
				</div>

				{/* Log Content */}
				<div className="flex-1 overflow-hidden px-6 py-4">
					{isLoading ? (
						<div className="flex items-center justify-center h-full">
							<div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent"></div>
							<p className="ml-3 text-sm text-gray-600">Loading logs...</p>
						</div>
					) : error ? (
						<div className="rounded-md bg-red-50 p-4">
							<p className="text-sm text-red-800">
								Error loading logs: {(error as Error).message}
							</p>
						</div>
					) : (
						<div
							ref={logContainerRef}
							className="h-full overflow-auto bg-gray-900 rounded-lg p-4 font-mono text-sm text-green-400"
						>
							<pre className="whitespace-pre-wrap break-words">{displayLogs}</pre>
						</div>
					)}
				</div>

				{/* Footer */}
				<div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
					<div className="flex items-center justify-between text-sm text-gray-500">
						<span>
							{isStreaming
								? `${streamLogs.length} lines (streaming)`
								: `${displayLogs.split("\n").length} lines`}
						</span>
						<button
							onClick={onClose}
							className="px-4 py-2 bg-gray-200 text-gray-700 rounded-md hover:bg-gray-300"
						>
							Close
						</button>
					</div>
				</div>
			</div>
		</div>
	);
}
