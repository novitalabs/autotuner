import { useCallback } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useWebSocket, WebSocketEvent } from "./useWebSocket";

/**
 * Hook for managing WebSocket connection to a specific task
 * Automatically invalidates React Query caches when events are received
 *
 * @param taskId - Task ID to monitor (null to disable connection)
 * @param enabled - Whether to enable the WebSocket connection (default: true)
 * @returns WebSocket state and controls
 *
 * @example
 * ```tsx
 * const { isConnected, lastMessage } = useTaskWebSocket(taskId);
 * ```
 */
export function useTaskWebSocket(taskId: number | null, enabled: boolean = true) {
	const queryClient = useQueryClient();

	// Get WebSocket URL (or null if taskId is null or disabled)
	const wsUrl = taskId && enabled
		? `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/api/ws/tasks/${taskId}`
		: null;

	// Handle incoming WebSocket messages
	const handleMessage = useCallback(
		(event: WebSocketEvent) => {
			console.log("[useTaskWebSocket] Received event:", event);

			// Invalidate relevant React Query caches based on event type
			// Use precise invalidation to minimize unnecessary refetches
			switch (event.type) {
				case "task_started":
				case "task_completed":
				case "task_failed":
					// Task-level changes: invalidate task list and specific task
					queryClient.invalidateQueries({ queryKey: ["tasks"] });
					if (taskId) {
						queryClient.invalidateQueries({ queryKey: ["task", taskId] });
					}
					break;

				case "task_progress":
					// Progress updates: only invalidate specific task (not full list)
					// This reduces refetches since task list doesn't need full refresh
					if (taskId) {
						queryClient.invalidateQueries({ queryKey: ["task", taskId] });
					}
					break;

				case "experiment_started":
				case "benchmark_started":
				case "benchmark_progress":
					// Experiment started/progress: only invalidate experiments
					// Don't refetch task list for intermediate states
					if (taskId) {
						queryClient.invalidateQueries({ queryKey: ["experiments", taskId] });
					}
					break;

				case "experiment_completed":
				case "experiment_failed":
					// Experiment completed: invalidate experiments + task (for stats)
					if (taskId) {
						queryClient.invalidateQueries({ queryKey: ["experiments", taskId] });
						queryClient.invalidateQueries({ queryKey: ["task", taskId] });
					}
					break;

				case "experiment_progress":
					// Fine-grained progress: only invalidate experiments
					if (taskId) {
						queryClient.invalidateQueries({ queryKey: ["experiments", taskId] });
					}
					break;

				case "connection_established":
					console.log("[useTaskWebSocket] Connected to task", taskId);
					// Initial connection: fetch fresh data
					queryClient.invalidateQueries({ queryKey: ["tasks"] });
					if (taskId) {
						queryClient.invalidateQueries({ queryKey: ["task", taskId] });
						queryClient.invalidateQueries({ queryKey: ["experiments", taskId] });
					}
					break;

				default:
					console.warn("[useTaskWebSocket] Unknown event type:", event.type);
			}
		},
		[queryClient, taskId]
	);

	// Use base WebSocket hook with task-specific configuration
	const websocket = useWebSocket(wsUrl, {
		autoConnect: true,
		autoReconnect: true,
		reconnectDelay: 1000,
		maxReconnectDelay: 10000,
		maxReconnectAttempts: 10,
		onMessage: handleMessage,
		onOpen: () => {
			console.log(`[useTaskWebSocket] Connected to task ${taskId}`);
		},
		onClose: () => {
			console.log(`[useTaskWebSocket] Disconnected from task ${taskId}`);
		},
		onError: (error) => {
			console.error(`[useTaskWebSocket] Error for task ${taskId}:`, error);
		},
	});

	return websocket;
}
