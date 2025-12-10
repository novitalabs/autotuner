/**
 * API service for agent chat functionality.
 */

import apiClient from "./api";
import type {
	AgentStatus,
	ChatSession,
	ChatMessage,
	AgentEventSubscription,
	ChatSessionCreateRequest,
	ChatMessageCreateRequest,
	AgentEventSubscriptionCreateRequest,
	SessionSyncRequest,
	SessionListItem,
	AuthorizationResponse,
} from "../types/agent";

export const agentApi = {
	// Status check
	getStatus: (): Promise<AgentStatus> => apiClient.get("/agent/status"),

	// Session management
	createSession: (data?: ChatSessionCreateRequest): Promise<ChatSession> =>
		apiClient.post("/agent/sessions", data || {}),

	getSession: (sessionId: string): Promise<ChatSession> =>
		apiClient.get(`/agent/sessions/${sessionId}`),

	listSessions: (limit?: number): Promise<SessionListItem[]> =>
		apiClient.get("/agent/sessions", {
			params: { limit: limit || 50 },
		}),

	syncSession: (data: SessionSyncRequest): Promise<{ status: string; session_id: string; message_count?: number }> =>
		apiClient.post("/agent/sessions/sync", data),

	// Title management
	generateTitle: (sessionId: string): Promise<{ title: string }> =>
		apiClient.post(`/agent/sessions/${sessionId}/title/generate`),

	updateTitle: (sessionId: string, title: string): Promise<ChatSession> =>
		apiClient.patch(`/agent/sessions/${sessionId}/title`, { title }),

	// Message management
	getMessages: (sessionId: string, limit?: number): Promise<ChatMessage[]> =>
		apiClient.get(`/agent/sessions/${sessionId}/messages`, {
			params: { limit: limit || 50 },
		}),

	sendMessage: (
		sessionId: string,
		data: ChatMessageCreateRequest
	): Promise<ChatMessage> =>
		apiClient.post(`/agent/sessions/${sessionId}/messages`, data),

	sendMessageStream: (
		sessionId: string,
		data: ChatMessageCreateRequest,
		onChunk: (chunk: {
			type: "content" | "tool_start" | "tool_results" | "final_response_start" | "complete" | "error" | "iteration_start" | "iteration_complete";
			content?: string;
			tool_calls?: any[];
			results?: any[];
			message?: ChatMessage;
			error?: string;
			iteration?: number;
			max_iterations?: number;
			tool_calls_count?: number;
		}) => void
	) => {
		// Use native EventSource for SSE
		const baseURL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
		const url = `${baseURL}/api/agent/sessions/${sessionId}/messages/stream`;

		return new Promise<ChatMessage>((resolve, reject) => {
			// Use fetch with streaming instead of EventSource for POST
			fetch(url, {
				method: "POST",
				headers: {
					"Content-Type": "application/json",
				},
				body: JSON.stringify(data),
			})
				.then((response) => {
					if (!response.ok) {
						throw new Error(`HTTP error! status: ${response.status}`);
					}

					const reader = response.body?.getReader();
					if (!reader) {
						throw new Error("No reader available");
					}

					const decoder = new TextDecoder();
					let buffer = "";

					function processChunk(): any {
						return reader!.read().then(({ done, value }) => {
							if (done) {
								return;
							}

							buffer += decoder.decode(value, { stream: true });
							const lines = buffer.split("\n");
							buffer = lines.pop() || "";

							for (const line of lines) {
								if (line.startsWith("data: ")) {
									try {
										const data = JSON.parse(line.slice(6));
										onChunk(data);

										if (data.type === "complete") {
											resolve(data.message);
											return;
										} else if (data.type === "error") {
											console.error("[agentApi] Error:", data.error);
											reject(new Error(data.error));
											return;
										}
									} catch (e) {
										console.error("Failed to parse SSE data:", line, e);
									}
								}
							}

							return processChunk();
						});
					}

					return processChunk();
				})
				.catch((error) => {
					console.error("[agentApi] Stream error:", error);
					reject(error);
				});
		});
	},

	// Event subscriptions
	subscribeToTask: (
		sessionId: string,
		data: AgentEventSubscriptionCreateRequest
	): Promise<AgentEventSubscription> =>
		apiClient.post(`/agent/sessions/${sessionId}/subscribe`, data),

	unsubscribeFromTask: (
		sessionId: string,
		taskId: number
	): Promise<{ message: string }> =>
		apiClient.delete(`/agent/sessions/${sessionId}/subscribe/${taskId}`),

	getSubscriptions: (sessionId: string): Promise<AgentEventSubscription[]> =>
		apiClient.get(`/agent/sessions/${sessionId}/subscriptions`),

	// Authorization
	grantAuthorization: (
		sessionId: string,
		scopes: string[],
		expiresAt?: string
	): Promise<AuthorizationResponse> =>
		apiClient.post(`/agent/sessions/${sessionId}/authorize`, {
			scopes,
			expires_at: expiresAt,
		}),

	revokeAuthorization: (
		sessionId: string,
		scopes: string[]
	): Promise<AuthorizationResponse> =>
		apiClient.post(`/agent/sessions/${sessionId}/revoke`, { scopes }),
};

export default agentApi;
