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
};

export default agentApi;
