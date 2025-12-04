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
} from "../types/agent";

export const agentApi = {
	// Status check
	getStatus: (): Promise<AgentStatus> => apiClient.get("/api/agent/status"),

	// Session management
	createSession: (data?: ChatSessionCreateRequest): Promise<ChatSession> =>
		apiClient.post("/api/agent/sessions", data || {}),

	getSession: (sessionId: string): Promise<ChatSession> =>
		apiClient.get(`/api/agent/sessions/${sessionId}`),

	// Message management
	getMessages: (sessionId: string, limit?: number): Promise<ChatMessage[]> =>
		apiClient.get(`/api/agent/sessions/${sessionId}/messages`, {
			params: { limit: limit || 50 },
		}),

	sendMessage: (
		sessionId: string,
		data: ChatMessageCreateRequest
	): Promise<ChatMessage> =>
		apiClient.post(`/api/agent/sessions/${sessionId}/messages`, data),

	// Event subscriptions
	subscribeToTask: (
		sessionId: string,
		data: AgentEventSubscriptionCreateRequest
	): Promise<AgentEventSubscription> =>
		apiClient.post(`/api/agent/sessions/${sessionId}/subscribe`, data),

	unsubscribeFromTask: (
		sessionId: string,
		taskId: number
	): Promise<{ message: string }> =>
		apiClient.delete(`/api/agent/sessions/${sessionId}/subscribe/${taskId}`),

	getSubscriptions: (sessionId: string): Promise<AgentEventSubscription[]> =>
		apiClient.get(`/api/agent/sessions/${sessionId}/subscriptions`),
};

export default agentApi;
