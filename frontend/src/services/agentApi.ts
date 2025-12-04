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
	getStatus: (): Promise<AgentStatus> => apiClient.get("/agent/status"),

	// Session management
	createSession: (data?: ChatSessionCreateRequest): Promise<ChatSession> =>
		apiClient.post("/agent/sessions", data || {}),

	getSession: (sessionId: string): Promise<ChatSession> =>
		apiClient.get(`/agent/sessions/${sessionId}`),

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
