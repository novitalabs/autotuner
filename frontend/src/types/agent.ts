/**
 * TypeScript interfaces for agent chat functionality.
 */

export interface AgentStatus {
	available: boolean;
	provider: string;
	model: string | null;
	missing_config: string[];
	message: string;
}

export interface ChatSession {
	id: number;
	session_id: string;
	user_id: string | null;
	context_summary: string | null;
	is_active: boolean;
	created_at: string;
	updated_at: string;
}

export interface ChatMessage {
	id: number;
	session_id: string;
	role: "user" | "assistant" | "system";
	content: string;
	tool_calls: Record<string, any> | null;
	metadata: Record<string, any> | null;
	token_count: number | null;
	created_at: string;
}

export interface AgentEventSubscription {
	id: number;
	session_id: string;
	task_id: number;
	event_types: string[];
	is_active: boolean;
	created_at: string;
	expires_at: string | null;
}

// Request types
export interface ChatSessionCreateRequest {
	user_id?: string;
}

export interface ChatMessageCreateRequest {
	content: string;
}

export interface AgentEventSubscriptionCreateRequest {
	task_id: number;
	event_types?: string[];
}
