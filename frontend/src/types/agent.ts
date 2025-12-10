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

export interface ToolCall {
	tool_name: string;
	args: Record<string, any>;
	id: string;
	status: "executing" | "executed" | "requires_auth" | "failed";
	result?: string;
	auth_scope?: string;
	error?: string;
}

export interface ChatMessage {
	id: number;
	session_id: string;
	role: "user" | "assistant" | "system";
	content: string;
	tool_calls: ToolCall[] | null;
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

// Session sync types
export interface MessageSync {
	role: string;
	content: string;
	created_at: string;
}

export interface SessionSyncRequest {
	session_id: string;
	created_at: string;
	messages: MessageSync[];
}

export interface SessionListItem {
	session_id: string;
	created_at: string;
	updated_at: string;
	last_message_preview: string;
	message_count: number;
}

// Authorization types
export interface ToolAuthorizationRequest {
	scopes: string[];
	expires_at?: string;
}

export interface ToolExecutionResult {
	success: boolean;
	result: string;
	tool_name: string;
	call_id?: string;
	requires_auth?: boolean;
	auth_scope?: string;
	authorized?: boolean;
}

export interface AuthorizationResponse {
	status: "granted" | "revoked";
	scopes: string[];
	tool_results?: ToolExecutionResult[];  // Results of pending tool calls executed after authorization
}

// Unified iteration display type
export interface IterationBlock {
	iteration: number;          // 1-based iteration number
	content: string;            // Text content for this iteration
	toolCalls: ToolCall[];      // Tool calls WITH results attached
	status: 'streaming' | 'complete';
}
