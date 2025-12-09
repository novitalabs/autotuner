/**
 * IndexedDB storage service for chat sessions and messages.
 * Provides offline-first storage with background sync to backend.
 */

import { openDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { ToolCall } from '../types/agent';

// Database schema types
interface ChatDB extends DBSchema {
	sessions: {
		key: string; // session_id UUID
		value: {
			session_id: string;
			created_at: string;
			updated_at: string;
			last_message_preview: string;
			message_count: number;
			synced_to_backend: boolean;
		};
		indexes: { 'by-updated': string };
	};
	messages: {
		key: string; // composite: session_id + timestamp
		value: {
			id: string; // unique key for the message
			session_id: string;
			role: 'user' | 'assistant' | 'system';
			content: string;
			tool_calls?: ToolCall[];
			created_at: string;
			synced_to_backend: boolean;
		};
		indexes: { 'by-session': string; 'by-created': string };
	};
}

// Data types
export interface SessionData {
	session_id: string;
	created_at: string;
	updated_at: string;
	title: string;
	last_message_preview: string;
	message_count: number;
	synced_to_backend: boolean;
}

// Re-export for convenience
export type { SessionData as Session };

export interface MessageData {
	id?: string;
	session_id: string;
	role: 'user' | 'assistant' | 'system';
	content: string;
	tool_calls?: ToolCall[];
	metadata?: {
		iteration_data?: Array<{
			iteration: number;
			content: string;
			tool_calls: ToolCall[];
		}>;
		iterations?: number;
		termination_reason?: string;
	};
	created_at: string;
	synced_to_backend: boolean;
}

class ChatStorageService {
	private dbPromise: Promise<IDBPDatabase<ChatDB>>;
	private readonly DB_NAME = 'inference-autotuner-chats';
	private readonly DB_VERSION = 1;

	constructor() {
		this.dbPromise = this.initDB();
	}

	private async initDB(): Promise<IDBPDatabase<ChatDB>> {
		return openDB<ChatDB>(this.DB_NAME, this.DB_VERSION, {
			upgrade(db) {
				// Create sessions store
				if (!db.objectStoreNames.contains('sessions')) {
					const sessionStore = db.createObjectStore('sessions', {
						keyPath: 'session_id',
					});
					sessionStore.createIndex('by-updated', 'updated_at');
				}

				// Create messages store
				if (!db.objectStoreNames.contains('messages')) {
					const messageStore = db.createObjectStore('messages', {
						keyPath: 'id',
					});
					messageStore.createIndex('by-session', 'session_id');
					messageStore.createIndex('by-created', 'created_at');
				}
			},
		});
	}

	// Session management methods

	async createSession(sessionId: string): Promise<void> {
		const db = await this.dbPromise;
		const now = new Date().toISOString();

		const session: SessionData = {
			session_id: sessionId,
			created_at: now,
			updated_at: now,
			title: '',
			last_message_preview: '',
			message_count: 0,
			synced_to_backend: false,
		};

		await db.add('sessions', session);
	}

	async getSession(sessionId: string): Promise<SessionData | undefined> {
		const db = await this.dbPromise;
		const session = await db.get('sessions', sessionId);

		// Handle legacy sessions without title field
		if (session && !('title' in session)) {
			(session as any).title = '';
		}

		return session as SessionData | undefined;
	}

	async listSessions(limit?: number): Promise<SessionData[]> {
		const db = await this.dbPromise;
		const index = db.transaction('sessions').store.index('by-updated');

		// Get all sessions sorted by updated_at descending
		const sessions = await index.getAll();
		sessions.reverse(); // Most recent first

		// Handle legacy sessions without title field and auto-generate from first user message
		for (const session of sessions) {
			if (!('title' in session)) {
				(session as any).title = '';
			}

			// If no title, try to use first user message as fallback
			if (!(session as any).title) {
				const messages = await this.getMessages(session.session_id, 10); // Get first 10 messages
				const firstUserMessage = messages.find(m => m.role === 'user');
				if (firstUserMessage) {
					// Use first 50 chars of first user message as temporary title
					(session as any).title = firstUserMessage.content.substring(0, 50).trim() +
						(firstUserMessage.content.length > 50 ? '...' : '');
				}
			}
		}

		if (limit) {
			return sessions.slice(0, limit) as SessionData[];
		}

		return sessions as SessionData[];
	}

	async deleteSession(sessionId: string): Promise<void> {
		const db = await this.dbPromise;
		const tx = db.transaction(['sessions', 'messages'], 'readwrite');

		// Delete session
		await tx.objectStore('sessions').delete(sessionId);

		// Delete all messages for this session
		const messageIndex = tx.objectStore('messages').index('by-session');
		const messages = await messageIndex.getAll(sessionId);

		for (const message of messages) {
			await tx.objectStore('messages').delete(message.id);
		}

		await tx.done;
	}

	async updateSessionTimestamp(sessionId: string): Promise<void> {
		const db = await this.dbPromise;
		const session = await db.get('sessions', sessionId);

		if (session) {
			session.updated_at = new Date().toISOString();
			await db.put('sessions', session);
		}
	}

	async updateSessionTitle(sessionId: string, title: string): Promise<void> {
		const db = await this.dbPromise;
		const session = await db.get('sessions', sessionId);

		if (session) {
			(session as any).title = title;
			session.updated_at = new Date().toISOString();
			await db.put('sessions', session);
		}
	}

	async markSessionSynced(sessionId: string): Promise<void> {
		const db = await this.dbPromise;
		const session = await db.get('sessions', sessionId);

		if (session) {
			session.synced_to_backend = true;
			await db.put('sessions', session);
		}
	}

	// Message management methods

	async saveMessage(sessionId: string, message: MessageData): Promise<void> {
		const db = await this.dbPromise;

		// Generate unique ID if not provided
		const messageWithId = {
			id: message.id || `${sessionId}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			session_id: sessionId,
			role: message.role,
			content: message.content,
			tool_calls: message.tool_calls,
			metadata: message.metadata,
			created_at: message.created_at,
			synced_to_backend: message.synced_to_backend,
		};

		await db.add('messages', messageWithId);

		// Update session metadata
		const session = await db.get('sessions', sessionId);
		if (session) {
			session.updated_at = message.created_at;
			session.message_count++;

			// Update last message preview (first 100 chars) - only use user messages
			if (message.role === 'user') {
				session.last_message_preview = message.content.substring(0, 100);
			}

			await db.put('sessions', session);
		}
	}

	async getMessages(
		sessionId: string,
		limit?: number
	): Promise<MessageData[]> {
		const db = await this.dbPromise;
		const index = db.transaction('messages').store.index('by-session');
		const messages = await index.getAll(sessionId);

		// Sort by created_at timestamp for reliable chronological order
		// This handles both frontend-generated IDs (sessionId-timestamp-random)
		// and backend-provided numeric IDs
		messages.sort((a, b) => {
			// Parse timestamps, handling both ISO formats:
			// - Frontend: "2024-01-15T10:30:00.000Z" (with Z)
			// - Backend:  "2024-01-15T10:30:00.123456" (without Z, but still UTC)
			// Add 'Z' suffix if not present to ensure correct UTC parsing
			const normalizeTimestamp = (ts: string): number => {
				if (!ts) return 0;
				// If timestamp doesn't end with Z or timezone offset, treat as UTC
				const normalized = ts.match(/[Z+-]/) ? ts : ts + 'Z';
				return new Date(normalized).getTime();
			};

			const timeA = normalizeTimestamp(a.created_at);
			const timeB = normalizeTimestamp(b.created_at);

			// If timestamps are the same (or invalid), maintain relative order:
			// user messages should come before assistant messages
			if (timeA === timeB || isNaN(timeA) || isNaN(timeB)) {
				// User messages before assistant messages
				if (a.role === 'user' && b.role === 'assistant') return -1;
				if (a.role === 'assistant' && b.role === 'user') return 1;
				// Fall back to ID comparison for same role (uses insertion order)
				return (a.id || '').localeCompare(b.id || '');
			}

			return timeA - timeB;
		});

		if (limit) {
			return messages.slice(-limit); // Get last N messages
		}

		return messages;
	}

	async getUnsyncedMessages(sessionId: string): Promise<MessageData[]> {
		const messages = await this.getMessages(sessionId);

		return messages.filter((msg) => !msg.synced_to_backend);
	}

	async markMessageSynced(messageId: string): Promise<void> {
		const db = await this.dbPromise;
		const message = await db.get('messages', messageId);

		if (message) {
			message.synced_to_backend = true;
			await db.put('messages', message);
		}
	}

	// Utility methods

	async clearAllData(): Promise<void> {
		const db = await this.dbPromise;
		const tx = db.transaction(['sessions', 'messages'], 'readwrite');

		await tx.objectStore('sessions').clear();
		await tx.objectStore('messages').clear();

		await tx.done;
	}

	async getStorageStats(): Promise<{
		sessions: number;
		messages: number;
	}> {
		const db = await this.dbPromise;
		const sessionCount = await db.count('sessions');
		const messageCount = await db.count('messages');

		return {
			sessions: sessionCount,
			messages: messageCount,
		};
	}
}

// Singleton instance
let chatStorageInstance: ChatStorageService | null = null;

export function getChatStorage(): ChatStorageService {
	if (!chatStorageInstance) {
		chatStorageInstance = new ChatStorageService();
	}
	return chatStorageInstance;
}

export default getChatStorage;
