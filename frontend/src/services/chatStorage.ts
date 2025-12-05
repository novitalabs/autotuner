/**
 * IndexedDB storage service for chat sessions and messages.
 * Provides offline-first storage with background sync to backend.
 */

import { openDB, type DBSchema, type IDBPDatabase } from 'idb';

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

		// Handle legacy sessions without title field
		sessions.forEach(session => {
			if (!('title' in session)) {
				(session as any).title = '';
			}
		});

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
		const messageWithId: Required<MessageData> = {
			id: message.id || `${sessionId}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
			session_id: sessionId,
			role: message.role,
			content: message.content,
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

		// Sort by ID (which contains timestamp) to ensure insertion order
		// ID format: sessionId-timestamp-random
		messages.sort((a, b) => {
			// Extract timestamp from ID
			const getTimestamp = (id: string) => {
				const parts = id.split('-');
				return parseInt(parts[parts.length - 2]) || 0;
			};
			return getTimestamp(a.id) - getTimestamp(b.id);
		});

		if (limit) {
			return messages.slice(-limit); // Get last N messages
		}

		return messages;
	}

	async getUnsyncedMessages(sessionId: string): Promise<MessageData[]> {
		const db = await this.dbPromise;
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
