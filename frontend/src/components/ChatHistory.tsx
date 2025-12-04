/**
 * Chat History Component
 * Displays recent chat sessions from IndexedDB.
 * Can be rendered inline in sidebar or as full-page list.
 */

import { useState, useEffect } from "react";
import { formatDistanceToNow } from "date-fns";
import { getChatStorage, type SessionData } from "../services/chatStorage";
import { Trash2 } from "lucide-react";

interface ChatHistoryProps {
	limit?: number; // Default 5 for sidebar, unlimited for AllChats page
	onSelectSession: (sessionId: string) => void;
}

export default function ChatHistory({ limit = 5, onSelectSession }: ChatHistoryProps) {
	const [sessions, setSessions] = useState<SessionData[]>([]);
	const [loading, setLoading] = useState(true);
	const [hoveredSession, setHoveredSession] = useState<string | null>(null);

	const loadSessions = async () => {
		try {
			const storage = getChatStorage();
			const sessions = await storage.listSessions(limit);
			setSessions(sessions);
		} catch (error) {
			console.error("Failed to load sessions:", error);
		} finally {
			setLoading(false);
		}
	};

	useEffect(() => {
		loadSessions();

		// Refresh sessions periodically
		const interval = setInterval(loadSessions, 5000);
		return () => clearInterval(interval);
	}, [limit]);

	const handleDelete = async (sessionId: string, event: React.MouseEvent) => {
		event.stopPropagation(); // Prevent session selection

		if (confirm("Delete this chat session?")) {
			try {
				const storage = getChatStorage();
				await storage.deleteSession(sessionId);
				await loadSessions(); // Refresh list
			} catch (error) {
				console.error("Failed to delete session:", error);
			}
		}
	};

	if (loading) {
		return (
			<div className="space-y-2 p-2">
				<div className="animate-pulse">
					<div className="h-12 bg-gray-200 rounded mb-2"></div>
					<div className="h-12 bg-gray-200 rounded mb-2"></div>
					<div className="h-12 bg-gray-200 rounded"></div>
				</div>
			</div>
		);
	}

	if (sessions.length === 0) {
		return (
			<div className="p-4 text-center text-gray-500 text-sm">
				No chat sessions yet. Start a new chat to begin!
			</div>
		);
	}

	return (
		<div className="space-y-2">
			{sessions.map((session) => (
				<div
					key={session.session_id}
					onClick={() => onSelectSession(session.session_id)}
					onMouseEnter={() => setHoveredSession(session.session_id)}
					onMouseLeave={() => setHoveredSession(null)}
					className="relative cursor-pointer p-3 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg text-sm transition-colors border border-transparent hover:border-gray-300 dark:hover:border-gray-600"
				>
					<div className="flex justify-between items-start">
						<div className="flex-1 min-w-0">
							<div className="font-medium truncate text-gray-900 dark:text-gray-100">
								{session.last_message_preview || "New conversation"}
							</div>
							<div className="text-xs text-gray-500 dark:text-gray-400 mt-1 flex items-center gap-2">
								<span>
									{formatDistanceToNow(new Date(session.updated_at), {
										addSuffix: true,
									})}
								</span>
								<span>•</span>
								<span>{session.message_count} messages</span>
							</div>
						</div>

						{hoveredSession === session.session_id && (
							<button
								onClick={(e) => handleDelete(session.session_id, e)}
								className="ml-2 p-1 hover:bg-red-100 dark:hover:bg-red-900 rounded transition-colors"
								title="Delete chat"
							>
								<Trash2 className="w-4 h-4 text-red-600 dark:text-red-400" />
							</button>
						)}
					</div>

					{!session.synced_to_backend && (
						<div className="absolute top-1 right-1 w-2 h-2 bg-blue-500 rounded-full" title="Not synced to backend"></div>
					)}
				</div>
			))}
		</div>
	);
}
