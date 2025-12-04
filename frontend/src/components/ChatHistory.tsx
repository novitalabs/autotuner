/**
 * Chat History Component
 * Displays recent chat sessions from IndexedDB.
 * Clean, minimal design matching the reference UI.
 */

import { useState, useEffect } from "react";
import { getChatStorage, type SessionData } from "../services/chatStorage";
import { Trash2 } from "lucide-react";

interface ChatHistoryProps {
	limit?: number; // Default 5 for sidebar, unlimited for AllChats page
	onSelectSession: (sessionId: string) => void;
	activeSessionId?: string; // Currently active session to highlight
}

export default function ChatHistory({ limit = 5, onSelectSession, activeSessionId }: ChatHistoryProps) {
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
			<div className="space-y-2 px-3 py-2">
				<div className="animate-pulse space-y-2">
					<div className="h-8 bg-gray-200 rounded"></div>
					<div className="h-8 bg-gray-200 rounded"></div>
					<div className="h-8 bg-gray-200 rounded"></div>
				</div>
			</div>
		);
	}

	if (sessions.length === 0) {
		return (
			<div className="px-3 py-4 text-center text-gray-500 text-sm">
				No chats yet
			</div>
		);
	}

	return (
		<div className="space-y-1">
			{sessions.map((session) => {
				const isActive = session.session_id === activeSessionId;

				return (
					<div
						key={session.session_id}
						onClick={() => onSelectSession(session.session_id)}
						onMouseEnter={() => setHoveredSession(session.session_id)}
						onMouseLeave={() => setHoveredSession(null)}
						className={`relative group cursor-pointer px-3 py-2 rounded text-sm transition-colors ${
							isActive
								? 'bg-blue-50 text-blue-700 font-medium'
								: 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
						}`}
					>
						<div className="flex items-center justify-between gap-2">
							<div className="flex-1 truncate">
								{session.last_message_preview || "New conversation"}
							</div>

							{hoveredSession === session.session_id && !isActive && (
								<button
									onClick={(e) => handleDelete(session.session_id, e)}
									className="flex-shrink-0 p-1 hover:bg-red-100 rounded transition-colors opacity-0 group-hover:opacity-100"
									title="Delete chat"
								>
									<Trash2 className="w-3.5 h-3.5 text-red-600" />
								</button>
							)}
						</div>
					</div>
				);
			})}
		</div>
	);
}
