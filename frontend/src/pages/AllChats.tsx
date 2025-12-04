/**
 * All Chats Page
 * Full-page view of all chat sessions with search/filter functionality.
 */

import { useState } from "react";
import ChatHistory from "../components/ChatHistory";
import { Search } from "lucide-react";

interface AllChatsProps {
	onNavigate: (sessionId: string) => void;
}

export default function AllChats({ onNavigate }: AllChatsProps) {
	const [searchQuery, setSearchQuery] = useState("");

	return (
		<div className="p-6 max-w-4xl mx-auto">
			<h1 className="text-2xl font-bold mb-6 text-gray-900 dark:text-gray-100">
				All Chats
			</h1>

			{/* Search bar */}
			<div className="mb-6 relative">
				<div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
					<Search className="h-5 w-5 text-gray-400" />
				</div>
				<input
					type="text"
					placeholder="Search chats..."
					value={searchQuery}
					onChange={(e) => setSearchQuery(e.target.value)}
					className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100"
				/>
			</div>

			{/* Chat list - unlimited */}
			<div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-4">
				<ChatHistory
					limit={undefined} // Load all sessions
					onSelectSession={onNavigate}
				/>
			</div>

			<div className="mt-4 text-sm text-gray-500 dark:text-gray-400 text-center">
				All chat sessions are stored locally in your browser
			</div>
		</div>
	);
}
