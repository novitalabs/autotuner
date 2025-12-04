/**
 * New Chat Page
 * Generates a session ID and navigates to the chat view.
 * Session is only created in IndexedDB when first message is sent.
 */

import { useEffect } from "react";

interface NewChatPageProps {
	onNavigate: (sessionId: string) => void;
}

export default function NewChatPage({ onNavigate }: NewChatPageProps) {
	useEffect(() => {
		// Generate UUID and navigate immediately
		// Session will be created in IndexedDB when first message is sent
		const sessionId = crypto.randomUUID();
		onNavigate(sessionId);
	}, [onNavigate]);

	return (
		<div className="flex items-center justify-center h-full">
			<div className="text-center">
				<div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
				<p className="mt-4 text-gray-600 dark:text-gray-400">
					Starting new chat...
				</p>
			</div>
		</div>
	);
}
