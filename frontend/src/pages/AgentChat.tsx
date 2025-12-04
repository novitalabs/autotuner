/**
 * Agent Chat page - With conversation history system
 * Supports URL parameters, IndexedDB storage, and background sync
 */

import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import agentApi from "../services/agentApi";
import { getChatStorage, type MessageData } from "../services/chatStorage";
import type { ChatMessage } from "../types/agent";

export default function AgentChat() {
	const [sessionId, setSessionId] = useState<string | null>(null);
	const [messages, setMessages] = useState<MessageData[]>([]);
	const [input, setInput] = useState("");
	const messagesEndRef = useRef<HTMLDivElement>(null);
	const queryClient = useQueryClient();
	const [loadingFromDb, setLoadingFromDb] = useState(true);

	// Parse session ID from URL hash
	useEffect(() => {
		const hash = window.location.hash.slice(1); // Remove #
		const params = new URLSearchParams(hash.split("?")[1] || "");
		const sessionParam = params.get("session");

		if (sessionParam) {
			setSessionId(sessionParam);
			loadMessagesFromIndexedDB(sessionParam);
		} else {
			// No session ID in URL - show empty state
			setLoadingFromDb(false);
		}
	}, []);

	// Load messages from IndexedDB (fast initial load)
	const loadMessagesFromIndexedDB = async (sid: string) => {
		try {
			const storage = getChatStorage();
			const msgs = await storage.getMessages(sid);

			// Messages are already sorted by insertion order (ID timestamp)
			setMessages(msgs);

			// Update session timestamp to move it to top of list
			await storage.updateSessionTimestamp(sid);
		} catch (error) {
			console.error("Failed to load messages from IndexedDB:", error);
		} finally {
			setLoadingFromDb(false);
		}
	};

	// Check agent status
	const { data: agentStatus, isLoading: statusLoading } = useQuery({
		queryKey: ["agent-status"],
		queryFn: () => agentApi.getStatus(),
		staleTime: 5000,
		refetchOnMount: true,
	});

	// Send message mutation with IndexedDB sync
	const sendMessageMutation = useMutation({
		mutationFn: async (content: string) => {
			if (!sessionId) throw new Error("No session ID");

			const storage = getChatStorage();
			let session = await storage.getSession(sessionId);

			// Create session in IndexedDB if it doesn't exist (first message)
			if (!session) {
				await storage.createSession(sessionId);
				session = await storage.getSession(sessionId);
			}

			// Sync full session to backend if not synced yet
			if (session && !session.synced_to_backend) {
				const allMessages = await storage.getMessages(sessionId);
				await agentApi.syncSession({
					session_id: sessionId,
					created_at: session.created_at,
					messages: allMessages.map(m => ({
						role: m.role,
						content: m.content,
						created_at: m.created_at,
					})),
				});
				await storage.markSessionSynced(sessionId);
			}

			// Save user message to IndexedDB (optimistic)
			const userMessage: MessageData = {
				session_id: sessionId,
				role: "user",
				content: content,
				created_at: new Date().toISOString(),
				synced_to_backend: false,
			};
			await storage.saveMessage(sessionId, userMessage);

			// Update local state immediately
			setMessages(prev => [...prev, userMessage]);

			// Send to backend
			const response = await agentApi.sendMessage(sessionId, { content });

			// Save assistant response to IndexedDB
			const assistantMessage: MessageData = {
				session_id: sessionId,
				role: "assistant",
				content: response.content,
				created_at: response.created_at,
				synced_to_backend: true,
			};
			await storage.saveMessage(sessionId, assistantMessage);

			// Update session timestamp
			await storage.updateSessionTimestamp(sessionId);

			return { userMessage, assistantMessage };
		},
		onSuccess: (data) => {
			// Update local state with assistant response
			setMessages(prev => [...prev, data.assistantMessage]);
			setInput("");
		},
		onError: (error) => {
			console.error("Failed to send message:", error);
			// TODO: Show error toast
		},
	});

	// Auto-scroll to bottom
	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages]);

	const handleSend = () => {
		if (!input.trim() || !sessionId) return;
		sendMessageMutation.mutate(input);
	};

	const handleKeyPress = (e: React.KeyboardEvent) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	};

	// Loading state
	if (statusLoading || loadingFromDb) {
		return (
			<div className="flex items-center justify-center h-full">
				<div className="text-gray-500">
					{statusLoading ? "Checking agent availability..." : "Loading messages..."}
				</div>
			</div>
		);
	}

	// Agent not available
	if (!agentStatus?.available) {
		return (
			<div className="flex items-center justify-center h-full bg-gray-50">
				<div className="max-w-2xl mx-auto p-8 bg-white rounded-lg shadow-md">
					<div className="text-center">
						<svg
							className="mx-auto h-16 w-16 text-gray-400"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<path
								strokeLinecap="round"
								strokeLinejoin="round"
								strokeWidth={2}
								d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
							/>
						</svg>
						<h3 className="mt-4 text-xl font-semibold text-gray-900">
							Agent Not Available
						</h3>
						<p className="mt-2 text-gray-600">{agentStatus?.message}</p>

						{agentStatus?.missing_config && agentStatus.missing_config.length > 0 && (
							<div className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4">
								<h4 className="text-sm font-semibold text-blue-900 mb-2">
									Configuration Required:
								</h4>
								<ul className="text-sm text-blue-800 space-y-1 text-left">
									{agentStatus.missing_config.map((config, idx) => (
										<li key={idx} className="flex items-start">
											<span className="mr-2">•</span>
											<span>{config}</span>
										</li>
									))}
								</ul>
							</div>
						)}

						<div className="mt-6 bg-gray-50 rounded-lg p-4 text-left">
							<h4 className="text-sm font-semibold text-gray-900 mb-2">
								Setup Instructions:
							</h4>
							<div className="text-sm text-gray-700 space-y-2">
								<p>
									<strong>For local models (vLLM/SGLang):</strong>
								</p>
								<pre className="bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# Set in .env file or environment
AGENT_PROVIDER=local
AGENT_BASE_URL=http://localhost:8000/v1
AGENT_MODEL=llama-3-70b-instruct`}
								</pre>

								<p className="mt-3">
									<strong>For Jiekou AI:</strong>
								</p>
								<pre className="bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# Set in .env file or environment
AGENT_PROVIDER=jiekou
AGENT_BASE_URL=https://api.jiekou.ai/v1
AGENT_API_KEY=your-jiekou-api-key
AGENT_MODEL=openai-gpt-oss-120b`}
								</pre>
								<p className="text-xs text-gray-600 mt-1">
									Note: Jiekou uses OpenAI-compatible API format. Get your API key from{" "}
									<a
										href="https://jiekou.ai"
										target="_blank"
										rel="noopener noreferrer"
										className="text-blue-600 hover:underline"
									>
										jiekou.ai
									</a>
								</p>

								<p className="mt-3">
									<strong>For Claude API:</strong>
								</p>
								<pre className="bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# Set in .env file or environment
AGENT_PROVIDER=claude
AGENT_API_KEY=your-anthropic-api-key
AGENT_MODEL=claude-3-5-sonnet-20241022`}
								</pre>

								<p className="mt-3">
									<strong>For OpenAI API:</strong>
								</p>
								<pre className="bg-gray-800 text-gray-100 p-3 rounded text-xs overflow-x-auto">
{`# Set in .env file or environment
AGENT_PROVIDER=openai
AGENT_API_KEY=your-openai-api-key
AGENT_MODEL=gpt-4`}
								</pre>
							</div>
							<p className="mt-4 text-sm text-gray-600">
								After configuration, restart the backend server.
							</p>
						</div>
					</div>
				</div>
			</div>
		);
	}

	// No session selected
	if (!sessionId) {
		return (
			<div className="flex items-center justify-center h-full bg-gray-50">
				<div className="text-center">
					<svg
						className="mx-auto h-24 w-24 text-gray-400"
						fill="none"
						viewBox="0 0 24 24"
						stroke="currentColor"
					>
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={2}
							d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
						/>
					</svg>
					<h3 className="mt-4 text-lg font-medium text-gray-900">
						No chat session selected
					</h3>
					<p className="mt-2 text-sm text-gray-500">
						Select a chat from the sidebar or create a new chat to get started
					</p>
				</div>
			</div>
		);
	}

	// Normal chat UI
	return (
		<div className="flex flex-col h-full bg-gray-50">
			{/* Header */}
			<div className="bg-white border-b px-6 py-4">
				<h1 className="text-2xl font-bold text-gray-900">Agent Chat</h1>
				<p className="text-sm text-gray-500 mt-1">
					Session: {sessionId.slice(0, 8)}...
				</p>
			</div>

			{/* Messages */}
			<div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
				{messages.length === 0 ? (
					<div className="text-center text-gray-400 mt-8">
						<p className="text-lg mb-2">👋 Welcome to Agent Chat!</p>
						<p>Send a message to get started</p>
					</div>
				) : (
					messages.map((message, idx) => (
						<MessageBubble key={message.id || idx} message={message} />
					))
				)}
				<div ref={messagesEndRef} />
			</div>

			{/* Input */}
			<div className="bg-white border-t px-6 py-4">
				<div className="flex gap-3">
					<textarea
						value={input}
						onChange={(e) => setInput(e.target.value)}
						onKeyDown={handleKeyPress}
						placeholder="Type a message... (Enter to send, Shift+Enter for new line)"
						className="flex-1 resize-none border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
						rows={1}
						style={{ minHeight: "44px", maxHeight: "200px" }}
						disabled={sendMessageMutation.isPending}
					/>
					<button
						onClick={handleSend}
						disabled={!input.trim() || sendMessageMutation.isPending}
						className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
					>
						{sendMessageMutation.isPending ? "Sending..." : "Send"}
					</button>
				</div>
			</div>
		</div>
	);
}

function MessageBubble({ message }: { message: MessageData }) {
	const isUser = message.role === "user";

	return (
		<div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
			<div
				className={`max-w-2xl rounded-lg px-4 py-3 ${
					isUser
						? "bg-blue-600 text-white"
						: "bg-white border border-gray-200 text-gray-900"
				}`}
			>
				<div className="text-sm whitespace-pre-wrap break-words">
					{message.content}
				</div>
				<div
					className={`text-xs mt-1 ${
						isUser ? "text-blue-100" : "text-gray-400"
					}`}
				>
					{new Date(message.created_at).toLocaleTimeString()}
				</div>
			</div>
		</div>
	);
}
