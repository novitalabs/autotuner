/**
 * Agent Chat page - With conversation history system
 * Supports URL parameters, IndexedDB storage, and background sync
 */

import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Edit2, ArrowUp, Loader2 } from "lucide-react";
import agentApi from "../services/agentApi";
import { getChatStorage, type MessageData } from "../services/chatStorage";
import type { ChatMessage } from "../types/agent";
import ToolCallCard from "../components/ToolCallCard";

export default function AgentChat() {
	const [sessionId, setSessionId] = useState<string | null>(null);
	const [messages, setMessages] = useState<MessageData[]>([]);
	const [sessionTitle, setSessionTitle] = useState<string>("");
	const [isEditingTitle, setIsEditingTitle] = useState(false);
	const [editTitleValue, setEditTitleValue] = useState("");
	const [input, setInput] = useState("");
	const [streamingContent, setStreamingContent] = useState("");
	const [isStreaming, setIsStreaming] = useState(false);
	const [toolCallStatus, setToolCallStatus] = useState<string | null>(null);
	const [currentIteration, setCurrentIteration] = useState<number>(0);
	const [maxIterations, setMaxIterations] = useState<number>(0);
	const messagesEndRef = useRef<HTMLDivElement>(null);
	const titleInputRef = useRef<HTMLInputElement>(null);
	const queryClient = useQueryClient();
	const [loadingFromDb, setLoadingFromDb] = useState(true);

	// Periodically refresh session title from IndexedDB
	useEffect(() => {
		if (!sessionId) return;

		const refreshTitle = async () => {
			try {
				const storage = getChatStorage();
				const session = await storage.getSession(sessionId);
				if (session?.title && session.title !== sessionTitle) {
					setSessionTitle(session.title);
				}
			} catch (error) {
				console.error("Failed to refresh title:", error);
			}
		};

		// Initial refresh
		refreshTitle();

		// Refresh every 2 seconds
		const interval = setInterval(refreshTitle, 2000);
		return () => clearInterval(interval);
	}, [sessionId, sessionTitle]);

	// Parse session ID from URL hash and listen for changes
	useEffect(() => {
		const handleHashChange = () => {
			const hash = window.location.hash.slice(1); // Remove #
			const params = new URLSearchParams(hash.split("?")[1] || "");
			const sessionParam = params.get("session");

			if (sessionParam) {
				setSessionId(sessionParam);
				setLoadingFromDb(true);
				loadMessagesFromIndexedDB(sessionParam);
			} else {
				// No session ID in URL - show empty state
				setSessionId(null);
				setMessages([]);
				setSessionTitle("");
				setLoadingFromDb(false);
			}
		};

		// Initial load
		handleHashChange();

		// Listen for hash changes (when switching sessions)
		window.addEventListener('hashchange', handleHashChange);
		return () => window.removeEventListener('hashchange', handleHashChange);
	}, []);

	// Load messages from IndexedDB (fast initial load)
	const loadMessagesFromIndexedDB = async (sid: string) => {
		try {
			const storage = getChatStorage();
			const msgs = await storage.getMessages(sid);

			// Messages are already sorted by insertion order (ID timestamp)
			setMessages(msgs);

			// Load session to get title
			const session = await storage.getSession(sid);
			if (session?.title) {
				setSessionTitle(session.title);
			}

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

	// Send message mutation with IndexedDB sync and streaming
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

			// HYBRID STREAMING MODE:
			// - Initial response: streamed for progressive display
			// - Tool execution: shown in real-time status
			// - Final response after tools: non-streamed (Jiekou API limitation with ToolMessage)
			//
			// Note: Full streaming with tool results doesn't work due to Jiekou API returning
			// 400 errors when ToolMessage entries are included in streaming requests.
			// The backend now uses non-streaming for the final synthesis after tools.

			// Reset streaming state
			setStreamingContent("");
			setIsStreaming(true);
			setToolCallStatus(null);
			setCurrentIteration(0);
			setMaxIterations(0);

			// Track accumulated content for error recovery
			let accumulatedContent = "";

			try {
				const response = await agentApi.sendMessageStream(
					sessionId,
					{ content },
					(chunk) => {
						if (chunk.type === "iteration_start") {
							// New iteration starting - just track iteration number, don't show status
							// Tool call and thinking are different concepts
							setCurrentIteration(chunk.iteration || 0);
							setMaxIterations(chunk.max_iterations || 0);
						} else if (chunk.type === "iteration_complete") {
							// Iteration completed - clear any tool status
							setToolCallStatus(null);
						} else if (chunk.type === "content") {
							// Append streaming content
							const newContent = chunk.content || "";
							accumulatedContent += newContent;
							setStreamingContent(prev => prev + newContent);
						} else if (chunk.type === "tool_start") {
							// Tool execution started
							const toolNames = chunk.tool_calls?.map(tc => tc.name).join(", ") || "";
							setToolCallStatus(`Calling tools: ${toolNames}...`);
						} else if (chunk.type === "tool_results") {
							setToolCallStatus("Processing tool results...");
						} else if (chunk.type === "final_response_start") {
							setToolCallStatus(null);
							// Don't reset accumulatedContent, but reset display
							setStreamingContent("");
						} else if (chunk.type === "error") {
							console.error("Stream error:", chunk.error);
							setToolCallStatus(null);
							setIsStreaming(false);
							setCurrentIteration(0);
							setMaxIterations(0);
						}
					}
				);

				setIsStreaming(false);
				setStreamingContent("");
				setToolCallStatus(null);
				setCurrentIteration(0);
				setMaxIterations(0);

				// Save assistant response to IndexedDB
				const assistantMessage: MessageData = {
					session_id: sessionId,
					role: "assistant",
					content: response.content,
					tool_calls: response.tool_calls || undefined,
					created_at: response.created_at,
					synced_to_backend: true,
				};
				await storage.saveMessage(sessionId, assistantMessage);

				// Update session timestamp
				await storage.updateSessionTimestamp(sessionId);

				return { userMessage, assistantMessage };
			} catch (error) {
				// Error occurred, but save what we got so far
				setIsStreaming(false);
				setStreamingContent("");
				setToolCallStatus(null);
				setCurrentIteration(0);
				setMaxIterations(0);

				if (accumulatedContent) {
					// Save partial response
					const partialMessage: MessageData = {
						session_id: sessionId,
						role: "assistant",
						content: accumulatedContent + "\n\n[Error: Response incomplete due to server error]",
						created_at: new Date().toISOString(),
						synced_to_backend: false,
					};
					await storage.saveMessage(sessionId, partialMessage);

					// Update local state
					setMessages(prev => [...prev, partialMessage]);
				}

				throw error;
			}
		},
		onSuccess: (data) => {
			// Update local state with assistant response
			setMessages(prev => [...prev, data.assistantMessage]);

			// Generate title after first user message (message count = 2: 1 user + 1 assistant)
			// Only generate if no title exists yet (don't override user-edited titles)
			if (sessionId) {
				try {
					const storage = getChatStorage();

					// Check if this is the first message exchange
					setTimeout(async () => {
						try {
							const msgs = await storage.getMessages(sessionId);
							const session = await storage.getSession(sessionId);

							// Generate title if: (1) no title yet, AND (2) we now have exactly 2 messages (1 user + 1 assistant)
							if ((!session?.title || session.title.trim() === '') && msgs.length === 2) {
								const response = await agentApi.generateTitle(sessionId);
								await storage.updateSessionTitle(sessionId, response.title);
								setSessionTitle(response.title);  // Update UI immediately
								console.debug(`Generated title: ${response.title}`);
							} else if (session?.title) {
								console.debug(`Session already has title: "${session.title}", skipping generation`);
							}
						} catch (error) {
							console.error("Failed to generate title:", error);
							// Silent fail - not critical to chat flow
						}
					}, 100);  // Small delay to not block UI
				} catch (error) {
					console.error("Failed to trigger title generation:", error);
				}
			}
		},
		onError: (error) => {
			console.error("Failed to send message:", error);
			// TODO: Show error toast
		},
	});

	// Auto-scroll to bottom
	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages, streamingContent, toolCallStatus]);

	const handleSend = () => {
		if (!input.trim() || !sessionId) return;
		const content = input;
		setInput("");  // Clear input immediately when sending
		sendMessageMutation.mutate(content);
	};

	const handleKeyPress = (e: React.KeyboardEvent) => {
		if (e.key === "Enter" && !e.shiftKey) {
			e.preventDefault();
			handleSend();
		}
	};

	const handleStartEditTitle = () => {
		setEditTitleValue(sessionTitle || "");
		setIsEditingTitle(true);
	};

	const handleSaveTitle = async () => {
		if (!sessionId) {
			setIsEditingTitle(false);
			return;
		}

		const newTitle = editTitleValue.trim();
		if (!newTitle) {
			setIsEditingTitle(false);
			return;
		}

		try {
			await agentApi.updateTitle(sessionId, newTitle);
			const storage = getChatStorage();
			await storage.updateSessionTitle(sessionId, newTitle);
			setSessionTitle(newTitle);
			setIsEditingTitle(false);
		} catch (error) {
			console.error("Failed to update title:", error);
			setIsEditingTitle(false);
		}
	};

	const handleCancelEditTitle = () => {
		setIsEditingTitle(false);
		setEditTitleValue("");
	};

	// Auto-focus title input when entering edit mode
	useEffect(() => {
		if (isEditingTitle && titleInputRef.current) {
			titleInputRef.current.focus();
			titleInputRef.current.select();
		}
	}, [isEditingTitle]);

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
				<div className="group/title flex items-center gap-3">
					{isEditingTitle ? (
						// Edit mode
						<input
							ref={titleInputRef}
							type="text"
							value={editTitleValue}
							onChange={(e) => setEditTitleValue(e.target.value)}
							onBlur={handleSaveTitle}
							onKeyDown={(e) => {
								if (e.key === 'Enter') {
									e.currentTarget.blur(); // Trigger onBlur to save
								}
								if (e.key === 'Escape') {
									handleCancelEditTitle();
								}
							}}
							className="flex-1 text-2xl font-bold text-gray-900 px-2 py-1 border border-blue-500 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
							placeholder="Enter title..."
						/>
					) : (
						// Display mode
						<>
							<h1 className="flex-1 text-2xl font-bold text-gray-900">
								{sessionTitle || "New Chat"}
							</h1>
							<button
								onClick={handleStartEditTitle}
								className="p-2 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded transition-all opacity-0 group-hover/title:opacity-100"
								title="Edit title"
							>
								<Edit2 className="w-5 h-5" />
							</button>
						</>
					)}
				</div>
			</div>

			{/* Session ID - below header, only visible on hover */}
			<div className="group relative">
				<div className="px-6 py-1 text-xs text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity">
					Session ID: {sessionId.slice(0, 8)}
				</div>
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

				{/* Streaming message */}
				{isStreaming && streamingContent && (
					<div className="flex justify-start">
						<div className="group relative max-w-3xl">
							<div className="rounded-lg px-4 py-3 bg-white border border-gray-200 text-gray-900">
								<div className="text-sm whitespace-pre-wrap break-words">
									{streamingContent}
									<span className="inline-block w-2 h-4 ml-1 bg-blue-600 animate-pulse"></span>
								</div>
							</div>
						</div>
					</div>
				)}

				{/* Tool calling status */}
				{toolCallStatus && (
					<div className="flex justify-start">
						<div className="rounded-lg px-4 py-2 bg-yellow-50 border border-yellow-200 text-yellow-800 text-sm flex items-center gap-2">
							<Loader2 className="w-4 h-4 animate-spin" />
							{toolCallStatus}
						</div>
					</div>
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
						className="p-3 bg-blue-600 text-white rounded-full hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
						title={sendMessageMutation.isPending ? "Sending..." : "Send message"}
					>
						{sendMessageMutation.isPending ? (
							<Loader2 className="w-5 h-5 animate-spin" />
						) : (
							<ArrowUp className="w-5 h-5" />
						)}
					</button>
				</div>
			</div>
		</div>
	);
}

function MessageBubble({ message }: { message: MessageData }) {
	const isUser = message.role === "user";

	const handleAuthorize = (scope: string) => {
		// TODO: Implement authorization flow
		alert(`Authorization for ${scope} will be implemented in next step`);
	};

	return (
		<div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
			<div className="group relative max-w-3xl">
				{/* Message bubble */}
				<div
					className={`rounded-lg px-4 py-3 ${
						isUser
							? "bg-blue-600 text-white"
							: "bg-white border border-gray-200 text-gray-900"
					}`}
				>
					{/* Message content */}
					{message.content && (
						<div className="text-sm whitespace-pre-wrap break-words">
							{message.content}
						</div>
					)}

					{/* Tool calls - only for assistant messages */}
					{!isUser && message.tool_calls && message.tool_calls.length > 0 && (
						<div className="mt-2">
							{message.tool_calls.map((toolCall: any, index: number) => (
								<ToolCallCard
									key={`${toolCall.id}-${index}`}
									toolCall={toolCall}
									onAuthorize={handleAuthorize}
								/>
							))}
						</div>
					)}
				</div>
				{/* Timestamp - below bubble, only visible on hover */}
				<div
					className={`text-xs mt-1 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity ${
						isUser ? "text-right" : "text-left"
					}`}
				>
					{new Date(message.created_at).toLocaleTimeString()}
				</div>
			</div>
		</div>
	);
}
