/**
 * Agent Chat page - Phase 1: Basic chat UI with echo bot
 */

import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import agentApi from "../services/agentApi";
import type { ChatMessage, AgentStatus } from "../types/agent";

export default function AgentChat() {
	const [sessionId, setSessionId] = useState<string | null>(null);
	const [input, setInput] = useState("");
	const messagesEndRef = useRef<HTMLDivElement>(null);
	const queryClient = useQueryClient();

	// Check agent status first
	const { data: agentStatus, isLoading: statusLoading } = useQuery({
		queryKey: ["agent-status"],
		queryFn: () => agentApi.getStatus(),
		staleTime: 60000, // Cache for 1 minute
	});

	// Create session on mount (only if agent is available)
	useEffect(() => {
		if (agentStatus?.available && !sessionId) {
			const initSession = async () => {
				const session = await agentApi.createSession();
				setSessionId(session.session_id);
			};
			initSession();
		}
	}, [agentStatus, sessionId]);

	// Fetch messages
	const { data: messages = [], isLoading } = useQuery({
		queryKey: ["agent-messages", sessionId],
		queryFn: () => agentApi.getMessages(sessionId!),
		enabled: !!sessionId && agentStatus?.available,
		refetchInterval: 2000, // Poll for new messages
	});

	// Send message mutation
	const sendMessageMutation = useMutation({
		mutationFn: (content: string) =>
			agentApi.sendMessage(sessionId!, { content }),
		onSuccess: () => {
			queryClient.invalidateQueries({ queryKey: ["agent-messages", sessionId] });
			setInput("");
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
	if (statusLoading) {
		return (
			<div className="flex items-center justify-center h-full">
				<div className="text-gray-500">Checking agent availability...</div>
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

	// Session initializing
	if (!sessionId) {
		return (
			<div className="flex items-center justify-center h-full">
				<div className="text-gray-500">Initializing chat session...</div>
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
					Phase 1: Echo Bot (LangChain integration coming in Phase 2)
				</p>
			</div>

			{/* Messages */}
			<div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
				{isLoading ? (
					<div className="text-center text-gray-500">Loading messages...</div>
				) : messages.length === 0 ? (
					<div className="text-center text-gray-400 mt-8">
						<p className="text-lg mb-2">👋 Welcome to Agent Chat!</p>
						<p>Send a message to get started</p>
					</div>
				) : (
					messages.map((message: ChatMessage) => (
						<MessageBubble key={message.id} message={message} />
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

function MessageBubble({ message }: { message: ChatMessage }) {
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
