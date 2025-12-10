/**
 * Agent Chat page - With conversation history system
 * Supports URL parameters, IndexedDB storage, and background sync
 */

import { useState, useEffect, useRef } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { Edit2, ArrowUp, Loader2 } from "lucide-react";
import agentApi from "../services/agentApi";
import { getChatStorage, type MessageData } from "../services/chatStorage";
import type { IterationBlock, ToolExecutionResult } from "../types/agent";
import IterationBlockComponent from "../components/IterationBlock";
import { enrichIterationData } from "../utils/iterationHelpers";
import { useTimezone } from "../contexts/TimezoneContext";

/**
 * Parse timestamp string to Date, treating timestamps without timezone as UTC.
 * Backend returns UTC timestamps without Z suffix (e.g., "2025-12-09T11:58:53.762000").
 * Frontend generates UTC timestamps with Z suffix (e.g., "2025-12-09T12:00:00.000Z").
 */
function parseTimestamp(timestamp: string): Date {
	if (!timestamp) return new Date();
	// Check if timestamp already has timezone indicator at the END
	// Z suffix, or +HH:MM / -HH:MM offset (but not the dashes in date part)
	const hasTimezone = timestamp.endsWith('Z') || /[+-]\d{2}:\d{2}$/.test(timestamp) || /[+-]\d{4}$/.test(timestamp);
	const normalized = hasTimezone ? timestamp : timestamp + "Z";
	return new Date(normalized);
}

export default function AgentChat() {
	const [sessionId, setSessionId] = useState<string | null>(null);
	const [messages, setMessages] = useState<MessageData[]>([]);
	const [sessionTitle, setSessionTitle] = useState<string>("");
	const [isEditingTitle, setIsEditingTitle] = useState(false);
	const [editTitleValue, setEditTitleValue] = useState("");
	const [input, setInput] = useState("");
	const [_isStreaming, setIsStreaming] = useState(false);
	const [streamingIterations, setStreamingIterations] = useState<IterationBlock[]>([]);
	const streamingIterationsRef = useRef<IterationBlock[]>([]); // Ref to avoid closure trap
	const [_currentIteration, setCurrentIteration] = useState<number>(0);
	const [_maxIterations, setMaxIterations] = useState<number>(0);
	const messagesEndRef = useRef<HTMLDivElement>(null);
	const titleInputRef = useRef<HTMLInputElement>(null);
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
			setIsStreaming(true);
			setStreamingIterations([]);
			streamingIterationsRef.current = []; // Reset ref too
			setCurrentIteration(0);
			setMaxIterations(0);

			// Track accumulated content for error recovery
			let accumulatedContent = "";

			try {
				const response = await agentApi.sendMessageStream(
					sessionId,
					{ content },
					(chunk) => {
						// Debug: Log all incoming chunks
						console.log('[SSE Debug] Received chunk:', chunk.type, chunk);

						if (chunk.type === "iteration_start") {
							// New iteration starting - create new IterationBlock
							setCurrentIteration(chunk.iteration || 0);
							setMaxIterations(chunk.max_iterations || 0);

							// Use ref as source of truth
							const currentIterations = streamingIterationsRef.current;
							const iterNum = chunk.iteration || currentIterations.length + 1;

							// Prevent duplicate iterations (same iteration number)
							if (currentIterations.some(iter => iter.iteration === iterNum)) {
								console.debug(`Skipping duplicate iteration_start for iteration ${iterNum}`);
								return;
							}

							const newState = [...currentIterations, {
								iteration: iterNum,
								content: "",
								toolCalls: [],
								status: 'streaming' as const
							}];
							streamingIterationsRef.current = newState;
							setStreamingIterations(newState);
						} else if (chunk.type === "iteration_complete") {
							// Iteration completed - mark as complete
							const currentIterations = streamingIterationsRef.current;
							if (currentIterations.length > 0) {
								const updated = [...currentIterations];
								updated[updated.length - 1] = {
									...updated[updated.length - 1],
									status: 'complete'
								};
								streamingIterationsRef.current = updated;
								setStreamingIterations(updated);
							}
						} else if (chunk.type === "content") {
							// Append streaming content to current iteration
							const newContent = chunk.content || "";
							accumulatedContent += newContent;

							const currentIterations = streamingIterationsRef.current;
							if (currentIterations.length > 0) {
								const updated = [...currentIterations];
								updated[updated.length - 1] = {
									...updated[updated.length - 1],
									content: updated[updated.length - 1].content + newContent
								};
								streamingIterationsRef.current = updated;
								setStreamingIterations(updated);
							}
						} else if (chunk.type === "tool_start") {
							// Tool execution started - add executing tool calls to current iteration
							if (chunk.tool_calls) {
								const executingTools = chunk.tool_calls.map(tc => ({
									tool_name: tc.tool_name || tc.name,
									args: tc.args,
									id: tc.id,
									status: "executing" as const
								}));

								// Use ref as source of truth to avoid React StrictMode double-invocation issues
								const currentIterations = streamingIterationsRef.current;

								if (currentIterations.length > 0) {
									const currentIter = currentIterations[currentIterations.length - 1];

									// Check if we already have tool calls with these IDs (prevent duplicates)
									const existingIds = new Set(currentIter.toolCalls.map(tc => tc.id));
									const newTools = executingTools.filter(tc => !existingIds.has(tc.id));

									if (newTools.length === 0 && currentIter.toolCalls.length > 0) {
										// All tools already exist, skip update
										return;
									}

									// Update ref first
									const updated = [...currentIterations];
									updated[updated.length - 1] = {
										...currentIter,
										toolCalls: [...currentIter.toolCalls, ...newTools]
									};
									streamingIterationsRef.current = updated;

									// Then sync to state
									setStreamingIterations(updated);
								}
							}
						} else if (chunk.type === "tool_results") {
							// Update tool calls to executed status with results
							if (chunk.results && Array.isArray(chunk.results)) {
								const results = chunk.results;
								console.log('[Auth Debug] Received tool_results (raw):', JSON.stringify(results, null, 2));

								// Use ref as source of truth
								const currentIterations = streamingIterationsRef.current;
								console.log('[Auth Debug] Current iterations toolCalls:', currentIterations.map(i => i.toolCalls));
								if (currentIterations.length > 0) {
									const currentIter = currentIterations[currentIterations.length - 1];

									// Skip if all tools are already completed (prevent duplicate updates)
									const allCompleted = currentIter.toolCalls.every(
										tc => tc.status === "executed" || tc.status === "failed" || tc.status === "requires_auth"
									);
									if (allCompleted && currentIter.toolCalls.length > 0) {
										return;
									}

									const updatedToolCalls = currentIter.toolCalls.map(tc => {
										// Don't update already completed tools
										if (tc.status === "executed" || tc.status === "failed" || tc.status === "requires_auth") {
											return tc;
										}

										const result = results.find(r =>
											r.call_id === tc.id || r.tool_name === tc.tool_name
										);

										if (result) {
											// Check if this is an authorization requirement
											console.log('[Auth Debug] Processing result for tool:', tc.tool_name, 'result:', result);
											if (result.requires_auth && !result.authorized) {
												console.log('[Auth Debug] Tool requires auth:', result);
												console.log('[Auth Debug] Setting auth_scope to:', result.auth_scope);
												return {
													...tc,
													status: "requires_auth" as const,
													auth_scope: result.auth_scope,
													result: result.result
												};
											}
											return {
												...tc,
												status: result.success ? "executed" as const : "failed" as const,
												result: result.result,
												error: result.success ? undefined : result.result
											};
										}
										return tc;
									});

									console.log('[Auth Debug] Updated tool calls:', updatedToolCalls);

									// Update ref first
									const updated = [...currentIterations];
									updated[updated.length - 1] = {
										...currentIter,
										toolCalls: updatedToolCalls
									};
									streamingIterationsRef.current = updated;

									// Then sync to state
									setStreamingIterations(updated);
								}
							}
						} else if (chunk.type === "final_response_start") {
							// Start of final response after tool execution
						} else if (chunk.type === "error") {
							// Error from backend - append error text to current iteration content
							// Don't save here - let the catch block handle final saving
							console.error("Stream error:", chunk.error);

							const errorText = `\n\n❌ **Error:** ${chunk.error || "Unknown error occurred"}`;

							// Append error to current iteration's content using ref
							const currentIterations = streamingIterationsRef.current;
							let updated: typeof currentIterations;

							if (currentIterations.length > 0) {
								updated = [...currentIterations];
								updated[updated.length - 1] = {
									...updated[updated.length - 1],
									content: updated[updated.length - 1].content + errorText,
									status: 'complete' as const
								};
							} else {
								// No iterations yet - create one with error
								updated = [{
									iteration: 1,
									content: errorText.trim(),
									toolCalls: [],
									status: 'complete' as const
								}];
							}
							streamingIterationsRef.current = updated;
							setStreamingIterations(updated);

							// Note: agentApi will reject the promise, catch block will handle saving
						}
					}
				);

				setIsStreaming(false);

				// Use metadata from backend response if available, otherwise build from streaming state
				// Backend metadata is more authoritative as it comes from the saved database record
				const currentIterations = streamingIterationsRef.current;
				let metadata = response.metadata;
				let allToolCalls: any[] = [];

				console.log('[Save Debug] response:', response);
				console.log('[Save Debug] response.metadata:', response.metadata);
				console.log('[Save Debug] response.tool_calls:', response.tool_calls);

				// Prefer backend tool_calls if available (they have complete status/result/auth_scope)
				if (response.tool_calls && response.tool_calls.length > 0) {
					console.log('[Save Debug] Using backend tool_calls');
					allToolCalls = response.tool_calls;
				} else if (currentIterations.length > 0) {
					// No backend metadata - build from streaming state (fallback)
					metadata = {
						iteration_data: currentIterations.map(iter => ({
							iteration: iter.iteration,
							content: iter.content,
							tool_calls: iter.toolCalls.map(tc => ({
								tool_name: tc.tool_name,
								args: tc.args,
								id: tc.id
							}))
						})),
						iterations: currentIterations.length
					};
					// Extract all tool calls from iterations (with complete result/error/status)
					allToolCalls = currentIterations.flatMap(iter =>
						iter.toolCalls.map(tc => ({
							tool_name: tc.tool_name,
							args: tc.args,
							id: tc.id,
							status: tc.status,
							result: tc.result,
							error: tc.error,
							auth_scope: tc.auth_scope  // Include auth_scope!
						}))
					);
				}

				// Save assistant response to IndexedDB
				// Use backend ID to ensure consistency between IndexedDB and state
				const assistantMessage: MessageData = {
					id: response.id ? String(response.id) : undefined,
					session_id: sessionId,
					role: "assistant",
					content: response.content,
					tool_calls: allToolCalls.length > 0 ? allToolCalls : undefined,
					metadata: metadata && Object.keys(metadata).length > 0 ? metadata : undefined,
					created_at: response.created_at,
					synced_to_backend: true,
				};
				await storage.saveMessage(sessionId, assistantMessage);

				// Clear streaming state after saving
				setStreamingIterations([]);
				setCurrentIteration(0);
				setMaxIterations(0);

				// Update session timestamp
				await storage.updateSessionTimestamp(sessionId);

				return { userMessage, assistantMessage };
			} catch (error) {
				// Error occurred - streamingIterations contains content + error text (appended by error event handler)
				setIsStreaming(false);

				// Get current iterations (error text already appended by error event handler)
				const currentIterations = streamingIterationsRef.current;

				// Build metadata with iteration_data (content already includes error text)
				const metadata: any = {};
				if (currentIterations.length > 0) {
					metadata.iteration_data = currentIterations.map(iter => ({
						iteration: iter.iteration,
						content: iter.content,  // Already contains error text from error event handler
						tool_calls: iter.toolCalls.map(tc => ({
							tool_name: tc.tool_name,
							args: tc.args,
							id: tc.id
						}))
					}));
					metadata.iterations = currentIterations.length;
				}

				const allToolCalls = currentIterations.flatMap(iter =>
					iter.toolCalls.map(tc => ({
						tool_name: tc.tool_name,
						args: tc.args,
						id: tc.id,
						status: tc.status,
						result: tc.result,
						error: tc.error
					}))
				);

				// Use last iteration's content (which already includes error text)
				let errorContent = "";
				if (currentIterations.length > 0) {
					errorContent = currentIterations[currentIterations.length - 1].content;
				} else if (accumulatedContent) {
					// Fallback if no iterations
					let errorMessage = "Response incomplete due to server error";
					if (error instanceof Error) {
						errorMessage = error.message;
					}
					errorContent = accumulatedContent + `\n\n❌ **Error:** ${errorMessage}`;
				} else {
					// No content at all
					let errorMessage = "Response incomplete due to server error";
					if (error instanceof Error) {
						errorMessage = error.message;
					}
					errorContent = `❌ **Error:** ${errorMessage}`;
				}

				const errorMessageData: MessageData = {
					session_id: sessionId,
					role: "assistant",
					content: errorContent,
					tool_calls: allToolCalls.length > 0 ? allToolCalls : undefined,
					metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
					created_at: new Date().toISOString(),
					synced_to_backend: false,
				};

				try {
					await storage.saveMessage(sessionId, errorMessageData);
				} catch (saveError) {
					console.error("Failed to save error message to IndexedDB:", saveError);
				}

				setStreamingIterations([]);
				streamingIterationsRef.current = [];
				setCurrentIteration(0);
				setMaxIterations(0);

				return {
					userMessage,
					assistantMessage: errorMessageData as any
				};
			}
		},
		onSuccess: (data) => {
			// Add assistant response to local state (works for both success and error cases)
			// Deduplicate by ID or content+timestamp to prevent duplicate messages
			setMessages(prev => {
				const newMsg = data.assistantMessage;
				const exists = prev.some(msg =>
					(msg.id && newMsg.id && msg.id === newMsg.id) ||
					(msg.content === newMsg.content && msg.created_at === newMsg.created_at && msg.role === newMsg.role)
				);
				if (exists) {
					console.debug("Skipping duplicate assistant message");
					return prev;
				}
				return [...prev, newMsg];
			});

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
			// Error already handled in catch block - state and IndexedDB already updated
		},
	});

	// Auto-scroll to bottom
	useEffect(() => {
		messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
	}, [messages, streamingIterations]);

	const handleSend = () => {
		if (!input.trim() || !sessionId) return;
		const content = input;
		setInput("");  // Clear input immediately when sending
		sendMessageMutation.mutate(content);
	};

	// Handle authorization request from tool calls
	const handleAuthorize = async (scope: string) => {
		if (!sessionId) return;

		try {
			const result = await agentApi.grantAuthorization(sessionId, [scope]);
			console.log(`Authorization granted for scope: ${scope}`);

			// If there were pending tool calls, they were executed automatically
			// Update the last message's tool call status to show results
			if (result.tool_results && result.tool_results.length > 0) {
				console.log("Tool results received after authorization:", result.tool_results);

				// Update streaming iterations if currently streaming
				if (streamingIterationsRef.current.length > 0) {
					const currentIterations = streamingIterationsRef.current;
					const updated = currentIterations.map(iter => ({
						...iter,
						toolCalls: iter.toolCalls.map(tc => {
							if (tc.status !== "requires_auth") return tc;
							const matchingResult = result.tool_results?.find(
								(r: ToolExecutionResult) => r.tool_name === tc.tool_name || r.call_id === tc.id
							);
							if (matchingResult) {
								return {
									...tc,
									status: matchingResult.success ? "executed" as const : "failed" as const,
									result: matchingResult.result,
									error: matchingResult.success ? undefined : matchingResult.result
								};
							}
							return tc;
						})
					}));
					streamingIterationsRef.current = updated;
					setStreamingIterations(updated);
				}

				// Update the messages to reflect the executed tool calls
				setMessages(prevMessages => {
					// Find the last assistant message with requires_auth tool calls
					let foundIndex = -1;
					for (let i = prevMessages.length - 1; i >= 0; i--) {
						const msg = prevMessages[i];
						if (msg.role === "assistant" && msg.tool_calls) {
							const hasAuthRequired = msg.tool_calls.some(tc => tc.status === "requires_auth");
							if (hasAuthRequired) {
								foundIndex = i;
								break;
							}
						}
					}

					if (foundIndex === -1) {
						return prevMessages; // No message to update
					}

					// Create a new array with the updated message (immutable update)
					return prevMessages.map((msg, idx) => {
						if (idx !== foundIndex) return msg;

						// Create a new message object with updated tool_calls
						return {
							...msg,
							tool_calls: msg.tool_calls?.map(tc => {
								const matchingResult = result.tool_results?.find(
									(r: ToolExecutionResult) => r.tool_name === tc.tool_name || r.call_id === tc.id
								);
								if (matchingResult) {
									return {
										...tc,
										status: matchingResult.success ? "executed" as const : "failed" as const,
										result: matchingResult.result,
										error: matchingResult.success ? undefined : matchingResult.result
									};
								}
								return tc;
							})
						};
					});
				});

				// Also update IndexedDB
				const storage = getChatStorage();
				const storedMessages = await storage.getMessages(sessionId);
				for (let i = storedMessages.length - 1; i >= 0; i--) {
					const msg = storedMessages[i];
					if (msg.role === "assistant" && msg.tool_calls) {
						const hasAuthRequired = msg.tool_calls.some(tc => tc.status === "requires_auth");
						if (hasAuthRequired) {
							msg.tool_calls = msg.tool_calls.map(tc => {
								const matchingResult = result.tool_results?.find(
									(r: ToolExecutionResult) => r.tool_name === tc.tool_name || r.call_id === tc.id
								);
								if (matchingResult) {
									return {
										...tc,
										status: matchingResult.success ? "executed" as const : "failed" as const,
										result: matchingResult.result,
										error: matchingResult.success ? undefined : matchingResult.result
									};
								}
								return tc;
							});
							await storage.saveMessage(sessionId, msg);
							break;
						}
					}
				}

				// Clear streaming iterations after authorization completes to avoid duplicate rendering
				// The tool call is now in messages state, so streamingIterations should be cleared
				streamingIterationsRef.current = [];
				setStreamingIterations([]);
			}

			// If LLM provided a continuation response, add it as a new message
			if (result.llm_continuation) {
				console.log("LLM continuation received:", result.llm_continuation);

				const storage = getChatStorage();

				// Create the continuation message with tool call results
				const continuationMessage: MessageData = {
					session_id: sessionId,
					role: "assistant",
					content: result.llm_continuation,
					tool_calls: result.tool_results?.map(r => ({
						tool_name: r.tool_name,
						args: {},
						id: r.call_id || r.tool_name,
						status: r.success ? "executed" as const : "failed" as const,
						result: r.success ? r.result : undefined,
						error: r.success ? undefined : r.result
					})),
					created_at: new Date().toISOString(),
					synced_to_backend: true,
				};

				// Add to local state
				setMessages(prev => [...prev, continuationMessage]);

				// Save to IndexedDB
				await storage.saveMessage(sessionId, continuationMessage);
			}
		} catch (error) {
			console.error("Failed to grant authorization:", error);
		}
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
						<MessageBubble key={message.id || idx} message={message} onAuthorize={handleAuthorize} />
					))
				)}

				{/* Streaming iterations (real-time) - unified display */}
				{streamingIterations.length > 0 && (
					<div className="flex justify-start">
						<div className="group relative max-w-3xl">
							<div className="rounded-lg px-4 py-3 bg-white border border-gray-200 text-gray-900">
								{streamingIterations.map((iter, idx) => (
									<IterationBlockComponent
										key={iter.iteration}
										iteration={iter}
										showHeader={streamingIterations.length > 1}
										isStreaming={idx === streamingIterations.length - 1 && iter.status === 'streaming'}
										onAuthorize={handleAuthorize}
									/>
								))}
							</div>
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

function MessageBubble({ message, onAuthorize }: { message: MessageData; onAuthorize?: (scope: string) => void }) {
	const { timezone } = useTimezone();
	const isUser = message.role === "user";

	// Format time using the timezone from context
	const displayTime = (ts: string) => {
		const date = parseTimestamp(ts);
		return date.toLocaleTimeString('en-US', {
			timeZone: timezone,
			hour: '2-digit',
			minute: '2-digit',
			hour12: false
		});
	};

	// User messages - simple display
	if (isUser) {
		return (
			<div className="flex justify-end">
				<div className="group relative max-w-3xl">
					<div className="rounded-lg px-4 py-3 bg-blue-600 text-white">
						<div className="text-sm whitespace-pre-wrap break-words">
							{message.content}
						</div>
					</div>
					{/* Timestamp - below bubble, only visible on hover */}
					<div className="text-xs mt-1 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity text-right">
						{displayTime(message.created_at)}
					</div>
				</div>
			</div>
		);
	}

	// Assistant messages - convert to IterationBlock format
	let iterations: IterationBlock[];

	if (message.metadata?.iteration_data) {
		// New format: enrich with results
		iterations = enrichIterationData(
			message.metadata.iteration_data,
			message.tool_calls || []
		);
	} else if (message.tool_calls && message.tool_calls.length > 0) {
		// Old format: single iteration
		iterations = [{
			iteration: 1,
			content: message.content,
			toolCalls: message.tool_calls,
			status: 'complete'
		}];
	} else {
		// Plain message
		iterations = [{
			iteration: 1,
			content: message.content,
			toolCalls: [],
			status: 'complete'
		}];
	}

	return (
		<div className="flex justify-start">
			<div className="group relative max-w-3xl">
				<div className="rounded-lg px-4 py-3 bg-white border border-gray-200 text-gray-900">
					{iterations.map((iter) => (
						<IterationBlockComponent
							key={iter.iteration}
							iteration={iter}
							showHeader={iterations.length > 1}
							isStreaming={false}
							onAuthorize={onAuthorize}
						/>
					))}
				</div>
				{/* Timestamp - below bubble, only visible on hover */}
				<div className="text-xs mt-1 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity text-left">
					{displayTime(message.created_at)}
				</div>
			</div>
		</div>
	);
}
