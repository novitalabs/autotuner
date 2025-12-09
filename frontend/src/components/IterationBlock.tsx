/**
 * IterationBlock component - Unified display for both streaming and historical iterations
 *
 * Renders a single iteration with content and tool calls in chronological order.
 * Works identically for real-time streaming and historical message display.
 */

import type { IterationBlock as IterationBlockType } from "../types/agent";
import ToolCallCard from "./ToolCallCard";

interface IterationBlockProps {
	iteration: IterationBlockType;
	showHeader: boolean;     // true if multiple iterations exist
	isStreaming: boolean;    // true for active streaming iteration
}

export default function IterationBlock({
	iteration,
	showHeader,
	isStreaming
}: IterationBlockProps) {
	return (
		<div>
		{/* Iteration header - only show if multiple iterations AND not the first one */}
		{showHeader && iteration.iteration > 1 && (
			<hr className="mt-6 mb-3 border-gray-300" />
		)}

			{/* Content section - thinking/reasoning text */}
			{iteration.content && (
				<div className="text-sm whitespace-pre-wrap break-words">
					{iteration.content}
					{/* Blinking cursor for active streaming */}
					{isStreaming && (
						<span className="inline-block w-2 h-4 ml-1 bg-blue-600 animate-pulse" />
					)}
				</div>
			)}

			{/* Tool calls section - execution history */}
			{iteration.toolCalls.length > 0 && (
				<div className={iteration.content ? "mt-3 space-y-2" : "space-y-2"}>
					{iteration.toolCalls.map((toolCall, idx) => (
						<ToolCallCard key={toolCall.id || idx} toolCall={toolCall} />
					))}
				</div>
			)}
		</div>
	);
}
