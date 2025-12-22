/**
 * IterationBlock component - Unified display for both streaming and historical iterations
 *
 * Renders a single iteration with content and tool calls in chronological order.
 * Works identically for real-time streaming and historical message display.
 */

import type { IterationBlock as IterationBlockType } from "../types/agent";
import ToolCallCard from "./ToolCallCard";
import StreamingMarkdown from "./StreamingMarkdown";

interface IterationBlockProps {
	iteration: IterationBlockType;
	showHeader: boolean;     // true if multiple iterations exist
	isStreaming: boolean;    // true for active streaming iteration
	onAuthorize?: (scope: string) => void;  // callback for authorization requests
	isAuthorizing?: boolean;  // whether authorization is in progress
}

export default function IterationBlock({
	iteration,
	showHeader,
	isStreaming,
	onAuthorize,
	isAuthorizing
}: IterationBlockProps) {
	return (
		<div>
		{/* Iteration header - only show if multiple iterations */}
		{showHeader && (
			<div className={iteration.iteration > 1 ? "mt-4 pt-3 border-t border-gray-200" : ""}>
			</div>
		)}

			{/* Content section - thinking/reasoning text */}
			{iteration.content && (
				<StreamingMarkdown
					content={iteration.content}
					isStreaming={isStreaming}
				/>
			)}

			{/* Tool calls section - execution history */}
			{iteration.toolCalls.length > 0 && (
				<div className={iteration.content ? "mt-3 space-y-2" : "space-y-2"}>
					{iteration.toolCalls.map((toolCall, idx) => (
						<ToolCallCard
							key={toolCall.id || idx}
							toolCall={toolCall}
							onAuthorize={onAuthorize}
							isAuthorizing={isAuthorizing}
						/>
					))}
				</div>
			)}
		</div>
	);
}
