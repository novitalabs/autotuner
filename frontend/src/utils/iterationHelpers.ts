/**
 * Helper functions for iteration data transformation and enrichment
 */

import type { IterationBlock, ToolCall } from "../types/agent";

/**
 * Enriches iteration data from message metadata with tool results
 *
 * Backend stores iteration_data WITHOUT results, and tool results separately in message.tool_calls.
 * This function matches them together by tool call ID.
 *
 * @param iterationData - From message.metadata.iteration_data
 * @param toolCallsWithResults - From message.tool_calls (top-level array with results)
 * @returns IterationBlock array with results attached to tool calls
 */
export function enrichIterationData(
	iterationData: Array<{
		iteration: number;
		content: string;
		tool_calls: Array<{ tool_name: string; args: any; id: string }>;
	}>,
	toolCallsWithResults: ToolCall[]
): IterationBlock[] {
	return iterationData.map(iter => {
		// Match tool calls by ID to attach results
		const enrichedToolCalls: ToolCall[] = iter.tool_calls.map(tc => {
			// Find matching result by call ID (primary) or tool_name (fallback)
			const withResult = toolCallsWithResults.find(
				r => r.id === tc.id || (r.tool_name === tc.tool_name && !r.id)
			);

			return {
				tool_name: tc.tool_name,
				args: tc.args,
				id: tc.id,
				status: withResult?.status || ("executed" as const),
				result: withResult?.result,
				error: withResult?.error
			};
		});

		return {
			iteration: iter.iteration,
			content: iter.content,
			toolCalls: enrichedToolCalls,
			status: 'complete' as const
		};
	});
}
