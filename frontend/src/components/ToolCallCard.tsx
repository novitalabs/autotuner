/**
 * ToolCallCard component - displays a tool execution
 */

import { useState } from "react";
import { ChevronDown, ChevronRight, CheckCircle, AlertCircle, XCircle, Lock, Loader2 } from "lucide-react";
import YAML from 'yaml';
import type { ToolCall } from "../types/agent";

interface ToolCallCardProps {
	toolCall: ToolCall;
	onAuthorize?: (scope: string) => void;
}

export default function ToolCallCard({ toolCall, onAuthorize }: ToolCallCardProps) {
	const [isExpanded, setIsExpanded] = useState(false);

	const getStatusIcon = () => {
		switch (toolCall.status) {
			case "executing":
				return <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />;
			case "executed":
				return <CheckCircle className="w-4 h-4 text-green-600" />;
			case "requires_auth":
				return <Lock className="w-4 h-4 text-yellow-600" />;
			case "failed":
				return <XCircle className="w-4 h-4 text-red-600" />;
			default:
				return <AlertCircle className="w-4 h-4 text-gray-600" />;
		}
	};

	const getStatusColor = () => {
		switch (toolCall.status) {
			case "executing":
				return "border-blue-200 bg-blue-50";
			case "executed":
				return "border-green-200 bg-green-50";
			case "requires_auth":
				return "border-yellow-200 bg-yellow-50";
			case "failed":
				return "border-red-200 bg-red-50";
			default:
				return "border-gray-200 bg-gray-50";
		}
	};

	// Format function call as: function_name(arg1=value1, arg2=value2)
	const formatFunctionCall = (): string => {
		const args = Object.entries(toolCall.args)
			.filter(([key]) => key !== 'db') // Filter out db parameter
			.map(([key, value]) => {
				let formattedValue: string;
				if (typeof value === 'string') {
					// String values in quotes, truncate if too long
					const strValue = value.length > 50 ? value.substring(0, 47) + '...' : value;
					formattedValue = `"${strValue}"`;
				} else if (typeof value === 'object' && value !== null) {
					// Objects as JSON, truncate if too long
					const jsonStr = JSON.stringify(value);
					formattedValue = jsonStr.length > 50 ? jsonStr.substring(0, 47) + '...' : jsonStr;
				} else {
					formattedValue = String(value);
				}
				return `${key}=${formattedValue}`;
			});

		return `${toolCall.tool_name}(${args.join(', ')})`;
	};

	// Format result as YAML using yaml package
	const formatResult = (): string => {
		if (!toolCall.result) return '';

		// Try to parse as JSON first
		try {
			const parsed = JSON.parse(toolCall.result);
			// Use yaml package to stringify
			return YAML.stringify(parsed, {
				indent: 2,
				lineWidth: 0, // Don't wrap lines
				minContentWidth: 0,
			});
		} catch {
			// If not JSON, return as-is
			return toolCall.result;
		}
	};

	return (
		<div className={`border rounded-lg p-3 mt-2 ${getStatusColor()}`}>
			{/* Header - function call format */}
			<div className="flex items-start gap-2">
				<button
					onClick={() => setIsExpanded(!isExpanded)}
					className="mt-0.5 hover:bg-white/50 rounded p-0.5 transition-colors flex-shrink-0"
				>
					{isExpanded ? (
						<ChevronDown className="w-4 h-4" />
					) : (
						<ChevronRight className="w-4 h-4" />
					)}
				</button>

				<div className="flex-1 min-w-0">
					<div className="flex items-center gap-2 mb-1">
						{getStatusIcon()}
						<span className="font-mono text-sm font-medium truncate">
							{formatFunctionCall()}
						</span>
					</div>

					{/* Authorization button for requires_auth status */}
					{toolCall.status === "requires_auth" && toolCall.auth_scope && onAuthorize && (
						<button
							onClick={() => onAuthorize(toolCall.auth_scope!)}
							className="mt-2 px-3 py-1 bg-yellow-600 text-white text-xs rounded hover:bg-yellow-700 transition-colors"
						>
							Grant {toolCall.auth_scope} Permission
						</button>
					)}

					{/* Error message for failed status (collapsed view) */}
					{!isExpanded && toolCall.status === "failed" && toolCall.error && (
						<div className="mt-2 text-xs text-red-700 truncate">
							Error: {toolCall.error}
						</div>
					)}
				</div>
			</div>

			{/* Expanded content */}
			{isExpanded && (
				<div className="mt-3 pl-6 space-y-3">
					{/* Full error message for failed status */}
					{toolCall.status === "failed" && toolCall.error && (
						<div>
							<div className="text-xs font-semibold text-red-700 mb-1">
								Error:
							</div>
							<pre className="text-xs text-red-700 bg-red-100 rounded p-2 overflow-x-auto whitespace-pre-wrap">
								{toolCall.error}
							</pre>
						</div>
					)}

					{/* Result in YAML format */}
					{toolCall.status === "executed" && toolCall.result && (
						<div>
							<div className="text-xs font-semibold text-gray-700 mb-1">
								Return:
							</div>
							<pre className="text-xs bg-white/50 rounded p-2 overflow-x-auto max-h-64 overflow-y-auto whitespace-pre-wrap font-mono">
								{formatResult()}
							</pre>
						</div>
					)}

					{/* Auth scope */}
					{toolCall.status === "requires_auth" && toolCall.auth_scope && (
						<div>
							<div className="text-xs font-semibold text-gray-700 mb-1">
								Required Permission:
							</div>
							<div className="text-xs bg-white/50 rounded p-2">
								{toolCall.auth_scope}
							</div>
						</div>
					)}

					{/* Call ID */}
					<div className="text-xs text-gray-400 font-mono">
						call_id: {toolCall.id}
					</div>
				</div>
			)}
		</div>
	);
}
