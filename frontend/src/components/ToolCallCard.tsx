/**
 * ToolCallCard component - displays a tool execution
 */

import { useState } from "react";
import { ChevronDown, ChevronRight, CheckCircle, AlertCircle, XCircle, Lock, Loader2 } from "lucide-react";
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

	const getStatusText = () => {
		switch (toolCall.status) {
			case "executing":
				return "Executing...";
			case "executed":
				return "Executed";
			case "requires_auth":
				return "Authorization Required";
			case "failed":
				return "Failed";
			default:
				return "Unknown";
		}
	};

	const formatJson = (obj: any): string => {
		try {
			return JSON.stringify(obj, null, 2);
		} catch {
			return String(obj);
		}
	};

	return (
		<div className={`border rounded-lg p-3 mt-2 ${getStatusColor()}`}>
			{/* Header - always visible */}
			<div className="flex items-start gap-2">
				<button
					onClick={() => setIsExpanded(!isExpanded)}
					className="mt-0.5 hover:bg-white/50 rounded p-0.5 transition-colors"
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
						<span className="font-mono text-sm font-semibold">
							{toolCall.tool_name}
						</span>
						<span className="text-xs text-gray-500">
							{getStatusText()}
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

					{/* Error message for failed status */}
					{toolCall.status === "failed" && toolCall.error && (
						<div className="mt-2 text-xs text-red-700 bg-red-100 rounded p-2">
							{toolCall.error}
						</div>
					)}
				</div>
			</div>

			{/* Expanded content */}
			{isExpanded && (
				<div className="mt-3 pl-6 space-y-2">
					{/* Arguments */}
					{Object.keys(toolCall.args).length > 0 && (
						<div>
							<div className="text-xs font-semibold text-gray-700 mb-1">
								Arguments:
							</div>
							<pre className="text-xs bg-white/50 rounded p-2 overflow-x-auto">
								{formatJson(toolCall.args)}
							</pre>
						</div>
					)}

					{/* Result */}
					{toolCall.status === "executed" && toolCall.result && (
						<div>
							<div className="text-xs font-semibold text-gray-700 mb-1">
								Result:
							</div>
							<pre className="text-xs bg-white/50 rounded p-2 overflow-x-auto max-h-48 overflow-y-auto">
								{toolCall.result}
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
					<div className="text-xs text-gray-400">
						ID: {toolCall.id}
					</div>
				</div>
			)}
		</div>
	);
}
