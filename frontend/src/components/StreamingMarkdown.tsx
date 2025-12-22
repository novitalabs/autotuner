/**
 * StreamingMarkdown component - Renders markdown with streaming support
 *
 * Converts complete paragraphs (split by empty lines) to HTML while streaming,
 * keeping the last incomplete paragraph as plain text until finished.
 * Supports GitHub Flavored Markdown (tables, strikethrough, etc.)
 */

import { useState, useCallback, createContext, useContext } from "react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

interface StreamingMarkdownProps {
	content: string;
	isStreaming: boolean;
}

/** Context to pass original markdown source to custom components */
const MarkdownSourceContext = createContext<string>("");

/** Copy button component */
function CopyButton({
	text,
	variant = "dark",
}: {
	text: string;
	variant?: "dark" | "light";
}) {
	const [copied, setCopied] = useState(false);

	const handleCopy = useCallback(async () => {
		await navigator.clipboard.writeText(text);
		setCopied(true);
		setTimeout(() => setCopied(false), 2000);
	}, [text]);

	const colorClass =
		variant === "dark"
			? "text-gray-400 hover:text-gray-200"
			: "text-gray-400 hover:text-gray-600";

	return (
		<button
			onClick={handleCopy}
			className={`flex items-center gap-1 text-xs transition-colors ${colorClass}`}
		>
			{copied ? (
				<>
					<svg
						className="w-3.5 h-3.5"
						fill="none"
						stroke="currentColor"
						viewBox="0 0 24 24"
					>
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={2}
							d="M5 13l4 4L19 7"
						/>
					</svg>
					Copied
				</>
			) : (
				<>
					<svg
						className="w-3.5 h-3.5"
						fill="none"
						stroke="currentColor"
						viewBox="0 0 24 24"
					>
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth={2}
							d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
						/>
					</svg>
					Copy
				</>
			)}
		</button>
	);
}

/** Extract code block from markdown source */
function extractCodeBlock(source: string): string {
	const match = source.match(/```[\w]*\n([\s\S]*?)```/);
	return match ? match[0] : source;
}

/** Extract table from markdown source */
function extractTable(source: string): string {
	const lines = source.split("\n");
	const tableLines: string[] = [];
	for (const line of lines) {
		if (line.trim().startsWith("|") || line.trim().match(/^\|?[-:| ]+\|$/)) {
			tableLines.push(line);
		}
	}
	return tableLines.join("\n");
}

/** Custom code block with header */
function CodeBlock({
	children,
	...props
}: React.HTMLAttributes<HTMLPreElement> & { children?: React.ReactNode }) {
	const source = useContext(MarkdownSourceContext);

	// Extract code content and language
	const codeElement = children as React.ReactElement<{
		className?: string;
		children?: string;
	}>;
	const className = codeElement?.props?.className || "";
	const language = className.replace("language-", "") || "code";

	// Get original markdown for copying
	const originalCode = extractCodeBlock(source);

	return (
		<div className="rounded-lg overflow-hidden border border-gray-700 my-4">
			<div className="flex items-center justify-between px-3 py-1.5 bg-gray-800 border-b border-gray-700">
				<span className="text-xs text-gray-400 font-medium">{language}</span>
				<CopyButton text={originalCode} variant="dark" />
			</div>
			<pre
				{...props}
				className="!mt-0 !rounded-t-none bg-gray-900 text-gray-100 p-3 overflow-x-auto text-sm"
			>
				{children}
			</pre>
		</div>
	);
}

/** Custom table with header */
function TableBlock({
	children,
	...props
}: React.TableHTMLAttributes<HTMLTableElement> & {
	children?: React.ReactNode;
}) {
	const source = useContext(MarkdownSourceContext);

	// Get original markdown for copying
	const originalTable = extractTable(source);

	return (
		<div className="rounded-lg overflow-hidden border border-gray-200 my-4 w-fit">
			<div className="flex items-center justify-between px-3 py-1.5 bg-gray-50 border-b border-gray-200">
				<span className="text-xs text-gray-500 font-medium">Table</span>
				<CopyButton text={originalTable} variant="light" />
			</div>
			<table
				{...props}
				className="border-collapse [&_th]:border [&_th]:border-gray-200 [&_th]:px-2 [&_th]:py-1.5 [&_th]:bg-gray-50 [&_th]:font-semibold [&_th]:text-left [&_td]:border [&_td]:border-gray-200 [&_td]:px-2 [&_td]:py-1.5 [&_tr:first-child_th]:border-t-0 [&_tr:first-child_td]:border-t-0 [&_tr_th:first-child]:border-l-0 [&_tr_td:first-child]:border-l-0 [&_tr_th:last-child]:border-r-0 [&_tr_td:last-child]:border-r-0 [&_tr:last-child_th]:border-b-0 [&_tr:last-child_td]:border-b-0"
			>
				{children}
			</table>
		</div>
	);
}

/** Custom components for ReactMarkdown */
const markdownComponents: Components = {
	pre: CodeBlock,
	table: TableBlock,
};

/**
 * Split content into blocks, preserving tables and code blocks as single units.
 * Tables and code blocks should not be split even if they contain empty-looking lines.
 */
function splitIntoBlocks(content: string): string[] {
	const blocks: string[] = [];
	const lines = content.split("\n");
	let currentBlock: string[] = [];
	let inCodeBlock = false;
	let inTable = false;

	for (let i = 0; i < lines.length; i++) {
		const line = lines[i];
		const trimmedLine = line.trim();

		// Track code block state
		if (trimmedLine.startsWith("```")) {
			inCodeBlock = !inCodeBlock;
			currentBlock.push(line);
			continue;
		}

		// Track table state (line starts with |)
		if (trimmedLine.startsWith("|") || trimmedLine.match(/^\|?[-:| ]+\|$/)) {
			inTable = true;
			currentBlock.push(line);
			continue;
		}

		// End table on non-table line
		if (inTable && !trimmedLine.startsWith("|") && trimmedLine !== "") {
			inTable = false;
		}

		// Empty line outside of code block/table = block boundary
		if (trimmedLine === "" && !inCodeBlock && !inTable) {
			if (currentBlock.length > 0) {
				blocks.push(currentBlock.join("\n"));
				currentBlock = [];
			}
		} else {
			currentBlock.push(line);
		}
	}

	// Don't forget the last block
	if (currentBlock.length > 0) {
		blocks.push(currentBlock.join("\n"));
	}

	return blocks;
}

export default function StreamingMarkdown({
	content,
	isStreaming,
}: StreamingMarkdownProps) {
	// Split content into blocks (paragraphs, tables, code blocks)
	const blocks = splitIntoBlocks(content);

	// All but the last block are "complete" - render as markdown
	const completeBlocks = isStreaming ? blocks.slice(0, -1) : blocks;
	// Last block may be incomplete if still streaming
	const incompleteBlock = isStreaming ? blocks[blocks.length - 1] : null;

	return (
		<div
			className={[
				// Base prose styling
				"prose prose-slate max-w-none",
				// Paragraph spacing
				"prose-p:my-3 prose-p:leading-relaxed",
				// Headings - more prominent with better spacing
				"prose-headings:font-semibold prose-headings:text-gray-800",
				"prose-h1:text-2xl prose-h1:mt-8 prose-h1:mb-5",
				"prose-h2:text-xl prose-h2:mt-7 prose-h2:mb-4",
				"prose-h3:text-lg prose-h3:mt-6 prose-h3:mb-3",
				"prose-h4:text-base prose-h4:mt-5 prose-h4:mb-2",
				// Lists with proper spacing
				"prose-ul:my-3 prose-ul:pl-5 prose-ol:my-3 prose-ol:pl-5",
				"prose-li:my-1.5 prose-li:leading-relaxed",
				// Inline code styling
				"prose-code:bg-gray-100 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-gray-700 prose-code:text-sm prose-code:font-medium",
				"prose-code:before:content-none prose-code:after:content-none",
				// Code blocks
				"prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-pre:rounded-lg prose-pre:p-4 prose-pre:my-4 prose-pre:overflow-x-auto",
				// Tables - clean design with borders
				"prose-table:my-4 prose-table:border-collapse",
				"prose-thead:bg-gray-50",
				"prose-th:px-2 prose-th:py-1.5 prose-th:text-left prose-th:font-semibold prose-th:text-gray-700 prose-th:border prose-th:border-gray-200",
				"prose-td:px-2 prose-td:py-1.5 prose-td:border prose-td:border-gray-200",
				"prose-tr:hover:bg-gray-50",
				// Blockquotes
				"prose-blockquote:border-l-4 prose-blockquote:border-primary-500 prose-blockquote:pl-4 prose-blockquote:italic prose-blockquote:text-gray-600 prose-blockquote:my-4",
				// Links
				"prose-a:text-primary-600 prose-a:no-underline hover:prose-a:underline",
				// Strong/bold text
				"prose-strong:font-semibold prose-strong:text-gray-800",
			].join(" ")}
		>
			{completeBlocks.map((block, idx) => (
				<MarkdownSourceContext.Provider key={idx} value={block}>
					<div className="mb-5 [&>h1]:mt-6 [&>h1]:mb-4 [&>h2]:mt-5 [&>h2]:mb-3 [&>h3]:mt-4 [&>h3]:mb-2 [&>h4]:mt-3 [&>h4]:mb-2 [&>p]:my-3 [&>ul]:my-3 [&>ol]:my-3 [&>blockquote]:my-4">
						<ReactMarkdown
							remarkPlugins={[remarkGfm]}
							components={markdownComponents}
						>
							{block}
						</ReactMarkdown>
					</div>
				</MarkdownSourceContext.Provider>
			))}
			{incompleteBlock && (
				<span className="whitespace-pre-wrap">{incompleteBlock}</span>
			)}
			{isStreaming && (
				<span className="inline-block w-2 h-4 ml-1 bg-emerald-500 animate-pulse" />
			)}
		</div>
	);
}
