/**
 * Tests for GLM tool_calls stripping in before_provider_request.
 *
 * These tests verify the fix for the ZAI/vLLM crash:
 *   "'list object' has no attribute 'items'"
 *   "'str object' has no attribute 'items'"
 *
 * The crash occurs when an assistant message containing a `tool_calls` field
 * is sent to GLM models on Makora's vLLM. The ZAI chat template calls .items()
 * on the tool_calls list (or the JSON-string arguments field), causing an
 * AttributeError that leaks into the HTTP 400 response body.
 *
 * Fix: strip tool_calls from assistant messages and convert them to GLM-native
 * XML text in content, which the model natively understands in conversation
 * history.
 *
 * IMPORTANT — Known limitations and risks of this approach:
 *
 * 1. ORPHANED TOOL RESULTS: When tool_calls are stripped from an assistant
 *    message, the subsequent role:"tool" messages become "orphaned" — their
 *    tool_call_id no longer references a tool_calls entry in any assistant
 *    message. OpenAI's API spec requires every role:"tool" message to have a
 *    matching tool_call_id in a preceding assistant message.
 *    - Impact: vLLM's GLM chat template renders role:"tool" messages via its
 *      tool-observation branch, which does NOT validate tool_call_id linkage
 *      (unlike OpenAI's strict API). The observation is rendered as plain text.
 *    - Risk: If vLLM tightens validation in a future version, orphaned tool
 *      results could be rejected. Mitigation: monitor vLLM releases; if they
 *      add strict validation, convert role:"tool" messages to role:"user"
 *      text as well.
 *
 * 2. NO TOOL EXECUTION HINT: By converting tool_calls to text, the model loses
 *    the structured signal that it previously called a tool. In follow-up turns,
 *    the model sees its prior tool call as natural text, which is how GLM was
 *    originally trained to handle tool calls (the XML format IS its native
 *    representation).
 *    - Impact: Minimal for GLM — the XML format is the model's native tool-call
 *      grammar, not a workaround. The model treats it as if it emitted the call
 *      in text mode.
 *
 * 3. DOWNSTREAM EXTENSION INTERFERENCE: If another extension's context hook
 *    or before_provider_request handler expects tool_calls to be present in
 *    the payload, stripping them here could break that extension.
 *    - Impact: Low. before_provider_request handlers run in extension load
 *      order; this extension should be loaded first to ensure the strip
 *      happens before any other handler inspects the payload.
 *    - Mitigation: Document load-order requirement in README.
 *
 * 4. FUTURE UPSTREAM FIX: If Makora/vLLM fixes the .items() crash upstream,
 *    this transform becomes a no-op (the XML text is still valid GLM input,
 *    so the model handles it fine either way). The transform can be safely
 *    removed once the upstream fix is confirmed.
 */

import { describe, expect, it } from "vitest";
import { stripGlmToolCalls, toolCallToGlmXml, isObject } from "../index.js";

// ─── Helper: construct a GLM sentinel tag from parts ───
// Using String.fromCharCode to avoid any encoding issues in test source
const TC_OPEN = String.fromCharCode(60) + "tool_call" + String.fromCharCode(62);
const TC_CLOSE = String.fromCharCode(60, 47) + "tool_call" + String.fromCharCode(62);
const AK_OPEN = String.fromCharCode(60) + "arg_key" + String.fromCharCode(62);
const AK_CLOSE = String.fromCharCode(60, 47) + "arg_key" + String.fromCharCode(62);
const AV_OPEN = String.fromCharCode(60) + "arg_value" + String.fromCharCode(62);
const AV_CLOSE = String.fromCharCode(60, 47) + "arg_value" + String.fromCharCode(62);

/** Build expected GLM XML for a single tool call */
function expectedGlmXml(name: string, args: Record<string, string>): string {
	const argLines = Object.entries(args).map(
		([k, v]) => `${AK_OPEN}${k}${AK_CLOSE}\n${AV_OPEN}${v}${AV_CLOSE}`,
	);
	return `${TC_OPEN}${name}\n${argLines.join("\n")}\n${TC_CLOSE}`;
}

// ─── Type definitions for test payloads ───

interface OpenAIToolCall {
	id: string;
	type: "function";
	function: {
		name: string;
		arguments: string;
	};
}

interface ChatMessage {
	role: string;
	content: string | null;
	tool_calls?: OpenAIToolCall[];
	tool_call_id?: string;
}

interface ChatPayload {
	model: string;
	messages: ChatMessage[];
}

// ─── isObject tests ───

describe("isObject", () => {
	it("returns true for plain objects", () => {
		expect(isObject({})).toBe(true);
		expect(isObject({ a: 1 })).toBe(true);
	});

	it("returns false for null, arrays, and primitives", () => {
		expect(isObject(null)).toBe(false);
		expect(isObject(undefined)).toBe(false);
		expect(isObject([1, 2])).toBe(false);
		expect(isObject("string")).toBe(false);
		expect(isObject(42)).toBe(false);
	});
});

// ─── toolCallToGlmXml tests ───

describe("toolCallToGlmXml", () => {
	it("converts a single tool call with string args to GLM XML", () => {
		const tc: Record<string, unknown> = {
			id: "call_1",
			type: "function",
			function: {
				name: "get_weather",
				arguments: JSON.stringify({ city: "Paris" }),
			},
		};
		const result = toolCallToGlmXml(tc);
		expect(result).toContain("get_weather");
		expect(result).toContain("city");
		expect(result).toContain("Paris");
		expect(result).toBe(expectedGlmXml("get_weather", { city: "Paris" }));
	});

	it("handles non-string argument values by JSON-stringifying them", () => {
		const tc: Record<string, unknown> = {
			id: "call_2",
			type: "function",
			function: {
				name: "search",
				arguments: JSON.stringify({ limit: 10, tags: ["a", "b"] }),
			},
		};
		const result = toolCallToGlmXml(tc);
		expect(result).toContain("limit");
		expect(result).toContain("10");
		expect(result).toContain("tags");
		expect(result).toContain('["a","b"]');
	});

	it("handles empty arguments", () => {
		const tc: Record<string, unknown> = {
			id: "call_3",
			type: "function",
			function: { name: "ping", arguments: "{}" },
		};
		const result = toolCallToGlmXml(tc);
		expect(result).toContain("ping");
		// No arg_key blocks when args is empty
		expect(result).not.toContain("arg_key");
	});

	it("handles invalid JSON arguments gracefully", () => {
		const tc: Record<string, unknown> = {
			id: "call_4",
			type: "function",
			function: { name: "broken", arguments: "not valid json" },
		};
		const result = toolCallToGlmXml(tc);
		expect(result).toContain("broken");
		// Should not crash, just produce no arg lines
		expect(result).not.toContain("arg_key");
	});

	it("handles missing function field", () => {
		const tc: Record<string, unknown> = { id: "call_5", type: "function" };
		const result = toolCallToGlmXml(tc);
		expect(result).toBeDefined();
		expect(typeof result).toBe("string");
	});
});

// ─── stripGlmToolCalls tests ───

describe("stripGlmToolCalls", () => {
	// ─── BEFORE/AFTER: Single tool call with preceding text content ───
	it("BEFORE: assistant has tool_calls field → AFTER: tool_calls stripped, XML in content", () => {
		const before: ChatPayload = {
			model: "zai-org/GLM-5.2-FP8",
			messages: [
				{ role: "user", content: "What's the weather in Paris?" },
				{
					role: "assistant",
					content: "Let me check the weather for you.",
					tool_calls: [
						{
							id: "call_1",
							type: "function",
							function: {
								name: "get_weather",
								arguments: JSON.stringify({ city: "Paris" }),
							},
						},
					],
				},
				{ role: "tool", tool_call_id: "call_1", content: '{"temp":22}' },
				{ role: "user", content: "Thanks!" },
			],
		};

		// BEFORE: assistant message has tool_calls
		const beforeAssistant = before.messages[1]!;
		expect(beforeAssistant.tool_calls).toBeDefined();
		expect(beforeAssistant.tool_calls!.length).toBe(1);
		expect(beforeAssistant.content).toBe("Let me check the weather for you.");

		const after = stripGlmToolCalls(before as unknown as Record<string, unknown>) as unknown as ChatPayload;

		// AFTER: tool_calls removed, content has GLM XML appended
		const afterAssistant = after.messages[1]!;
		expect(afterAssistant.tool_calls).toBeUndefined();
		expect(typeof afterAssistant.content).toBe("string");
		expect(afterAssistant.content).toContain("Let me check the weather for you.");
		expect(afterAssistant.content).toContain("get_weather");
		expect(afterAssistant.content).toContain("city");
		expect(afterAssistant.content).toContain("Paris");
	});

	// ─── BEFORE/AFTER: Assistant with null content + tool_calls ───
	it("BEFORE: assistant has null content + tool_calls → AFTER: XML replaces null content", () => {
		const before: ChatPayload = {
			model: "zai-org/GLM-5.1-FP8",
			messages: [
				{
					role: "assistant",
					content: null,
					tool_calls: [
						{
							id: "c1",
							type: "function",
							function: {
								name: "bash",
								arguments: JSON.stringify({ command: "ls -la" }),
							},
						},
					],
				},
				{ role: "tool", tool_call_id: "c1", content: "file1.txt\nfile2.txt" },
			],
		};

		// BEFORE
		expect(before.messages[0]!.content).toBeNull();
		expect(before.messages[0]!.tool_calls).toBeDefined();

		const after = stripGlmToolCalls(before as unknown as Record<string, unknown>) as unknown as ChatPayload;

		// AFTER
		const afterAssistant = after.messages[0]!;
		expect(afterAssistant.tool_calls).toBeUndefined();
		expect(typeof afterAssistant.content).toBe("string");
		expect(afterAssistant.content).toContain("bash");
		expect(afterAssistant.content).toContain("command");
		expect(afterAssistant.content).toContain("ls -la");
	});

	// ─── BEFORE/AFTER: Multiple tool calls in one assistant message ───
	it("BEFORE: multiple tool_calls → AFTER: all converted to XML blocks", () => {
		const before: ChatPayload = {
			model: "zai-org/GLM-5.2-FP8",
			messages: [
				{
					role: "assistant",
					content: "",
					tool_calls: [
						{
							id: "c1",
							type: "function",
							function: {
								name: "read",
								arguments: JSON.stringify({ path: "/a" }),
							},
						},
						{
							id: "c2",
							type: "function",
							function: {
								name: "read",
								arguments: JSON.stringify({ path: "/b" }),
							},
						},
					],
				},
			],
		};

		const after = stripGlmToolCalls(before as unknown as Record<string, unknown>) as unknown as ChatPayload;

		const content = after.messages[0]!.content as string;
		// Two tool_call blocks
		const tcMatches = content.match(/tool_call/g);
		expect(tcMatches).not.toBeNull();
		expect(tcMatches!.length).toBeGreaterThanOrEqual(2);
		expect(content).toContain("read");
		expect(content).toContain("/a");
		expect(content).toContain("/b");
	});

	// ─── No-op: payload without tool_calls ───
	it("returns payload unchanged when no assistant has tool_calls", () => {
		const payload: ChatPayload = {
			model: "zai-org/GLM-5.2-FP8",
			messages: [
				{ role: "user", content: "hi" },
				{ role: "assistant", content: "hello" },
			],
		};

		const result = stripGlmToolCalls(payload as unknown as Record<string, unknown>);
		expect(result).toBe(payload); // Same reference — no modification
	});

	// ─── No-op: non-assistant messages with tool_calls-like fields ───
	it("does not modify tool result messages", () => {
		const payload: ChatPayload = {
			model: "zai-org/GLM-5.2-FP8",
			messages: [
				{
					role: "assistant",
					content: "checking",
					tool_calls: [
						{
							id: "c1",
							type: "function",
							function: { name: "bash", arguments: JSON.stringify({ command: "pwd" }) },
						},
					],
				},
				{ role: "tool", tool_call_id: "c1", content: "/home/user" },
			],
		};

		const after = stripGlmToolCalls(payload as unknown as Record<string, unknown>) as unknown as ChatPayload;

		// Tool result message should be unchanged
		expect(after.messages[1]!.role).toBe("tool");
		expect(after.messages[1]!.content).toBe("/home/user");
		expect(after.messages[1]!.tool_call_id).toBe("c1");
	});

	// ─── Edge case: empty tool_calls array ───
	it("does not modify assistant with empty tool_calls array", () => {
		const payload: ChatPayload = {
			model: "zai-org/GLM-5.2-FP8",
			messages: [
				{
					role: "assistant",
					content: "hello",
					tool_calls: [],
				},
			],
		};

		const result = stripGlmToolCalls(payload as unknown as Record<string, unknown>) as unknown as ChatPayload;
		// Empty array → no modification (xmlBlocks.length === 0)
		expect(result.messages[0]!.tool_calls).toEqual([]);
		expect(result.messages[0]!.content).toBe("hello");
	});

	// ─── Edge case: messages not an array ───
	it("returns payload unchanged when messages is not an array", () => {
		const payload = { model: "test", messages: "not an array" };
		const result = stripGlmToolCalls(payload as unknown as Record<string, unknown>);
		expect(result).toBe(payload);
	});

	// ─── Edge case: null/non-object messages ───
	it("handles null and non-object messages gracefully", () => {
		const payload = {
			model: "test",
			messages: [null, "string", 42, { role: "assistant", content: "ok" }],
		};
		// Should not crash
		const result = stripGlmToolCalls(payload as unknown as Record<string, unknown>);
		expect(result).toBeDefined();
	});

	// ─── Multi-turn conversation with multiple tool-call rounds ───
	it("strips tool_calls from multiple assistant messages in a multi-turn conversation", () => {
		const before: ChatPayload = {
			model: "zai-org/GLM-5.2-FP8",
			messages: [
				{ role: "user", content: "Read file A then file B" },
				{
					role: "assistant",
					content: "Reading file A.",
					tool_calls: [
						{
							id: "c1",
							type: "function",
							function: { name: "read", arguments: JSON.stringify({ path: "/a" }) },
						},
					],
				},
				{ role: "tool", tool_call_id: "c1", content: "content A" },
				{
					role: "assistant",
					content: "Reading file B.",
					tool_calls: [
						{
							id: "c2",
							type: "function",
							function: { name: "read", arguments: JSON.stringify({ path: "/b" }) },
						},
					],
				},
				{ role: "tool", tool_call_id: "c2", content: "content B" },
			],
		};

		const after = stripGlmToolCalls(before as unknown as Record<string, unknown>) as unknown as ChatPayload;

		// First assistant: tool_calls stripped
		expect(after.messages[1]!.tool_calls).toBeUndefined();
		expect(after.messages[1]!.content).toContain("Reading file A.");
		expect(after.messages[1]!.content).toContain("/a");

		// Second assistant: tool_calls stripped
		expect(after.messages[3]!.tool_calls).toBeUndefined();
		expect(after.messages[3]!.content).toContain("Reading file B.");
		expect(after.messages[3]!.content).toContain("/b");

		// Tool results untouched
		expect(after.messages[2]!.content).toBe("content A");
		expect(after.messages[4]!.content).toBe("content B");
	});

	// ─── Preserves non-tool_calls fields on assistant messages ───
	it("preserves other fields on assistant messages (role, reasoning_content)", () => {
		const payload = {
			model: "zai-org/GLM-5.2-FP8",
			messages: [
				{
					role: "assistant",
					content: "thinking...",
					reasoning_content: "Let me analyze this.",
					tool_calls: [
						{
							id: "c1",
							type: "function",
							function: { name: "bash", arguments: JSON.stringify({ command: "echo hi" }) },
						},
					],
				},
			],
		};

		const after = stripGlmToolCalls(payload as unknown as Record<string, unknown>) as {
			messages: Array<Record<string, unknown>>;
		};

		expect(after.messages[0]!.role).toBe("assistant");
		expect(after.messages[0]!.reasoning_content).toBe("Let me analyze this.");
		expect(after.messages[0]!.tool_calls).toBeUndefined();
	});
});
