/**
 * Makora Provider Extension
 *
 * Registers Makora (inference.makora.com) as a custom provider using the
 * OpenAI completions API.
 *
 * Makora is an inference optimization platform serving open-weight models via
 * a unified OpenAI-compatible API at https://inference.makora.com/v1. Each
 * model is hosted on vLLM and speaks the standard OpenAI chat completions
 * protocol. Most models use the shared provider baseUrl; models not yet
 * on the unified endpoint retain a per-model `baseUrl` override.
 *
 * Model resolution strategy: static models.json merged with custom-models.json
 *
 * Reasoning notes:
 *   - DeepSeek V4 Pro: reasoning via chat_template_kwargs.thinking on vLLM.
 *     pi sends thinking: { type } via the "deepseek" thinkingFormat, but vLLM
 *     ignores that — the before_provider_request hook rewrites the payload to
 *     use chat_template_kwargs: { thinking: true } instead.
 *     Returns reasoning_content field.
 *   - DeepSeek V4 Flash: reasoning via include_reasoning +
 *     chat_template_kwargs.thinking on vLLM.
 *     The before_provider_request hook rewrites the payload to replace
 *     thinking: { type } with include_reasoning: true +
 *     chat_template_kwargs: { thinking: true }.
 *     include_reasoning alone returns reasoning: null on this vLLM build.
 *     Returns reasoning field.
 *   - GLM 5.1 FP8: reasoning via chat_template_kwargs.enable_thinking.
 *     NOTE: vLLM may leak chain-of-thought into content instead of the
 *     reasoning field on some builds. See
 *     https://github.com/vllm-project/vllm/issues/31319
 *     Also: vLLM's streaming parser omits delta.tool_calls when the model
 *     calls tools, finishing with finish_reason: "tool_calls" but an empty
 *     delta. Setting zaiToolStream: true sends tool_stream: true in the
 *     request, which forces vLLM to use the explicit tool streaming path
 *     that correctly emits tool call chunks.
 *   - GPT-OSS 120B: reasoning always on; returns `reasoning` field.
 *   - Kimi K2.6 NVFP4: reasoning always on by default; returns `reasoning`
 *     field. Can be toggled via enable_thinking.
 *   - Qwen 3.6 models: reasoning via chat_template_kwargs.enable_thinking;
 *     returns `reasoning` field.
 *   - Llama 3.3 70B: not a reasoning model.
 *
 * Developer role is NOT supported by any of the chat templates on Makora's
 * vLLM deployment (prompts with role: "developer" are silently dropped).
 * supportsDeveloperRole is set to false for all models.
 *
 * Usage:
 *   # Option 1: Store in auth.json (recommended)
 *   # Add to ~/.pi/agent/auth.json:
 *   #   "makora": { "type": "api_key", "key": "your-api-key" }
 *
 *   # Option 2: Set as environment variable
 *   export MAKORA_OPTIMIZE_TOKEN=your-api-key
 *
 *   # Run pi with the extension
 *   pi -e /path/to/pi-makora-provider
 *
 * Then use /model to select from available models.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import modelsData from "./models.json" with { type: "json" };
import customModelsData from "./custom-models.json" with { type: "json" };
import patchData from "./patch.json" with { type: "json" };

// ── Inline types for client-side tool call repair ──────────────────────
// The makora provider has no runtime dependencies, so we inline these
// instead of importing from @earendil-works/pi-ai.

interface ToolCallBlock {
  type: "toolCall";
  id: string;
  name: string;
  arguments: Record<string, any>;
  thoughtSignature?: string;
}

interface TextBlock {
  type: "text";
  text: string;
  textSignature?: string;
}

interface ThinkingBlock {
  type: "thinking";
  thinking: string;
  thinkingSignature?: string;
}

type ContentBlock = TextBlock | ThinkingBlock | ToolCallBlock;

interface AssistantMsg {
  role: "assistant";
  content: ContentBlock[];
  stopReason?: string;
  model?: string;
  responseModel?: string;
}

// Types

interface JsonModel {
  id: string;
  name: string;
  reasoning: boolean;
  input: string[];
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };
  contextWindow: number;
  maxTokens: number;
  baseUrl?: string;
  notes?: string;
  thinkingLevelMap?: Record<string, string | null>;
  headers?: Record<string, string>;
  vision?: {
    maxImagesPerRequest?: number;
  };
  compat?: {
    supportsDeveloperRole?: boolean;
    supportsStore?: boolean;
    maxTokensField?: "max_completion_tokens" | "max_tokens";
    thinkingFormat?:
      | "openai"
      | "openrouter"
      | "deepseek"
      | "together"
      | "zai"
      | "qwen"
      | "qwen-chat-template";
    supportsReasoningEffort?: boolean;
    requiresReasoningContentOnAssistantMessages?: boolean;
    requiresToolResultName?: boolean;
    requiresAssistantAfterToolResult?: boolean;
    cacheControlFormat?: "anthropic";
  };
}

interface PatchEntry {
  name?: string;
  reasoning?: boolean;
  input?: string[];
  cost?: {
    input?: number;
    output?: number;
    cacheRead?: number;
    cacheWrite?: number;
  };
  contextWindow?: number;
  maxTokens?: number;
  baseUrl?: string;
  notes?: string;
  thinkingLevelMap?: Record<string, string | null>;
  headers?: Record<string, string>;
  compat?: Record<string, unknown>;
}

type PatchMap = Record<string, PatchEntry>;

// Patch Application

function applyPatch(model: JsonModel, patch: PatchEntry): JsonModel {
  const result = { ...model };

  if (patch.name !== undefined) result.name = patch.name;
  if (patch.reasoning !== undefined) result.reasoning = patch.reasoning;
  if (patch.input !== undefined) result.input = patch.input;
  if (patch.contextWindow !== undefined) result.contextWindow = patch.contextWindow;
  if (patch.maxTokens !== undefined) result.maxTokens = patch.maxTokens;
  if (patch.baseUrl !== undefined) result.baseUrl = patch.baseUrl;
  if (patch.notes !== undefined) result.notes = patch.notes;
  if (patch.thinkingLevelMap !== undefined) result.thinkingLevelMap = { ...patch.thinkingLevelMap };
  if (patch.headers !== undefined) result.headers = { ...patch.headers };

  if (patch.cost) {
    result.cost = {
      input: patch.cost.input ?? result.cost.input,
      output: patch.cost.output ?? result.cost.output,
      cacheRead: patch.cost.cacheRead ?? result.cost.cacheRead,
      cacheWrite: patch.cost.cacheWrite ?? result.cost.cacheWrite,
    };
  }
  if (patch.compat) {
    result.compat = { ...(result.compat || {}), ...patch.compat };
  }

  if (!result.reasoning && result.compat?.thinkingFormat) {
    delete result.compat.thinkingFormat;
  }
  if (result.compat && Object.keys(result.compat).length === 0) {
    delete result.compat;
  }

  return result;
}

/** Merge static models with any user-defined custom models */
function buildModels(
  base: JsonModel[],
  custom: JsonModel[],
  patch: PatchMap
): JsonModel[] {
  const modelMap = new Map<string, JsonModel>();

  for (const model of base) {
    modelMap.set(model.id, model);
  }

  for (const [id, patchEntry] of Object.entries(patch)) {
    const existing = modelMap.get(id);
    if (existing) {
      modelMap.set(id, applyPatch(existing, patchEntry));
    }
  }

  for (const model of custom) {
    const existing = modelMap.get(model.id);
    const patchEntry = patch[model.id];
    if (existing && patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else if (existing) {
      modelMap.set(model.id, model);
    } else if (patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else {
      modelMap.set(model.id, model);
    }
  }

  return Array.from(modelMap.values());
}

// Extension Entry Point

const PROVIDER_ID = "makora";
const BASE_URL = "https://inference.makora.com/v1";

const DS_PRO_ID = "deepseek-ai/DeepSeek-V4-Pro";
const DS_FLASH_ID = "deepseek-ai/DeepSeek-V4-Flash";

const DS_VLLM_MODELS = new Set([DS_PRO_ID, DS_FLASH_ID]);

// Models that need client-side tool call repair.
// vLLM is missing --enable-auto-tool-choice and --tool-call-parser for these,
// so tool call tokens pass through as raw text when skip_special_tokens is false.
const KIMI_K2_ID = "nvidia/Kimi-K2.6-NVFP4";
const QWEN_36_27B_ID = "unsloth/Qwen3.6-27B-NVFP4";
const QWEN_36_35B_ID = "unsloth/Qwen3.6-35B-A3B-NVFP4";

const CLIENT_SIDE_TOOL_PARSE_MODELS = new Set([
  KIMI_K2_ID,
  QWEN_36_27B_ID,
  QWEN_36_35B_ID,
]);

/**
 * Intercept the request payload for models that need client-side fixes.
 *
 * DeepSeek V4: pi's "deepseek" thinkingFormat sends `thinking: { type: "enabled" }`
 * which Makora's vLLM ignores. Rewrite to vLLM-native params.
 *
 * Kimi K2.6 / Qwen 3.6: vLLM lacks --enable-auto-tool-choice and --tool-call-parser.
 * Set skip_special_tokens: false so raw tool call tokens pass through in content.
 */
function rewritePayload(payload: Record<string, unknown>): Record<string, unknown> {
  const model = payload.model as string | undefined;
  if (!model || !DS_VLLM_MODELS.has(model)) return payload;

  const p = { ...payload };

  // Remove the DeepSeek API-style `thinking` param that vLLM ignores
  delete p.thinking;

  if (model === DS_PRO_ID) {
    // DS Pro: chat_template_kwargs.thinking + reasoning_effort
    const ctq = (p.chat_template_kwargs as Record<string, unknown>) ?? {};
    p.chat_template_kwargs = { ...ctq, thinking: true };
  } else if (model === DS_FLASH_ID) {
    // DS Flash: include_reasoning + chat_template_kwargs.thinking + reasoning_effort
    // vLLM requires *both* include_reasoning and chat_template_kwargs.thinking:
    // include_reasoning alone returns reasoning: null.
    p.include_reasoning = true;
    const ctq = (p.chat_template_kwargs as Record<string, unknown>) ?? {};
    p.chat_template_kwargs = { ...ctq, thinking: true };
  }

  return p;
}

/**
 * Set skip_special_tokens for Kimi/Qwen so raw tool call tokens (e.g.
 * `<|tool_call_begin|>`) survive vLLM's streaming parser and appear in content.
 */
function rewriteToolCallPayload(payload: Record<string, unknown>): Record<string, unknown> {
  const model = payload.model as string | undefined;
  if (!model || !CLIENT_SIDE_TOOL_PARSE_MODELS.has(model)) return payload;

  const p = { ...payload };
  p.skip_special_tokens = false;
  return p;
}

// ── Client-side tool call parsers ────────────────────────────────────

/** Parse Kimi K2.6 native tool call format from raw text content. */
function parseKimiToolCalls(text: string, callIdBase: string): { calls: ToolCallBlock[]; remaining: string } {
  const calls: ToolCallBlock[] = [];
  let remaining = text;
  let callIndex = 0;

  const beginMarker = "<|tool_call_begin|>";
  const endMarker = "<|tool_call_end|>";

  while (true) {
    const beginIdx = remaining.indexOf(beginMarker);
    if (beginIdx === -1) break;

    const afterBegin = remaining.slice(beginIdx + beginMarker.length);
    const endIdx = afterBegin.indexOf(endMarker);
    if (endIdx === -1) break;

    const inner = afterBegin.slice(0, endIdx).trim();
    const newlineIdx = inner.indexOf("\n");

    let name: string;
    let argsStr: string;

    if (newlineIdx === -1) {
      name = inner.trim();
      argsStr = "{}";
    } else {
      name = inner.slice(0, newlineIdx).trim();
      argsStr = inner.slice(newlineIdx + 1).trim();
    }

    let args: Record<string, any>;
    try {
      args = JSON.parse(argsStr);
    } catch {
      remaining = remaining.slice(beginIdx + beginMarker.length + endIdx + endMarker.length);
      continue;
    }

    calls.push({
      type: "toolCall" as const,
      id: `${callIdBase}_kimi_${callIndex++}`,
      name,
      arguments: args,
    });

    remaining = remaining.slice(0, beginIdx) + afterBegin.slice(endIdx + endMarker.length);
  }

  return { calls, remaining: remaining.trim() };
}

/** Parse Qwen 3.6 hermes-style tool call format from raw text content. */
function parseQwenToolCalls(text: string, callIdBase: string): { calls: ToolCallBlock[]; remaining: string } {
  const calls: ToolCallBlock[] = [];
  let callIndex = 0;

  const regex = /<function=([^>]+)>([\s\S]*?)<\/function>/g;

  let match: RegExpExecArray | null;
  while ((match = regex.exec(text)) !== null) {
    const name = match[1].trim();
    const argsStr = match[2].trim();

    let args: Record<string, any>;
    try {
      args = JSON.parse(argsStr);
    } catch {
      continue;
    }

    calls.push({
      type: "toolCall" as const,
      id: `${callIdBase}_qwen_${callIndex++}`,
      name,
      arguments: args,
    });
  }

  const remaining = text.replace(regex, "").trim();
  return { calls, remaining };
}

/**
 * Detect and parse tool call tokens from an assistant message's text content.
 * Returns a new message with ToolCall blocks replacing the raw token text.
 */
function repairToolCalls(
  message: AssistantMsg,
  model: string
): AssistantMsg {
  const content = message.content;
  const newContent: ContentBlock[] = [];
  let hasToolCalls = false;

  // Accumulate text across blocks for parsing (tool call tokens may span blocks)
  const textBlocks: { index: number; block: TextBlock }[] = [];

  for (let i = 0; i < content.length; i++) {
    const block = content[i];
    if (block.type === "text") {
      textBlocks.push({ index: i, block });
    }
  }

  if (textBlocks.length === 0) return message;

  // Combine all text blocks for parsing
  const combinedText = textBlocks.map((t) => t.block.text).join("");

  // Choose parser based on model
  const isKimi = model === KIMI_K2_ID;
  const isQwen =
    model === QWEN_36_27B_ID || model === QWEN_36_35B_ID;

  if (!isKimi && !isQwen) return message;

  const callIdBase = `call_${Date.now()}`;
  const result = isKimi
    ? parseKimiToolCalls(combinedText, callIdBase)
    : parseQwenToolCalls(combinedText, callIdBase);

  if (result.calls.length === 0) return message;

  hasToolCalls = true;

  // Rebuild content: non-text blocks as-is, text content replaced with
  // remaining text + tool call blocks
  let textBlockIdx = 0;
  for (let i = 0; i < content.length; i++) {
    const block = content[i];
    if (block.type !== "text") {
      newContent.push(block);
      continue;
    }
    textBlockIdx++;
    // Only inject remaining text and tool calls after the last text block
    if (textBlockIdx === textBlocks.length) {
      if (result.remaining) {
        newContent.push({ type: "text", text: result.remaining });
      }
      for (const call of result.calls) {
        newContent.push(call);
      }
    }
  }

  return {
    ...message,
    content: newContent,
    stopReason: hasToolCalls ? "toolUse" : message.stopReason,
  };
}

export default function (pi: ExtensionAPI) {
  const embeddedModels = modelsData as JsonModel[];
  const customModels = customModelsData as JsonModel[];
  const patches = patchData as PatchMap;

  const models = buildModels(embeddedModels, customModels, patches);

  // apiKey resolution order: auth.json ("makora" key) → MAKORA_OPTIMIZE_TOKEN env var
  pi.registerProvider(PROVIDER_ID, {
    name: "Makora",
    baseUrl: BASE_URL,
    apiKey: "$MAKORA_OPTIMIZE_TOKEN",
    api: "openai-completions",
    models,
  });

  pi.on("before_provider_request", (event) => {
    const payload = event.payload as Record<string, unknown> | undefined;
    if (!payload || typeof payload.model !== "string") return;
    let p = rewritePayload(payload);
    p = rewriteToolCallPayload(p);
    return p;
  });

  pi.on("message_end", (event) => {
    const msg = event.message;
    if (msg.role !== "assistant") return;

    const model =
      (msg as AssistantMsg).model ||
      (msg as Record<string, unknown>).responseModel ||
      "";
    if (!model || !CLIENT_SIDE_TOOL_PARSE_MODELS.has(model as string)) return;

    const repaired = repairToolCalls(msg as AssistantMsg, model as string);
    if (repaired !== msg) {
      return { message: repaired };
    }
  });
}

