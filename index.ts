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
 *   - GLM 5.2 FP8: reasoning via chat_template_kwargs.enable_thinking.
 *     Thinking levels (minimal/low/medium/high/max) are copied from the
 *     neuralwatt provider's GLM 5.2 configuration and mapped through pi's
 *     qwen-chat-template thinkingFormat. Same vLLM streaming-parser issue as
 *     GLM 5.1 (delta.tool_calls omitted, finish_reason "tool_calls" with an
 *     empty delta → empty arguments); setGlmToolStream sends tool_stream: true.
 *   - GPT-OSS 120B: reasoning always on; returns `reasoning` field.
 *   - Kimi K2.6 NVFP4 / Kimi K2.7 Code: reasoning always on by default;
 *     returns `reasoning` field. Can be toggled via enable_thinking.
 *   - Qwen 3.6 models: reasoning via chat_template_kwargs.enable_thinking;
 *     returns `reasoning` field.
 *   - MiniMax M3 MXFP8: reasoning via chat_template_kwargs.enable_thinking;
 *     returns reasoning_content field.
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

/** Type guard: non-null, non-array object. */
export function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

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
const MINIMAX_M3_ID = "MiniMaxAI/MiniMax-M3-MXFP8";

const DS_VLLM_MODELS = new Set([DS_PRO_ID, DS_FLASH_ID]);
const ENABLE_THINKING_VLLM_MODELS = new Set([MINIMAX_M3_ID]);

/**
 * Makora's GLM models, built from the same models list this extension
 * registers. `before_provider_request` is GLOBAL in pi — it fires for every
 * loaded provider's requests, not just makora's — and its event carries only
 * `payload` (no `provider` field to gate on). So the GLM tool_calls strip MUST
 * be scoped to makora's own GLM model ids. A loose `/glm/i` regex would also
 * rewrite tool_calls for every sibling provider that serves any GLM model
 * (baseten, io, lilac, neuralwatt, tensorix, fireworks, crofai, hypercharm,
 * wafer, parasail, umans, opencode, ...), silently breaking their tool calling
 * and clobbering the input their own before_provider_request handlers expect.
 *
 * Derived (not hardcoded) so it auto-tracks models.json refreshes — models.json
 * is auto-generated by scripts/update-models.js. All makora models run on vLLM,
 * so every GLM id here shares the ZAI chat-template `.items()` crash that
 * stripGlmToolCalls works around.
 */
const allMakoraModels = buildModels(
  modelsData as JsonModel[],
  customModelsData as JsonModel[],
  patchData as PatchMap,
);
const GLM_VLLM_MODELS = new Set(
  allMakoraModels.filter((m) => /glm/i.test(m.id)).map((m) => m.id),
);

/** Whether `model` is one of makora's GLM models — the only ids the
 *  before_provider_request hook should strip tool_calls for. Exported so the
 *  scoping can be unit-tested in isolation. */
export function isMakoraGlmVllmModel(model: string): boolean {
  return GLM_VLLM_MODELS.has(model);
}

/**
 * Intercept the request payload for models that need vLLM-specific thinking
 * param rewrites.
 *
 * pi's "deepseek" thinkingFormat sends `thinking: { type: "enabled" }` which
 * is the official DeepSeek API format — but Makora's vLLM deployment ignores
 * it. vLLM requires different params depending on the model:
 *   - DS V4 Pro:  `chat_template_kwargs: { thinking: true }` + `reasoning_effort`
 *   - DS V4 Flash: `include_reasoning: true` + `chat_template_kwargs: { thinking: true }`
 *     + `reasoning_effort`. `include_reasoning` alone returns `reasoning: null`
 *     on this vLLM build — both params are required.
 *   - MiniMax M3: `chat_template_kwargs: { enable_thinking: true }` +
 *     `reasoning_effort`. Returns `reasoning_content` field.
 *
 * This hook rewrites the payload accordingly.
 */
function rewriteVllmPayload(payload: Record<string, unknown>): Record<string, unknown> {
  const model = payload.model as string | undefined;
  if (!model) return payload;

  const p = { ...payload };

  if (DS_VLLM_MODELS.has(model)) {
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
  } else if (ENABLE_THINKING_VLLM_MODELS.has(model)) {
    // Models using chat_template_kwargs.enable_thinking (e.g. MiniMax M3)
    delete p.thinking;
    const ctq = (p.chat_template_kwargs as Record<string, unknown>) ?? {};
    p.chat_template_kwargs = { ...ctq, enable_thinking: true };
  }

  return p;
}

/**
 * GLM models on Makora's vLLM crash with a leaked Python AttributeError
 * (`'list object' has no attribute 'items'` or `'str object' has no attribute 'items'`)
 * when any assistant message in the request contains a `tool_calls` field.
 * The ZAI/vLLM chat template calls `.items()` on the tool_calls list (or on the
 * JSON-string `arguments` field), which raises AttributeError and leaks into the
 * HTTP 400 response body.
 *
 * Fix: for makora's GLM models only (see GLM_VLLM_MODELS / isMakoraGlmVllmModel),
 * strip `tool_calls` from assistant messages in the before_provider_request hook
 * and convert them back to GLM's native
 * `<tool_call>` XML text in `content`. The model natively understands this
 * format in conversation history, and the `role: "tool"` result messages that
 * follow are rendered fine by the chat template's tool-observation branch.
 *
 * If upstream fixes both the streaming parser and the 500/400 crash, this
 * transform becomes a harmless no-op (the XML text is still valid GLM input).
 */

export function toolCallToGlmXml(tc: Record<string, unknown>): string {
  const fn = (isObject(tc.function) ? tc.function : {}) as Record<string, unknown>;
  const name = typeof fn.name === "string" ? fn.name : "";
  const argsStr = typeof fn.arguments === "string" ? fn.arguments : "{}";
  let args: Record<string, unknown>;
  try {
    args = JSON.parse(argsStr) as Record<string, unknown>;
  } catch {
    args = {};
  }
  if (!isObject(args)) args = {};
  const argLines = Object.entries(args).map(
    ([key, value]) =>
      `<arg_key>${key}</arg_key>\n<arg_value>${
        typeof value === "string" ? value : JSON.stringify(value)
      }</arg_value>`,
  );
  return `<tool_call>${name}\n${argLines.join("\n")}\n</tool_call>`;
}

export function stripGlmToolCalls(payload: Record<string, unknown>): Record<string, unknown> {
  const messages = payload.messages;
  if (!Array.isArray(messages)) return payload;

  let modified = false;
  const newMessages = messages.map((msg) => {
    if (!isObject(msg)) return msg;
    if (msg.role !== "assistant") return msg;

    const toolCalls = msg.tool_calls;
    if (!Array.isArray(toolCalls) || toolCalls.length === 0) return msg;

    const xmlBlocks = toolCalls
      .filter((tc): tc is Record<string, unknown> => isObject(tc))
      .map(toolCallToGlmXml);
    if (xmlBlocks.length === 0) return msg;

    const toolCallText = xmlBlocks.join("\n");
    // pi's openai-completions provider always serializes assistant content as a
    // plain string, but guard against null/array content for robustness.
    const existingContent = typeof msg.content === "string" ? msg.content : "";
    const newContent = existingContent ? `${existingContent}\n${toolCallText}` : toolCallText;

    modified = true;
    const rest: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(msg)) {
      if (k !== "tool_calls") rest[k] = v;
    }
    return { ...rest, content: newContent };
  });

  if (!modified) return payload;
  return { ...payload, messages: newMessages };
}

/**
 * GLM models on Makora's vLLM stream tool calls unreliably: the streaming
 * parser omits `delta.tool_calls`, finishing with `finish_reason:
 * "tool_calls"` but an empty delta, so the aggregated tool call arrives with
 * empty `arguments` (pi then rejects it: "command: must have required
 * properties command"). Setting `tool_stream: true` in the request forces
 * vLLM's explicit tool-streaming path, which emits tool-call chunks with
 * populated arguments.
 *
 * Gated on makora's GLM model ids via isMakoraGlmVllmModel (the same gate as
 * stripGlmToolCalls): the .items() crash and the empty-delta streaming bug
 * are both vLLM ZAI-chat-template issues affecting the same model set, and
 * all makora models run on vLLM. If upstream vLLM fixes either, both
 * transforms become no-ops and can be removed together. Exported for testing.
 */
export function setGlmToolStream(payload: Record<string, unknown>): Record<string, unknown> {
  return { ...payload, tool_stream: true };
}

export default function (pi: ExtensionAPI) {
  const models = allMakoraModels;

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

    let result = rewriteVllmPayload(payload);
    if (isMakoraGlmVllmModel(payload.model)) {
      result = stripGlmToolCalls(result);
      result = setGlmToolStream(result);
    }
    return result;
  });
}

