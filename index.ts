/**
 * Makora Provider Extension
 *
 * Registers Makora (inference.makora.com) as a custom provider using the
 * OpenAI completions API.
 *
 * Makora is an inference optimization platform serving open-weight models via
 * per-model URL slugs under https://inference.makora.com/{slug}/v1. Each
 * model is hosted on vLLM and speaks the standard OpenAI chat completions
 * protocol. Because models live at separate endpoints, every model carries
 * its own `baseUrl` override.
 *
 * Model resolution strategy: Static + patch + custom
 *   1. Embedded models.json served at registration time (zero-latency)
 *   2. patch.json applied on top for per-field overrides
 *   3. custom-models.json merged in for user-added models
 *
 * SWR is not applicable because Makora does not expose a /v1/models endpoint.
 * If one becomes available in the future, the stale-while-revalidate pattern
 * from other pi-*-provider extensions can be added here.
 *
 * Merge order: embedded → apply patch.json → merge custom-models.json
 *
 * Reasoning notes:
 *   - DeepSeek V4 Flash/Pro: reasoning NOT supported on Makora's current
 *     vLLM deployment (reasoning_content is always null).
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
 *   - HY3 Preview, Llama 3.3 70B: not reasoning models.
 *
 * Developer role is NOT supported by any of the chat templates on Makora's
 * vLLM deployment (prompts with role: "developer" are silently dropped).
 * supportsDeveloperRole is set to false for all models.
 *
 * Tool use notes:
 *   - Kimi K2.6 NVFP4: vLLM's Kimi deployment does not have
 *     --enable-auto-tool-choice or --tool-call-parser set. When tools
 *     are present in the payload without an explicit tool_choice, vLLM
 *     defaults tool_choice to "auto" internally, triggering a 400.
 *     The before_provider_request hook forces tool_choice: "none" to
 *     avoid the implicit default-to-auto. Tool calling is effectively
 *     disabled for this model until Makora enables the server flags.
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

// ─── Types ────────────────────────────────────────────────────────────────

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
  thinkingLevelMap?: Record<string, string | null>;
  headers?: Record<string, string>;
  vision?: {
    maxImagesPerRequest?: number;
  };
  compat?: {
    supportsDeveloperRole?: boolean;
    supportsStore?: boolean;
    maxTokensField?: "max_completion_tokens" | "max_tokens";
    thinkingFormat?: "openai" | "openrouter" | "deepseek" | "together" | "zai" | "qwen" | "qwen-chat-template";
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
  thinkingLevelMap?: Record<string, string | null>;
  headers?: Record<string, string>;
  compat?: Record<string, unknown>;
}

type PatchMap = Record<string, PatchEntry>;

// ─── Patch Application ───────────────────────────────────────────────────

function applyPatch(model: JsonModel, patch: PatchEntry): JsonModel {
  const result = { ...model };

  if (patch.name !== undefined) result.name = patch.name;
  if (patch.reasoning !== undefined) result.reasoning = patch.reasoning;
  if (patch.input !== undefined) result.input = patch.input;
  if (patch.contextWindow !== undefined) result.contextWindow = patch.contextWindow;
  if (patch.maxTokens !== undefined) result.maxTokens = patch.maxTokens;
  if (patch.baseUrl !== undefined) result.baseUrl = patch.baseUrl;
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

/** Full pipeline: base models → patch → custom → result */
function buildModels(base: JsonModel[], custom: JsonModel[], patch: PatchMap): JsonModel[] {
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

// ─── Extension Entry Point ──────────────────────────────────────────────

const PROVIDER_ID = "makora";
const BASE_URL = "https://inference.makora.com";

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

  // Force tool_choice: "none" for models whose vLLM deployment lacks
  // --enable-auto-tool-choice and --tool-call-parser. When tools are
  // present in the payload without an explicit tool_choice, vLLM
  // defaults tool_choice to "auto" internally, which triggers a 400.
  // Setting it to "none" explicitly avoids the default-to-auto path.
  // The model can still call tools through its chat template, but vLLM
  // will not parse tool call outputs without --tool-call-parser, so
  // tool use is effectively disabled for these models regardless.
  const NO_AUTO_TOOL_CHOICE_MODELS = new Set([
    "nvidia/Kimi-K2.6-NVFP4",
  ]);

  pi.on("before_provider_request", (event, _ctx) => {
    const payload = event.payload as Record<string, unknown>;
    const modelId = payload.model as string | undefined;
    if (!modelId || !NO_AUTO_TOOL_CHOICE_MODELS.has(modelId)) return;
    if (!("tools" in payload)) return;

    return { ...payload, tool_choice: "none" };
  });
}
