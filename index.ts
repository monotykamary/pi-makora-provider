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
 *   - DeepSeek V4 Pro: returns `reasoning` field.
 *   - DeepSeek V4 Flash: returns `reasoning` field.
 *   - GLM 5.2 FP8 / NVFP4: returns `reasoning` field.
 *   - Kimi K2.7 Code: returns `reasoning` field.
 *   - Qwen 3.6 models: returns `reasoning` field.
 *   - Llama 3.3 70B: not a reasoning model.
 *
 * A death-loop guard (see ./death-loop-guard.ts) is registered alongside the
 * provider. It watches the assistant text stream on the GLM 5.2 family and,
 * if the model falls into a degenerate repetition loop — an unbroken run of a
 * single character or token, spaced or not (observed: `!!!!`, `0000`, `0 0 0`)
 * — it aborts the runaway generation and resumes the agentic loop invisibly
 * via agent.prompt([]) (the pi-invisible-continue pattern); no new user
 * message is injected.
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
import { registerDeathLoopGuard } from "./death-loop-guard.js";

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
    /** Extra keys merged into vLLM `chat_template_kwargs` on every request.
     *  Used for preserved-thinking flags like `preserve_thinking` / `clear_thinking`
     *  that some chat templates require for multi-turn reasoning continuity. */
    chatTemplateKwargs?: Record<string, unknown>;
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

const allMakoraModels = buildModels(
  modelsData as JsonModel[],
  customModelsData as JsonModel[],
  patchData as PatchMap,
);

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

  // Abort runaway repetition loops — a single character or token, spaced or
  // not (e.g. '!!!!', '0000', '0 0 0') — on the GLM 5.2 family and resume the
  // agentic loop invisibly (no new user message). See ./death-loop-guard.ts.
  registerDeathLoopGuard(pi);
}
