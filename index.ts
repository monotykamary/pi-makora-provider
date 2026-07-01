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
 * provider. It watches the assistant text stream on the GLM 5.2 family with
 * four detectors — character run, token run, trailing-unit run, and
 * normalized-line run — and, if the model falls into a degenerate repetition
 * loop (observed: `!!!!`, `0000`, `0 0 0`, `{},{},{}`, and a structured
 * log-line loop), it aborts the runaway generation, removes the toxic message
 * from the agent's transcript (so it can't bias later turns), and resumes the
 * agentic loop invisibly via agent.prompt([]) (the pi-invisible-continue
 * pattern); no new user message is injected.
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
import type { SimpleStreamOptions, AssistantMessageEventStream } from "@earendil-works/pi-ai";
import { clampThinkingLevel } from "@earendil-works/pi-ai";
import { streamOpenAICompletions } from "@earendil-works/pi-ai/compat";
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
    /** When set, onPayload copies each assistant message's `reasoning` field into
     *  this field name before sending. Needed when a model's chat template reads
     *  prior reasoning from a different field than the one pi-ai sends it in.
     *  e.g. Kimi K2.7's template reads `reasoning_content`, but pi-ai sends
     *  `reasoning` (the field the model returns) — without the copy, preserve_thinking
     *  can't render the prior trace. */
    assistantReasoningField?: string;
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

// Thinking-off + preserved-thinking via streamSimple
//
// Makora's reasoning models need BOTH a thinking on/off switch AND multi-turn
// reasoning continuity (`preserve_thinking`). pi-ai's built-in `thinkingFormat`
// branches are mutually exclusive: the `chat-template` branch emits
// `chat_template_kwargs` (so it can carry `preserve_thinking`) but never a
// top-level `reasoning_effort`; the OpenAI fallback emits `reasoning_effort`
// (the lever GLM 5.2 and Qwen 3.6 actually respond to for off) but never
// `chat_template_kwargs`. No single format emits both.
//
// Behavioral E2E (see test-thinking-triggers.ts / test-preserve-deterministic.ts):
//   GLM 5.2   — off ONLY via top-level `reasoning_effort: "none"`;
//               `enable_thinking`/`thinking`/chat_template_kwargs toggles are ignored.
//               Multi-turn continuity uses `clear_thinking: false` (NOT preserve_thinking
//               — that flag is inert for GLM); toggled with the thinking switch
//               (false when on = preserve, true when off = clear).
//   Qwen 3.6  — off via top-level `reasoning_effort: "none"` (or `enable_thinking`).
//               Multi-turn continuity uses `preserve_thinking: true`; reads the
//               `reasoning` field pi-ai sends. Toggled with the thinking switch.
//   Kimi K2.7 — off via namespaced `chat_template_kwargs.thinking: false`.
//               Multi-turn continuity uses `preserve_thinking: true`, BUT the
//               template reads `reasoning_content` while pi-ai sends `reasoning`
//               (Makora doesn't gateway-alias the fields) — so onPayload also
//               copies each assistant message's `reasoning` → `reasoning_content`.
//   preserve_thinking IS functional (deterministic E2E: Qwen 3.6 27B recalls
//     1.00 with it on, 0.00 off) — it gates whether the prior reasoning trace
//     is rendered into the next turn's prompt.
//
// So this provider registers a `streamSimple` wrapper that delegates to pi-ai's
// `streamOpenAICompletions` (keeping all its streaming/tool-calling/caching) and
// uses pi-ai's `onPayload` hook — which runs AFTER `buildParams` — to inject the
// `chat_template_kwargs` that the chosen `thinkingFormat` branch can't reach.
// `buildParams` still owns `reasoning_effort` (on → mapped effort; off → "none");
// `onPayload` owns `chat_template_kwargs` (preserve_thinking + Kimi's `thinking`).
// Per-model config lives in `compat.chatTemplateKwargs` (patch.json), using the
// same `{ "$var": "thinking.enabled" }` schema pi-ai's chat-template format uses.

/** Resolve one `chatTemplateKwargs` value (scalar or `{ $var }`) against the
 *  current thinking state. Mirrors pi-ai's `resolveChatTemplateKwargValue` so
 *  onPayload-injected values match what the built-in chat-template format would
 *  produce — but injected after buildParams so they coexist with reasoning_effort. */
export function resolveChatTemplateKwarg(
  value: unknown,
  model: JsonModel,
  thinkingOn: boolean,
  reasoningEffort: string | undefined,
): unknown {
  // Scalars pass through unchanged (e.g. static `preserve_thinking: true`).
  if (value === null || typeof value !== "object") {
    return value;
  }
  const spec = value as { $var?: string; omitWhenOff?: boolean; invert?: boolean };
  if (spec.omitWhenOff && !thinkingOn) {
    return undefined;
  }
  if (spec.$var === "thinking.enabled") {
    return spec.invert ? !thinkingOn : thinkingOn;
  }
  if (spec.$var === "thinking.effort") {
    const mapped = thinkingOn
      ? model.thinkingLevelMap?.[reasoningEffort as string]
      : model.thinkingLevelMap?.off;
    if (mapped === undefined) return reasoningEffort;
    return typeof mapped === "string" ? mapped : undefined;
  }
  return undefined;
}

/** Custom streamSimple: delegate to pi-ai's OpenAI completions streamer and
 *  inject per-model `chat_template_kwargs` (preserve_thinking / clear_thinking +
 *  Kimi's thinking toggle) via the onPayload hook, and alias each assistant
 *  message's reasoning field when a template reads a different field name.
 *  Models with neither chatTemplateKwargs nor assistantReasoningField (DeepSeek,
 *  Llama, Gemma) pass through unchanged — onPayload is not registered. */
export function streamMakora(
  model: any,
  context: any,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const apiKey = options?.apiKey || "";
  if (!apiKey) {
    throw new Error(
      `No API key for Makora. Add it to ~/.pi/agent/auth.json under "makora", ` +
        `set the MAKORA_OPTIMIZE_TOKEN env var, or use --api-key.`,
    );
  }

  // pi-ai's streamer reads `model.api` to pick the OpenAI completions client. Our
  // provider registers under `api: "makora"` (so pi routes to this streamSimple);
  // override to `openai-completions` here so streamOpenAICompletions uses the
  // standard client. Per-model baseUrl overrides (per-slug endpoints) are kept.
  const makoraModel = { ...model, api: "openai-completions", baseUrl: model.baseUrl || BASE_URL };

  // pi hands streamSimple providers the raw thinking selection as
  // `options.reasoning` (a ThinkingLevel). The raw streamOpenAICompletions only
  // reads `options.reasoningEffort`, so replicate the clamp+convert pi-ai's own
  // streamSimple wrapper does — otherwise reasoning_effort never reaches the
  // body and thinking levels silently do nothing. "off" → undefined (off).
  const clampedReasoning = options?.reasoning
    ? clampThinkingLevel(makoraModel, options.reasoning)
    : undefined;
  const reasoningEffort = clampedReasoning === "off" ? undefined : clampedReasoning;
  const thinkingOn = reasoningEffort !== undefined;
  const { reasoning: _reasoning, ...streamOptions } = options ?? {};

  // Inject chat_template_kwargs after buildParams via onPayload. Any caller-
  // supplied onPayload is chained first so it can inspect/replace the payload;
  // our injection then merges into whatever chat_template_kwargs already exist.
  // We also register onPayload when the model needs its prior-reasoning field
  // aliased (assistantReasoningField) — e.g. Kimi's template reads
  // `reasoning_content` but pi-ai sends `reasoning`; the copy makes the
  // preserved trace visible to the template so preserve_thinking can render it.
  const userOnPayload = (streamOptions as any).onPayload;
  const extraKwargs = (makoraModel as JsonModel).compat?.chatTemplateKwargs;
  const hasExtraKwargs =
    !!extraKwargs && typeof extraKwargs === "object" && Object.keys(extraKwargs).length > 0;
  const assistantReasoningField = (makoraModel as JsonModel).compat?.assistantReasoningField;
  const needsFieldCopy = !!assistantReasoningField;
  const onPayload =
    hasExtraKwargs || needsFieldCopy || userOnPayload
      ? async (params: any, mdl: any) => {
          let p = params;
          if (userOnPayload) {
            const next = await userOnPayload(p, mdl);
            if (next !== undefined) p = next;
          }
          if (needsFieldCopy && thinkingOn) {
            // Only alias the prior reasoning field when thinking is ON — when the
            // user turns thinking off we must NOT carry prior reasoning forward
            // (preserve_thinking is inert for Kimi's field rendering; the field's
            // presence is what gates continuity). Kimi's template ignores the
            // `reasoning` field, so skipping the copy leaves nothing to render.
            const field = assistantReasoningField!;
            const msgs = Array.isArray(p?.messages) ? p.messages : [];
            p = {
              ...p,
              messages: msgs.map((m: any) =>
                m && m.role === "assistant" &&
                typeof m.reasoning === "string" && m.reasoning.length > 0 &&
                m[field] === undefined
                  ? { ...m, [field]: m.reasoning }
                  : m
              ),
            };
          }
          if (hasExtraKwargs) {
            const resolved: Record<string, unknown> = {};
            for (const [k, v] of Object.entries(extraKwargs!)) {
              const r = resolveChatTemplateKwarg(v, makoraModel, thinkingOn, reasoningEffort as string | undefined);
              if (r !== undefined) resolved[k] = r;
            }
            p = {
              ...p,
              chat_template_kwargs: {
                ...(p?.chat_template_kwargs ?? {}),
                ...resolved,
              },
            };
          }
          return p;
        }
      : undefined;

  return streamOpenAICompletions(makoraModel, context, {
    ...streamOptions,
    reasoningEffort,
    apiKey,
    ...(onPayload ? { onPayload } : {}),
  });
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

  // apiKey resolution order: auth.json ("makora" key) → MAKORA_OPTIMIZE_TOKEN env var.
  // `api: "makora"` + `streamSimple` routes every Makora model through streamMakora
  // (above), which delegates to pi-ai's OpenAI completions streamer and injects
  // per-model chat_template_kwargs via onPayload.
  pi.registerProvider(PROVIDER_ID, {
    name: "Makora",
    baseUrl: BASE_URL,
    apiKey: "$MAKORA_OPTIMIZE_TOKEN",
    api: "makora",
    streamSimple: streamMakora,
    models,
  });

  // Abort runaway repetition loops — a single character, a spaced token, a
  // delimiter-joined unit, or a structured line/template loop (e.g. '!!!!',
  // '0000', '0 0 0', '{},{}', a log-line loop) — on the GLM 5.2 family; remove
  // the toxic output from the agent's transcript and resume the agentic loop
  // invisibly (no new user message). See ./death-loop-guard.ts.
  registerDeathLoopGuard(pi);
}
