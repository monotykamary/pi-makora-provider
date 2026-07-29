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
 *
 * Settings Configuration:
 *   /makora-settings opens a settings panel to toggle Preserved Thinking per
 *   reasoning model. Preserve Thinking keeps each turn's reasoning trace in the
 *   next prompt (better multi-turn recall for coding); Clear Thinking lets the
 *   template drop older reasoning (saves tokens, can hurt recall / cause
 *   overthinking). Selections persist to ~/.pi/agent/extensions/makora.json
 *   (modelOverrides, applied on top of patch.json) and take effect immediately.
 */

import { getAgentDir, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { SimpleStreamOptions, AssistantMessageEventStream } from "@earendil-works/pi-ai/compat";
import { clampThinkingLevel, streamOpenAICompletions } from "@earendil-works/pi-ai/compat";
import modelsData from "./models.json" with { type: "json" };
import customModelsData from "./custom-models.json" with { type: "json" };
import patchData from "./patch.json" with { type: "json" };
import deprecatedData from "./deprecated-models.json" with { type: "json" };
import { registerNanCollapseGuard, nanCanary, canaryEnabled, isGuardedModel } from "./nan-collapse-guard.js";
import fs from "fs";
import path from "path";

// Types

interface JsonModel {
  id: string;
  name: string;
  reasoning: boolean;
  input: ("text" | "image")[];
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
  input?: ("text" | "image")[];
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

// User Configuration: ~/.pi/agent/extensions/makora.json lets a user override model
// properties per id ON TOP of patch.json + custom-models.json (so they win).
// Recursively deep-merges `compat` (incl. nested `chatTemplateKwargs`) and
// `thinkingLevelMap` (toggle one flag without redeclaring the rest), replaces
// scalars and arrays. Lets a user pin chatTemplateKwargs.preserve_thinking /
// clear_thinking to a fixed boolean via /makora-settings, overriding the default
// `{ $var: "thinking.enabled" }` schema that ties preserve to the thinking switch.
interface ModelOverride {
  thinkingLevelMap?: Record<string, string | null>;
  compat?: Record<string, unknown>;
}

interface MakoraConfig {
  modelOverrides?: Record<string, ModelOverride>;
}

const CONFIG_PATH = path.join(getAgentDir(), "extensions", "makora.json");
const DEFAULT_CONFIG: MakoraConfig = { modelOverrides: {} };

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

// Recursively deep-merge `override` into `base`. Plain objects merge key-by-key
// (so a user can toggle a single chatTemplateKwargs flag without redeclaring the
// rest); arrays and non-plain-object values (booleans, { $var } specs) replace
// the base value wholesale.
function deepMerge<T>(base: T, override: unknown): T {
  if (!isPlainObject(override) || !isPlainObject(base)) return override as T;
  const result: Record<string, unknown> = { ...(base as Record<string, unknown>) };
  for (const [k, v] of Object.entries(override)) {
    result[k] = isPlainObject(v) && isPlainObject(result[k]) ? deepMerge(result[k], v) : v;
  }
  return result as T;
}

// Validate user-supplied modelOverrides from the config file. Non-object ids and
// non-object overrides are dropped silently so a malformed file doesn't crash
// model registration.
function parseModelOverrides(raw: unknown): Record<string, ModelOverride> | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const result: Record<string, ModelOverride> = {};
  for (const [id, override] of Object.entries(raw as Record<string, unknown>)) {
    if (typeof id !== "string" || !override || typeof override !== "object" || Array.isArray(override)) continue;
    const o = override as Record<string, unknown>;
    const parsed: ModelOverride = {};
    if (o.thinkingLevelMap && typeof o.thinkingLevelMap === "object") {
      const m: Record<string, string | null> = {};
      for (const [k, v] of Object.entries(o.thinkingLevelMap as Record<string, unknown>)) {
        if (v === null) m[k] = null;
        else if (typeof v === "string") m[k] = v;
      }
      if (Object.keys(m).length > 0) parsed.thinkingLevelMap = m;
    }
    if (o.compat && typeof o.compat === "object") parsed.compat = o.compat as Record<string, unknown>;
    if (Object.keys(parsed).length > 0) result[id] = parsed;
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

// Reads ~/.pi/agent/extensions/makora.json. Missing file → populate with defaults
// so the user can discover it, then return defaults. An existing-but-invalid file
// is left untouched (defaults returned) so a user's typo isn't silently wiped.
// Loaded lazily via getConfig() (not at import) so importing the module has no
// filesystem side effects and unit tests can import the pure helpers safely.
function loadConfig(): MakoraConfig {
  let rawText: string;
  try {
    rawText = fs.readFileSync(CONFIG_PATH, "utf8");
  } catch {
    try {
      fs.mkdirSync(path.dirname(CONFIG_PATH), { recursive: true });
      fs.writeFileSync(CONFIG_PATH, JSON.stringify(DEFAULT_CONFIG, null, 2) + "\n");
    } catch {
      // Write failure is non-fatal — defaults still work in memory
    }
    return { ...DEFAULT_CONFIG };
  }
  try {
    const raw = JSON.parse(rawText);
    return { modelOverrides: parseModelOverrides(raw.modelOverrides) };
  } catch {
    return { ...DEFAULT_CONFIG };
  }
}

let config: MakoraConfig | undefined;
function getConfig(): MakoraConfig {
  if (!config) config = loadConfig();
  return config;
}

// Read-modify-write the validated config and refresh the in-memory cache. Used by
// /makora-settings so a toggle takes effect immediately (builtModels() reads
// getConfig()) without a restart, and without clobbering the user's other
// modelOverrides — the mutator receives loadConfig() (validated), so overrides
// hand-edited since startup survive the spread. The file is normalized to a
// discoverable shape (modelOverrides: {} always present).
function updateConfig(mutator: (cfg: MakoraConfig) => MakoraConfig): MakoraConfig {
  const next = mutator(loadConfig());
  const toWrite: MakoraConfig = { modelOverrides: next.modelOverrides ?? {} };
  try {
    fs.mkdirSync(path.dirname(CONFIG_PATH), { recursive: true });
    fs.writeFileSync(CONFIG_PATH, JSON.stringify(toWrite, null, 2) + "\n");
  } catch {
    // Write failure is non-fatal — the in-memory cache below still updates.
  }
  config = next;
  return next;
}

// Resolve whether a model's preserve/clear flag currently means "preserve
// reasoning across turns". Handles both the plain-boolean overrides a user pins
// via /makora-settings AND the default `{ $var: "thinking.enabled" }` schema in
// patch.json (which ties preserve to the thinking switch). The default is treated
// as preserved — preserve-when-thinking-on is the operative behavior the E2E
// tests verify — so the settings panel shows the default as "Preserve Thinking"
// until a user pins it.
export function isPreserved(
  flag: "clear_thinking" | "preserve_thinking",
  value: unknown,
): boolean {
  if (typeof value === "boolean") {
    return flag === "clear_thinking" ? value === false : value === true;
  }
  if (isPlainObject(value) && (value as { $var?: string }).$var === "thinking.enabled") {
    const invert = (value as { invert?: boolean }).invert === true;
    // $var resolves to thinkingOn (or !thinkingOn when inverted). Preserved means
    // the flag's effective value favors keeping reasoning when thinking is on.
    if (flag === "clear_thinking") return invert; // clear_thinking = !thinkingOn → preserved when on
    return !invert; // preserve_thinking = thinkingOn → preserved when on
  }
  return false;
}

// Apply a user-supplied modelOverride (from makora.json) on top of a built model.
// Recursively deep-merges compat (incl. nested chatTemplateKwargs) /
// thinkingLevelMap so a user can pin a single flag (e.g. chatTemplateKwargs.
// preserve_thinking) without redeclaring the rest; replaces scalars and arrays.
// No reasoning-cleanup (unlike applyPatch) — the override is authoritative.
function applyModelOverride(model: JsonModel, override: ModelOverride): JsonModel {
  const result = { ...model };
  for (const [key, value] of Object.entries(override)) {
    (result as any)[key] = isPlainObject(value) && isPlainObject((result as any)[key])
      ? deepMerge((result as any)[key], value)
      : value;
  }
  return result;
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

/** Merge static models with custom models, patch overrides, and user overrides. */
function buildModels(
  base: JsonModel[],
  custom: JsonModel[],
  patch: PatchMap,
  overrides: Record<string, ModelOverride> = {},
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

  // User-supplied modelOverrides (from ~/.pi/agent/extensions/makora.json) applied
  // LAST so they win over patch.json + custom-models.json. Deep-merges compat
  // (incl. chatTemplateKwargs) / thinkingLevelMap so a user can pin a single
  // chatTemplateKwargs flag (e.g. preserve_thinking) without redeclaring the rest.
  for (const [id, override] of Object.entries(overrides)) {
    const existing = modelMap.get(id);
    if (existing) {
      modelMap.set(id, applyModelOverride(existing, override));
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
  const canaryOn = canaryEnabled() && isGuardedModel(makoraModel);
  const onPayload =
    hasExtraKwargs || needsFieldCopy || userOnPayload || canaryOn
      ? async (params: any, mdl: any) => {
          // Optional logprobs canary (MAKORA_NAN_CANARY=1): probe the same payload
          // with logprobs:true/max_tokens:1; a 400-"nan" response flags a NaN
          // prefill for the engine team. Diagnostic only (fire-and-forget) — the
          // in-stream onset guard handles recovery. See ./nan-collapse-guard.ts.
          if (canaryOn) {
            void nanCanary(
              params as Record<string, unknown>,
              makoraModel.baseUrl,
              apiKey,
              makoraModel.id,
            );
          }
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

// Grace period for delisted models. When the provider API stops listing a
// model, update-models.js moves its last-known definition into
// deprecated-models.json (stamped with deprecatedAt) instead of dropping it.
// For 14 days the model keeps working here so in-flight sessions and saved
// model settings do not break; afterwards it is evicted permanently.
const DEPRECATED_MODEL_TTL_MS = 14 * 24 * 60 * 60 * 1000;

// Grace-period deprecated models with deprecation metadata stripped.
function activeDeprecatedModels(): JsonModel[] {
  const now = Date.now();
  const result: JsonModel[] = [];
  for (const entry of Object.values(deprecatedData as Record<string, JsonModel & { deprecatedAt?: string }>)) {
    if (!entry?.id) continue;
    const removedAt = Date.parse(entry.deprecatedAt ?? "");
    if (Number.isNaN(removedAt) || now - removedAt > DEPRECATED_MODEL_TTL_MS) continue;
    const model = { ...entry } as JsonModel & { deprecatedAt?: string };
    delete model.deprecatedAt;
    result.push(model);
  }
  return result;
}

// Append grace-period deprecated models the list does not already have (live data wins).
function withDeprecated(models: JsonModel[]): JsonModel[] {
  const seen = new Set(models.map((m) => m.id));
  const extras = activeDeprecatedModels().filter((m) => !seen.has(m.id));
  return extras.length > 0 ? [...models, ...extras] : models;
}

// Extension Entry Point

const PROVIDER_ID = "makora";
const BASE_URL = "https://inference.makora.com/v1";

export default function (pi: ExtensionAPI) {
  const embeddedModels = modelsData as JsonModel[];
  const customModels = customModelsData as JsonModel[];
  const patches = patchData as PatchMap;

  function activeOverrides(): Record<string, ModelOverride> {
    return getConfig().modelOverrides ?? {};
  }

  function builtModels(): JsonModel[] {
    return withDeprecated(buildModels(embeddedModels, customModels, patches, activeOverrides()));
  }

  // apiKey resolution order: auth.json ("makora" key) → MAKORA_OPTIMIZE_TOKEN env var.
  // `api: "makora"` + `streamSimple` routes every Makora model through streamMakora
  // (above), which delegates to pi-ai's OpenAI completions streamer and injects
  // per-model chat_template_kwargs via onPayload.
  function makeProviderConfig(models: JsonModel[]) {
    return {
      name: "Makora",
      baseUrl: BASE_URL,
      apiKey: "$MAKORA_OPTIMIZE_TOKEN",
      api: "makora",
      streamSimple: streamMakora,
      models,
    };
  }

  pi.registerProvider(PROVIDER_ID, makeProviderConfig(builtModels()));

  // Guard against the GLM-5.2 NVFP4/FP8 long-context NaN-logit collapse — the
  // "death loop" root cause: cold/long-context prefill through the NVFP4 MoE
  // path produces NaN logits at ~10k+ tokens → argmax-of-NaN ("!" / "{},") fixed
  // point → infinite repetition; streaming silently aborts. Scoped to the
  // known-bad models + a token gate, detects the NaN signature at reasoning
  // onset, and recovers by reducing context below the threshold before
  // resuming (the old blanket detector trimmed only the symptom and recurred
  // 6×). Mitigation only — the fix belongs in the vLLM NVFP4 prefill kernel.
  // See ./nan-collapse-guard.ts.
  registerNanCollapseGuard(pi);

  // Preserved-thinking state + notify
  //
  // Deferred model_select notify timer — cleared on rapid re-switch and on
  // session_shutdown so only the latest switch notifies.
  let modelSelectNotifyTimer: ReturnType<typeof setTimeout> | null = null;
  const MODEL_SELECT_NOTIFY_DELAY_MS = 250;

  // Collect each reasoning model that carries a preserve/clear flag, with its
  // current preserved state resolved from the build pipeline (config as source of
  // truth, not the bound model's compat). Handles both the default
  // `{ $var: "thinking.enabled" }` schema and plain-boolean user overrides.
  function collectPreserveState(): Array<{ id: string; name: string; flag: "clear_thinking" | "preserve_thinking"; preserved: boolean }> {
    const resolved = builtModels();
    const out: Array<{ id: string; name: string; flag: "clear_thinking" | "preserve_thinking"; preserved: boolean }> = [];
    for (const m of resolved) {
      const kwargs = m.compat?.chatTemplateKwargs;
      if (!kwargs || typeof kwargs !== "object") continue;
      if ("clear_thinking" in kwargs) {
        out.push({ id: m.id, name: m.name || m.id, flag: "clear_thinking", preserved: isPreserved("clear_thinking", kwargs.clear_thinking) });
      } else if ("preserve_thinking" in kwargs) {
        out.push({ id: m.id, name: m.name || m.id, flag: "preserve_thinking", preserved: isPreserved("preserve_thinking", kwargs.preserve_thinking) });
      }
    }
    return out;
  }

  // Notify preserved-thinking state for a preserve-flag model. Computed from the
  // build pipeline (config as source of truth), deferred so pi core's (and other
  // extensions') notifications land first, and cancelled on re-switch/shutdown so
  // only the latest shows. Always level "info" — the text conveys the coding/prose
  // tradeoff.
  function notifyPreservedThinkingFor(model: any, ctx: any): void {
    if (!model || model.provider !== PROVIDER_ID) return;
    const entry = collectPreserveState().find((e) => e.id === model.id);
    if (!entry) return;
    const flagValue = entry.flag === "clear_thinking" ? !entry.preserved : entry.preserved;
    const msg = entry.preserved
      ? `Preserved thinking ON for ${entry.name} (${entry.flag}: ${flagValue}) — suited for coding, but not for prose. Open /makora-settings to change.`
      : `Preserved thinking OFF for ${entry.name} (${entry.flag}: ${flagValue}) — reasoning trimmed each turn (lighter; better for prose). Open /makora-settings to change.`;
    if (modelSelectNotifyTimer) clearTimeout(modelSelectNotifyTimer);
    modelSelectNotifyTimer = setTimeout(() => {
      modelSelectNotifyTimer = null;
      try { ctx.ui.notify(msg, "info"); } catch { /* notify is a no-op without a UI runner */ }
    }, MODEL_SELECT_NOTIFY_DELAY_MS);
  }

  // Re-register on session_start so a user-edited makora.json (e.g. toggled
  // modelOverrides) takes effect this session, and notify preserved-thinking
  // state for the active model (model_select may not fire on startup).
  pi.on("session_start", async (_event, ctx) => {
    config = loadConfig();
    pi.registerProvider(PROVIDER_ID, makeProviderConfig(builtModels()));
    notifyPreservedThinkingFor(ctx.model, ctx);
  });

  pi.on("model_select", async (event, ctx) => {
    notifyPreservedThinkingFor(event.model ?? ctx.model, ctx);
  });

  pi.on("session_shutdown", () => {
    if (modelSelectNotifyTimer) { clearTimeout(modelSelectNotifyTimer); modelSelectNotifyTimer = null; }
  });

  // /makora-settings: settings UI (mirrors pi core /settings)
  //
  // Opens a SettingsList (lazy-imported from pi-tui) via ctx.ui.custom(). Toggles
  // write to ~/.pi/agent/extensions/makora.json (modelOverrides), refresh the
  // in-memory config, and re-register the provider so the change takes effect
  // immediately. Pinned booleans override the default `$var: thinking.enabled`
  // schema, decoupling preserve from the thinking switch.
  pi.registerCommand("makora-settings", {
    description: "Configure Makora: preserved thinking per model",
    async handler(_args, ctx) {
      if (ctx.mode !== "tui") {
        if (!ctx.hasUI) {
          ctx.ui.notify("/makora-settings requires a UI (TUI or GUI).", "error");
          return;
        }
        const fresh = collectPreserveState();
        if (fresh.length === 0) {
          ctx.ui.notify("No models support preserved thinking.", "info");
          return;
        }
        const modelPick = await ctx.ui.select(
          "Makora preserved thinking \u2014 pick a model",
          fresh.map((e) => `${e.name}: ${e.preserved ? "Preserve Thinking" : "Clear Thinking"}`),
        );
        if (modelPick === undefined) return;
        const entry = fresh.find((e) => modelPick.startsWith(`${e.name}:`));
        if (!entry) return;
        const v = await ctx.ui.select(entry.name, ["Preserve Thinking", "Clear Thinking"]);
        if (v === undefined) return;
        const preservedOn = v === "Preserve Thinking";
        const flagValue = entry.flag === "clear_thinking" ? !preservedOn : preservedOn;
        updateConfig((cfg) => {
          const overrides = cfg.modelOverrides ?? (cfg.modelOverrides = {});
          const ov = overrides[entry.id] ?? (overrides[entry.id] = {});
          const compat = ov.compat ?? (ov.compat = {});
          const kwargs = compat.chatTemplateKwargs ?? (compat.chatTemplateKwargs = {});
          kwargs[entry.flag] = flagValue;
          return cfg;
        });
        pi.registerProvider(PROVIDER_ID, makeProviderConfig(builtModels()));
        ctx.ui.notify(`Preserved thinking ${preservedOn ? "on" : "off"} for ${entry.name} \u2014 takes effect now.`, "info");
        return;
      }
      const { SettingsList, Container } = await import("@earendil-works/pi-tui");
      const { getSettingsListTheme, DynamicBorder } = await import("@earendil-works/pi-coding-agent");

      await ctx.ui.custom((_tui, theme, _kb, done) => {
        const border = () => new DynamicBorder((s: string) => theme.fg("border", s));
        // SettingsList left-aligns the value column after the widest label (capped
        // at 30 cols). A label wider than 30 shifts that row's value out of
        // alignment, so cap model-name labels.
        const truncateLabel = (s: string) => (s.length > 30 ? s.slice(0, 27) + "..." : s);

        const items: any[] = [
          {
            id: "preserved-thinking",
            label: "Preserved thinking ›",
            description: "Per-model Preserve Thinking / Clear Thinking (full-history reasoning). Preserve Thinking keeps all turns' reasoning; Clear Thinking lets the template drop older reasoning (saves tokens, can hurt multi-turn recall / cause overthinking). Pins a fixed value that overrides the default (which ties preserve to the thinking switch).",
            currentValue: "configure",
            submenu: (_currentValue: string, subDone: (v?: string) => void) => {
              // Re-read state on each open so toggles from a previous visit (which
              // wrote makora.json + refreshed config) are reflected — a snapshot
              // captured at panel-open time would show stale values after a toggle.
              const fresh = collectPreserveState();
              const subItems = fresh.map((e) => ({
                id: `preserve:${e.id}`,
                label: truncateLabel(e.name),
                description: `${e.id} — Preserve Thinking keeps full reasoning history across turns; Clear Thinking lets the template drop older reasoning (saves tokens, can hurt multi-turn recall / cause overthinking).`,
                currentValue: e.preserved ? "Preserve Thinking" : "Clear Thinking",
                values: ["Preserve Thinking", "Clear Thinking"],
              }));
              const subList = new SettingsList(
                subItems,
                Math.min(subItems.length + 2, 15),
                getSettingsListTheme(),
                (id: string, newValue: string) => {
                  const modelId = id.slice("preserve:".length);
                  const entry = fresh.find((p) => p.id === modelId);
                  if (!entry) return;
                  const preservedOn = newValue === "Preserve Thinking";
                  const flagValue = entry.flag === "clear_thinking" ? !preservedOn : preservedOn;
                  updateConfig((cfg) => {
                    const overrides = cfg.modelOverrides ?? (cfg.modelOverrides = {});
                    const ov = overrides[modelId] ?? (overrides[modelId] = {});
                    const compat = ov.compat ?? (ov.compat = {});
                    const kwargs = compat.chatTemplateKwargs ?? (compat.chatTemplateKwargs = {});
                    kwargs[entry.flag] = flagValue;
                    return cfg;
                  });
                  pi.registerProvider(PROVIDER_ID, makeProviderConfig(builtModels()));
                  ctx.ui.notify(`Preserved thinking ${preservedOn ? "on" : "off"} for ${entry.name} — takes effect now.`, "info");
                },
                () => subDone(),
                { enableSearch: true },
              );
              // The outer container's borders already frame the panel; return the
              // list directly so we don't render a second border pair.
              return subList;
            },
          },
        ];

        const container = new Container();
        container.addChild(border());

        const settingsList = new SettingsList(
          items,
          Math.min(items.length + 2, 15),
          getSettingsListTheme(),
          (_id: string, _newValue: string) => {
            // Top-level items are submenus (preserved-thinking); per-model toggles
            // are handled inside the submenu callback above.
          },
          () => done(undefined),
          { enableSearch: true },
        );
        container.addChild(settingsList);
        container.addChild(border());

        return {
          render(width: number) {
            return container.render(width);
          },
          invalidate() {
            container.invalidate();
          },
          handleInput(data: string) {
            settingsList.handleInput?.(data);
          },
        };
      });
    },
  });
}

export { parseModelOverrides, loadConfig, getConfig, updateConfig, applyModelOverride, buildModels };
export type { MakoraConfig, ModelOverride };
