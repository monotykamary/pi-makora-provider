/**
 * Tests for /makora-settings: per-model preserved-thinking overrides.
 *
 * /makora-settings pins chatTemplateKwargs.preserve_thinking / clear_thinking to
 * a fixed boolean in ~/.pi/agent/extensions/makora.json (modelOverrides), which
 * deep-merges on top of patch.json and wins. The pinned boolean overrides the
 * default `{ $var: "thinking.enabled" }` schema (which ties preserve to the
 * thinking switch), decoupling preserve from on/off. These tests cover the pure
 * helpers (isPreserved, parseModelOverrides, applyModelOverride, buildModels)
 * plus the streamMakora end-to-end showing a pinned boolean flows through
 * onPayload regardless of thinking state — no filesystem access required.
 */

import { describe, expect, it } from "vitest";
import {
  isPreserved,
  parseModelOverrides,
  applyModelOverride,
  buildModels,
  streamMakora,
} from "../index.js";
import { __resetStreamCalls, __setClamp, __streamCalls } from "./__mocks__/pi-ai.js";
import patchData from "../patch.json" with { type: "json" };
import type { JsonModel, ModelOverride } from "../index.js";

const glmVar = { $var: "thinking.enabled", invert: true };
const preserveVar = { $var: "thinking.enabled" };

function baseModel(id: string, compat: JsonModel["compat"]): JsonModel {
  return {
    id,
    name: id,
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 131072,
    maxTokens: 16384,
    ...(compat ? { compat } : {}),
  };
}

const ctx = { messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }] };

describe("isPreserved", () => {
  it("clear_thinking: plain false → preserved, plain true → cleared", () => {
    expect(isPreserved("clear_thinking", false)).toBe(true);
    expect(isPreserved("clear_thinking", true)).toBe(false);
  });

  it("preserve_thinking: plain true → preserved, plain false → cleared", () => {
    expect(isPreserved("preserve_thinking", true)).toBe(true);
    expect(isPreserved("preserve_thinking", false)).toBe(false);
  });

  it("$var default (GLM clear_thinking inverted) → preserved (preserve when thinking on)", () => {
    expect(isPreserved("clear_thinking", glmVar)).toBe(true);
  });

  it("$var default (Qwen/Kimi preserve_thinking) → preserved", () => {
    expect(isPreserved("preserve_thinking", preserveVar)).toBe(true);
  });

  it("a non-inverted $var clear_thinking → not preserved (clear when thinking on)", () => {
    expect(isPreserved("clear_thinking", { $var: "thinking.enabled" })).toBe(false);
  });

  it("an inverted $var preserve_thinking → not preserved", () => {
    expect(isPreserved("preserve_thinking", { $var: "thinking.enabled", invert: true })).toBe(false);
  });

  it("unknown value → not preserved", () => {
    expect(isPreserved("clear_thinking", "yes")).toBe(false);
    expect(isPreserved("preserve_thinking", undefined)).toBe(false);
  });
});

describe("parseModelOverrides", () => {
  it("returns undefined for non-object / array input", () => {
    expect(parseModelOverrides(null)).toBeUndefined();
    expect(parseModelOverrides("x")).toBeUndefined();
    expect(parseModelOverrides([])).toBeUndefined();
  });

  it("parses compat.chatTemplateKwargs overrides", () => {
    const out = parseModelOverrides({
      "zai-org/GLM-5.2-FP8": { compat: { chatTemplateKwargs: { clear_thinking: true } } },
    });
    expect(out).toEqual({
      "zai-org/GLM-5.2-FP8": { compat: { chatTemplateKwargs: { clear_thinking: true } } },
    });
  });

  it("parses thinkingLevelMap with null + string values", () => {
    const out = parseModelOverrides({
      m1: { thinkingLevelMap: { off: "none", high: null } },
    });
    expect(out).toEqual({ m1: { thinkingLevelMap: { off: "none", high: null } } });
  });

  it("drops non-object override values", () => {
    const out = parseModelOverrides({
      good: { compat: { chatTemplateKwargs: { preserve_thinking: false } } },
      bad: "nope",
      arr: [1, 2],
    });
    expect(out).toEqual({ good: { compat: { chatTemplateKwargs: { preserve_thinking: false } } } });
  });

  it("returns undefined when nothing valid remains", () => {
    expect(parseModelOverrides({ a: "x", b: 1 })).toBeUndefined();
  });
});

describe("applyModelOverride", () => {
  it("pins a plain boolean over a $var schema (deep-merge chatTemplateKwargs)", () => {
    const model = baseModel("glm", {
      supportsReasoningEffort: true,
      chatTemplateKwargs: { clear_thinking: glmVar },
    });
    const out = applyModelOverride(model, { compat: { chatTemplateKwargs: { clear_thinking: true } } });
    expect(out.compat?.chatTemplateKwargs?.clear_thinking).toBe(true);
    // other compat fields preserved
    expect(out.compat?.supportsReasoningEffort).toBe(true);
    // original model not mutated
    expect((model.compat?.chatTemplateKwargs as any).clear_thinking).toEqual(glmVar);
  });

  it("preserves sibling kwargs (Kimi thinking stays $var when pinning preserve_thinking)", () => {
    const model = baseModel("kimi", {
      supportsReasoningEffort: false,
      assistantReasoningField: "reasoning_content",
      chatTemplateKwargs: { thinking: preserveVar, preserve_thinking: preserveVar },
    });
    const out = applyModelOverride(model, { compat: { chatTemplateKwargs: { preserve_thinking: false } } });
    expect(out.compat?.chatTemplateKwargs).toEqual({ thinking: preserveVar, preserve_thinking: false });
    expect(out.compat?.assistantReasoningField).toBe("reasoning_content");
  });
});

describe("buildModels with user overrides", () => {
  const patches = patchData as Record<string, any>;

  it("override beats patch.json: pins clear_thinking boolean over the $var default", () => {
    const base = [baseModel("zai-org/GLM-5.2-FP8", undefined)];
    const overrides: Record<string, ModelOverride> = {
      "zai-org/GLM-5.2-FP8": { compat: { chatTemplateKwargs: { clear_thinking: true } } },
    };
    const [glm] = buildModels(base, [], patches, overrides);
    expect(glm.compat?.chatTemplateKwargs?.clear_thinking).toBe(true);
    // patch-derived fields still present (not wiped by the partial override)
    expect(glm.compat?.supportsReasoningEffort).toBe(true);
    expect(glm.reasoning).toBe(true);
    expect(glm.thinkingLevelMap?.off).toBe("none");
  });

  it("without overrides, the patch.json $var default is kept", () => {
    const base = [baseModel("zai-org/GLM-5.2-FP8", undefined)];
    const [glm] = buildModels(base, [], patches);
    expect(glm.compat?.chatTemplateKwargs?.clear_thinking).toEqual(glmVar);
  });

  it("override on an unknown id is a no-op", () => {
    const base = [baseModel("zai-org/GLM-5.2-FP8", undefined)];
    const overrides: Record<string, ModelOverride> = {
      "does/not-exist": { compat: { chatTemplateKwargs: { preserve_thinking: false } } },
    };
    const [glm] = buildModels(base, [], patches, overrides);
    expect(glm.compat?.chatTemplateKwargs?.clear_thinking).toEqual(glmVar);
  });

  it("Qwen preserve_thinking can be pinned off over the $var default", () => {
    const base = [baseModel("unsloth/Qwen3.6-27B-NVFP4", undefined)];
    const overrides: Record<string, ModelOverride> = {
      "unsloth/Qwen3.6-27B-NVFP4": { compat: { chatTemplateKwargs: { preserve_thinking: false } } },
    };
    const [qwen] = buildModels(base, [], patches, overrides);
    expect(qwen.compat?.chatTemplateKwargs?.preserve_thinking).toBe(false);
  });
});

describe("streamMakora honors a pinned-boolean override (decoupled from thinking switch)", () => {
  // A GLM model as buildModels would produce AFTER a user pins clear_thinking: true
  // via /makora-settings (Clear Thinking). The default $var would resolve to
  // clear_thinking=false when thinking is ON; the pin must force true regardless.
  const glmPinnedClear = {
    id: "zai-org/GLM-5.2-FP8",
    provider: "makora",
    reasoning: true,
    input: ["text"],
    thinkingLevelMap: { minimal: null, low: null, medium: null, high: "high", xhigh: "max", off: "none" },
    compat: {
      supportsReasoningEffort: true,
      chatTemplateKwargs: { clear_thinking: true },
    },
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 131072,
    maxTokens: 16384,
  };

  it("injects clear_thinking=true even when thinking is ON (pin wins over thinking state)", async () => {
    __resetStreamCalls();
    __setClamp((_m: any, level: any) => level);
    streamMakora(glmPinnedClear, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: glmPinnedClear.id, reasoning_effort: "high" }, glmPinnedClear);
    expect(result.chat_template_kwargs).toEqual({ clear_thinking: true });
    expect(result.reasoning_effort).toBe("high");
  });

  it("a pinned-preserve Qwen (preserve_thinking: true) stays true even when thinking is OFF", async () => {
    __resetStreamCalls();
    __setClamp((_m: any, level: any) => level);
    const qwenPinnedPreserve = {
      id: "unsloth/Qwen3.6-27B-NVFP4",
      provider: "makora",
      reasoning: true,
      input: ["text"],
      thinkingLevelMap: { minimal: "low", xhigh: "high", off: "none" },
      compat: { supportsReasoningEffort: true, chatTemplateKwargs: { preserve_thinking: true } },
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 131072,
      maxTokens: 16384,
    };
    streamMakora(qwenPinnedPreserve, ctx, { apiKey: "sk-test", reasoning: "off" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: qwenPinnedPreserve.id, reasoning_effort: "none" }, qwenPinnedPreserve);
    // $var default would have yielded preserve_thinking=false when off; the pin holds true.
    expect(result.chat_template_kwargs).toEqual({ preserve_thinking: true });
  });
});
