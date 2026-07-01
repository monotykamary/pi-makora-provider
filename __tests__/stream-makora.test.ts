/**
 * Tests for streamMakora: thinking-off + preserved-thinking via streamSimple.
 *
 * streamMakora delegates to pi-ai's streamOpenAICompletions and injects per-model
 * chat_template_kwargs (preserve_thinking / clear_thinking + Kimi's thinking
 * toggle) through the onPayload hook, and aliases each assistant message's
 * reasoning field when a template reads a different field name (Kimi). These
 * tests verify the wiring — clamp → reasoningEffort, onPayload registration,
 * $var resolution (incl. invert), reasoning-field aliasing, merge semantics,
 * and patch.json config — without making real HTTP calls (pi-ai is stubbed).
 */

import { describe, expect, it, beforeEach } from "vitest";
import {
  __resetStreamCalls,
  __setClamp,
  __streamCalls,
} from "./__mocks__/pi-ai.js";
import { streamMakora, resolveChatTemplateKwarg } from "../index.js";
import patchData from "../patch.json" with { type: "json" };

// Models shaped as buildModels() produces them after patch.json is applied.
// Mirrors the effective config verified via scripts (see AGENTS.md data flow).

const glm52 = {
  id: "zai-org/GLM-5.2-FP8",
  provider: "makora",
  reasoning: true,
  input: ["text"],
  thinkingLevelMap: { minimal: null, low: null, medium: null, high: "high", xhigh: "max", off: "none" },
  compat: {
    supportsReasoningEffort: true,
    chatTemplateKwargs: { clear_thinking: { $var: "thinking.enabled", invert: true } },
  },
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 131072,
  maxTokens: 16384,
};

const qwen = {
  id: "unsloth/Qwen3.6-27B-NVFP4",
  provider: "makora",
  reasoning: true,
  input: ["text"],
  thinkingLevelMap: { minimal: "low", xhigh: "high", off: "none" },
  compat: {
    supportsReasoningEffort: true,
    chatTemplateKwargs: { preserve_thinking: { $var: "thinking.enabled" } },
  },
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 131072,
  maxTokens: 16384,
};

const kimi = {
  id: "moonshotai/Kimi-K2.7-Code",
  provider: "makora",
  reasoning: true,
  input: ["text", "image"],
  thinkingLevelMap: { minimal: "low", xhigh: "high" },
  compat: {
    supportsReasoningEffort: false,
    assistantReasoningField: "reasoning_content",
    chatTemplateKwargs: {
      thinking: { $var: "thinking.enabled" },
      preserve_thinking: { $var: "thinking.enabled" },
    },
  },
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 262144,
  maxTokens: 262144,
};

// DeepSeek: no chatTemplateKwargs, no assistantReasoningField → onPayload NOT registered.
const deepseek = {
  id: "deepseek-ai/DeepSeek-V4-Pro",
  provider: "makora",
  reasoning: true,
  input: ["text"],
  thinkingLevelMap: { minimal: null, low: null, medium: null, high: "high", xhigh: "max" },
  compat: { thinkingFormat: "deepseek", supportsReasoningEffort: true },
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 131072,
  maxTokens: 16384,
};

const ctx = { messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }] };

beforeEach(() => {
  __resetStreamCalls();
  __setClamp((_m: any, level: any) => level);
});

describe("streamMakora delegation", () => {
  it("delegates to streamOpenAICompletions with api overridden to openai-completions", () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    expect(__streamCalls).toHaveLength(1);
    expect(__streamCalls[0].model.api).toBe("openai-completions");
    expect(__streamCalls[0].model.id).toBe("zai-org/GLM-5.2-FP8");
  });

  it("preserves a per-model baseUrl override (per-slug endpoint)", () => {
    const withOverride = { ...glm52, baseUrl: "https://inference.makora.com/glm-slug/v1" };
    streamMakora(withOverride, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    expect(__streamCalls[0].model.baseUrl).toBe("https://inference.makora.com/glm-slug/v1");
  });

  it("forwards the resolved apiKey to streamOpenAICompletions", () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    expect(__streamCalls[0].options.apiKey).toBe("sk-test");
  });

  it("throws a helpful error when no apiKey is available", () => {
    expect(() => streamMakora(glm52, ctx, { reasoning: "high" } as any)).toThrow(/No API key for Makora/);
  });
});

describe("thinking level → reasoningEffort conversion", () => {
  it("converts raw reasoning to reasoningEffort (non-off)", () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    expect(__streamCalls[0].options.reasoningEffort).toBe("high");
    expect(__streamCalls[0].options.reasoning).toBeUndefined();
  });

  it("converts reasoning \"off\" to undefined reasoningEffort", () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "off" } as any);
    expect(__streamCalls[0].options.reasoningEffort).toBeUndefined();
  });

  it("uses clampThinkingLevel to clamp unsupported levels (minimal→high for GLM)", () => {
    __setClamp((_m: any, level: any) => (level === "minimal" ? "high" : level));
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "minimal" } as any);
    expect(__streamCalls[0].options.reasoningEffort).toBe("high");
  });

  it("omits reasoningEffort when no reasoning selection is passed", () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test" } as any);
    expect(__streamCalls[0].options.reasoningEffort).toBeUndefined();
  });
});

describe("onPayload registration", () => {
  it("registers onPayload when chatTemplateKwargs is set (GLM)", () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    expect(typeof __streamCalls[0].options.onPayload).toBe("function");
  });

  it("registers onPayload when chatTemplateKwargs is set (Kimi)", () => {
    streamMakora(kimi, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    expect(typeof __streamCalls[0].options.onPayload).toBe("function");
  });

  it("registers onPayload when only assistantReasoningField is set (no kwargs)", () => {
    const kimiNoKwargs = { ...kimi, compat: { supportsReasoningEffort: false, assistantReasoningField: "reasoning_content" } };
    streamMakora(kimiNoKwargs, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    expect(typeof __streamCalls[0].options.onPayload).toBe("function");
  });

  it("does NOT register onPayload when neither is set (DeepSeek)", () => {
    streamMakora(deepseek, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    expect(__streamCalls[0].options.onPayload).toBeUndefined();
  });
});

describe("onPayload $var resolution (thinking on)", () => {
  it("GLM: injects clear_thinking=false (preserve) alongside buildParams' reasoning_effort", async () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: glm52.id, messages: [], reasoning_effort: "high" }, glm52);
    expect(result.chat_template_kwargs).toEqual({ clear_thinking: false });
    expect(result.reasoning_effort).toBe("high");
  });

  it("Qwen: injects preserve_thinking=true alongside reasoning_effort", async () => {
    streamMakora(qwen, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: qwen.id, reasoning_effort: "high" }, qwen);
    expect(result.chat_template_kwargs).toEqual({ preserve_thinking: true });
    expect(result.reasoning_effort).toBe("high");
  });

  it("Kimi: injects thinking=true + preserve_thinking=true (no reasoning_effort)", async () => {
    streamMakora(kimi, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: kimi.id, messages: [] }, kimi);
    expect(result.chat_template_kwargs).toEqual({ thinking: true, preserve_thinking: true });
  });
});

describe("onPayload $var resolution (thinking off)", () => {
  it("GLM: clear_thinking=true (clear) alongside reasoning_effort: none", async () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "off" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: glm52.id, reasoning_effort: "none" }, glm52);
    expect(result.chat_template_kwargs).toEqual({ clear_thinking: true });
    expect(result.reasoning_effort).toBe("none");
  });

  it("Qwen: preserve_thinking=false alongside reasoning_effort: none", async () => {
    streamMakora(qwen, ctx, { apiKey: "sk-test", reasoning: "off" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: qwen.id, reasoning_effort: "none" }, qwen);
    expect(result.chat_template_kwargs).toEqual({ preserve_thinking: false });
  });

  it("Kimi: thinking=false + preserve_thinking=false", async () => {
    streamMakora(kimi, ctx, { apiKey: "sk-test", reasoning: "off" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: kimi.id, messages: [] }, kimi);
    expect(result.chat_template_kwargs).toEqual({ thinking: false, preserve_thinking: false });
  });
});

describe("onPayload assistantReasoningField aliasing (Kimi)", () => {
  it("copies each assistant message reasoning → reasoning_content", async () => {
    streamMakora(kimi, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const prior = "I deduced the answer is 42.";
    const result = await onPayload(
      { model: kimi.id, messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "42", reasoning: prior },
      ] },
      kimi,
    );
    const asst = result.messages.find((m: any) => m.role === "assistant");
    expect(asst.reasoning_content).toBe(prior);
    expect(asst.reasoning).toBe(prior);
  });

  it("does NOT overwrite a pre-existing reasoning_content field", async () => {
    streamMakora(kimi, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const existing = "pre-existing reasoning_content";
    const result = await onPayload(
      { model: kimi.id, messages: [{ role: "assistant", content: "x", reasoning: "r", reasoning_content: existing }] },
      kimi,
    );
    expect(result.messages[0].reasoning_content).toBe(existing);
  });

  it("skips assistant messages without a reasoning field", async () => {
    streamMakora(kimi, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload(
      { model: kimi.id, messages: [{ role: "assistant", content: "no reasoning here" }] },
      kimi,
    );
    expect(result.messages[0].reasoning_content).toBeUndefined();
  });

  it("does NOT alias when assistantReasoningField is absent (GLM/Qwen)", async () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload(
      { model: glm52.id, messages: [{ role: "assistant", content: "x", reasoning: "prior" }] },
      glm52,
    );
    expect(result.messages[0].reasoning_content).toBeUndefined();
  });

  it("does NOT copy the reasoning field when thinking is OFF (no continuity)", async () => {
    streamMakora(kimi, ctx, { apiKey: "sk-test", reasoning: "off" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const prior = "prior reasoning";
    const result = await onPayload(
      { model: kimi.id, messages: [{ role: "assistant", content: "x", reasoning: prior }] },
      kimi,
    );
    // thinking off → no copy → template (which ignores `reasoning`) sees nothing
    expect(result.messages[0].reasoning_content).toBeUndefined();
    expect(result.messages[0].reasoning).toBe(prior);
  });
});

describe("onPayload merge semantics", () => {
  it("merges into pre-existing chat_template_kwargs instead of clobbering", async () => {
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "high" } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const original = { model: glm52.id, chat_template_kwargs: { enable_thinking: true } };
    const result = await onPayload(original, glm52);
    expect(result.chat_template_kwargs).toEqual({ enable_thinking: true, clear_thinking: false });
  });

  it("chains a caller-supplied onPayload first, then injects kwargs", async () => {
    const userPayload = (p: any) => ({ ...p, userSawIt: true });
    streamMakora(glm52, ctx, { apiKey: "sk-test", reasoning: "high", onPayload: userPayload } as any);
    const onPayload = __streamCalls[0].options.onPayload;
    const result = await onPayload({ model: glm52.id }, glm52);
    expect(result.userSawIt).toBe(true);
    expect(result.chat_template_kwargs).toEqual({ clear_thinking: false });
  });
});

describe("resolveChatTemplateKwarg (pure helper)", () => {
  const model = glm52;

  it("passes scalars through unchanged", () => {
    expect(resolveChatTemplateKwarg(true, model, true, "high")).toBe(true);
    expect(resolveChatTemplateKwarg(false, model, false, undefined)).toBe(false);
    expect(resolveChatTemplateKwarg("static", model, true, "high")).toBe("static");
    expect(resolveChatTemplateKwarg(42, model, true, "high")).toBe(42);
    expect(resolveChatTemplateKwarg(null, model, true, "high")).toBe(null);
  });

  it("$var thinking.enabled → true when on, false when off", () => {
    expect(resolveChatTemplateKwarg({ $var: "thinking.enabled" }, model, true, "high")).toBe(true);
    expect(resolveChatTemplateKwarg({ $var: "thinking.enabled" }, model, false, undefined)).toBe(false);
  });

  it("$var thinking.enabled invert → false when on, true when off", () => {
    expect(resolveChatTemplateKwarg({ $var: "thinking.enabled", invert: true }, model, true, "high")).toBe(false);
    expect(resolveChatTemplateKwarg({ $var: "thinking.enabled", invert: true }, model, false, undefined)).toBe(true);
  });

  it("$var thinking.effort resolves to the mapped effort when on", () => {
    expect(resolveChatTemplateKwarg({ $var: "thinking.effort" }, glm52, true, "xhigh")).toBe("max");
  });

  it("$var thinking.effort resolves to thinkingLevelMap.off when off", () => {
    expect(resolveChatTemplateKwarg({ $var: "thinking.effort" }, glm52, false, undefined)).toBe("none");
  });

  it("omitWhenOff drops the key when thinking is off", () => {
    expect(resolveChatTemplateKwarg({ $var: "thinking.enabled", omitWhenOff: true }, model, false, undefined)).toBeUndefined();
  });

  it("omitWhenOff keeps the key when thinking is on", () => {
    expect(resolveChatTemplateKwarg({ $var: "thinking.enabled", omitWhenOff: true }, model, true, "high")).toBe(true);
  });
});

describe("patch.json thinking-off + preserve config (behavioral E2E-verified)", () => {
  const patches = patchData as Record<string, any>;

  // GLM 5.2: off via reasoning_effort "none"; multi-turn via clear_thinking (inverted).
  for (const id of ["zai-org/GLM-5.2-FP8", "zai-org/GLM-5.2-NVFP4"]) {
    it(`${id}: off via reasoning_effort, clear_thinking inverted, no thinkingFormat`, () => {
      expect(patches[id]?.compat?.thinkingFormat).toBeUndefined();
      expect(patches[id]?.compat?.supportsReasoningEffort).toBe(true);
      expect(patches[id]?.thinkingLevelMap?.off).toBe("none");
      expect(patches[id]?.compat?.chatTemplateKwargs).toEqual({
        clear_thinking: { $var: "thinking.enabled", invert: true },
      });
    });

    it(`${id}: minimal/low/medium clamp to high (null = unsupported)`, () => {
      const map = patches[id]?.thinkingLevelMap;
      expect(map?.minimal).toBeNull();
      expect(map?.low).toBeNull();
      expect(map?.medium).toBeNull();
      expect(map?.high).toBe("high");
      expect(map?.xhigh).toBe("max");
    });
  }

  // Qwen 3.6: off via reasoning_effort "none"; preserve_thinking toggled.
  for (const id of ["unsloth/Qwen3.6-27B-NVFP4", "unsloth/Qwen3.6-35B-A3B-NVFP4"]) {
    it(`${id}: off via reasoning_effort, preserve_thinking $var, no thinkingFormat`, () => {
      expect(patches[id]?.compat?.thinkingFormat).toBeUndefined();
      expect(patches[id]?.compat?.supportsReasoningEffort).toBe(true);
      expect(patches[id]?.thinkingLevelMap?.off).toBe("none");
      expect(patches[id]?.compat?.chatTemplateKwargs).toEqual({
        preserve_thinking: { $var: "thinking.enabled" },
      });
    });
  }

  // Kimi K2.7: namespaced thinking + preserve_thinking + reasoning_content alias.
  it("moonshotai/Kimi-K2.7-Code: namespaced thinking, preserve_thinking, reasoning_content alias", () => {
    const id = "moonshotai/Kimi-K2.7-Code";
    expect(patches[id]?.compat?.thinkingFormat).toBeUndefined();
    expect(patches[id]?.compat?.supportsReasoningEffort).toBe(false);
    expect(patches[id]?.compat?.assistantReasoningField).toBe("reasoning_content");
    expect(patches[id]?.compat?.chatTemplateKwargs).toEqual({
      thinking: { $var: "thinking.enabled" },
      preserve_thinking: { $var: "thinking.enabled" },
    });
  });

  // DeepSeek: unchanged.
  for (const id of ["deepseek-ai/DeepSeek-V4-Pro", "deepseek-ai/DeepSeek-V4-Flash"]) {
    it(`${id}: unchanged (deepseek format, no chatTemplateKwargs)`, () => {
      expect(patches[id]?.compat?.thinkingFormat).toBe("deepseek");
      expect(patches[id]?.compat?.chatTemplateKwargs).toBeUndefined();
    });
  }
});
