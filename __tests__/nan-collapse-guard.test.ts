/**
 * Tests for the NVFP4 NaN-collapse guard's pure detection logic.
 * (isOnsetCollapse / isCollapseMessage / isGuardedModel / extractText /
 *  trimAndReduceContext's reduction math — no Agent/pi wiring needed.)
 */
import { describe, it, expect } from "vitest";
import {
  isOnsetCollapse,
  isCollapseMessage,
  isGuardedModel,
  extractText,
  isOnsetCollapse as collapse,
  DEGENERATE_STUB_TEXT,
  NAN_RECOVERY_CUSTOM_TYPE,
  isNanRecoveryMarker,
} from "../nan-collapse-guard.js";

describe("isOnsetCollapse — NaN-argmax fixed point at reasoning onset", () => {
  it("trips on a single-char run (!) at/after 40 chars", () => {
    expect(isOnsetCollapse("!".repeat(40))).toBe(true);
    expect(isOnsetCollapse("!".repeat(64))).toBe(true);
    expect(isOnsetCollapse("!".repeat(15857))).toBe(true);
  });

  it("trips on a short multi-char unit ({}, / (); / 0 ) repeated ≥20×", () => {
    expect(isOnsetCollapse("{},".repeat(22))).toBe(true);   // colleague's {},{},{}
    expect(isOnsetCollapse("();".repeat(22))).toBe(true);
    expect(isOnsetCollapse("0 ".repeat(22))).toBe(true);    // "0 0 0 …"
    expect(isOnsetCollapse("{}".repeat(22))).toBe(true);
  });

  it("does NOT trip on a normal reasoning onset", () => {
    expect(isOnsetCollapse("Let me scan the word list and identify the an")).toBe(false);
    expect(isOnsetCollapse("So `lastOutputAt` is updated on output (line ")).toBe(false);
    expect(isOnsetCollapse("The user wants a JSON array of empty objects")).toBe(false);
    expect(isOnsetCollapse("1. Understand the Goal: The user wants to cha")).toBe(false);
  });

  it("does NOT trip on short / whitespace-led / sub-threshold repeats", () => {
    expect(isOnsetCollapse("!!!")).toBe(false);             // too short (<40)
    expect(isOnsetCollapse("    ".repeat(20))).toBe(false);  // whitespace-led
    expect(isOnsetCollapse("\n\n\n\n".repeat(12))).toBe(false);
  });
  it("trips on a 2-char unit repeated 20× (a repeated short token at onset is degenerate)", () => {
    expect(collapse("ab".repeat(20))).toBe(true);
  });

  it("ignores a unit shorter than 20 reps even if the window fills", () => {
    // 4-char unit "abcd" repeated 10× = 40 chars, only 10 reps → not a collapse.
    expect(isOnsetCollapse("abcd".repeat(10))).toBe(false);
  });
});

describe("isCollapseMessage — finalized assistant message scrubbing", () => {
  const stub = { role: "assistant", content: [{ type: "text", text: DEGENERATE_STUB_TEXT }] };
  it("matches the guard's own stub", () => {
    expect(isCollapseMessage(stub as any)).toBe(true);
  });
  it("matches a degenerate onset (15857 '!' reasoning)", () => {
    expect(isCollapseMessage({ role: "assistant", content: [{ type: "thinking", thinking: "!".repeat(15857) }] } as any)).toBe(true);
  });
  it("does not match a normal assistant message", () => {
    expect(isCollapseMessage({ role: "assistant", content: [{ type: "text", text: "Here is the fix…" }] } as any)).toBe(false);
  });
  it("ignores non-assistant roles", () => {
    expect(isCollapseMessage({ role: "user", content: "!".repeat(100) } as any)).toBe(false);
  });
});

describe("isGuardedModel", () => {
  it("guards the GLM-5.2 NVFP4/FP8 quants", () => {
    expect(isGuardedModel({ id: "zai-org/GLM-5.2-NVFP4" })).toBe(true);
    expect(isGuardedModel({ id: "zai-org/GLM-5.2-FP8" })).toBe(true);
  });
  it("does not guard other models", () => {
    expect(isGuardedModel({ id: "moonshotai/Kimi-K2.7-Code" })).toBe(false);
    expect(isGuardedModel({ id: "deepseek-ai/DeepSeek-V4-Pro" })).toBe(false);
  });
  it("handles null/undefined", () => {
    expect(isGuardedModel(null)).toBe(false);
    expect(isGuardedModel(undefined)).toBe(false);
  });
});

describe("extractText", () => {
  it("joins text + thinking parts", () => {
    expect(extractText("plain string")).toBe("plain string");
    expect(extractText([{ type: "text", text: "a" }, { type: "thinking", thinking: "b" }])).toBe("ab");
  });
  it("handles null/empty", () => {
    expect(extractText(null)).toBe("");
    expect(extractText([])).toBe("");
  });
});

describe("native recovery marker", () => {
  it("recognizes only the provider-invisible custom marker", () => {
    expect(isNanRecoveryMarker({ role: "custom", customType: NAN_RECOVERY_CUSTOM_TYPE })).toBe(true);
    expect(isNanRecoveryMarker({ role: "user", customType: NAN_RECOVERY_CUSTOM_TYPE })).toBe(false);
    expect(isNanRecoveryMarker({ role: "custom", customType: "other" })).toBe(false);
  });
});
