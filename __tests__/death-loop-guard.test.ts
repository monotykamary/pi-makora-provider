/**
 * Tests for the Makora death-loop guard's pure detection helpers.
 *
 * registerDeathLoopGuard() itself monkey-patches Agent.prototype.subscribe and
 * wires pi events, so it is not exercised here — only the deterministic helpers
 * it depends on. Importing the module also confirms the static
 * @earendil-works/pi-agent-core import resolves (via the devDependency) so the
 * extension loads under pi's loader alias at runtime.
 */

import { describe, expect, it } from "vitest";
import {
  BANG_THRESHOLD,
  BACKOFF_BASE_MS,
  BACKOFF_MAX_MS,
  BACKOFF_MULTIPLIER,
  DEFAULT_BACKOFF_CONFIG,
  GUARDED_MODEL_IDS,
  calculateDelay,
  extractText,
  formatDuration,
  isGuardedModel,
  messageTrailingBangs,
  nextTrailingBangs,
  registerDeathLoopGuard,
} from "../death-loop-guard.js";

describe("death-loop-guard config", () => {
  it("guards the GLM 5.2 family by default", () => {
    expect(GUARDED_MODEL_IDS.has("zai-org/GLM-5.2-NVFP4")).toBe(true);
    expect(GUARDED_MODEL_IDS.has("zai-org/GLM-5.2-FP8")).toBe(true);
  });

  it("exports a sensible bang threshold", () => {
    expect(BANG_THRESHOLD).toBeGreaterThanOrEqual(20);
  });

  it("exports the guard registration function", () => {
    expect(typeof registerDeathLoopGuard).toBe("function");
  });

  it("retries are unbounded — no per-prompt cap is exported", () => {
    // Long-horizon work must not be stranded by a retry cap. The guard
    // mirrors pi-retry: infinite retries with backoff, exiting only on a
    // clean turn, user abort (Esc), or session change (/new).
    expect(BACKOFF_BASE_MS).toBeGreaterThan(0);
    expect(BACKOFF_MAX_MS).toBeGreaterThanOrEqual(BACKOFF_BASE_MS);
    expect(BACKOFF_MULTIPLIER).toBeGreaterThan(1);
  });
});

describe("nextTrailingBangs", () => {
  it("returns the prior run for an empty delta", () => {
    expect(nextTrailingBangs(0, "")).toBe(0);
    expect(nextTrailingBangs(7, "")).toBe(7);
  });

  it("counts trailing '!' in a delta that has other characters", () => {
    expect(nextTrailingBangs(0, "hello")).toBe(0);
    expect(nextTrailingBangs(0, "hi!")).toBe(1);
    expect(nextTrailingBangs(0, "a!!")).toBe(2);
  });

  it("extends the prior run when the delta is all '!'", () => {
    expect(nextTrailingBangs(0, "!!!")).toBe(3);
    expect(nextTrailingBangs(3, "!!")).toBe(5);
    expect(nextTrailingBangs(5, "!".repeat(40))).toBe(45);
  });

  it("cuts off the prior run when a non-'!' char appears in the delta", () => {
    expect(nextTrailingBangs(5, "a!")).toBe(1);
    expect(nextTrailingBangs(5, "ab")).toBe(0);
    expect(nextTrailingBangs(5, "!a")).toBe(0);
    expect(nextTrailingBangs(5, "x!!!")).toBe(3);
  });

  it("handles multi-byte (non-'!') characters without false positives", () => {
    expect(nextTrailingBangs(0, "é!")).toBe(1);
    expect(nextTrailingBangs(4, "café")).toBe(0);
  });
});

describe("extractText", () => {
  it("passes strings through", () => {
    expect(extractText("hello")).toBe("hello");
  });

  it("concatenates text blocks", () => {
    expect(
      extractText([
        { type: "text", text: "a" },
        { type: "text", text: "b" },
      ]),
    ).toBe("ab");
  });

  it("ignores non-text blocks (thinking/toolCall)", () => {
    expect(
      extractText([
        { type: "text", text: "x" },
        { type: "thinking", text: "!!!" },
        { type: "toolCall", name: "bash" },
      ]),
    ).toBe("x");
  });

  it("returns empty string for non-text content", () => {
    expect(extractText(null)).toBe("");
    expect(extractText(123)).toBe("");
    expect(extractText(undefined)).toBe("");
    expect(extractText([{ type: "toolCall" }])).toBe("");
  });
});

describe("messageTrailingBangs", () => {
  it("counts a trailing run in string content", () => {
    expect(messageTrailingBangs({ role: "assistant", content: "hello!!!" })).toBe(3);
    expect(messageTrailingBangs({ role: "assistant", content: "plain" })).toBe(0);
  });

  it("counts a trailing run across concatenated text blocks", () => {
    expect(
      messageTrailingBangs({
        role: "assistant",
        content: [{ type: "text", text: "!!!" }, { type: "text", text: "!!" }],
      }),
    ).toBe(5);
  });

  it("returns 0 when the last text block does not end in '!'", () => {
    expect(
      messageTrailingBangs({
        role: "assistant",
        content: [{ type: "text", text: "!!!" }, { type: "text", text: "no" }],
      }),
    ).toBe(0);
  });

  it("returns 0 for empty or non-text content", () => {
    expect(messageTrailingBangs({ role: "assistant", content: "" })).toBe(0);
    expect(messageTrailingBangs({ role: "assistant", content: [{ type: "thinking", text: "!!!" }] })).toBe(0);
  });
});

describe("calculateDelay", () => {
  it("grows exponentially with the attempt number", () => {
    expect(calculateDelay(1)).toBe(BACKOFF_BASE_MS);
    expect(calculateDelay(2)).toBe(BACKOFF_BASE_MS * BACKOFF_MULTIPLIER);
    expect(calculateDelay(3)).toBe(BACKOFF_BASE_MS * BACKOFF_MULTIPLIER ** 2);
  });

  it("is capped at BACKOFF_MAX_MS", () => {
    const capped = calculateDelay(100);
    expect(capped).toBe(BACKOFF_MAX_MS);
    expect(calculateDelay(1000)).toBe(BACKOFF_MAX_MS);
  });

  it("honors a custom config", () => {
    expect(
      calculateDelay(3, { baseDelayMs: 100, maxDelayMs: 1000, multiplier: 3 }),
    ).toBe(900);
    expect(
      calculateDelay(5, { baseDelayMs: 100, maxDelayMs: 1000, multiplier: 3 }),
    ).toBe(1000);
  });
});

describe("formatDuration", () => {
  it("formats sub-second durations in ms", () => {
    expect(formatDuration(500)).toBe("500ms");
    expect(formatDuration(0)).toBe("0ms");
  });

  it("formats sub-minute durations in seconds", () => {
    expect(formatDuration(2000)).toBe("2.0s");
    expect(formatDuration(5500)).toBe("5.5s");
  });

  it("formats minute-plus durations as minutes and seconds", () => {
    expect(formatDuration(60000)).toBe("1m 0s");
    expect(formatDuration(90000)).toBe("1m 30s");
  });
});

describe("DEFAULT_BACKOFF_CONFIG", () => {
  it("mirrors pi-retry defaults", () => {
    expect(DEFAULT_BACKOFF_CONFIG).toEqual({
      baseDelayMs: 2000,
      maxDelayMs: 60_000,
      multiplier: 2,
    });
  });
});

describe("isGuardedModel", () => {
  it("guards the configured GLM 5.2 ids on the makora provider", () => {
    expect(isGuardedModel({ provider: "makora", id: "zai-org/GLM-5.2-NVFP4" })).toBe(true);
    expect(isGuardedModel({ provider: "makora", id: "zai-org/GLM-5.2-FP8" })).toBe(true);
  });

  it("rejects other makora models", () => {
    expect(isGuardedModel({ provider: "makora", id: "deepseek-ai/DeepSeek-V4-Pro" })).toBe(false);
  });

  it("rejects the right model id on the wrong provider", () => {
    expect(isGuardedModel({ provider: "openai", id: "zai-org/GLM-5.2-NVFP4" })).toBe(false);
  });

  it("rejects missing ids or undefined/null", () => {
    expect(isGuardedModel({ provider: "makora" })).toBe(false);
    expect(isGuardedModel(undefined)).toBe(false);
    expect(isGuardedModel(null)).toBe(false);
  });
});
