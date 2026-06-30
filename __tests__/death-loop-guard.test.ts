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
  REPEAT_THRESHOLD,
  TOKEN_REPEAT_THRESHOLD,
  TOKEN_REPEAT_BUFFER_CHARS,
  BACKOFF_BASE_MS,
  BACKOFF_MAX_MS,
  BACKOFF_MULTIPLIER,
  DEFAULT_BACKOFF_CONFIG,
  GUARDED_MODEL_IDS,
  IGNORED_REPEAT_CHARS,
  calculateDelay,
  extractText,
  formatDuration,
  isDegenerateRun,
  isDegenerateTokenRun,
  isGuardedModel,
  messageTrailingBangs,
  messageTrailingRun,
  nextTrailingBangs,
  nextTrailingRun,
  registerDeathLoopGuard,
  trailingTokenRun,
} from "../death-loop-guard.js";

describe("death-loop-guard config", () => {
  it("guards the GLM 5.2 family by default", () => {
    expect(GUARDED_MODEL_IDS.has("zai-org/GLM-5.2-NVFP4")).toBe(true);
    expect(GUARDED_MODEL_IDS.has("zai-org/GLM-5.2-FP8")).toBe(true);
  });

  it("exports a sensible repeat threshold", () => {
    expect(REPEAT_THRESHOLD).toBeGreaterThanOrEqual(20);
  });

  it("keeps BANG_THRESHOLD as a deprecated alias of REPEAT_THRESHOLD", () => {
    expect(BANG_THRESHOLD).toBe(REPEAT_THRESHOLD);
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

  it("ignores whitespace characters for repetition trips", () => {
    expect(IGNORED_REPEAT_CHARS.has(" ")).toBe(true);
    expect(IGNORED_REPEAT_CHARS.has("\t")).toBe(true);
    expect(IGNORED_REPEAT_CHARS.has("\n")).toBe(true);
    expect(IGNORED_REPEAT_CHARS.has("\r")).toBe(true);
    expect(IGNORED_REPEAT_CHARS.has("!")).toBe(false);
    expect(IGNORED_REPEAT_CHARS.has("0")).toBe(false);
  });

  it("exports token-repeat thresholds for spaced loops", () => {
    expect(TOKEN_REPEAT_THRESHOLD).toBeGreaterThanOrEqual(20);
    // Buffer must hold many copies of the repeated unit.
    expect(TOKEN_REPEAT_BUFFER_CHARS).toBeGreaterThanOrEqual(
      TOKEN_REPEAT_THRESHOLD * 2,
    );
  });
});

describe("nextTrailingRun", () => {
  it("returns the prior run unchanged for an empty delta", () => {
    expect(nextTrailingRun({ char: "", len: 0 }, "")).toEqual({ char: "", len: 0 });
    expect(nextTrailingRun({ char: "!", len: 7 }, "")).toEqual({ char: "!", len: 7 });
  });

  it("counts the trailing run of any single character", () => {
    expect(nextTrailingRun({ char: "", len: 0 }, "hello")).toEqual({ char: "o", len: 1 });
    expect(nextTrailingRun({ char: "", len: 0 }, "hi!")).toEqual({ char: "!", len: 1 });
    expect(nextTrailingRun({ char: "", len: 0 }, "a!!")).toEqual({ char: "!", len: 2 });
    // The colleague's case: a run of '0'.
    expect(nextTrailingRun({ char: "", len: 0 }, "code 0000")).toEqual({ char: "0", len: 4 });
    expect(nextTrailingRun({ char: "", len: 0 }, "0000")).toEqual({ char: "0", len: 4 });
  });

  it("extends the prior run when the delta is all the same character", () => {
    expect(nextTrailingRun({ char: "!", len: 0 }, "!!!")).toEqual({ char: "!", len: 3 });
    expect(nextTrailingRun({ char: "!", len: 3 }, "!!")).toEqual({ char: "!", len: 5 });
    expect(nextTrailingRun({ char: "!", len: 5 }, "!".repeat(40))).toEqual({ char: "!", len: 45 });
    expect(nextTrailingRun({ char: "0", len: 3 }, "00")).toEqual({ char: "0", len: 5 });
  });

  it("cuts off the prior run and switches character when a different one appears", () => {
    expect(nextTrailingRun({ char: "!", len: 5 }, "a!")).toEqual({ char: "!", len: 1 });
    expect(nextTrailingRun({ char: "!", len: 5 }, "ab")).toEqual({ char: "b", len: 1 });
    expect(nextTrailingRun({ char: "!", len: 5 }, "!a")).toEqual({ char: "a", len: 1 });
    expect(nextTrailingRun({ char: "!", len: 5 }, "x!!!")).toEqual({ char: "!", len: 3 });
    // '!' loop hands off to a '0' loop.
    expect(nextTrailingRun({ char: "!", len: 12 }, "0000")).toEqual({ char: "0", len: 4 });
  });

  it("tracks whitespace runs (filtering happens in isDegenerateRun, not here)", () => {
    expect(nextTrailingRun({ char: "", len: 0 }, "   ")).toEqual({ char: " ", len: 3 });
    expect(nextTrailingRun({ char: " ", len: 3 }, "  ")).toEqual({ char: " ", len: 5 });
    expect(nextTrailingRun({ char: "", len: 0 }, "\n\n\n")).toEqual({ char: "\n", len: 3 });
  });

  it("handles multi-byte (non-ASCII) characters without false run extension", () => {
    expect(nextTrailingRun({ char: "", len: 0 }, "é!")).toEqual({ char: "!", len: 1 });
    expect(nextTrailingRun({ char: "!", len: 4 }, "café")).toEqual({ char: "é", len: 1 });
  });
});

describe("nextTrailingBangs (deprecated '!'-only alias)", () => {
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

  it("returns 0 once the run switches away from '!'", () => {
    expect(nextTrailingBangs(12, "0000")).toBe(0);
  });
});

describe("isDegenerateRun", () => {
  it("is false below the threshold", () => {
    expect(isDegenerateRun({ char: "!", len: 39 })).toBe(false);
    expect(isDegenerateRun({ char: "0", len: 39 })).toBe(false);
  });

  it("trips at/above the threshold for any non-whitespace character", () => {
    expect(isDegenerateRun({ char: "!", len: 40 })).toBe(true);
    expect(isDegenerateRun({ char: "!", len: 45 })).toBe(true);
    expect(isDegenerateRun({ char: "0", len: 40 })).toBe(true);
    expect(isDegenerateRun({ char: "a", len: 40 })).toBe(true);
    expect(isDegenerateRun({ char: "-", len: 40 })).toBe(true);
  });

  it("never trips on whitespace runs, however long", () => {
    expect(isDegenerateRun({ char: " ", len: 40 })).toBe(false);
    expect(isDegenerateRun({ char: " ", len: 10_000 })).toBe(false);
    expect(isDegenerateRun({ char: "\t", len: 40 })).toBe(false);
    expect(isDegenerateRun({ char: "\n", len: 40 })).toBe(false);
    expect(isDegenerateRun({ char: "\r", len: 40 })).toBe(false);
  });

  it("is false for an empty run", () => {
    expect(isDegenerateRun({ char: "", len: 0 })).toBe(false);
    expect(isDegenerateRun({ char: "", len: 100 })).toBe(false);
  });

  it("honors a custom threshold", () => {
    expect(isDegenerateRun({ char: "0", len: 5 }, 10)).toBe(false);
    expect(isDegenerateRun({ char: "0", len: 10 }, 10)).toBe(true);
  });
});

describe("trailingTokenRun", () => {
  it("returns an empty run for empty or whitespace-only text", () => {
    expect(trailingTokenRun("")).toEqual({ token: "", count: 0 });
    expect(trailingTokenRun("   ")).toEqual({ token: "", count: 0 });
    expect(trailingTokenRun("\n\t")).toEqual({ token: "", count: 0 });
  });

  it("counts a trailing run of a single token", () => {
    expect(trailingTokenRun("hello")).toEqual({ token: "hello", count: 1 });
    expect(trailingTokenRun("foo bar")).toEqual({ token: "bar", count: 1 });
  });

  it("counts consecutive identical tokens separated by whitespace", () => {
    expect(trailingTokenRun("0 0 0")).toEqual({ token: "0", count: 3 });
    expect(trailingTokenRun("a 0 0 0")).toEqual({ token: "0", count: 3 });
    expect(trailingTokenRun("! ! ! !")).toEqual({ token: "!", count: 4 });
    expect(trailingTokenRun("!!!! !!!! !!!!")).toEqual({ token: "!!!!", count: 3 });
  });

  it("handles leading whitespace and a different leading token", () => {
    expect(trailingTokenRun("   0 0 0")).toEqual({ token: "0", count: 3 });
    expect(trailingTokenRun("x 0 0 0")).toEqual({ token: "0", count: 3 });
  });

  it("handles trailing whitespace without inflating the count", () => {
    expect(trailingTokenRun("0 0 0 ")).toEqual({ token: "0", count: 3 });
    expect(trailingTokenRun("0 0 0\n")).toEqual({ token: "0", count: 3 });
  });

  it("treats tabs and newlines as token separators", () => {
    expect(trailingTokenRun("0\t0\t0")).toEqual({ token: "0", count: 3 });
    expect(trailingTokenRun("0\n0\n0")).toEqual({ token: "0", count: 3 });
  });

  it("stops counting at the first different token", () => {
    expect(trailingTokenRun("0 0 0 1")).toEqual({ token: "1", count: 1 });
    expect(trailingTokenRun("00 0 0")).toEqual({ token: "0", count: 2 });
  });

  it("detects a 40-token spaced zero loop (the colleague case)", () => {
    expect(trailingTokenRun("0 ".repeat(40))).toEqual({ token: "0", count: 40 });
    expect(trailingTokenRun("x ".repeat(39) + "0 ".repeat(40))).toEqual({
      token: "0",
      count: 40,
    });
  });
});

describe("isDegenerateTokenRun", () => {
  it("is false below the threshold", () => {
    expect(isDegenerateTokenRun({ token: "0", count: 39 })).toBe(false);
  });

  it("trips at/above the threshold for any token", () => {
    expect(isDegenerateTokenRun({ token: "0", count: 40 })).toBe(true);
    expect(isDegenerateTokenRun({ token: "!", count: 40 })).toBe(true);
    expect(isDegenerateTokenRun({ token: "!!!!", count: 50 })).toBe(true);
  });

  it("is false for an empty token", () => {
    expect(isDegenerateTokenRun({ token: "", count: 0 })).toBe(false);
    expect(isDegenerateTokenRun({ token: "", count: 100 })).toBe(false);
  });

  it("honors a custom threshold", () => {
    expect(isDegenerateTokenRun({ token: "0", count: 5 }, 10)).toBe(false);
    expect(isDegenerateTokenRun({ token: "0", count: 10 }, 10)).toBe(true);
  });
});

describe("character vs token detection coverage", () => {
  it("a spaced zero loop is NOT a degenerate character run but IS a degenerate token run", () => {
    const spaced = "0 ".repeat(40);
    // Character run sees only the last '0' (the space resets it).
    let charRun = { char: "", len: 0 };
    for (const ch of spaced) charRun = nextTrailingRun(charRun, ch);
    expect(isDegenerateRun(charRun)).toBe(false);
    // Token run sees 40 consecutive '0' tokens.
    expect(isDegenerateTokenRun(trailingTokenRun(spaced))).toBe(true);
  });

  it("an unspaced zero loop IS a degenerate character run", () => {
    const unspaced = "0".repeat(40);
    let charRun = { char: "", len: 0 };
    for (const ch of unspaced) charRun = nextTrailingRun(charRun, ch);
    expect(isDegenerateRun(charRun)).toBe(true);
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

describe("messageTrailingRun", () => {
  it("counts a trailing run of any character in string content", () => {
    expect(messageTrailingRun({ role: "assistant", content: "hello!!!" })).toEqual({ char: "!", len: 3 });
    expect(messageTrailingRun({ role: "assistant", content: "plain" })).toEqual({ char: "n", len: 1 });
    expect(messageTrailingRun({ role: "assistant", content: "value 0000" })).toEqual({ char: "0", len: 4 });
  });

  it("counts a trailing run across concatenated text blocks", () => {
    expect(
      messageTrailingRun({
        role: "assistant",
        content: [{ type: "text", text: "!!!" }, { type: "text", text: "!!" }],
      }),
    ).toEqual({ char: "!", len: 5 });
  });

  it("switches character when the last text block ends differently", () => {
    expect(
      messageTrailingRun({
        role: "assistant",
        content: [{ type: "text", text: "!!!" }, { type: "text", text: "no" }],
      }),
    ).toEqual({ char: "o", len: 1 });
  });

  it("returns an empty run for empty or non-text content", () => {
    expect(messageTrailingRun({ role: "assistant", content: "" })).toEqual({ char: "", len: 0 });
    expect(
      messageTrailingRun({ role: "assistant", content: [{ type: "thinking", text: "!!!" }] }),
    ).toEqual({ char: "", len: 0 });
  });
});

describe("messageTrailingBangs (deprecated '!'-only alias)", () => {
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
