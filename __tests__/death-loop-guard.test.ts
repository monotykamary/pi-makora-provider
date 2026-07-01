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
  DEATH_LOOP_STUB_TEXT,
  GUARDED_MODEL_IDS,
  IGNORED_REPEAT_CHARS,
  LINE_REPEAT_BUFFER_CHARS,
  LINE_REPEAT_THRESHOLD,
  REPEAT_THRESHOLD,
  TOKEN_REPEAT_THRESHOLD,
  TOKEN_REPEAT_BUFFER_CHARS,
  UNIT_MAX_LENGTH,
  UNIT_REPEAT_BUFFER_CHARS,
  UNIT_REPEAT_THRESHOLD,
  calculateDelay,
  extractText,
  formatDuration,
  isDeathLoopMessage,
  isDegenerateLineRun,
  isDegenerateRun,
  isDegenerateTokenRun,
  isDegenerateUnitRun,
  isGuardedMessage,
  isGuardedModel,
  messageTrailingBangs,
  messageTrailingRun,
  nextTrailingBangs,
  nextTrailingRun,
  normalizeLine,
  registerDeathLoopGuard,
  trailingLineRun,
  trailingTokenRun,
  trailingUnitRun,
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

  it("exports line-repeat thresholds for structured loops", () => {
    expect(LINE_REPEAT_THRESHOLD).toBeGreaterThanOrEqual(20);
    // Buffer must hold many complete lines beyond the threshold.
    expect(LINE_REPEAT_BUFFER_CHARS).toBeGreaterThanOrEqual(
      LINE_REPEAT_THRESHOLD * 2,
    );
  });

  it("exports unit-repeat thresholds for delimiter-joined loops", () => {
    expect(UNIT_REPEAT_THRESHOLD).toBeGreaterThanOrEqual(20);
    expect(UNIT_MAX_LENGTH).toBeGreaterThanOrEqual(2);
    // Buffer must hold threshold copies of the longest candidate unit.
    expect(UNIT_REPEAT_BUFFER_CHARS).toBeGreaterThanOrEqual(
      UNIT_REPEAT_THRESHOLD * UNIT_MAX_LENGTH,
    );
  });

  it("exports a non-empty death-loop stub text", () => {
    expect(typeof DEATH_LOOP_STUB_TEXT).toBe("string");
    expect(DEATH_LOOP_STUB_TEXT.length).toBeGreaterThan(0);
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

  it("a structured log loop is NOT a char/token run but IS a line/template run", () => {
    const names = ["alice", "bob", "carol", "dave"];
    const lines: string[] = [];
    for (let i = 0; i < 100; i++)
      lines.push(
        `2025-11-12 11:31:42,${String(i).padStart(3, "0")} [0.0.0.0:54321] DEBUG: User logged in: ${names[i % names.length]}`,
      );
    const text = lines.join("\n") + "\n";
    // Last char is a newline (whitespace) -> no char run.
    expect(
      isDegenerateRun(messageTrailingRun({ role: "assistant", content: text })),
    ).toBe(false);
    // Last token differs per line -> no token run.
    expect(isDegenerateTokenRun(trailingTokenRun(text))).toBe(false);
    // But every line shares one normalized structure -> line run trips.
    expect(isDegenerateLineRun(trailingLineRun(text))).toBe(true);
  });
});

describe("normalizeLine", () => {
  it("maps identifiers to A and digit runs to #", () => {
    expect(normalizeLine("User logged in: alice")).toBe("A A A: A");
    expect(normalizeLine("User logged in: bob")).toBe("A A A: A");
  });

  it("maps pure numbers to #", () => {
    expect(normalizeLine("2025-11-12 11:31:42,000")).toBe("#-#-# #:#:#,#");
  });

  it("collapses runs of spaces/tabs to one and trims trailing whitespace", () => {
    expect(normalizeLine("a   b\tc   ")).toBe("A A A");
    expect(normalizeLine("\t\tx  ")).toBe(" A");
  });

  it("leaves structural punctuation intact", () => {
    expect(normalizeLine("[0.0.0.0:54321] DEBUG:")).toBe("[#.#.#.#:#] A:");
  });

  it("returns empty for an empty/whitespace-only line", () => {
    expect(normalizeLine("")).toBe("");
    expect(normalizeLine("   ")).toBe("");
    expect(normalizeLine("\t")).toBe("");
  });
});

describe("trailingLineRun", () => {
  const logLine = (i: number, name: string) =>
    `2025-11-12 11:31:42,${String(i).padStart(3, "0")} [0.0.0.0:54321] DEBUG: User logged in: ${name}`;
  const names = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"];
  const TEMPLATE = "#-#-# #:#:#,# [#.#.#.#:#] A: A A A: A";

  it("returns an empty run when there are no complete lines", () => {
    expect(trailingLineRun("")).toEqual({ template: "", count: 0 });
    expect(trailingLineRun("no newline here")).toEqual({ template: "", count: 0 });
  });

  it("counts a single complete line as a run of 1", () => {
    expect(trailingLineRun("hello\n")).toEqual({ template: "A", count: 1 });
  });

  it("counts consecutive lines with the same normalized structure", () => {
    const text =
      [0, 1, 2, 3].map((i) => logLine(i, names[i % names.length])).join("\n") + "\n";
    const run = trailingLineRun(text);
    expect(run.count).toBe(4);
    expect(run.template).toBe(TEMPLATE);
  });

  it("ignores the trailing partial (mid-stream) line", () => {
    const complete =
      [0, 1, 2].map((i) => logLine(i, names[i % names.length])).join("\n") + "\n";
    const text = complete + "partial line with no newline";
    expect(trailingLineRun(text).count).toBe(3);
  });

  it("stops at the first line with a different structure", () => {
    const text =
      "header line\n" +
      [0, 1, 2].map((i) => logLine(i, names[i % names.length])).join("\n") +
      "\n";
    const run = trailingLineRun(text);
    expect(run.count).toBe(3);
    expect(run.template).toBe(TEMPLATE);
  });

  it("skips trailing blank lines and does not trip on them", () => {
    const text =
      [0, 1, 2].map((i) => logLine(i, names[i % names.length])).join("\n") +
      "\n\n\n";
    expect(trailingLineRun(text).count).toBe(3);
  });

  it("detects a 100-line structural log loop (the colleague case)", () => {
    const lines: string[] = [];
    for (let i = 0; i < 100; i++) lines.push(logLine(i, names[i % names.length]));
    const text = lines.join("\n") + "\n";
    expect(trailingLineRun(text).count).toBe(100);
  });
});

describe("isDegenerateLineRun", () => {
  it("is false below the threshold", () => {
    expect(isDegenerateLineRun({ template: "A A A: A", count: 99 })).toBe(false);
  });

  it("trips at/above the threshold for a non-empty template", () => {
    expect(isDegenerateLineRun({ template: "A A A: A", count: 100 })).toBe(true);
    expect(isDegenerateLineRun({ template: "#-#-# #", count: 150 })).toBe(true);
  });

  it("never trips on a blank-line run", () => {
    expect(isDegenerateLineRun({ template: "", count: 100 })).toBe(false);
    expect(isDegenerateLineRun({ template: "", count: 10_000 })).toBe(false);
  });

  it("honors a custom threshold", () => {
    expect(isDegenerateLineRun({ template: "A", count: 5 }, 10)).toBe(false);
    expect(isDegenerateLineRun({ template: "A", count: 10 }, 10)).toBe(true);
  });
});

describe("trailingUnitRun", () => {
  it("returns an empty run for text too short to repeat", () => {
    expect(trailingUnitRun("")).toEqual({ unit: "", count: 0 });
    expect(trailingUnitRun("a")).toEqual({ unit: "", count: 0 });
  });

  it("detects a single-character unit (!!!!)", () => {
    expect(trailingUnitRun("!".repeat(40))).toEqual({ unit: "!", count: 40 });
  });

  it("detects a multi-char unit under a delimiter ({},{},{})", () => {
    expect(trailingUnitRun("{},".repeat(40))).toEqual({ unit: "{},", count: 40 });
  });

  it("detects a multi-char unit with no delimiter ({}{}{})", () => {
    expect(trailingUnitRun("{}".repeat(40))).toEqual({ unit: "{}", count: 40 });
  });

  it("detects a spaced unit (0 0 0)", () => {
    const run = trailingUnitRun("0 ".repeat(40));
    expect(run.count).toBeGreaterThanOrEqual(40);
    expect(run.unit).toBe("0 ");
  });

  it("ignores a leading partial block (mid-stream truncation)", () => {
    // A partial "{}" before the full "{},{}" blocks must not reduce the count.
    const text = "{}" + "{},".repeat(40);
    expect(trailingUnitRun(text).count).toBe(40);
  });

  it("returns a low count for non-repeating prose", () => {
    const run = trailingUnitRun(
      "The quick brown fox jumps over the lazy dog.",
    );
    expect(run.count).toBeLessThan(10);
  });

  it("honors a custom max length", () => {
    // Unit "ab" (length 2) is found when maxLength >= 2.
    expect(trailingUnitRun("ab".repeat(40), 2).count).toBe(40);
    // With maxLength 1 only single-char periods are considered -> no "ab" run.
    expect(trailingUnitRun("ab".repeat(40), 1).count).toBe(1);
  });
});

describe("isDegenerateUnitRun", () => {
  it("is false below the threshold", () => {
    expect(isDegenerateUnitRun({ unit: "{},", count: 39 })).toBe(false);
  });

  it("trips at/above the threshold for a non-whitespace unit", () => {
    expect(isDegenerateUnitRun({ unit: "{},", count: 40 })).toBe(true);
    expect(isDegenerateUnitRun({ unit: "!", count: 100 })).toBe(true);
  });

  it("never trips on a pure-whitespace unit", () => {
    expect(isDegenerateUnitRun({ unit: " ", count: 100 })).toBe(false);
    expect(isDegenerateUnitRun({ unit: "   ", count: 100 })).toBe(false);
  });

  it("is false for an empty unit", () => {
    expect(isDegenerateUnitRun({ unit: "", count: 0 })).toBe(false);
    expect(isDegenerateUnitRun({ unit: "", count: 100 })).toBe(false);
  });

  it("honors a custom threshold", () => {
    expect(isDegenerateUnitRun({ unit: "ab", count: 5 }, 10)).toBe(false);
    expect(isDegenerateUnitRun({ unit: "ab", count: 10 }, 10)).toBe(true);
  });
});

describe("delimiter-joined unit detection coverage", () => {
  it("'{},{},{}' is NOT a char/token/line run but IS a trailing-unit run", () => {
    const text = "{},".repeat(40);
    expect(
      isDegenerateRun(messageTrailingRun({ role: "assistant", content: text })),
    ).toBe(false);
    expect(isDegenerateTokenRun(trailingTokenRun(text))).toBe(false);
    expect(isDegenerateLineRun(trailingLineRun(text))).toBe(false);
    expect(isDegenerateUnitRun(trailingUnitRun(text))).toBe(true);
  });
});

describe("isDeathLoopMessage", () => {
  const stub = {
    role: "assistant",
    provider: "makora",
    model: "zai-org/GLM-5.2-NVFP4",
  };

  it("detects a character-run message", () => {
    expect(isDeathLoopMessage({ ...stub, content: "code " + "!".repeat(40) })).toBe(
      true,
    );
  });

  it("detects a spaced token-run message", () => {
    expect(isDeathLoopMessage({ ...stub, content: "0 ".repeat(40) })).toBe(true);
  });

  it("detects a delimiter-joined unit-loop message ({},{},{})", () => {
    expect(isDeathLoopMessage({ ...stub, content: "{},".repeat(40) })).toBe(true);
    expect(isDeathLoopMessage({ ...stub, content: "{}".repeat(40) })).toBe(true);
  });

  it("detects a structured line-loop message", () => {
    const names = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"];
    const lines: string[] = [];
    for (let i = 0; i < 100; i++)
      lines.push(
        `2025-11-12 11:31:42,${String(i).padStart(3, "0")} [0.0.0.0:54321] DEBUG: User logged in: ${names[i % names.length]}`,
      );
    expect(isDeathLoopMessage({ ...stub, content: lines.join("\n") + "\n" })).toBe(
      true,
    );
  });

  it("is false for clean prose", () => {
    expect(
      isDeathLoopMessage({
        ...stub,
        content: "The quick brown fox jumps over the lazy dog.",
      }),
    ).toBe(false);
  });

  it("ignores non-assistant messages", () => {
    expect(isDeathLoopMessage({ role: "user", content: "0 ".repeat(100) })).toBe(
      false,
    );
  });

  it("is false for empty content", () => {
    expect(isDeathLoopMessage({ ...stub, content: "" })).toBe(false);
    expect(
      isDeathLoopMessage({ ...stub, content: [{ type: "thinking", text: "x" }] }),
    ).toBe(false);
  });
});

describe("isGuardedMessage", () => {
  it("accepts a guarded GLM 5.2 message from the makora provider", () => {
    expect(
      isGuardedMessage({
        role: "assistant",
        provider: "makora",
        model: "zai-org/GLM-5.2-NVFP4",
      }),
    ).toBe(true);
    expect(
      isGuardedMessage({
        role: "assistant",
        provider: "makora",
        model: "zai-org/GLM-5.2-FP8",
      }),
    ).toBe(true);
  });

  it("rejects other makora models", () => {
    expect(
      isGuardedMessage({
        role: "assistant",
        provider: "makora",
        model: "deepseek-ai/DeepSeek-V4-Pro",
      }),
    ).toBe(false);
  });

  it("rejects the right model on the wrong provider", () => {
    expect(
      isGuardedMessage({
        role: "assistant",
        provider: "openai",
        model: "zai-org/GLM-5.2-NVFP4",
      }),
    ).toBe(false);
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
