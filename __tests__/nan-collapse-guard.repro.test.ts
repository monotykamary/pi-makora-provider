/**
 * Reproduction-knowledge integration test for the NVFP4 NaN-collapse guard.
 * Uses the EXACT collapse shapes we reproduced on the makora engine
 * (2026-06-27 localterm session: 6 reasoning-trace `!` runs at offset 0; the
 * colleague's `{},{},{}`; the `!`/`{}` task-independence) and real normal
 * reasoning onsets, plus the trim-only recovery (only the degenerate turn is
 * dropped — the user's session is preserved). See ./glm52-nan-collapse-report/ (stashed).
 */
import { describe, it, expect } from "vitest";
import {
  isOnsetCollapse,
  isCollapseMessage,
  isGuardedModel,
  trimAborted,
} from "../nan-collapse-guard.js";

// The 6 observed session hits — all reasoning-trace, offset 0, all `!`.
const SESSION_RUN_LENS = [15857, 694, 233, 124, 18057, 478];

describe("reproduction: detects the 6 actual session `!` collapses", () => {
  for (const len of SESSION_RUN_LENS) {
    it(`trips on the ${len}-char '!' reasoning run (session hit)`, () => {
      const reasoning = [{ type: "thinking", thinking: "!".repeat(len) }];
      expect(isOnsetCollapse("!".repeat(len))).toBe(true);
      expect(isCollapseMessage({ role: "assistant", content: reasoning } as any)).toBe(true);
    });
  }
});

describe("reproduction: detects the colleague's {},{},{} and other unit shapes", () => {
  it("trips on {},{},{} (colleague's manifestation)", () => {
    expect(isOnsetCollapse("{},".repeat(22))).toBe(true);
    expect(isCollapseMessage({ role: "assistant", content: [{ type: "thinking", thinking: "{},".repeat(22) }] } as any)).toBe(true);
  });
  it("trips on ();(); and 0 0 0 (other NaN-argmax unit shapes)", () => {
    expect(isOnsetCollapse("();".repeat(22))).toBe(true);
    expect(isOnsetCollapse("0 ".repeat(22))).toBe(true);
  });
});

describe("reproduction: does NOT trip on real normal reasoning onsets", () => {
  const normalOnsets = [
    "The user wants to change the sorting logic of a session switcher",
    "Let me scan the word list and identify all the animals in it.",
    "So `lastOutputAt` is updated on output (line 69) — I need to find",
    "1.  **Understand the Goal:** The user wants to change the sorting",
    "I need to find animal names in the list. Let me scan through it.",
  ];
  for (const onset of normalOnsets) {
    it(`normal onset → false: "${onset.slice(0, 40)}…"`, () => {
      expect(isOnsetCollapse(onset)).toBe(false);
      expect(isCollapseMessage({ role: "assistant", content: [{ type: "thinking", thinking: onset }] } as any)).toBe(false);
    });
  }
});

describe("reproduction: the recovery trims ONLY the degenerate turn (preserves the session)", () => {
  it("drops the trailing collapse assistant message and nothing else", () => {
    const messages: any[] = [
      { role: "system", content: "s" },
      { role: "user", content: "do the task" },
      { role: "assistant", content: [{ type: "text", text: "working..." }] },
      { role: "assistant", content: [{ type: "thinking", thinking: "!".repeat(256) }] },
    ];
    const agent = { state: { messages } } as any;
    trimAborted(agent);
    expect(agent.state.messages.length).toBe(3);
    expect(agent.state.messages[2].content[0].text).toBe("working...");
  });

  it("drops a trailing aborted (stopReason) assistant message", () => {
    const messages = [
      { role: "user", content: "u" },
      { role: "assistant", content: [{ type: "text", text: "!!!" }], stopReason: "aborted" },
    ] as any;
    const agent = { state: { messages } } as any;
    trimAborted(agent);
    expect(agent.state.messages.length).toBe(1);
  });

  it("does NOT trim valid context — a long session's real turns are all preserved", () => {
    const messages: any[] = [
      { role: "system", content: "s" },
      { role: "user", content: "u" },
    ];
    for (let i = 0; i < 10; i++) {
      messages.push({ role: "assistant", content: [{ type: "text", text: `turn ${i}` }] });
      messages.push({ role: "tool", tool_call_id: `c${i}`, content: "X".repeat(8000) });
    }
    messages.push({ role: "assistant", content: [{ type: "thinking", thinking: "!".repeat(256) }] });
    const before = messages.length;
    const agent = { state: { messages } } as any;
    trimAborted(agent);
    // ONLY the collapse (1 message) is dropped — all 10 turns + 10 tool results kept.
    expect(agent.state.messages.length).toBe(before - 1);
    expect(agent.state.messages[0].role).toBe("system");
    expect(agent.state.messages[1].role).toBe("user");
    expect(agent.state.messages[agent.state.messages.length - 1].role).toBe("tool");
  });

  it("is a no-op when the last message is not a collapse/aborted assistant", () => {
    const messages = [
      { role: "user", content: "u" },
      { role: "assistant", content: [{ type: "text", text: "normal answer" }] },
    ] as any;
    const agent = { state: { messages } } as any;
    trimAborted(agent);
    expect(agent.state.messages.length).toBe(2);
  });
});
