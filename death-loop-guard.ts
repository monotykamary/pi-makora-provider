/**
 * Death-loop guard for Makora reasoning models.
 *
 * Some Makora models (notably GLM 5.2 NVFP4 / FP8) occasionally fall into a
 * degenerate repetition loop, emitting an unbroken run of '!' characters
 * (e.g. "!!!!...") that consumes the whole response. This guard watches the
 * streamed assistant output (both the visible answer and the reasoning
 * trace); when that run is detected it aborts the runaway generation, drops
 * the partial (toxic) assistant message from the transcript, and resumes the
 * agentic loop invisibly via agent.prompt([]) — the same pattern
 * pi-invisible-continue uses, so no new user message pollutes the context.
 *
 * Why trim the aborted message: on abort, pi finalizes the in-flight
 * assistant message WITH its accumulated '!!!' content and stopReason
 * "aborted" into the transcript. Resuming from that context would re-feed
 * the toxic text to the model and likely re-trigger the loop. Dropping the
 * last (aborted) assistant message leaves the context ending at the prior
 * user/toolResult message — a clean continuation point.
 *
 * Module resolution: @earendil-works/pi-agent-core is a devDependency only
 * (types + test resolution). At runtime pi's extension loader aliases that
 * specifier to its bundled copy, so the Agent class patched below is the
 * SAME class AgentSession uses. A static import is required — jiti's alias
 * applies to static imports (which it rewrites to its own resolver) but not
 * to native dynamic import() calls.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Agent } from "@earendil-works/pi-agent-core";

const PROVIDER_ID = "makora";

/** Makora model IDs to guard. Add ids to widen coverage, or include "*"
 *  to guard every Makora model. Defaults to the GLM 5.2 family (the known
 *  offender). Kept exported so tests and downstream forks can introspect. */
export const GUARDED_MODEL_IDS = new Set<string>([
  "zai-org/GLM-5.2-NVFP4",
  "zai-org/GLM-5.2-FP8",
]);

/** Trip after this many consecutive '!' characters in the streamed answer.
 *  40 is far above anything normal prose or code produces. */
export const BANG_THRESHOLD = 40;

/** Max invisible recoveries per user prompt, to bound abort/continue thrash. */
export const MAX_RECOVERS_PER_RUN = 3;

export interface GuardedMessage {
  role: string;
  stopReason?: string;
  content?: unknown;
}

export interface GuardedAgent {
  abort(): void;
  waitForIdle(): Promise<void>;
  prompt(input: unknown[] | string): Promise<void>;
  state: { messages: GuardedMessage[] };
}

let _agent: GuardedAgent | null = null;

/** Mutex held only across the abort+trim critical section — NOT across the
 *  resumed prompt([]) run, so the resumed stream stays monitored. */
let _recovering = false;

/** Trailing '!' run length in the current text block. */
let _trailingBangs = 0;

/** Latch: already tripped for the current assistant message. */
let _tripped = false;

/** Recoveries performed in the current user-initiated agent run. */
let _recoversThisRun = 0;

export function isGuardedModel(
  model: { provider?: string; id?: string } | undefined | null,
): boolean {
  if (!model) return false;
  if (model.provider !== PROVIDER_ID) return false;
  if (GUARDED_MODEL_IDS.has("*")) return true;
  return model.id != null && GUARDED_MODEL_IDS.has(model.id);
}

/** Update the trailing-'!' run length given a new text delta. O(len(delta)).
 *  Trailing run depends only on the delta's suffix: if the delta contains any
 *  non-'!' char, the prior run is cut off at that char; if the delta is all
 *  '!', it extends the prior run. */
export function nextTrailingBangs(prev: number, delta: string): number {
  const len = delta.length;
  if (len === 0) return prev;
  let i = len - 1;
  while (i >= 0 && delta.charCodeAt(i) === 0x21) i--; // '!' === 0x21
  const trailingInDelta = len - 1 - i;
  return i >= 0 ? trailingInDelta : prev + len;
}

export function extractText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    let out = "";
    for (const block of content) {
      if (
        block &&
        typeof block === "object" &&
        (block as { type?: string }).type === "text" &&
        typeof (block as { text?: unknown }).text === "string"
      ) {
        out += (block as { text: string }).text;
      }
    }
    return out;
  }
  return "";
}

/** Trailing '!' run length of a finalized message's text content. */
export function messageTrailingBangs(msg: GuardedMessage): number {
  const text = extractText(msg.content);
  let i = text.length - 1;
  while (i >= 0 && text.charCodeAt(i) === 0x21) i--;
  return text.length - 1 - i;
}

async function recover(agent: GuardedAgent): Promise<void> {
  _recovering = true;
  try {
    agent.abort();
    await agent.waitForIdle();
    const msgs = agent.state.messages;
    const last = msgs[msgs.length - 1];
    if (
      last &&
      last.role === "assistant" &&
      (last.stopReason === "aborted" || messageTrailingBangs(last) >= BANG_THRESHOLD)
    ) {
      // Drop the toxic partial so it isn't re-fed to the model on resume.
      // The state setter copies the array, leaving a clean transcript that
      // ends at the prior user/toolResult message.
      agent.state.messages = msgs.slice(0, -1);
    }
  } finally {
    // Release before resuming so the resumed stream stays monitored.
    _recovering = false;
  }
  try {
    // Invisible continue: fresh agent loop, no new message injected.
    await agent.prompt([]);
  } catch {
    // "Agent is already processing" or other transient error — best effort.
  }
}

export function registerDeathLoopGuard(pi: ExtensionAPI): void {
  // Capture the live Agent instance by chaining Agent.prototype.subscribe.
  // subscribe() fires when AgentSession attaches — on every fresh session
  // and every resume — so _agent always points at the active Agent. Chain
  // any prior patch (e.g. pi-invisible-continue) so both extensions coexist.
  const proto = Agent.prototype as unknown as {
    subscribe: (this: GuardedAgent, ...args: unknown[]) => unknown;
  };
  const origSubscribe = proto.subscribe;
  proto.subscribe = function (this: GuardedAgent, ...args: unknown[]) {
    _agent = this;
    return origSubscribe.apply(this, args);
  };

  pi.on("session_start", () => {
    _recovering = false;
    _trailingBangs = 0;
    _tripped = false;
    _recoversThisRun = 0;
  });

  // before_agent_start fires only for user prompts (the AgentSession path),
  // not for the recovery's direct agent.prompt([]), so the recovery cap is
  // bounded per user prompt instead of reset on every recovery continuation.
  pi.on("before_agent_start", () => {
    _recovering = false;
    _trailingBangs = 0;
    _tripped = false;
    _recoversThisRun = 0;
  });

  pi.on("message_start", (event) => {
    if (event.message?.role === "assistant") {
      _trailingBangs = 0;
      _tripped = false;
    }
  });

  pi.on("message_update", (event, ctx) => {
    if (_recovering || _tripped) return;
    const ame = event.assistantMessageEvent;
    if (ame.type === "text_start" || ame.type === "thinking_start") {
      // New content block (answer or reasoning) — trailing run starts fresh.
      _trailingBangs = 0;
      return;
    }
    // Watch both the visible answer and the reasoning trace: GLM 5.2 is a
    // reasoning model, and the loop can surface in either.
    if (ame.type !== "text_delta" && ame.type !== "thinking_delta") return;
    if (!isGuardedModel(ctx.model)) return;

    _trailingBangs = nextTrailingBangs(_trailingBangs, ame.delta);
    if (_trailingBangs < BANG_THRESHOLD) return;

    if (_recoversThisRun >= MAX_RECOVERS_PER_RUN) {
      _tripped = true;
      ctx.ui.notify(
        "Makora death-loop guard: runaway '!' output detected, but the " +
          "recovery limit for this prompt was reached — stopping " +
          "intervention. Try /continue or rephrase.",
        "warning",
      );
      return;
    }
    _tripped = true;
    _recoversThisRun++;
    const agent = _agent;
    if (!agent) {
      ctx.ui.notify(
        "Makora death-loop guard: runaway '!' output detected but the Agent " +
          "instance was not captured; cannot recover automatically.",
        "warning",
      );
      return;
    }
    ctx.ui.notify(
      `Makora death-loop guard: aborting runaway '!' output and resuming ` +
        `(${_recoversThisRun}/${MAX_RECOVERS_PER_RUN}).`,
      "warning",
    );
    // Detach: the handler must return so the run can unwind to idle before
    // recover() awaits waitForIdle() and calls prompt([]).
    void recover(agent);
  });
}
