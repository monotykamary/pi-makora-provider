/**
 * NVFP4 long-context NaN-collapse guard.
 *
 * Replaces the old blanket death-loop detector with a targeted guard built on
 * the root cause we diagnosed: at ~10k+ prompt tokens the GLM-5.2-NVFP4/FP8
 * cold (or long-context) prefill through the NVFP4 quantized MoE path produces
 * NaN logits. argmax-of-NaN returns the lowest token id ("!" here, "{},{}" on
 * other deployments) and that token becomes a stable fixed point → infinite
 * repetition, finish_reason "length"; streaming silently aborts (empty stream);
 * requesting logprobs surfaces it as HTTP 400 "Out of range float values are not
 * JSON compliant: nan". See ./glm52-nan-collapse-report/ (stashed) + vLLM
 * #31856 / #47042.
 *
 * The ONLY change from the old blanket death-loop guard is a CLEARER repetition
 * signal: instead of four output pattern-matchers, this detects the NaN-argmax
 * onset fixed-point in the reasoning trace (the first ~64 chars = one short
 * unit repeated) scoped to the GLM-5.2 NVFP4/FP8 quants. The recovery is
 * UNCHANGED from the old guard: trim the degenerate message and resume
 * indefinitely via agent.prompt([]) (invisible-continue) with backoff until a
 * clean turn, user abort, or session change. We deliberately do NOT trim valid
 * context — removing the user's session to "fix" the recurrence would destroy
 * the session. (The recurrence is the engine bug re-triggering as context
 * grows; the guard catches each collapse early at onset instead of after a
 * 15k-token `!` run.) An optional logprobs canary (MAKORA_NAN_CANARY) flags NaN
 * prefills for the engine team.
 *
 * Mitigation only — the NaN originates in the vLLM NVFP4 long-context prefill
 * kernel (engine bug). The fix belongs in the engine; this guard keeps the
 * toxic output out of the transcript and stops the runaway generation early.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Agent } from "@earendil-works/pi-agent-core";

// ---- Configuration ---------------------------------------------------------

/** Model IDs vulnerable to the NVFP4 long-context NaN collapse. Add "*" to
 *  guard every makora model. Defaults to the GLM-5.2 NVFP4/FP8 quants. */
export const GUARDED_MODEL_IDS = new Set<string>([
  "zai-org/GLM-5.2-NVFP4",
  "zai-org/GLM-5.2-FP8",
]);

/** Prompt-token floor for the bug. The NVFP4 NaN collapse is observed from
 *  ~8.9k tokens (intermittent) and reliably ≥~10k; below ~6k it doesn't occur.
 *  Used to gate the EMPTY-STREAM detector (an empty turn at <this is likely
 *  legitimate, not a NaN abort) and to label the recovery notify. Onset
 *  fixed-point detection is NOT gated — it's near-zero false-positive and the
 *  collapse threshold is fuzzy, so we check every guarded-model reasoning
 *  onset (cost is one ~64-char check per turn). */
export const NAN_COLLAPSE_TOKEN_THRESHOLD = 6_000;

/** The logprobs canary is expensive (re-prefills), so only probe at this size
 *  where collapse is reliable. Opt-in via MAKORA_NAN_CANARY. */
export const CANARY_TOKEN_THRESHOLD = 10_000;

/** Onset detection window: we only inspect the FIRST this-many chars of each
 *  reasoning/text block. The collapse is at token 0, so this is enough. */
const ONSET_WINDOW_CHARS = 64;
/** Need at least this many chars before deciding the onset is a collapse. */
const ONSET_MIN_CHARS = 40;
/** A short unit (1..UNIT_MAX_LEN chars) repeated this many times at onset =
 *  NaN-argmax fixed point ("!", "{},", "();", ...). */
const ONSET_MIN_REPS = 20;
const UNIT_MAX_LEN = 3;

/** Clean replacement swapped in at message_end so a collapse is removed from
 *  the saved transcript; the context handler also strips this stub. */
export const DEGENERATE_STUB_TEXT =
  "[Makora NaN-collapse guard: discarded a degenerate NVFP4 long-context " +
  "prefill (NaN logits) and reduced context before resuming.]";

const BACKOFF_BASE_MS = 2000;
const BACKOFF_MAX_MS = 60_000;
const BACKOFF_MULTIPLIER = 2;

// ---- Types (loose — event shapes are complex) ------------------------------

interface GuardedMessage {
  role?: string;
  content?: unknown;
  tool_calls?: unknown;
  stopReason?: string;
  model?: string;
}
interface GuardedAgent {
  state: { messages: GuardedMessage[] };
  abort(): void;
  waitForIdle(): Promise<void>;
  prompt(input: unknown[] | string): Promise<void>;
}

// ---- Module state (one active session at a time, like the old guard) -------

let _agent: GuardedAgent | null = null;
let _notifyFn: ((msg: string, level: "info" | "warning" | "error") => void) | null = null;
let _weAborted = false;   // we triggered the abort (vs user Esc)
let _userAborted = false; // user aborted (Esc) — exit recovery
let _recovering = false;  // recovery mutex
let _tripped = false;     // detection fired this turn
let _sessionGeneration = 0; // bump on session_start to cancel stale loops

let _lastPromptTokens = 0;     // estimated prompt size of the current LLM call
let _onsetText = "";           // accumulated onset chars for the current block
let _onsetChecked = false;      // past the onset window (decided) for this block

// ---- Helpers ---------------------------------------------------------------

export function isGuardedModel(model: { id?: string } | null | undefined): boolean {
  if (!model) return false;
  if (GUARDED_MODEL_IDS.has("*")) return true;
  return model.id != null && GUARDED_MODEL_IDS.has(model.id);
}

/** Whether the opt-in logprobs canary is enabled (env MAKORA_NAN_CANARY). */
export function canaryEnabled(): boolean {
  const v = (typeof process !== "undefined" && process.env?.MAKORA_NAN_CANARY) || "";
  return v === "1" || v.toLowerCase() === "true" || v.toLowerCase() === "on";
}

/** Rough token estimate from message content. Used only for gating/recovery,
 *  not billing — char/4 is close enough to the ~10k threshold decisions. */
export function estimateTokensMessages(messages: GuardedMessage[] | unknown): number {
  if (!Array.isArray(messages)) return 0;
  let chars = 0;
  for (const m of messages) {
    chars += extractText((m as GuardedMessage).content).length;
    const tc = (m as GuardedMessage).tool_calls;
    if (Array.isArray(tc)) for (const c of tc) chars += JSON.stringify(c ?? "").length;
  }
  return Math.ceil(chars / 4);
}

/** Extract text from a message content that may be a string or a list of
 *  parts ({type:"text"|"thinking", text|thinking}). */
export function extractText(content: unknown): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    let out = "";
    for (const p of content) {
      if (p && typeof p === "object") {
        const o = p as Record<string, unknown>;
        if (typeof o.text === "string") out += o.text;
        else if (typeof o.thinking === "string") out += o.thinking;
        else if (typeof o.content === "string") out += o.content;
      } else if (typeof p === "string") out += p;
    }
    return out;
  }
  return "";
}

/**
 * Is the first ONSET_WINDOW_CHARS chars of `text` a single short unit repeated
 * ONSET_MIN_REPS+ times? That is the NaN-argmax fixed point at reasoning onset
 * ("!", "{},", "();", ...). Whitespace-led runs are ignored (legitimate).
 * Onset-targeted: only the first ~64 chars, so normal repetitive output later
 * in a turn (tables, CSVs, JSON) is never inspected.
 */
export function isOnsetCollapse(text: string): boolean {
  const s = text.slice(0, ONSET_WINDOW_CHARS);
  if (s.length < ONSET_MIN_CHARS) return false;
  if (/\s/.test(s[0])) return false;
  for (let p = 1; p <= UNIT_MAX_LEN; p++) {
    if (s.length < p * ONSET_MIN_REPS) continue;
    let ok = true;
    for (let i = p; i < s.length; i++) {
      if (s[i] !== s[i % p]) { ok = false; break; }
    }
    if (ok) return true;
  }
  return false;
}

/** A finalized assistant message that was a collapse (onset fixed-point or our
 *  aborted stub). Used by the message_end scrubber + context handler + trim. */
export function isCollapseMessage(msg: GuardedMessage): boolean {
  if (msg.role !== "assistant") return false;
  const text = extractText(msg.content);
  if (text === DEGENERATE_STUB_TEXT) return true;
  return isOnsetCollapse(text);
}

/** Recovery trim: drop ONLY the trailing aborted/collapse assistant message so
 *  the resumed prompt([]) sends a clean continuation context ending at the
 *  prior user/toolResult. Matches the old guard — we do NOT trim valid context
 *  (removing the user's session would destroy it). The degenerate turn itself
 *  is the only thing removed. */
export function trimAborted(agent: GuardedAgent): void {
  const msgs = agent.state.messages;
  const last = msgs[msgs.length - 1];
  if (
    last &&
    last.role === "assistant" &&
    (last.stopReason === "aborted" || isCollapseMessage(last))
  ) {
    agent.state.messages = msgs.slice(0, -1);
  }
}

function calculateDelay(attempt: number): number {
  const d = BACKOFF_BASE_MS * Math.pow(BACKOFF_MULTIPLIER, attempt - 1);
  return Math.min(d, BACKOFF_MAX_MS);
}
function formatDuration(ms: number): string {
  return ms >= 1000 ? `${Math.round(ms / 1000)}s` : `${ms}ms`;
}
async function interruptibleSleep(ms: number, gen: number): Promise<boolean> {
  const step = 100;
  for (let elapsed = 0; elapsed < ms; elapsed += step) {
    if (_sessionGeneration !== gen || _userAborted) return true;
    await new Promise((r) => setTimeout(r, Math.min(step, ms - elapsed)));
  }
  return false;
}

// ---- Recovery loop ---------------------------------------------------------

/** Loops until a clean turn, user abort, or session change. Backoff sleep is
 *  interruptible (100ms poll) so Esc and /new take effect quickly. */
async function triggerRecovery(): Promise<void> {
  if (!_agent || _userAborted || _recovering) return;
  _recovering = true;
  const myGeneration = _sessionGeneration;
  try {
    await _agent.waitForIdle();
    if (_userAborted || _sessionGeneration !== myGeneration) return;

    trimAborted(_agent);

    let attempt = 0;
    while (true) {
      if (_userAborted || _sessionGeneration !== myGeneration) return;
      attempt++;
      const delay = calculateDelay(attempt);
      const overThreshold = _lastPromptTokens >= NAN_COLLAPSE_TOKEN_THRESHOLD;
      if (_notifyFn) {
        _notifyFn(
          `Makora NaN-collapse guard: aborted a degenerate NVFP4 long-context ` +
            `prefill${overThreshold ? ` (~${_lastPromptTokens} prompt tokens)` : ""}, ` +
            `resuming (retry ${attempt}, backoff ${formatDuration(delay)})...`,
          "warning",
        );
      }
      const interrupted = await interruptibleSleep(delay, myGeneration);
      if (interrupted || _userAborted || _sessionGeneration !== myGeneration) return;

      _weAborted = false;
      _tripped = false;
      try {
        await _agent.prompt([]);
      } catch {
        return; // "Agent is already processing" or transient — bail.
      }
      if (_userAborted || _sessionGeneration !== myGeneration) return;
      if (_weAborted) { // collapsed again — trim the degenerate turn and loop
        trimAborted(_agent);
        continue;
      }
      return; // clean turn
    }
  } finally {
    if (_sessionGeneration === myGeneration) _recovering = false;
  }
}

function trip(ctx?: { ui?: { notify?: (m: string, l: "info" | "warning" | "error") => void } }): void {
  if (_tripped) return;
  _tripped = true;
  _weAborted = true;
  const agent = _agent;
  if (!agent) {
    ctx?.ui?.notify?.(
      "Makora NaN-collapse guard: NVFP4 NaN collapse detected but the Agent " +
        "was not captured; cannot recover automatically.",
      "warning",
    );
    return;
  }
  agent.abort();
  if (!_recovering) void triggerRecovery();
}

// ---- Optional logprobs canary (provider-level preflight) ------------------

/** Probe the same payload with logprobs:true, max_tokens:1. If the engine
 *  returns 400 "nan", the prefill NaN'd — warn (and the in-stream onset guard
 *  will recover). Diagnostic/metric only; opt-in via MAKORA_NAN_CANARY. */
export async function nanCanary(
  params: Record<string, unknown>,
  baseUrl: string,
  apiKey: string,
  modelId: string,
): Promise<boolean> {
  // Skip below threshold — the NVFP4 prefill is numerically stable there, so
  // the probe can't detect anything and only adds latency.
  const estTokens = estimateTokensMessages(params.messages as GuardedMessage[]);
  if (estTokens < CANARY_TOKEN_THRESHOLD) return false;
  const body = { ...params, max_tokens: 1, logprobs: true, top_logprobs: 1, stream: false };
  delete (body as Record<string, unknown>).stream_options;
  try {
    const res = await fetch(`${baseUrl}/chat/completions`, {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.status === 400) {
      const t = await res.text().catch(() => "");
      if (t.toLowerCase().includes("nan")) {
        // eslint-disable-next-line no-console
        console.warn(
          `[makora-nan-canary] NaN prefill detected for ${modelId} ` +
            `(logprobs probe → HTTP 400 "nan"). The in-stream guard will recover.`,
        );
        return true;
      }
    }
  } catch {
    // Network/timeout — ignore; the in-stream guard handles recovery.
  }
  return false;
}

// ---- Registration ----------------------------------------------------------

export function registerNanCollapseGuard(pi: ExtensionAPI): void {
  // Capture the live Agent by chaining Agent.prototype.subscribe (fires on
  // every fresh session + resume). Chain any prior patch (pi-invisible-continue,
  // pi-retry) so all coexist.
  const proto = Agent.prototype as unknown as {
    subscribe: (this: GuardedAgent, ...args: unknown[]) => unknown;
  };
  const origSubscribe = proto.subscribe;
  proto.subscribe = function (this: GuardedAgent, ...args: unknown[]) {
    _agent = this;
    return origSubscribe.apply(this, args);
  };

  pi.on("session_start", () => {
    _sessionGeneration++;
    _recovering = false;
    _tripped = false;
    _weAborted = false;
    _userAborted = false;
    _lastPromptTokens = 0;
  });

  // before_agent_start fires only for user prompts (not recovery's prompt([])),
  // so resets bound detection to the current user turn.
  pi.on("before_agent_start", () => {
    _tripped = false;
    _weAborted = false;
    _userAborted = false;
  });

  pi.on("message_start", (event: any) => {
    if (event.message?.role === "assistant") {
      _tripped = false;
      _weAborted = false;
      _onsetText = "";
      _onsetChecked = false;
    }
  });

  // Onset fixed-point detection: the first ~64 chars of the reasoning trace
  // being one short unit repeated = NaN-argmax collapse ("!", "{},", "();").
  // Scoped to the REASONING trace only — the NVFP4 NaN is a prefill bug, so it
  // hits the first generated tokens = the thinking trace (all 6 observed hits
  // were thinking at offset 0). Watching the visible-text channel too would
  // false-positive on legitimate markdown (a "----" horizontal rule is 40
  // identical chars). Scoped to guarded models; NOT token-gated — the collapse
  // threshold is fuzzy (~8.9k+) and onset detection is near-zero false-positive,
  // so every guarded-model reasoning onset is checked. This clearer signal is
  // the only change vs the old blanket detector.
  pi.on("message_update", (event: any, ctx: any) => {
    _notifyFn = (message, level) => ctx.ui?.notify?.(message, level);
    if (_recovering || _tripped) return;
    const ame = event.assistantMessageEvent;
    if (!ame) return;

    if (ame.type === "thinking_start") {
      _onsetText = "";
      _onsetChecked = false;
      return;
    }
    if (ame.type !== "thinking_delta") return;
    if (!isGuardedModel(ctx.model)) return;

    if (!_onsetChecked) {
      _onsetText += ame.delta ?? "";
      if (_onsetText.length > ONSET_WINDOW_CHARS) _onsetText = _onsetText.slice(0, ONSET_WINDOW_CHARS);
      if (_onsetText.length >= ONSET_MIN_CHARS) {
        if (isOnsetCollapse(_onsetText)) {
          trip(ctx);
          return;
        }
        if (_onsetText.length >= ONSET_WINDOW_CHARS) _onsetChecked = true;
      }
    }
  });

  // User abort (Esc) — a non-we-aborted aborted turn. Stops recovery.
  pi.on("turn_end", (event: any) => {
    const msg = event.message;
    if (msg?.role === "assistant" && msg.stopReason === "aborted" && !_weAborted) {
      _userAborted = true;
    }
  });
  pi.on("turn_end", (_event: any, ctx: any) => {
    if (!_notifyFn) _notifyFn = (message, level) => ctx.ui?.notify?.(message, level);
  });

  // Scrub a collapse from the saved transcript (replace with a clean stub
  // before it is saved / shown). Recovery itself is the trim + retry loop
  // (triggerRecovery); this just keeps the toxic `!` out of the saved record.
  pi.on("message_end", (event: any) => {
    const msg = event.message as GuardedMessage | undefined;
    if (!msg || msg.role !== "assistant") return;
    if (!isGuardedModel({ id: msg.model })) return;
    if (!isCollapseMessage(msg) && !_weAborted) return;
    return {
      message: { ...event.message, content: [{ type: "text", text: DEGENERATE_STUB_TEXT }] },
    };
  });

  // Before each LLM call: strip any collapse/stub from the trailing context so
  // the model never re-sees toxic output, and refresh the prompt-token estimate
  // used to label the recovery notify. (The recovery's prompt([]) path is
  // covered by the state.messages trim; this covers user-prompt turns and
  // resumes.)
  pi.on("context", (event: any) => {
    const messages = event.messages;
    if (messages && messages.length > 0) {
      _lastPromptTokens = estimateTokensMessages(messages);
      const scanFrom = Math.max(0, messages.length - 32);
      let changed = false;
      const filtered = messages.filter((m, i) => {
        if (i < scanFrom || m.role !== "assistant") return true;
        const text = extractText(m.content);
        if (text === DEGENERATE_STUB_TEXT || (isCollapseMessage(m) && text.length >= ONSET_MIN_CHARS)) {
          changed = true;
          return false;
        }
        return true;
      });
      if (changed) return { messages: filtered };
    } else {
      _lastPromptTokens = 0;
    }
    return undefined;
  });
}
