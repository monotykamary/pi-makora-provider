/**
 * Death-loop guard for Makora reasoning models.
 *
 * Some Makora models (notably GLM 5.2 NVFP4 / FP8) occasionally fall into a
 * degenerate repetition loop, emitting an unbroken run of a single character
 * (observed: '!' -> "!!!!..." and '0' -> "0000...") or a repeated token
 * (observed: '0' -> "0 0 0 ...") that consumes the whole response. This guard
 * watches the
 * streamed assistant output (both the visible answer and the reasoning
 * trace); when that run is detected it aborts the runaway generation, drops
 * the partial (toxic) assistant message from the transcript, and resumes the
 * agentic loop invisibly via agent.prompt([]) — the same pattern
 * pi-invisible-continue / pi-retry uses, so no new user message pollutes the
 * context.
 *
 * Infinite retries with exponential backoff. Long-horizon agent work can trip
 * the loop many times in one session; capping retries would strand the agent
 * mid-task. Instead the recovery loops indefinitely (like pi-retry) until the
 * model produces a clean turn, the user aborts (Esc), or the session changes
 * (/new, /resume). Backoff (2s→60s, 2×) paces retries and gives the user a
 * window to intervene; interruptible sleep polls every 100ms so Esc and /new
 * take effect within 100ms instead of waiting out the full delay.
 *
 * Why trim the aborted message: on abort, pi finalizes the in-flight
 * assistant message WITH its accumulated '!!!' content and stopReason
 * "aborted" into the transcript. Resuming from that context would re-feed
 * the toxic text to the model and likely re-trigger the loop. Dropping the
 * last (aborted) assistant message leaves the context ending at the prior
 * user/toolResult message — a clean continuation point.
 *
 * Distinguishing our abort from a user Esc: both call agent.abort(), so both
 * surface as stopReason "aborted" at turn_end. The handler sets _weAborted
 * before aborting; turn_end only treats an abort as user-initiated (setting
 * _userAborted to exit the loop) when _weAborted is false. Caveat: if the
 * user hits Esc in the ~ms between our trip and the abort completing, the
 * abort is mis-attributed to us and one unwanted retry fires. The backoff
 * window lets the user Esc again, which then works normally.
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

/** Trip after this many consecutive identical characters in the streamed
 *  answer. 40 is far above anything normal prose or code produces for any
 *  single non-whitespace character. */
export const REPEAT_THRESHOLD = 40;

/** @deprecated alias kept for downstream forks; prefer REPEAT_THRESHOLD.
 *  The guard now catches runs of any non-whitespace character, not just '!'. */
export const BANG_THRESHOLD = REPEAT_THRESHOLD;

/** Characters whose repetition we never trip on. Whitespace can legitimately
 *  repeat (code indentation, blank lines, markdown padding), so a long
 *  whitespace run is not treated as a degenerate loop. Every other character
 *  is a candidate. Exported for tests/introspection. */
export const IGNORED_REPEAT_CHARS = new Set<string>([
  " ",
  "\t",
  "\n",
  "\r",
  "\f",
  "\v",
]);

/** A trailing run of one repeated character in streamed text. `char` is ""
 *  when `len` === 0. */
export interface TrailingRun {
  char: string;
  len: number;
}

/** Trip after this many consecutive identical whitespace-delimited tokens.
 *  Catches spaced repetition ("0 0 0 ...", "!!!! !!!! ...") that the
 *  single-character run can't see (the separator resets it). 40 mirrors
 *  REPEAT_THRESHOLD. */
export const TOKEN_REPEAT_THRESHOLD = 40;

/** Bounded tail kept for token-run detection. Must hold >=
 *  TOKEN_REPEAT_THRESHOLD copies of the repeated unit; 1024 comfortably fits 40
 *  copies of tokens up to ~25 chars. */
export const TOKEN_REPEAT_BUFFER_CHARS = 1024;

/** Exponential backoff for recovery retries. Mirrors pi-retry's defaults:
 *  2s base, 60s cap, 2× multiplier. Tunable via these constants. */
export const BACKOFF_BASE_MS = 2000;
export const BACKOFF_MAX_MS = 60_000;
export const BACKOFF_MULTIPLIER = 2;

export interface BackoffConfig {
  baseDelayMs: number;
  maxDelayMs: number;
  multiplier: number;
}

export const DEFAULT_BACKOFF_CONFIG: BackoffConfig = {
  baseDelayMs: BACKOFF_BASE_MS,
  maxDelayMs: BACKOFF_MAX_MS,
  multiplier: BACKOFF_MULTIPLIER,
};

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

/** Mutex: only one recovery loop may be in-flight at a time. The
 *  message_update handler still aborts mid-stream while this is held, but
 *  only the loop driver re-issues prompt([]). */
let _recovering = false;

/** Trailing run of one repeated character in the current text block. */
let _trailingRunChar = "";
let _trailingRunLen = 0;

/** Bounded trailing buffer for token-level repetition detection (e.g.
 *  "0 0 0 ...", "!!!! !!!! ..."). Single-char runs don't see spaced
 *  repetition, so we also tokenize the recent tail and count consecutive
 *  identical tokens. Capped to keep memory/cost bounded on long streams. */
let _tokenRepeatBuffer = "";

/** Reset all per-turn/per-block detection state. */
function resetDetection(): void {
  _trailingRunChar = "";
  _trailingRunLen = 0;
  _tokenRepeatBuffer = "";
}

/** Latch: already tripped for the current assistant message. */
let _tripped = false;

/** True when WE aborted the current turn (death-loop), false on user Esc.
 *  Reset on message_start of each new assistant turn. */
let _weAborted = false;

/** True when the user cancelled (Esc). Set in turn_end, cleared on
 *  session_start and fresh successful turns. Stops the recovery loop. */
let _userAborted = false;

/** Session generation counter: incremented on every session_start. The
 *  recovery loop captures it on entry and exits when it changes (/new,
 *  /resume), so a stale loop never drives a new session. */
let _sessionGeneration = 0;

/** notify() captured fresh from the most recent event ctx, since
 *  ctx.ui.notify isn't available inside the loop driver. Mirrors pi-retry. */
let _notifyFn:
  | ((message: string, level: "info" | "warning" | "error") => void)
  | null = null;

export function isGuardedModel(
  model: { provider?: string; id?: string } | undefined | null,
): boolean {
  if (!model) return false;
  if (model.provider !== PROVIDER_ID) return false;
  if (GUARDED_MODEL_IDS.has("*")) return true;
  return model.id != null && GUARDED_MODEL_IDS.has(model.id);
}

/** Update the trailing single-character run given a new text delta.
 *  O(len(delta)). Tracks runs of any one character (so the guard catches a
 *  degenerate loop regardless of which character the model fixates on),
 *  including whitespace -- the trip decision in isDegenerateRun filters those.
 *
 *  Trailing run depends only on the delta's suffix: if the delta ends with a
 *  different character than the prior run, the prior run is cut off and the
 *  new run is the delta's trailing run of its last character; if the delta is
 *  entirely the prior run's character, it extends the prior run. */
export function nextTrailingRun(prev: TrailingRun, delta: string): TrailingRun {
  const len = delta.length;
  if (len === 0) return prev;
  const lastCode = delta.charCodeAt(len - 1);
  let i = len - 1;
  while (i >= 0 && delta.charCodeAt(i) === lastCode) i--;
  const lastChar = delta[len - 1];
  const trailingInDelta = len - 1 - i;
  // Delta is entirely one char and it matches the prior run: extend it.
  if (i < 0 && prev.len > 0 && prev.char === lastChar) {
    return { char: prev.char, len: prev.len + len };
  }
  return { char: lastChar, len: trailingInDelta };
}

/** @deprecated '!'-only alias kept for downstream forks. Returns the trailing
 *  '!' run length (0 once the run switches to any other character). */
export function nextTrailingBangs(prev: number, delta: string): number {
  const run = nextTrailingRun({ char: "!", len: prev }, delta);
  return run.char === "!" ? run.len : 0;
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

/** Trailing single-character run of a finalized message's text content. */
export function messageTrailingRun(msg: GuardedMessage): TrailingRun {
  const text = extractText(msg.content);
  if (text.length === 0) return { char: "", len: 0 };
  const lastCode = text.charCodeAt(text.length - 1);
  let i = text.length - 1;
  while (i >= 0 && text.charCodeAt(i) === lastCode) i--;
  return { char: text[text.length - 1], len: text.length - 1 - i };
}

/** @deprecated '!'-only alias kept for downstream forks. */
export function messageTrailingBangs(msg: GuardedMessage): number {
  const run = messageTrailingRun(msg);
  return run.char === "!" ? run.len : 0;
}

/** True when a trailing run is long enough to be a degenerate loop. Whitespace
 *  runs never count (see IGNORED_REPEAT_CHARS). Exported so the trip decision
 *  is unit-testable independently of the streaming handler. */
export function isDegenerateRun(
  run: TrailingRun,
  threshold: number = REPEAT_THRESHOLD,
): boolean {
  return (
    run.len >= threshold &&
    run.char !== "" &&
    !IGNORED_REPEAT_CHARS.has(run.char)
  );
}

/** A trailing run of one repeated whitespace-delimited token. `token` is ""
 *  when `count` === 0. */
export interface TokenRun {
  token: string;
  count: number;
}

/** Is `code` a whitespace code unit we split tokens on? Mirrors
 *  IGNORED_REPEAT_CHARS (space, tab, LF, CR, FF, VT). */
function isWhitespaceChar(code: number): boolean {
  switch (code) {
    case 0x20: case 0x09: case 0x0a: case 0x0d: case 0x0c: case 0x0b:
      return true;
    default:
      return false;
  }
}

/** Trailing run of one repeated whitespace-delimited token in `text`. Walks
 *  backward from the end: skips trailing whitespace, captures the last token,
 *  then counts consecutive preceding copies separated only by whitespace.
 *  O(run length * token length). Used on a bounded tail buffer, so cost stays
 *  bounded regardless of total stream length. */
export function trailingTokenRun(text: string): TokenRun {
  const len = text.length;
  let end = len;
  while (end > 0 && isWhitespaceChar(text.charCodeAt(end - 1))) end--;
  if (end === 0) return { token: "", count: 0 };
  let start = end;
  while (start > 0 && !isWhitespaceChar(text.charCodeAt(start - 1))) start--;
  const token = text.slice(start, end);
  let count = 1;
  let i = start;
  while (i > 0) {
    let j = i;
    while (j > 0 && isWhitespaceChar(text.charCodeAt(j - 1))) j--;
    if (j === 0) break;
    let prevEnd = j;
    let prevStart = prevEnd;
    while (prevStart > 0 && !isWhitespaceChar(text.charCodeAt(prevStart - 1))) {
      prevStart--;
    }
    if (text.slice(prevStart, prevEnd) !== token) break;
    count++;
    i = prevStart;
  }
  return { token, count };
}

/** True when a trailing token run is long enough to be a degenerate loop.
 *  Tokens are non-whitespace by construction, so no whitespace exclusion is
 *  needed (unlike isDegenerateRun). Exported for unit testing. */
export function isDegenerateTokenRun(
  run: TokenRun,
  threshold: number = TOKEN_REPEAT_THRESHOLD,
): boolean {
  return run.count >= threshold && run.token !== "";
}

/** Exponential backoff delay, capped at maxDelayMs. Mirrors pi-retry. */
export function calculateDelay(
  attempt: number,
  config: BackoffConfig = DEFAULT_BACKOFF_CONFIG,
): number {
  const delay = config.baseDelayMs * Math.pow(config.multiplier, attempt - 1);
  return Math.min(delay, config.maxDelayMs);
}

/** Format a duration for user-facing messages. Mirrors pi-retry. */
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60000);
  const seconds = ((ms % 60000) / 1000).toFixed(0);
  return `${minutes}m ${seconds}s`;
}

/** Interruptible sleep: polls _userAborted and _sessionGeneration every
 *  100ms. Returns true if interrupted (Esc or /new), false if the full delay
 *  elapsed. Mirrors pi-retry. */
function interruptibleSleep(ms: number, generation: number): Promise<boolean> {
  if (ms <= 0) return Promise.resolve(false);
  return new Promise((resolve) => {
    const checkInterval = 100;
    let elapsed = 0;
    const timer = setInterval(() => {
      elapsed += checkInterval;
      if (_userAborted || _sessionGeneration !== generation) {
        clearInterval(timer);
        resolve(true);
      } else if (elapsed >= ms) {
        clearInterval(timer);
        resolve(false);
      }
    }, checkInterval);
  });
}

/** Remove a trailing aborted/death-loop assistant message so the next
 *  prompt([]) sends a clean context ending at the prior user/toolResult. */
function trimAbortedDeathLoop(agent: GuardedAgent): void {
  const msgs = agent.state.messages;
  const last = msgs[msgs.length - 1];
  if (
    last &&
    last.role === "assistant" &&
    (last.stopReason === "aborted" ||
      isDegenerateRun(messageTrailingRun(last)) ||
      isDegenerateTokenRun(trailingTokenRun(extractText(last.content))))
  ) {
    agent.state.messages = msgs.slice(0, -1);
  }
}

/** Recovery loop driver — the core. Loops until a clean turn, user abort,
 *  or session change. Backoff sleep happens AFTER each prompt([]) settles,
 *  so it does not block the agent; Esc and /new take effect within 100ms. */
async function triggerRecovery(): Promise<void> {
  if (!_agent) return;
  if (_userAborted) return;
  if (_recovering) return;
  _recovering = true;

  const myGeneration = _sessionGeneration;

  try {
    await _agent.waitForIdle();
    if (_userAborted || _sessionGeneration !== myGeneration) return;

    // Trim the aborted !!! message from the initial trip.
    trimAbortedDeathLoop(_agent);

    let attempt = 0;
    // Loop until success, user abort, or session change.
    while (true) {
      if (_userAborted || _sessionGeneration !== myGeneration) return;

      attempt++;
      const delay = calculateDelay(attempt);
      if (_notifyFn) {
        _notifyFn(
          `Makora death-loop guard: resuming after runaway repeated-character ` +
            `output (retry ${attempt}, backoff ${formatDuration(delay)})...`,
          "warning",
        );
      }

      // Interruptible sleep with backoff BEFORE the retry. Lets Esc and
      // /new take effect within 100ms instead of waiting the full delay.
      const interrupted = await interruptibleSleep(delay, myGeneration);
      if (interrupted) return;
      if (_userAborted || _sessionGeneration !== myGeneration) return;

      // _weAborted is also reset on message_start, but reset here too in
      // case message_start already fired before we reached this point.
      _weAborted = false;
      try {
        await _agent.prompt([]);
      } catch {
        // "Agent is already processing" or other transient error — bail.
        return;
      }
      if (_userAborted || _sessionGeneration !== myGeneration) return;

      // Did the resumed turn death-loop again? If we aborted it, trim and
      // loop; otherwise it completed cleanly (or the user aborted) — exit.
      if (_weAborted) {
        trimAbortedDeathLoop(_agent);
        continue;
      }
      return;
    }
  } finally {
    // Only release the mutex if the session hasn't changed. If /new fired,
    // a new recovery loop may already own it — resetting here would clobber.
    if (_sessionGeneration === myGeneration) {
      _recovering = false;
    }
  }
}

export function registerDeathLoopGuard(pi: ExtensionAPI): void {
  // Capture the live Agent instance by chaining Agent.prototype.subscribe.
  // subscribe() fires when AgentSession attaches — on every fresh session
  // and every resume — so _agent always points at the active Agent. Chain
  // any prior patch (e.g. pi-invisible-continue, pi-retry) so all coexist.
  const proto = Agent.prototype as unknown as {
    subscribe: (this: GuardedAgent, ...args: unknown[]) => unknown;
  };
  const origSubscribe = proto.subscribe;
  proto.subscribe = function (this: GuardedAgent, ...args: unknown[]) {
    _agent = this;
    return origSubscribe.apply(this, args);
  };

  pi.on("session_start", () => {
    // Bump generation so any in-flight recovery loop from a previous session
    // exits within 100ms (during backoff) or right after prompt([]) returns.
    _sessionGeneration++;
    _recovering = false;
    resetDetection();
    _tripped = false;
    _weAborted = false;
    _userAborted = false;
  });

  // before_agent_start fires only for user prompts (the AgentSession path),
  // not for the recovery's direct agent.prompt([]), so these resets bound
  // detection to the current user prompt without clearing mid-recovery.
  pi.on("before_agent_start", () => {
    resetDetection();
    _tripped = false;
    _weAborted = false;
    // A new user prompt is fresh activity — clear a stale user-abort flag.
    _userAborted = false;
  });

  pi.on("message_start", (event) => {
    if (event.message?.role === "assistant") {
      resetDetection();
      _tripped = false;
      _weAborted = false;
    }
  });

  pi.on("message_update", (event, ctx) => {
    // Refresh notify on every handler so it stays current after session
    // switches (a stale ctx goes invalid). Same approach as pi-retry.
    _notifyFn = (message, level) => ctx.ui.notify(message, level);

    if (_recovering || _tripped) return;
    const ame = event.assistantMessageEvent;
    if (ame.type === "text_start" || ame.type === "thinking_start") {
      // New content block (answer or reasoning) — detection starts fresh.
      resetDetection();
      return;
    }
    // Watch both the visible answer and the reasoning trace: GLM 5.2 is a
    // reasoning model, and the loop can surface in either.
    if (ame.type !== "text_delta" && ame.type !== "thinking_delta") return;
    if (!isGuardedModel(ctx.model)) return;

    const run = nextTrailingRun(
      { char: _trailingRunChar, len: _trailingRunLen },
      ame.delta,
    );
    _trailingRunChar = run.char;
    _trailingRunLen = run.len;

    // Token-level repetition: a spaced loop like "0 0 0 ..." or "!!!! !!!!"
    // never builds a single-char run (the separator resets it), so track the
    // recent tail as whitespace-delimited tokens and count consecutive copies.
    _tokenRepeatBuffer = (_tokenRepeatBuffer + ame.delta).slice(
      -TOKEN_REPEAT_BUFFER_CHARS,
    );
    const tokenRun = trailingTokenRun(_tokenRepeatBuffer);

    if (!isDegenerateRun(run) && !isDegenerateTokenRun(tokenRun)) return;

    // Trip: abort the runaway stream and kick off the recovery loop (once,
    // mutex-gated). The loop driver handles re-detection on resumed turns.
    _tripped = true;
    _weAborted = true;
    const agent = _agent;
    if (!agent) {
      ctx.ui.notify(
        "Makora death-loop guard: runaway repeated-character output detected " +
          "but the Agent instance was not captured; cannot recover automatically.",
        "warning",
      );
      return;
    }
    agent.abort();
    // Detach: the handler must return so the run unwinds to idle before the
    // loop driver awaits waitForIdle(). Only kick off the loop if it isn't
    // already running; otherwise the running loop catches this abort.
    if (!_recovering) {
      void triggerRecovery();
    }
  });

  // Detect user aborts via turn_end. Our own death-loop abort also surfaces
  // as stopReason "aborted", so _weAborted gates this: only a non-we-aborted
  // aborted turn is treated as a user Esc and stops the recovery loop.
  pi.on("turn_end", (event) => {
    const msg = event.message as GuardedMessage | undefined;
    if (msg?.role === "assistant" && msg.stopReason === "aborted" && !_weAborted) {
      _userAborted = true;
    }
  });

  // Also refresh notify on turn_end so the loop driver has a fresh fn even
  // if message_update hasn't fired yet this session.
  pi.on("turn_end", (_event, ctx) => {
    if (!_notifyFn) {
      _notifyFn = (message, level) => ctx.ui.notify(message, level);
    }
  });
}
