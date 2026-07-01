/**
 * Test: streamMakora combined payload E2E (GLM / Qwen / Kimi)
 *
 * The unit tests (stream-makora.test.ts) verify streamMakora wires onPayload
 * correctly with a mocked streamer. The trigger probe (test-thinking-triggers.ts)
 * proved `reasoning_effort: "none"` turns GLM/Qwen off and that namespaced
 * `chat_template_kwargs.thinking:false` turns Kimi off — but in isolation.
 *
 * This test sends the EXACT combined payload streamMakora produces — a top-level
 * `reasoning_effort` (from buildParams' OpenAI fallback) ALONGSIDE a namespaced
 * `chat_template_kwargs.preserve_thinking` (from the onPayload hook), which no
 * built-in thinkingFormat emits — and confirms the combination is accepted and
 * behaves correctly:
 *
 *   GLM/Qwen ON : reasoning_effort: <effort> + chat_template_kwargs.preserve_thinking: true  → reasons
 *   GLM/Qwen OFF: reasoning_effort: "none"     + chat_template_kwargs.preserve_thinking: false → no reasoning
 *   Kimi ON    : chat_template_kwargs.thinking: true,  preserve_thinking: true  → reasons
 *   Kimi OFF   : chat_template_kwargs.thinking: false, preserve_thinking: false → no reasoning
 *
 * A neutral prompt is used (the "think step by step" prompt can force reasoning
 * past an off-switch and confound the result).
 *
 * Usage:
 *   MAKORA_OPTIMIZE_TOKEN=your-key npx tsx test-stream-makora-e2e.ts
 */

const BASE_URL = "https://inference.makora.com/v1";
const API_KEY = process.env.MAKORA_OPTIMIZE_TOKEN!;
const TIMEOUT_MS = 120_000;

if (!API_KEY) {
  console.error("Set MAKORA_OPTIMIZE_TOKEN env var");
  process.exit(1);
}

// Effective configs after patch.json (verified via scripts). Mirrors what
// streamMakora receives from the model registry at runtime.
interface ModelSpec {
  id: string;
  name: string;
  family: "glm" | "qwen" | "kimi";
  /** ON effort sent as top-level reasoning_effort (GLM/Qwen). Kimi sends none. */
  onEffort?: string;
}

const MODELS: ModelSpec[] = [
  { id: "zai-org/GLM-5.2-FP8", name: "GLM 5.2 FP8", family: "glm", onEffort: "high" },
  { id: "zai-org/GLM-5.2-NVFP4", name: "GLM 5.2 NVFP4", family: "glm", onEffort: "high" },
  { id: "unsloth/Qwen3.6-27B-NVFP4", name: "Qwen 3.6 27B", family: "qwen", onEffort: "high" },
  { id: "unsloth/Qwen3.6-35B-A3B-NVFP4", name: "Qwen 3.6 35B", family: "qwen", onEffort: "high" },
  { id: "moonshotai/Kimi-K2.7-Code", name: "Kimi K2.7 Code", family: "kimi" },
];

const PROMPT =
  "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. " +
  "How much does the ball cost? Give the answer.";

interface Variant {
  label: "on" | "off";
  /** Build the exact payload streamMakora would send for this (model, thinking state). */
  build: (m: ModelSpec) => Record<string, unknown>;
}

const VARIANTS: Variant[] = [
  {
    label: "on",
    build: (m) => {
      // buildParams (OpenAI fallback, no thinkingFormat) emits reasoning_effort
      // for GLM/Qwen; for Kimi (supportsReasoningEffort:false) it emits nothing.
      const payload: Record<string, unknown> = {
        model: m.id,
        messages: [{ role: "user", content: PROMPT }],
        max_tokens: 1024,
        stream: false,
      };
      if (m.family !== "kimi") payload.reasoning_effort = m.onEffort;
      // onPayload injects chat_template_kwargs:
      //   GLM/Qwen: { preserve_thinking: true }
      //   Kimi:     { thinking: true, preserve_thinking: true }
      const kwargs: Record<string, unknown> = { preserve_thinking: true };
      if (m.family === "kimi") kwargs.thinking = true;
      payload.chat_template_kwargs = kwargs;
      return payload;
    },
  },
  {
    label: "off",
    build: (m) => {
      const payload: Record<string, unknown> = {
        model: m.id,
        messages: [{ role: "user", content: PROMPT }],
        max_tokens: 1024,
        stream: false,
      };
      if (m.family !== "kimi") payload.reasoning_effort = "none";
      const kwargs: Record<string, unknown> = { preserve_thinking: false };
      if (m.family === "kimi") kwargs.thinking = false;
      payload.chat_template_kwargs = kwargs;
      return payload;
    },
  },
];

async function chatCompletion(
  payload: Record<string, unknown>
): Promise<{ reasoningLen: number; contentLen: number; error?: string }> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!res.ok) {
      const body = await res.text();
      return { reasoningLen: 0, contentLen: 0, error: `HTTP ${res.status}: ${body.slice(0, 300)}` };
    }
    const data = (await res.json()) as {
      choices?: { message?: { content?: string; reasoning?: string; reasoning_content?: string } }[];
    };
    const msg = data.choices?.[0]?.message;
    return {
      reasoningLen: (msg?.reasoning ?? msg?.reasoning_content ?? "").length,
      contentLen: (msg?.content ?? "").length,
    };
  } catch (err: unknown) {
    const m = err instanceof Error ? err.message : String(err);
    return { reasoningLen: 0, contentLen: 0, error: m.includes("aborted") ? `Timeout` : m };
  } finally {
    clearTimeout(timer);
  }
}

const OFF_THRESHOLD = 50;

async function main() {
  console.log("=== streamMakora Combined Payload E2E (GLM / Qwen / Kimi) ===\n");
  let allPass = true;

  for (const m of MODELS) {
    console.log(`► ${m.name} (${m.id})`);
    const results: Record<string, { reasoningLen: number; error?: string }> = {};
    for (const v of VARIANTS) {
      const payload = v.build(m);
      process.stdout.write(`  [${v.label}] payload=${JSON.stringify({ reasoning_effort: payload.reasoning_effort ?? "-", chat_template_kwargs: payload.chat_template_kwargs })} ... `);
      const r = await chatCompletion(payload);
      results[v.label] = r;
      console.log(r.error ? `ERROR ${r.error.slice(0, 80)}` : `reasoning=${r.reasoningLen}B content=${r.contentLen}B`);
    }
    const on = results.on;
    const off = results.off;
    const onReasons = !on.error && on.reasoningLen >= OFF_THRESHOLD;
    const offSuppressed = !off.error && off.reasoningLen < OFF_THRESHOLD;
    const ok = onReasons && offSuppressed;
    if (!ok) allPass = false;
    console.log(`  → ON reasons: ${onReasons ? "YES" : "no"}  | OFF suppressed: ${offSuppressed ? "YES" : "no"}  | ${ok ? "✅ PASS" : "❌ FAIL"}\n`);
  }

  console.log(allPass ? "✅ All models: combined payload (reasoning_effort + chat_template_kwargs) turns thinking on/off correctly." : "❌ Some models failed — see above.");
  process.exit(allPass ? 0 : 1);
}

main();
