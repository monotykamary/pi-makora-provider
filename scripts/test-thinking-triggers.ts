/**
 * Test: Thinking-Off Trigger Probe for GLM 5.2 + Qwen 3.6
 *
 * The companion test (test-thinking-off.ts) showed these four models ignore
 * `chat_template_kwargs.thinking` / `chat_template_kwargs.enable_thinking` —
 * reasoning persists regardless. This probe tries every *other* thinking
 * lever pi-ai knows about, to find what (if anything) actually turns
 * thinking off for these models.
 *
 * Candidates (top-level unless noted), each mirroring a pi `thinkingFormat`:
 *
 *   OFF candidates:
 *     1. toplevel_enable_thinking_false   enable_thinking: false                 (qwen)
 *     2. toplevel_thinking_false          thinking: false                        (bool)
 *     3. toplevel_thinking_disabled       thinking: { type: "disabled" }         (zai)
 *     4. toplevel_thinking_disabled_clear  thinking: { type:"disabled", clear_thinking:true } (zai)
 *     5. reasoning_effort_none             reasoning_effort: "none"               (openai off)
 *     6. reasoning_effort_low              reasoning_effort: "low"                (openai floor)
 *     7. string_thinking_none              thinking: "none"                       (string-thinking)
 *     8. string_thinking_off              thinking: "off"                        (string)
 *     9. kitchen_sink                     all of the above off + chat_template_kwargs off
 *
 *   ON control:
 *    10. on_control                       chat_template_kwargs: { enable_thinking: true }
 *        (confirms the model CAN reason, so an empty OFF result means the
 *        trigger worked — not that the model simply never reasons)
 *
 * A trigger "works" ⇔ ON control yields reasoning AND that OFF candidate
 * suppresses it (reasoning length ~0). If every OFF candidate still reasons,
 * the model is effectively always-thinking on this deployment and pi should
 * map "thinking off" to null / lowest effort rather than a real off-switch.
 *
 * Usage:
 *   MAKORA_OPTIMIZE_TOKEN=your-key npx tsx test-thinking-triggers.ts
 */

const BASE_URL = "https://inference.makora.com/v1";
const API_KEY = process.env.MAKORA_OPTIMIZE_TOKEN!;
const TIMEOUT_MS = 120_000;

if (!API_KEY) {
  console.error("Set MAKORA_OPTIMIZE_TOKEN env var");
  process.exit(1);
}

interface ModelSpec {
  id: string;
  name: string;
}

const MODELS: ModelSpec[] = [
  { id: "zai-org/GLM-5.2-FP8", name: "GLM 5.2 FP8" },
  { id: "zai-org/GLM-5.2-NVFP4", name: "GLM 5.2 NVFP4" },
  { id: "unsloth/Qwen3.6-27B-NVFP4", name: "Qwen 3.6 27B" },
  { id: "unsloth/Qwen3.6-35B-A3B-NVFP4", name: "Qwen 3.6 35B" },
];

// Neutral reasoning-inducing prompt — invites reasoning without literally
// asking to "think step by step", so a suppressed trace is meaningful.
const PROMPT =
  "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. " +
  "How much does the ball cost? Give the answer.";

interface Variant {
  label: string;
  source: string;
  apply: (payload: Record<string, unknown>) => void;
}

const VARIANTS: Variant[] = [
  {
    label: "toplevel_enable_thinking_false",
    source: "qwen format",
    apply: (p) => { p.enable_thinking = false; },
  },
  {
    label: "toplevel_thinking_false",
    source: "bool",
    apply: (p) => { p.thinking = false; },
  },
  {
    label: "toplevel_thinking_disabled",
    source: "zai format",
    apply: (p) => { p.thinking = { type: "disabled" }; },
  },
  {
    label: "toplevel_thinking_disabled_clear",
    source: "zai format (clear)",
    apply: (p) => { p.thinking = { type: "disabled", clear_thinking: true }; },
  },
  {
    label: "reasoning_effort_none",
    source: "openai off",
    apply: (p) => { p.reasoning_effort = "none"; },
  },
  {
    label: "reasoning_effort_low",
    source: "openai floor",
    apply: (p) => { p.reasoning_effort = "low"; },
  },
  {
    label: "string_thinking_none",
    source: "string-thinking off",
    apply: (p) => { p.thinking = "none"; },
  },
  {
    label: "string_thinking_off",
    source: "string off",
    apply: (p) => { p.thinking = "off"; },
  },
  {
    label: "kitchen_sink",
    source: "all off combined",
    apply: (p) => {
      p.enable_thinking = false;
      p.thinking = { type: "disabled", clear_thinking: true };
      p.reasoning_effort = "none";
      p.chat_template_kwargs = { enable_thinking: false, thinking: false };
    },
  },
  {
    label: "on_control",
    source: "known on (chat_template_kwargs.enable_thinking:true)",
    apply: (p) => { p.chat_template_kwargs = { enable_thinking: true }; },
  },
];

// API helper

async function chatCompletion(
  model: ModelSpec,
  variant: Variant
): Promise<{ contentLen: number; reasoningLen: number; error?: string }> {
  const payload: Record<string, unknown> = {
    model: model.id,
    messages: [{ role: "user", content: PROMPT }],
    max_tokens: 1024,
    stream: false,
  };
  variant.apply(payload);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(`${BASE_URL}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.text();
      return { contentLen: 0, reasoningLen: 0, error: `HTTP ${res.status}: ${body.slice(0, 300)}` };
    }

    const data = (await res.json()) as {
      choices?: {
        message?: {
          content?: string;
          reasoning?: string;
          reasoning_content?: string;
        };
      }[];
    };

    const msg = data.choices?.[0]?.message;
    const reasoning = msg?.reasoning ?? msg?.reasoning_content ?? "";
    const content = msg?.content ?? "";
    return { contentLen: content.length, reasoningLen: reasoning.length };
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    return { contentLen: 0, reasoningLen: 0, error: msg.includes("aborted") ? `Timeout after ${TIMEOUT_MS / 1000}s` : msg };
  } finally {
    clearTimeout(timer);
  }
}

// Test runner

interface VariantResult {
  variant: string;
  source: string;
  reasoningLen: number;
  contentLen: number;
  error?: string;
}

interface ModelResult {
  model: string;
  id: string;
  variants: VariantResult[];
  /** Triggers that cleanly suppressed reasoning (reasoning ~0 while on_control has reasoning). */
  workingTriggers: string[];
  onControlReasoningLen: number;
  alwaysThinking: boolean;
}

async function testModel(model: ModelSpec): Promise<ModelResult> {
  const result: ModelResult = { model: model.name, id: model.id, variants: [], workingTriggers: [], onControlReasoningLen: 0, alwaysThinking: true };
  console.log(`\n► ${model.name} (${model.id})`);

  const onVariant = VARIANTS.find((v) => v.label === "on_control")!;
  for (const variant of VARIANTS) {
    process.stdout.write(`  [${variant.label}] ... `);
    const r = await chatCompletion(model, variant);
    result.variants.push({ variant: variant.label, source: variant.source, ...r });
    const reason = r.error
      ? `ERROR ${r.error.slice(0, 80)}`
      : `reasoning=${r.reasoningLen}B content=${r.contentLen}B`;
    console.log(reason);
  }

  const byLabel = (label: string) => result.variants.find((v) => v.variant === label)!;
  const on = byLabel("on_control");
  result.onControlReasoningLen = on.reasoningLen;

  const onHasReasoning = !on.error && on.reasoningLen > 0;
  if (!onHasReasoning) {
    console.log(`  ⚠️  on_control produced no reasoning (${on.reasoningLen}B) — cannot validate off suppression; treating as always-thinking inconclusive`);
  }

  for (const v of result.variants) {
    if (v.variant === "on_control") continue;
    if (v.error) continue;
    // Suppressed: reasoning essentially absent. Use a small threshold (e.g. <50B)
    // to tolerate stray whitespace/preamble while still counting as "off".
    if (v.reasoningLen < 50 && onHasReasoning) {
      result.workingTriggers.push(v.variant);
    }
  }

  // Always-thinking if every non-error OFF candidate still produced substantial reasoning.
  const offCandidates = result.variants.filter((v) => v.variant !== "on_control" && !v.error);
  result.alwaysThinking = onHasReasoning && offCandidates.length > 0 && offCandidates.every((v) => v.reasoningLen >= 50);

  if (result.workingTriggers.length > 0) {
    console.log(`  ✅ working off-trigger(s): ${result.workingTriggers.join(", ")}`);
  } else {
    console.log(`  → no single off-trigger suppressed reasoning (on_control=${on.reasoningLen}B)`);
  }
  if (result.alwaysThinking) console.log(`  ⊘ appears always-thinking on this deployment`);

  return result;
}

// Main

async function main() {
  console.log("=== Thinking-Off Trigger Probe (GLM 5.2 + Qwen 3.6) ===");
  console.log(`Models: ${MODELS.length}  | Variants per model: ${VARIANTS.length}\n`);

  const results: ModelResult[] = [];
  for (const model of MODELS) {
    results.push(await testModel(model));
  }

  // Summary table
  console.log("\n\n=== Summary ===\n");
  console.log("| Model | on_control | working off-triggers | always-thinking? |");
  console.log("|-------|------------|-----------------------|------------------|");
  for (const r of results) {
    const wt = r.workingTriggers.length > 0 ? r.workingTriggers.join("; ") : "—";
    console.log(`| ${r.model} | ${r.onControlReasoningLen}B | ${wt} | ${r.alwaysThinking ? "YES" : "no"} |`);
  }

  // Per-variant detail
  console.log("\n=== Per-variant reasoning length (bytes) ===\n");
  const labels = VARIANTS.map((v) => v.label);
  console.log(`| Model | ${labels.join(" | ")} |`);
  console.log(`|-------|${labels.map(() => "-----").join("|")}|`);
  for (const r of results) {
    const cells = labels.map((l) => {
      const v = r.variants.find((x) => x.variant === l)!;
      return v.error ? "ERR" : String(v.reasoningLen);
    });
    console.log(`| ${r.model} | ${cells.join(" | ")} |`);
  }

  // Verdict
  const anyWorking = results.some((r) => r.workingTriggers.length > 0);
  const allAlwaysThinking = results.every((r) => r.alwaysThinking);

  console.log("\n=== Verdict ===\n");
  if (anyWorking) {
    console.log("✅ Found working off-triggers:");
    for (const r of results) {
      if (r.workingTriggers.length > 0) {
        const sources = r.workingTriggers.map((t) => {
          const v = VARIANTS.find((x) => x.label === t)!;
          return `${t} [${v.source}]`;
        });
        console.log(`  - ${r.model}: ${sources.join(", ")}`);
      } else {
        console.log(`  - ${r.model}: no off-trigger found${r.alwaysThinking ? " (always-thinking)" : ""}`);
      }
    }
  } else if (allAlwaysThinking) {
    console.log("⊘ All four models are always-thinking on this Makora deployment — no off-trigger suppresses reasoning.");
    console.log("  pi should map 'thinking off' to null / lowest effort (not a real off-switch) for these families.");
  } else {
    console.log("⚠️  No clean off-trigger found, and not all models confirmed always-thinking — see per-variant table.");
  }

  process.exit(anyWorking ? 0 : allAlwaysThinking ? 0 : 1);
}

main();
