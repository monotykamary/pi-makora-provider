/**
 * Test: Thinking-Off via chat_template_kwargs Across Makora Models
 *
 * Probes whether a uniform `chat_template_kwargs.thinking` toggle can turn
 * thinking off for every model in this provider — reasoning and non-reasoning
 * alike — answering whether the provider can emit `thinking` (keyed to pi's
 * thinking switch) for all models.
 *
 * For each model, a reasoning-provoking prompt is sent under four kwarg
 * variants and the returned reasoning length is measured:
 *
 *   1. {}                       baseline (default template behavior)
 *   2. {enable_thinking:false}  what the current `qwen-chat-template` format sends when off
 *   3. {thinking:false}         the proposed uniform "off" key
 *   4. {thinking:true}          control — confirms the model CAN reason, so
 *                               that an empty `thinking:false` result is
 *                               meaningful (the toggle worked) rather than the
 *                               model simply never reasoning
 *
 * Verdict per reasoning model:
 *   `thinking` key works   ⇔  thinking:true → reasoning present
 *                        AND  thinking:false → reasoning absent
 *   (likewise for `enable_thinking`, for comparison)
 *
 * Non-reasoning models: the kwarg must be tolerated (no HTTP error) across all
 * variants — their templates have no `thinking` variable, so vLLM should
 * ignore it. If any error, the kwarg cannot be blanket-applied.
 *
 * Usage:
 *   MAKORA_OPTIMIZE_TOKEN=your-key npx tsx test-thinking-off.ts
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
  reasoning: boolean;
  /** Extra top-level payload keys merged into every request for this model
   *  (e.g. DeepSeek V4 Flash needs `include_reasoning: true` to return the
   *  reasoning field, independent of whether thinking is on). */
  extraPayload?: Record<string, unknown>;
}

const MODELS: ModelSpec[] = [
  { id: "deepseek-ai/DeepSeek-V4-Pro", name: "DeepSeek V4 Pro", reasoning: true },
  {
    id: "deepseek-ai/DeepSeek-V4-Flash",
    name: "DeepSeek V4 Flash",
    reasoning: true,
    extraPayload: { include_reasoning: true },
  },
  { id: "zai-org/GLM-5.2-FP8", name: "GLM 5.2 FP8", reasoning: true },
  { id: "zai-org/GLM-5.2-NVFP4", name: "GLM 5.2 NVFP4", reasoning: true },
  { id: "moonshotai/Kimi-K2.7-Code", name: "Kimi K2.7 Code", reasoning: true },
  { id: "unsloth/Qwen3.6-27B-NVFP4", name: "Qwen 3.6 27B NVFP4", reasoning: true },
  { id: "unsloth/Qwen3.6-35B-A3B-NVFP4", name: "Qwen 3.6 35B A3B", reasoning: true },
  { id: "google/gemma-4-26B-A4B", name: "Gemma 4 26B", reasoning: false },
  { id: "meta-llama/Llama-3.3-70B-Instruct", name: "Llama 3.3 70B", reasoning: false },
  { id: "amd/Llama-3.3-70B-Instruct-FP8-KV", name: "Llama 3.3 70B FP8", reasoning: false },
];

interface Variant {
  label: string;
  kwargs: Record<string, unknown> | null; // null = omit chat_template_kwargs entirely
}

const VARIANTS: Variant[] = [
  { label: "baseline", kwargs: null },
  { label: "enable_thinking:false", kwargs: { enable_thinking: false } },
  { label: "thinking:false", kwargs: { thinking: false } },
  { label: "thinking:true", kwargs: { thinking: true } },
];

const PROMPT = "What is 17 * 23? Think step by step, then give the final answer.";

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
    ...model.extraPayload,
  };
  if (variant.kwargs !== null) {
    payload.chat_template_kwargs = variant.kwargs;
  }

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
  reasoningLen: number;
  contentLen: number;
  error?: string;
}

interface ModelResult {
  model: string;
  id: string;
  reasoning: boolean;
  variants: VariantResult[];
  /** `thinking` key toggles reasoning off/on (for reasoning models). */
  thinkingKeyWorks?: boolean;
  /** `enable_thinking` key toggles reasoning off/on (for reasoning models). */
  enableThinkingKeyWorks?: boolean;
  /** Non-reasoning model tolerates the kwarg without error. */
  toleratesKwarg?: boolean;
  /** Could not be tested (e.g. baseline access denied) — kwarg effect unknown. */
  inconclusive?: boolean;
  inconclusiveReason?: string;
}

async function testModel(model: ModelSpec): Promise<ModelResult> {
  const result: ModelResult = { model: model.name, id: model.id, reasoning: model.reasoning, variants: [] };
  console.log(`\n► ${model.name} (${model.id})  [reasoning=${model.reasoning}]`);

  for (const variant of VARIANTS) {
    process.stdout.write(`  [${variant.label}] ... `);
    const r = await chatCompletion(model, variant);
    result.variants.push({ variant: variant.label, ...r });
    const reason = r.error
      ? `ERROR ${r.error.slice(0, 80)}`
      : `reasoning=${r.reasoningLen}B content=${r.contentLen}B`;
    console.log(reason);
  }

  const byLabel = (label: string) => result.variants.find((v) => v.variant === label)!;
  const offVariant = byLabel("thinking:false");
  const onVariant = byLabel("thinking:true");
  const enOffVariant = byLabel("enable_thinking:false");
  const baseVariant = byLabel("baseline");

  if (model.reasoning) {
    // A key "works" if turning it on yields reasoning AND turning it off suppresses it.
    const thinkingOnHasReasoning = !onVariant.error && onVariant.reasoningLen > 0;
    const thinkingOffNoReasoning = !offVariant.error && offVariant.reasoningLen === 0;
    result.thinkingKeyWorks = thinkingOnHasReasoning && thinkingOffNoReasoning;

    // For enable_thinking, we don't have an explicit on-control with the same key,
    // so judge off against the thinking:true control: it "works" if it suppresses
    // reasoning that we know the model can produce.
    const enOffNoReasoning = !enOffVariant.error && enOffVariant.reasoningLen === 0;
    result.enableThinkingKeyWorks = thinkingOnHasReasoning && enOffNoReasoning;

    // If every variant errored, the model is untestable (e.g. access denied).
    if (result.variants.every((v) => v.error)) {
      result.inconclusive = true;
      result.inconclusiveReason = baseVariant.error?.slice(0, 120);
    }

    console.log(
      `  → thinking key works: ${result.thinkingKeyWorks ? "YES" : "no"}  | ` +
        `enable_thinking key works: ${result.enableThinkingKeyWorks ? "YES" : "no"}  | ` +
        `baseline reasoning=${baseVariant.reasoningLen}B`
    );
  } else {
    // A baseline error (e.g. HTTP 403 access denied) means we can't test the
    // kwarg at all — it's inconclusive, not a kwarg intolerance.
    if (baseVariant.error) {
      result.inconclusive = true;
      result.inconclusiveReason = baseVariant.error.slice(0, 120);
      console.log(`  → inconclusive (baseline error: ${result.inconclusiveReason})`);
    } else {
      const kwargVariants = result.variants.filter((v) => v.variant !== "baseline");
      const anyKwargError = kwargVariants.some((v) => v.error);
      result.toleratesKwarg = !anyKwargError;
      console.log(`  → tolerates kwarg: ${result.toleratesKwarg ? "YES" : "no"}`);
    }
  }

  return result;
}

// Main

async function main() {
  console.log("=== Thinking-Off via chat_template_kwargs Test ===");
  console.log(`Models: ${MODELS.length}  | Variants per model: ${VARIANTS.length}\n`);

  const results: ModelResult[] = [];
  for (const model of MODELS) {
    results.push(await testModel(model));
  }

  // Summary table
  console.log("\n\n=== Summary ===\n");
  console.log("| Model | Reasoning | thinking key | enable_thinking key | tolerates kwarg | inconclusive |");
  console.log("|-------|-----------|--------------|---------------------|-----------------|--------------|");
  for (const r of results) {
    if (r.reasoning) {
      const inc = r.inconclusive ? "YES" : "";
      console.log(
        `| ${r.model} | yes | ${r.thinkingKeyWorks ? "WORKS" : "no"} | ${r.enableThinkingKeyWorks ? "WORKS" : "no"} | — | ${inc} |`
      );
    } else {
      console.log(`| ${r.model} | no | — | — | ${r.toleratesKwarg ? "YES" : r.inconclusive ? "—" : "NO"} | ${r.inconclusive ? "YES" : ""} |`);
    }
  }

  // Verdict
  const reasoningModels = results.filter((r) => r.reasoning && !r.inconclusive);
  const nonReasoning = results.filter((r) => !r.reasoning && !r.inconclusive);
  const inconclusive = results.filter((r) => r.inconclusive);

  const allThinkingKeyWork = reasoningModels.length > 0 && reasoningModels.every((r) => r.thinkingKeyWorks);
  const allTolerate = nonReasoning.every((r) => r.toleratesKwarg);

  console.log("\n=== Verdict ===\n");
  console.log(`Reasoning models where 'thinking' key turns thinking off+on: ${reasoningModels.filter((r) => r.thinkingKeyWorks).length}/${reasoningModels.length}`);
  console.log(`Non-reasoning models that tolerate the kwarg: ${nonReasoning.filter((r) => r.toleratesKwarg).length}/${nonReasoning.length}`);
  if (inconclusive.length > 0) {
    console.log(`Inconclusive (untestable, e.g. access denied): ${inconclusive.length}`);
    for (const i of inconclusive) console.log(`  - ${i.model}: ${i.inconclusiveReason}`);
  }

  if (allThinkingKeyWork && allTolerate) {
    console.log("\n✅ YES — this provider can emit `chat_template_kwargs.thinking` (toggled by pi's thinking switch) for ALL testable models.");
  } else {
    console.log("\n⚠️  NO — a single `thinking` key does not uniformly work for all models:");
    if (reasoningModels.length > 0 && !allThinkingKeyWork) {
      const fails = reasoningModels.filter((r) => !r.thinkingKeyWorks);
      console.log("   `thinking` key does NOT cleanly toggle for:");
      for (const f of fails) console.log(`     - ${f.model} (reasoning still present under thinking:false — may read 'enable_thinking', be always-thinking, or need another key)`);
    }
    if (!allTolerate) {
      const fails = nonReasoning.filter((r) => !r.toleratesKwarg);
      console.log("   non-reasoning models that ERRORED on the kwarg:");
      for (const f of fails) {
        const err = f.variants.find((v) => v.error && v.variant !== "baseline");
        console.log(`     - ${f.model}: ${err?.error?.slice(0, 120)}`);
      }
    }
  }

  const exitFails = (reasoningModels.some((r) => !r.thinkingKeyWorks) ? 1 : 0) | (nonReasoning.some((r) => !r.toleratesKwarg) ? 1 : 0);
  process.exit(exitFails);
}

main();
