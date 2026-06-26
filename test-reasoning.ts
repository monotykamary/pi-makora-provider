/**
 * Test: Preserved Reasoning Across Makora Models
 *
 * For each reasoning model, this script:
 *   1. Sends a chat completion request that mimics pi's payload and verifies
 *      the reasoning field is present
 *   2. Sends a follow-up request that includes the assistant's prior message
 *      with its reasoning preserved — and verifies the model continues reasoning
 *
 * Usage:
 *   MAKORA_OPTIMIZE_TOKEN=your-key npx tsx test-reasoning.ts
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
  /** Field the model returns reasoning in */
  reasoningResponseField: "reasoning" | "reasoning_content";
  /** Build the full request payload for this model */
  buildPayload: (messages: Record<string, unknown>[]) => Record<string, unknown>;
}

const MODELS: ModelSpec[] = [
  {
    id: "deepseek-ai/DeepSeek-V4-Pro",
    name: "DeepSeek V4 Pro",
    reasoningResponseField: "reasoning",
    buildPayload: (messages) => ({
      model: "deepseek-ai/DeepSeek-V4-Pro",
      messages,
      max_tokens: 1024,
      stream: false,
      // Mimics rewriteDsVllmPayload
      chat_template_kwargs: { thinking: true },
    }),
  },
  {
    id: "deepseek-ai/DeepSeek-V4-Flash",
    name: "DeepSeek V4 Flash",
    reasoningResponseField: "reasoning",
    buildPayload: (messages) => ({
      model: "deepseek-ai/DeepSeek-V4-Flash",
      messages,
      max_tokens: 1024,
      stream: false,
      // Mimics rewriteDsVllmPayload — both required
      include_reasoning: true,
      chat_template_kwargs: { thinking: true },
    }),
  },
  {
    id: "zai-org/GLM-5.2-FP8",
    name: "GLM 5.2 FP8",
    reasoningResponseField: "reasoning",
    buildPayload: (messages) => ({
      model: "zai-org/GLM-5.2-FP8",
      messages,
      max_tokens: 1024,
      stream: false,
      // Mimics qwen-chat-template thinkingFormat
      chat_template_kwargs: { enable_thinking: true },
    }),
  },
  {
    id: "zai-org/GLM-5.2-NVFP4",
    name: "GLM 5.2 NVFP4",
    reasoningResponseField: "reasoning",
    buildPayload: (messages) => ({
      model: "zai-org/GLM-5.2-NVFP4",
      messages,
      max_tokens: 1024,
      stream: false,
      // Mimics qwen-chat-template thinkingFormat
      chat_template_kwargs: { enable_thinking: true },
    }),
  },
  {
    id: "moonshotai/Kimi-K2.7-Code",
    name: "Kimi K2.7 Code",
    reasoningResponseField: "reasoning",
    buildPayload: (messages) => ({
      model: "moonshotai/Kimi-K2.7-Code",
      messages,
      max_tokens: 1024,
      stream: false,
      // Mimics qwen-chat-template thinkingFormat + chatTemplateKwargs.preserve_thinking
      chat_template_kwargs: { enable_thinking: true, preserve_thinking: true },
    }),
  },
  {
    id: "unsloth/Qwen3.6-27B-NVFP4",
    name: "Qwen 3.6 27B NVFP4",
    reasoningResponseField: "reasoning",
    buildPayload: (messages) => ({
      model: "unsloth/Qwen3.6-27B-NVFP4",
      messages,
      max_tokens: 1024,
      stream: false,
      // Mimics qwen-chat-template thinkingFormat + chatTemplateKwargs.preserve_thinking
      chat_template_kwargs: { enable_thinking: true, preserve_thinking: true },
    }),
  },
  {
    id: "unsloth/Qwen3.6-35B-A3B-NVFP4",
    name: "Qwen 3.6 35B A3B NVFP4",
    reasoningResponseField: "reasoning",
    buildPayload: (messages) => ({
      model: "unsloth/Qwen3.6-35B-A3B-NVFP4",
      messages,
      max_tokens: 1024,
      stream: false,
      // Mimics qwen-chat-template thinkingFormat + chatTemplateKwargs.preserve_thinking
      chat_template_kwargs: { enable_thinking: true, preserve_thinking: true },
    }),
  },
];

// API helper

async function chatCompletion(
  model: ModelSpec,
  messages: Record<string, unknown>[]
): Promise<{ content: string; reasoning: string }> {
  const payload = model.buildPayload(messages);

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
      throw new Error(`HTTP ${res.status}: ${body.slice(0, 500)}`);
    }

    const data = (await res.json()) as {
      choices: {
        message: {
          content?: string;
          reasoning?: string;
          reasoning_content?: string;
        };
      }[];
    };

    const msg = data.choices?.[0]?.message;
    const content = msg?.content ?? "";
    const reasoning =
      model.reasoningResponseField === "reasoning_content"
        ? msg?.reasoning_content ?? ""
        : msg?.reasoning ?? "";

    return { content, reasoning };
  } finally {
    clearTimeout(timer);
  }
}

// Test runner

type TestResult = "PASS" | "FAIL";

interface ModelTestResult {
  model: string;
  turn1ReasoningPresent: TestResult;
  turn1ReasoningLen: number;
  turn2ReasoningPresent: TestResult;
  turn2ReasoningLen: number;
  turn1ContentLen: number;
  turn2ContentLen: number;
  error?: string;
}

async function testModel(model: ModelSpec): Promise<ModelTestResult> {
  const result: ModelTestResult = {
    model: model.name,
    turn1ReasoningPresent: "FAIL",
    turn1ReasoningLen: 0,
    turn2ReasoningPresent: "FAIL",
    turn2ReasoningLen: 0,
    turn1ContentLen: 0,
    turn2ContentLen: 0,
  };

  try {
    // Turn 1: simple reasoning question
    const messages1: Record<string, unknown>[] = [
      {
        role: "user",
        content:
          "What is 17 * 23? Think step by step, then give the final answer.",
      },
    ];

    console.log(`  [Turn 1] Sending request...`);
    const r1 = await chatCompletion(model, messages1);
    result.turn1ReasoningLen = r1.reasoning.length;
    result.turn1ContentLen = r1.content.length;
    result.turn1ReasoningPresent =
      r1.reasoning.length > 0 ? "PASS" : "FAIL";

    console.log(
      `  [Turn 1] content=${r1.content.length}B, reasoning=${r1.reasoning.length}B`
    );
    console.log(
      `  [Turn 1] Reasoning preview: ${r1.reasoning.slice(0, 120).replace(/\n/g, " ")}${r1.reasoning.length > 120 ? "..." : ""}`
    );

    if (result.turn1ReasoningPresent === "FAIL") {
      result.error = "No reasoning returned on turn 1";
      return result;
    }

    // Turn 2: follow-up with preserved reasoning
    // Build the assistant message with reasoning in the same field the model
    // returns it in — matching pi's requiresReasoningContentOnAssistantMessages
    // behavior for DS Pro, and the standard multi-turn format for others.
    const assistantMsg: Record<string, unknown> = {
      role: "assistant",
      content: r1.content,
    };
    if (model.reasoningResponseField === "reasoning_content") {
      assistantMsg.reasoning_content = r1.reasoning;
    } else {
      assistantMsg.reasoning = r1.reasoning;
    }

    const messages2: Record<string, unknown>[] = [
      messages1[0],
      assistantMsg,
      {
        role: "user",
        content: "Now add 5 to that result. Think step by step again.",
      },
    ];

    console.log(`  [Turn 2] Sending preserved-reasoning request...`);
    const r2 = await chatCompletion(model, messages2);
    result.turn2ReasoningLen = r2.reasoning.length;
    result.turn2ContentLen = r2.content.length;
    result.turn2ReasoningPresent =
      r2.reasoning.length > 0 ? "PASS" : "FAIL";

    console.log(
      `  [Turn 2] content=${r2.content.length}B, reasoning=${r2.reasoning.length}B`
    );
    console.log(
      `  [Turn 2] Reasoning preview: ${r2.reasoning.slice(0, 120).replace(/\n/g, " ")}${r2.reasoning.length > 120 ? "..." : ""}`
    );

    if (result.turn2ReasoningPresent === "FAIL") {
      result.error = "No reasoning returned on turn 2 (preserved)";
    }
  } catch (err: unknown) {
    result.error =
      err instanceof Error ? err.message : String(err);
    if (result.error.includes("aborted")) {
      result.error = `Timeout after ${TIMEOUT_MS / 1000}s`;
    }
  }

  return result;
}

// Main

async function main() {
  console.log("=== Preserved Reasoning Test ===\n");
  console.log(`Models to test: ${MODELS.length}\n`);

  const results: ModelTestResult[] = [];

  for (const model of MODELS) {
    console.log(`\n► ${model.name} (${model.id})`);
    console.log(`  Reasoning response field: ${model.reasoningResponseField}`);
    const r = await testModel(model);
    results.push(r);
  }

  // Summary table
  console.log("\n\n=== Summary ===\n");
  console.log(
    "| Model | Resp Field | T1 Reasoning | T1 Len | T2 Reasoning | T2 Len | Error |"
  );
  console.log(
    "|-------|-----------|-------------|--------|-------------|--------|--------|"
  );

  for (const r of results) {
    const spec = MODELS.find((m) => m.name === r.model)!;
    console.log(
      `| ${r.model} | ${spec.reasoningResponseField} | ${r.turn1ReasoningPresent} | ${r.turn1ReasoningLen} | ${r.turn2ReasoningPresent} | ${r.turn2ReasoningLen} | ${r.error ?? ""} |`
    );
  }

  const passes = results.filter(
    (r) =>
      r.turn1ReasoningPresent === "PASS" &&
      r.turn2ReasoningPresent === "PASS"
  ).length;
  const fails = results.length - passes;

  console.log(`\n${passes}/${results.length} models pass both turns.`);

  if (fails > 0) {
    console.log(`\n❌ ${fails} model(s) failed:`);
    for (const r of results) {
      if (
        r.turn1ReasoningPresent !== "PASS" ||
        r.turn2ReasoningPresent !== "PASS"
      ) {
        console.log(
          `  - ${r.model}: T1=${r.turn1ReasoningPresent} T2=${r.turn2ReasoningPresent}${r.error ? ` (${r.error})` : ""}`
        );
      }
    }
  } else {
    console.log(`\n✅ All models pass preserved reasoning.`);
  }

  process.exit(fails > 0 ? 1 : 0);
}

main();
