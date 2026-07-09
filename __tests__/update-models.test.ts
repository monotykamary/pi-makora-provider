import { describe, expect, it } from "vitest";
import { DISPLAY_NAME_MAP, generateDisplayName } from "../scripts/update-models.js";

const currentNames: Array<[string, string]> = [
  ["amd/Llama-3.3-70B-Instruct-FP8-KV", "Llama 3.3 70B FP8"],
  ["deepseek-ai/DeepSeek-V4-Flash", "DeepSeek V4 Flash"],
  ["deepseek-ai/DeepSeek-V4-Pro", "DeepSeek V4 Pro"],
  ["google/gemma-4-26B-A4B", "Gemma 4 26B A4B"],
  ["meta-llama/Llama-3.3-70B-Instruct", "Llama 3.3 70B Instruct"],
  ["moonshotai/Kimi-K2.7-Code", "Kimi K2.7 Code"],
  ["openai/gpt-oss-120b", "GPT-OSS 120B"],
  ["unsloth/Qwen3.6-27B-NVFP4", "Qwen 3.6 27B NVFP4"],
  ["unsloth/Qwen3.6-35B-A3B-NVFP4", "Qwen 3.6 35B A3B NVFP4"],
  ["zai-org/GLM-5.2-FP8", "GLM 5.2 FP8"],
  ["zai-org/GLM-5.2-NVFP4", "GLM 5.2 NVFP4"],
];

describe("update-models display-name map", () => {
  it.each(currentNames)("maps %s to %s", (id, expected) => {
    expect(generateDisplayName(id)).toBe(expected);
  });

  it("does not keep stale Makora model ids in the curated map", () => {
    expect(DISPLAY_NAME_MAP).not.toHaveProperty("nvidia/Kimi-K2.6-NVFP4");
    expect(DISPLAY_NAME_MAP).not.toHaveProperty("zai-org/GLM-5.1-FP8");
    expect(DISPLAY_NAME_MAP).not.toHaveProperty("MiniMaxAI/MiniMax-M3-MXFP8");
  });

  it("falls back to a readable title-cased model suffix for unknown ids", () => {
    expect(generateDisplayName("example-org/new-model-alpha")).toBe("New Model Alpha");
  });
});
