import { describe, expect, it } from "vitest";
import models from "../models.json" with { type: "json" };

describe("Makora v1 model limits", () => {
  it("uses DeepSeek V4 Flash's advertised maximum output length", () => {
    const model = models.find(({ id }) => id === "deepseek-ai/DeepSeek-V4-Flash");

    expect(model).toBeDefined();
    expect(model?.maxTokens).toBe(384000);
  });
});
