import { describe, expect, it } from "vitest";
import models from "../models.json" with { type: "json" };

describe("Makora v1 model limits", () => {
  it("uses DeepSeek V4.1 Flash's advertised maximum output length", () => {
    // V4 Flash moved to the rolling graveyard; check its current successor.
    const model = models.find(({ id }) => id === "deepseek-ai/DeepSeek-V4.1-Flash");

    expect(model).toBeDefined();
    // The 2026-10-18 provider catalog raised this limit from 393216.
    expect(model?.maxTokens).toBe(943718);
  });
});
