import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { afterEach, beforeEach, expect, it, vi } from "vitest";

vi.mock("../models.json", () => {
  const base = { reasoning: true, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } };
  return { default: [
    { ...base, id: "fixture", name: "Fixture", contextWindow: 1048576, maxTokens: 0, compat: { chatTemplateKwargs: { preserve_thinking: true } } },
    { ...base, id: "short", name: "Short", contextWindow: 256, maxTokens: 0 },
    { ...base, id: "published", name: "Published", contextWindow: 65536, maxTokens: 8192 },
  ] };
});

let home: string;
let config: string;
beforeEach(() => {
  home = fs.mkdtempSync(path.join(os.tmpdir(), "makora-settings-"));
  vi.stubEnv("PI_CODING_AGENT_DIR", home);
  config = path.join(home, "extensions", "makora.json");
  vi.resetModules();
});
afterEach(() => {
  vi.unstubAllEnvs();
  fs.rmSync(home, { recursive: true, force: true });
});

async function start() {
  const { default: init } = await import("../index.js");
  let handler!: (args: string, ctx: unknown) => Promise<void>;
  const registerProvider = vi.fn();
  init({ on: vi.fn(), registerProvider, registerCommand: (_name, command) => { handler = (args, ctx) => command.handler(args, ctx as Parameters<typeof command.handler>[1]); } } as Parameters<typeof init>[0]);
  return { registerProvider, edit: (select: (title: string, items: string[]) => string | undefined, notify = vi.fn()) => handler("", { mode: "rpc", hasUI: true, ui: { select, notify } }) };
}

it("registers a conservative budget only when the catalog has no positive output limit", async () => {
  const { registerProvider } = await start();
  const models = registerProvider.mock.calls[0][1].models;
  expect(models.find((m: { id: string }) => m.id === "fixture").maxTokens).toBe(4096);
  expect(models.find((m: { id: string }) => m.id === "short").maxTokens).toBe(256);
  expect(models.find((m: { id: string }) => m.id === "published").maxTokens).toBe(8192);
});

it("keeps UI writes scoped to the selected flag after unrelated edits during the prompt", async () => {
  const { edit } = await start();
  const newer = {
    futureField: { data: true },
    modelOverrides: {
      fixture: { futureOverride: 17, compat: { chatTemplateKwargs: { parallel_edit: true } } },
      other: { futureOverride: 18 },
    },
  };
  await edit((_title, items) => {
    if (items[0] !== "Preserve Thinking") return items.find((item) => item.startsWith("Fixture:"));
    fs.writeFileSync(config, JSON.stringify(newer));
    return "Clear Thinking";
  });
  expect(JSON.parse(fs.readFileSync(config, "utf8"))).toEqual({
    ...newer, modelOverrides: { ...newer.modelOverrides, fixture: { futureOverride: 17, compat: { chatTemplateKwargs: { parallel_edit: true, preserve_thinking: false } } } },
  });
});

it("does not publish or re-register if config becomes malformed during the prompt", async () => {
  const { edit, registerProvider } = await start();
  const notify = vi.fn();
  await edit((_title, items) => {
    if (items[0] !== "Preserve Thinking") return items.find((item) => item.startsWith("Fixture:"));
    fs.writeFileSync(config, "{not-json");
    return "Clear Thinking";
  }, notify);
  expect(fs.readFileSync(config, "utf8")).toBe("{not-json");
  expect(registerProvider).toHaveBeenCalledOnce();
  expect(notify).toHaveBeenCalledWith(expect.stringContaining("Could not save Makora settings"), "error");
});
