#!/usr/bin/env node

/**
 * Generates the Models section of README.md from models.json + patch.json + custom-models.json.
 *
 * Replaces the table between the <!-- MODELS_TABLE_START --> and
 * <!-- MODELS_TABLE_END --> markers in README.md. Idempotent — run anytime.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

const modelsPath = resolve(__dirname, "models.json");
const patchPath = resolve(__dirname, "patch.json");
const customPath = resolve(__dirname, "custom-models.json");
const readmePath = resolve(__dirname, "README.md");

const models = JSON.parse(readFileSync(modelsPath, "utf-8"));
const patch = JSON.parse(readFileSync(patchPath, "utf-8"));
const custom = JSON.parse(readFileSync(customPath, "utf-8"));
const readme = readFileSync(readmePath, "utf-8");

// Merge: apply patch over models (same logic as index.ts)
function applyPatchToModel(model, p) {
  const result = { ...model };
  if (p.name !== undefined) result.name = p.name;
  if (p.reasoning !== undefined) result.reasoning = p.reasoning;
  if (p.notes !== undefined) result.notes = p.notes;
  if (p.contextWindow !== undefined) result.contextWindow = p.contextWindow;
  if (p.maxTokens !== undefined) result.maxTokens = p.maxTokens;
  if (p.baseUrl !== undefined) result.baseUrl = p.baseUrl;
  if (p.compat) result.compat = { ...(result.compat || {}), ...p.compat };
  return result;
}

const modelMap = new Map();
for (const m of models) modelMap.set(m.id, m);
for (const [id, p] of Object.entries(patch)) {
  const existing = modelMap.get(id);
  if (existing) modelMap.set(id, applyPatchToModel(existing, p));
}
for (const m of custom) {
  const existing = modelMap.get(m.id);
  const p = patch[m.id];
  if (!existing) modelMap.set(m.id, p ? applyPatchToModel(m, p) : m);
}

const merged = Array.from(modelMap.values());

// Generate table
const header = "| Model | ID | Reasoning | Notes |";
const divider = "|-------|----|-----------|-------|";

const rows = merged.map((m) => {
  const reasoning = m.reasoning ? "Yes" : "No";
  const notes = m.notes || "";
  return `| ${m.name} | \`${m.id}\` | ${reasoning} | ${notes} |`;
});

const table = [header, divider, ...rows].join("\n");

// Replace in README between markers
const startMarker = "<!-- MODELS_TABLE_START -->";
const endMarker = "<!-- MODELS_TABLE_END -->";

const startIdx = readme.indexOf(startMarker);
const endIdx = readme.indexOf(endMarker);

if (startIdx === -1 || endIdx === -1) {
  console.error("README.md is missing <!-- MODELS_TABLE_START --> or <!-- MODELS_TABLE_END --> markers.");
  process.exit(1);
}

const before = readme.slice(0, startIdx + startMarker.length);
const after = readme.slice(endIdx);

const newReadme = before + "\n" + table + "\n" + after;

writeFileSync(readmePath, newReadme, "utf-8");
console.log("README.md models table updated.");
