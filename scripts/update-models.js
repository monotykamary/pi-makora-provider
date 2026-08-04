#!/usr/bin/env node

/**
 * Update Makora models from API
 *
 * Fetches models from https://inference.makora.com/v1/models and updates:
 * - models.json: Provider model definitions (auto-generated from API)
 * - README.md: Model table between <!-- MODELS_TABLE_START --> / <!-- MODELS_TABLE_END -->
 *
 * The Makora /v1/models API returns model id, max_model_len (context window),
 * and max_output_length. It does NOT provide pricing, reasoning mode, compat,
 * thinkingLevelMap, or notes. Those come from patch.json.
 *
 * Data flow:
 *   models.json       → auto-generated from Makora /v1/models (model discovery)
 *   patch.json        → manual overrides (reasoning, compat, notes, limits, etc.)
 *   custom-models.json → models not available via the API
 *
 * Merge order for README: models.json → apply patch.json → merge custom-models.json
 *
 * API key: the stored `makora` credential in ~/.pi/agent/auth.json wins, then
 * the MAKORA_OPTIMIZE_TOKEN environment variable. The script refuses to run without one.
 */

import fs from 'fs';
import os from 'os';
import { execSync } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// pi's agent directory: PI_CODING_AGENT_DIR (with ~ expansion) or ~/.pi/agent.
function piAgentDir() {
  const envDir = process.env.PI_CODING_AGENT_DIR;
  if (envDir) {
    return envDir.startsWith('~/') || envDir === '~'
      ? path.join(os.homedir(), envDir.slice(1))
      : envDir;
  }
  return path.join(os.homedir(), '.pi', 'agent');
}

const AUTH_JSON_PATH = path.join(piAgentDir(), 'auth.json');

/**
 * Resolve a configured value using pi's semantics (resolve-config-value.ts in
 * pi-mono): "!command" runs via the shell (10s timeout) and uses trimmed
 * stdout; "$VAR" / "${VAR}" interpolate environment variables ("$$" escapes a
 * literal "$", "$!" a literal "!"); anything else is a literal. Returns
 * undefined when a referenced env var is unset or a command fails.
 */
function resolveConfigValue(config, env) {
  if (typeof config !== 'string' || config.length === 0) return undefined;
  if (config.startsWith('!')) {
    try {
      const out = execSync(config.slice(1), {
        encoding: 'utf8',
        timeout: 10000,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      return out.trim() || undefined;
    } catch {
      return undefined;
    }
  }
  const ENV_NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
  let resolved = '';
  let index = 0;
  while (index < config.length) {
    const dollar = config.indexOf('$', index);
    if (dollar < 0) {
      resolved += config.slice(index);
      break;
    }
    resolved += config.slice(index, dollar);
    const next = config[dollar + 1];
    let name;
    if (next === '$' || next === '!') {
      resolved += next;
      index = dollar + 2;
      continue;
    } else if (next === '{') {
      const end = config.indexOf('}', dollar + 2);
      if (end < 0) {
        resolved += '$';
        index = dollar + 1;
        continue;
      }
      const inner = config.slice(dollar + 2, end);
      if (!ENV_NAME_RE.test(inner)) {
        resolved += config.slice(dollar, end + 1);
        index = end + 1;
        continue;
      }
      name = inner;
      index = end + 1;
    } else {
      const match = config.slice(dollar + 1).match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (!match) {
        resolved += '$';
        index = dollar + 1;
        continue;
      }
      name = match[0];
      index = dollar + 1 + name.length;
    }
    const value = (env && env[name]) || process.env[name] || undefined;
    if (value === undefined) return undefined;
    resolved += value;
  }
  return resolved;
}

/**
 * The API key, resolved the way pi itself resolves it for this provider: the
 * stored `makora` credential in ~/.pi/agent/auth.json wins, then
 * the MAKORA_OPTIMIZE_TOKEN environment variable.
 */
function resolveApiKey() {
  try {
    const auth = JSON.parse(fs.readFileSync(AUTH_JSON_PATH, 'utf8'));
    const credential = auth?.makora;
    if (credential && credential.type === 'api_key' && typeof credential.key === 'string') {
      const key = resolveConfigValue(credential.key, credential.env);
      if (key) return key;
    }
  } catch {
    // Missing or unparseable auth.json: fall through to the env var.
  }
  return process.env.MAKORA_OPTIMIZE_TOKEN || undefined;
}

const MODELS_API_URL = 'https://inference.makora.com/v1/models';
const MODELS_JSON_PATH = path.join(__dirname, '..', 'models.json');
const PATCH_JSON_PATH = path.join(__dirname, '..', 'patch.json');
const CUSTOM_MODELS_JSON_PATH = path.join(__dirname, '..', 'custom-models.json');
const README_PATH = path.join(__dirname, '..', 'README.md');

// ─── Helpers ─────────────────────────────────────────────────────────────────

function loadJson(filePath) {
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf8'));
  } catch {
    return {};
  }
}

function saveJson(filePath, data) {
  fs.writeFileSync(filePath, JSON.stringify(data, null, 2) + '\n');
  console.log(`✓ Saved ${path.basename(filePath)}`);
}

// ─── API fetch ───────────────────────────────────────────────────────────────

async function fetchModels() {
  const apiKey = resolveApiKey();
  if (!apiKey) {
    throw new Error('No API key found: no `makora` credential resolved from ' + AUTH_JSON_PATH + ' and MAKORA_OPTIMIZE_TOKEN is not set');
  }

  console.log(`Fetching models from ${MODELS_API_URL}...`);
  const response = await fetch(MODELS_API_URL, {
    headers: { 'Authorization': `Bearer ${apiKey}` },
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  const models = data.data || [];
  console.log(`✓ Fetched ${models.length} models from API`);
  return models;
}

// ─── DisplayName generation ────────────────────────────────────────────────

const DISPLAY_NAME_MAP = {
  'deepseek-ai/DeepSeek-V4-Flash': 'DeepSeek V4 Flash',
  'deepseek-ai/DeepSeek-V4-Pro': 'DeepSeek V4 Pro',
  'zai-org/GLM-5.1-FP8': 'GLM 5.1 FP8',
  'zai-org/GLM-5.2-FP8': 'GLM 5.2 FP8',
  'openai/gpt-oss-120b': 'GPT-OSS 120B',
  'nvidia/Kimi-K2.6-NVFP4': 'Kimi K2.6 NVFP4',
  'MiniMaxAI/MiniMax-M3-MXFP8': 'MiniMax M3 MXFP8',
  'meta-llama/Llama-3.3-70B-Instruct': 'Llama 3.3 70B Instruct',
  'unsloth/Qwen3.6-27B-NVFP4': 'Qwen 3.6 27B NVFP4',
  'unsloth/Qwen3.6-35B-A3B-NVFP4': 'Qwen 3.6 35B A3B NVFP4',
};

function generateDisplayName(id) {
  if (DISPLAY_NAME_MAP[id]) return DISPLAY_NAME_MAP[id];

  // Fallback: strip org prefix, replace hyphens with spaces, title-case
  const modelPart = id.includes('/') ? id.split('/').slice(1).join('/') : id;
  return modelPart
    .replace(/-/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}

// ─── Transform API model → models.json entry ────────────────────────────────

function transformApiModel(apiModel) {
  const id = apiModel.id;
  const name = generateDisplayName(id);
  const contextWindow = apiModel.max_model_len || 0;
  const maxTokens = apiModel.max_output_length || 0;

  return {
    id,
    name,
    reasoning: false,
    input: ['text'],
    cost: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
    },
    contextWindow,
    maxTokens,
    compat: {
      supportsDeveloperRole: false,
      supportsStore: false,
      maxTokensField: 'max_completion_tokens',
    },
  };
}

// ─── Patch & Custom Models ──────────────────────────────────────────────────

function applyPatch(model, patch) {
  const result = { ...model };
  if (patch.name !== undefined) result.name = patch.name;
  if (patch.reasoning !== undefined) result.reasoning = patch.reasoning;
  if (patch.input !== undefined) result.input = patch.input;
  if (patch.contextWindow !== undefined) result.contextWindow = patch.contextWindow;
  if (patch.maxTokens !== undefined) result.maxTokens = patch.maxTokens;
  if (patch.baseUrl !== undefined) result.baseUrl = patch.baseUrl;
  if (patch.notes !== undefined) result.notes = patch.notes;
  if (patch.thinkingLevelMap !== undefined) result.thinkingLevelMap = { ...patch.thinkingLevelMap };
  if (patch.headers !== undefined) result.headers = { ...patch.headers };
  if (patch.cost) {
    result.cost = {
      input: patch.cost.input ?? result.cost.input,
      output: patch.cost.output ?? result.cost.output,
      cacheRead: patch.cost.cacheRead ?? result.cost.cacheRead,
      cacheWrite: patch.cost.cacheWrite ?? result.cost.cacheWrite,
    };
  }
  if (patch.compat) {
    result.compat = { ...(result.compat || {}), ...patch.compat };
  }
  if (!result.reasoning && result.compat?.thinkingFormat) {
    delete result.compat.thinkingFormat;
  }
  if (result.compat && Object.keys(result.compat).length === 0) {
    delete result.compat;
  }
  return result;
}

function buildModels(baseModels, customModels, patchData) {
  const modelMap = new Map();

  for (const model of baseModels) {
    modelMap.set(model.id, model);
  }

  for (const [id, patchEntry] of Object.entries(patchData)) {
    const existing = modelMap.get(id);
    if (existing) {
      modelMap.set(id, applyPatch(existing, patchEntry));
    }
  }

  for (const model of customModels) {
    const existing = modelMap.get(model.id);
    const patchEntry = patchData[model.id];
    if (existing && patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else if (existing) {
      modelMap.set(model.id, model);
    } else if (patchEntry) {
      modelMap.set(model.id, applyPatch(model, patchEntry));
    } else {
      modelMap.set(model.id, model);
    }
  }

  return Array.from(modelMap.values());
}

// ─── README generation ──────────────────────────────────────────────────────

function generateReadmeTable(models) {
  const header = '| Model | ID | Reasoning | Notes |';
  const divider = '|-------|----|-----------|-------|';

  const rows = models.map(m => {
    const reasoning = m.reasoning ? 'Yes' : 'No';
    const notes = m.notes || '';
    return `| ${m.name} | \`${m.id}\` | ${reasoning} | ${notes} |`;
  });

  return [header, divider, ...rows].join('\n');
}

function updateReadme(models) {
  let readme = fs.readFileSync(README_PATH, 'utf8');

  const newTable = generateReadmeTable(models);

  const startMarker = '<!-- MODELS_TABLE_START -->';
  const endMarker = '<!-- MODELS_TABLE_END -->';

  const startIdx = readme.indexOf(startMarker);
  const endIdx = readme.indexOf(endMarker);

  if (startIdx === -1 || endIdx === -1) {
    console.error('README.md is missing <!-- MODELS_TABLE_START --> or <!-- MODELS_TABLE_END --> markers.');
    process.exit(1);
  }

  const before = readme.slice(0, startIdx + startMarker.length);
  const after = readme.slice(endIdx);

  readme = before + '\n' + newTable + '\n' + after;

  fs.writeFileSync(README_PATH, readme);
  console.log(`✓ Updated README.md with ${models.length} models`);
}

// ─── Main ────────────────────────────────────────────────────────────────────

// Grace period for delisted models: update-models.js moves models the API no
// longer lists into deprecated-models.json (stamped with deprecatedAt) instead
// of dropping them; the runtime appends them back so sessions and saved model
// settings keep working, and after 14 days they are evicted permanently.
const DEPRECATED_MODEL_TTL_MS = 14 * 24 * 60 * 60 * 1000;

/**
 * Reconcile deprecated-models.json against the freshly fetched model list.
 * - in old models.json but not the API: moved into the deprecated file
 *   (deprecatedAt = now; preserved on repeat runs so the grace clock is not reset)
 * - back in the API: resurrected (dropped from the deprecated file)
 * - deprecatedAt older than 14 days: evicted permanently
 * Must run BEFORE the new models.json is written; it reads the old file itself.
 */
function updateDeprecatedModels(modelsJsonPath, newModels) {
  const deprecatedPath = path.join(path.dirname(modelsJsonPath), 'deprecated-models.json');

  let oldModels = [];
  try {
    const parsed = JSON.parse(fs.readFileSync(modelsJsonPath, 'utf8'));
    if (Array.isArray(parsed)) oldModels = parsed;
  } catch { /* first run: no previous models.json */ }

  let deprecated = {};
  try {
    const parsed = JSON.parse(fs.readFileSync(deprecatedPath, 'utf8'));
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) deprecated = parsed;
  } catch { /* no graveyard yet */ }

  const currentIds = new Set(newModels.map(m => m.id));
  const now = new Date().toISOString();
  const added = [];
  const resurrected = [];
  const evicted = [];

  for (const old of oldModels) {
    if (old && old.id && !currentIds.has(old.id) && !deprecated[old.id]) {
      deprecated[old.id] = { ...old, deprecatedAt: now };
      added.push(old.id);
    }
  }

  for (const [id, entry] of Object.entries(deprecated)) {
    if (currentIds.has(id)) {
      delete deprecated[id];
      resurrected.push(id);
      continue;
    }
    const removedAt = Date.parse(entry && entry.deprecatedAt ? entry.deprecatedAt : '');
    if (Number.isNaN(removedAt) || Date.now() - removedAt > DEPRECATED_MODEL_TTL_MS) {
      delete deprecated[id];
      evicted.push(id);
    }
  }

  if (added.length > 0 || resurrected.length > 0 || evicted.length > 0) {
    fs.writeFileSync(deprecatedPath, JSON.stringify(deprecated, null, 2) + '\n');
    console.log('Updated deprecated-models.json ' + JSON.stringify({ added, resurrected, evicted }));
  }
}

async function main() {
  try {
    const apiModels = await fetchModels();

    // Load existing models.json for change detection
    const existingModels = loadJson(MODELS_JSON_PATH);
    const existingIds = new Set(
      (Array.isArray(existingModels) ? existingModels : []).map(m => m.id)
    );

    // Transform API models to models.json format (pure API regeneration)
    let models = apiModels.map(m => transformApiModel(m));

    // Sort by id for deterministic output
    models.sort((a, b) => a.id.localeCompare(b.id));

    // Live API is authoritative — models absent from API are removed
    // Move delisted models to deprecated-models.json BEFORE models.json is overwritten
    updateDeprecatedModels(MODELS_JSON_PATH, models);
    saveJson(MODELS_JSON_PATH, models);

    // Load and process custom models
    const customModels = Array.isArray(loadJson(CUSTOM_MODELS_JSON_PATH))
      ? loadJson(CUSTOM_MODELS_JSON_PATH)
      : [];

    // Find custom models that now appear in upstream (remove from custom)
    const upstreamIds = new Set(models.map(m => m.id));
    const duplicates = customModels.filter(m => upstreamIds.has(m.id));
    if (duplicates.length > 0) {
      console.log(`\nFound ${duplicates.length} custom model(s) now available upstream:`);
      for (const dup of duplicates) {
        console.log(`  - ${dup.id} (${dup.name})`);
      }
      const cleaned = customModels.filter(m => !upstreamIds.has(m.id));
      saveJson(CUSTOM_MODELS_JSON_PATH, cleaned);
      customModels.length = 0;
      customModels.push(...cleaned);
    }

    // Build full model list for README: base → patch → custom
    const patchData = loadJson(PATCH_JSON_PATH);
    const readmeModels = buildModels(models, customModels, patchData);
    readmeModels.sort((a, b) => a.name.localeCompare(b.name));

    // Update README
    updateReadme(readmeModels);

    // Log new models (not in existing models.json or patch.json)
    const newIds = new Set(models.map(m => m.id));
    const added = [...newIds].filter(id => !existingIds.has(id));
    const removed = [...existingIds].filter(id => !newIds.has(id));

    console.log('\n--- Summary ---');
    console.log(`Total models: ${models.length} upstream + ${customModels.length} custom`);
    console.log(`Patches: ${Object.keys(patchData).length}`);
    if (added.length > 0) {
      console.log(`New models: ${added.join(', ')} — add to patch.json for overrides`);
    }
    if (removed.length > 0) {
      console.log(`Removed models: ${removed.join(', ')} — move to custom-models.json if still needed`);
    }

    console.log('\nDone!');
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
