#!/usr/bin/env node

/**
 * Update Makora models from API
 *
 * Fetches models from https://inference.makora.com/v1/models and updates:
 * - models.json: Provider model definitions (auto-generated from API)
 * - README.md: Model table between <!-- MODELS_TABLE_START --> / <!-- MODELS_TABLE_END -->
 *
 * The Makora /v1/models API returns model id and max_model_len (context window),
 * but does NOT provide: pricing, max output tokens, reasoning mode, compat,
 * thinkingLevelMap, or notes. Those come from patch.json.
 *
 * Data flow:
 *   models.json       → auto-generated from Makora /v1/models (model discovery)
 *   patch.json        → manual overrides (reasoning, compat, notes, limits, etc.)
 *   custom-models.json → models not available via the API
 *
 * Merge order for README: models.json → apply patch.json → merge custom-models.json
 *
 * Requires MAKORA_OPTIMIZE_TOKEN environment variable.
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

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
  const apiKey = process.env.MAKORA_OPTIMIZE_TOKEN;
  if (!apiKey) {
    throw new Error('MAKORA_OPTIMIZE_TOKEN environment variable is required');
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
  'openai/gpt-oss-120b': 'GPT-OSS 120B',
  'nvidia/Kimi-K2.6-NVFP4': 'Kimi K2.6 NVFP4',
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
    maxTokens: 0,
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
