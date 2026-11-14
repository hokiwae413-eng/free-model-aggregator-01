// The trial brief: three JSON documents inside one prompt file, plus the
// catalog probe that turns the provider directory into the registered catalog.
//
// The brief carries, in prose plus JSON:
//   - free-model-aggregator-01/providers/v1 : the ten simulated providers
//     (id, wire, auth scheme, key env var, key, base path)
//   - free-model-aggregator-01/ratings/v1   : the sealed quality table
//     (rating_milli, context length, tools/vision/json caps, free flag)
//   - free-model-aggregator-01/workload/v1  : the twenty requests, in order
//     (api, stream, weight, requirements)
//
// The probe asks each provider's catalog route in its own wire. The catalog
// route is free (no budget) and deterministic, so the catalog built here is
// the same document `--list` prints and the register body carries.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

export interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthKind;
  /** Base path (e.g. /p/groq) or an absolute URL. */
  base: string;
  keyEnv: string;
  key: string;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  contextLength: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface CatalogDoc {
  providers: CatalogProvider[];
}

export interface RatingRow {
  model: string;
  provider: string;
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingsDoc {
  ratings: RatingRow[];
  best: { id: string; rating: number }[];
}

export interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface WorkloadRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Need;
}

export interface Brief {
  providers: ProviderEntry[];
  ratings: RatingsDoc | null;
  workload: WorkloadRow[];
  error: string | null;
}

export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  status: number;
  models: ListedModel[];
  error: string | null;
}

export interface Caps {
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** Join a base path to the env origin; an absolute base is used as-is. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const root = envUrl.replace(/\/+$/, '');
  if (b === '') return root;
  return `${root}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers carrying one provider's key, in its own scheme. */
export function authHeaders(auth: AuthKind, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

function strOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** Index of the `}` that closes the object opening at `start`, or -1. */
function closeBrace(text: string, start: number): number {
  let depth = 0;
  let inStr = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inStr = false;
    } else if (ch === '"') inStr = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function schemaDocs(text: string): Json[] {
  const out: Json[] = [];
  for (let at = text.indexOf('{'); at !== -1; at = text.indexOf('{', at + 1)) {
    const end = closeBrace(text, at);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(at, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own; keep scanning.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderEntry | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none'
      ? v.auth
      : 'bearer';
  // The directory writes `path` (/p/<id>); accept base_path / base_url / base too.
  const base =
    strOf(v.path) !== ''
      ? strOf(v.path)
      : strOf(v.base_path) !== ''
        ? strOf(v.base_path)
        : strOf(v.base_url) !== ''
          ? strOf(v.base_url)
          : strOf(v.base);
  return {
    id,
    name: strOf(v.name) !== '' ? strOf(v.name) : id,
    wire,
    auth,
    base,
    keyEnv: strOf(v.key_env),
    key: strOf(v.key),
  };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = strOf(v.model);
  const provider = strOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: numOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadOf(v: unknown): WorkloadRow | null {
  if (!isRecord(v)) return null;
  const id = strOf(v.id);
  if (id === '') return null;
  const req = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: req.tools === true,
      vision: req.vision === true,
      json: req.json === true,
      ...(typeof req.min_context === 'number' && Number.isFinite(req.min_context)
        ? { minContext: req.min_context }
        : {}),
    },
  };
}

/** Read the brief at `path`: provider directory, quality table, workload. */
export function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: null, workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocs(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingsDoc | null =
    ratingsDoc === null || !Array.isArray(ratingsDoc.ratings)
      ? null
      : {
          ratings: ratingsDoc.ratings.flatMap((v) => {
            const r = ratingOf(v);
            return r === null ? [] : [r];
          }),
          best: Array.isArray(ratingsDoc.best)
            ? ratingsDoc.best.flatMap((v) => {
                if (!isRecord(v) || strOf(v.id) === '') return [];
                return [{ id: strOf(v.id), rating: numOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload: WorkloadRow[] =
    workloadDoc === null || !Array.isArray(workloadDoc.manifest)
      ? []
      : workloadDoc.manifest.flatMap((v) => {
          const w = workloadOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, workload, error: null };
}

/** The quality-table row for one provider's model, when the table has it. */
export function ratingRow(ratings: RatingsDoc | null, providerId: string, modelId: string): RatingRow | null {
  if (ratings === null) return null;
  return ratings.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the
 * sealed truth), the listing's own word where it does not, paid otherwise.
 * A paid model is never routed.
 */
export function isFreeModel(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table first, the listing second. */
export function capsOf(model: ListedModel, row: RatingRow | null): Caps {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's caps meet a request's requirements. Unknown never meets. */
export function meetsNeed(need: Need, caps: Caps): boolean {
  if (need.tools === true && caps.tools !== true) return false;
  if (need.vision === true && caps.vision !== true) return false;
  if (need.json === true && caps.json !== true) return false;
  if (need.minContext !== undefined && (caps.contextLength === null || caps.contextLength < need.minContext))
    return false;
  return true;
}

/** A listing's price block: true when every price is zero, null when silent. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModel(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? strOf(v.name).replace(/^models\//, '') : strOf(v.id);
  if (id === '') return null;
  const context = numOrNull(v.context_length) ?? numOrNull(v.context_window) ?? numOrNull(v.inputTokenLimit);
  return {
    id,
    displayName:
      strOf(v.display_name) !== ''
        ? strOf(v.display_name)
        : strOf(v.displayName) !== ''
          ? strOf(v.displayName)
          : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    contextLength: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

/** The models one catalog answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModel(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire; any failure excludes it. */
export async function probeProvider(
  envUrl: string,
  provider: ProviderEntry,
  timeoutMs = 5000,
): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = {
    accept: 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

const byId = (a: ListedModel, b: ListedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/**
 * The one catalog: every provider that answered, every model it listed.
 * base_url is the directory's own path (/p/<id>), never joined to the env
 * origin: the env's origin differs between a live pass and its replay, and
 * the catalog is part of the state a replay must reproduce byte for byte.
 */
export function buildCatalog(probes: readonly Probe[], ratings: RatingsDoc | null): CatalogDoc {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort(byId)) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRow(ratings, probe.provider.id, model.id);
      models.push({
        id: model.id,
        free: isFreeModel(model, row),
        contextLength: row?.contextLength ?? model.contextLength ?? null,
      });
    }
    providers.push({
      id: probe.provider.id,
      base_url: probe.provider.base,
      key_env: probe.provider.keyEnv,
      wire: probe.provider.wire,
      models,
    });
  }
  return { providers };
}
