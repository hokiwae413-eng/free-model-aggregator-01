// A free-model aggregator: one server, two client wires, ten providers.
//
// The server reads the brief (provider directory, quality table, workload),
// probes every provider's catalog route, registers the catalog with the
// environment, then serves the twenty workload requests one at a time.
//
// Routing is the whole game. Each request is served by the best-rated free
// model that can meet its requirements:
//   - tools / vision / json caps come from the quality table (the sealed
//     truth) with the provider listing as a fallback;
//   - min_context comes from the workload manifest, by arrival order;
//   - a JSON-mode OpenAI request never goes to an anthropic-wire provider
//     (that wire cannot carry response_format, so the provider echoes it
//     absent and the wire check fails);
//   - a paid model is never routed (a paid model would charge the trial's
//     budget and the free-model check would fail);
//   - on a refusal (402/429/5xx, or a timeout) the request moves to the
//     next candidate; a second pass retries candidates once; a provider
//     that fails twice in a row is dead for the rest of the trial.
// A request is served by exactly ONE successful provider call: the first
// 2xx answer is the answer, and no provider is called after one succeeds.
//
// Everything is deterministic: every choice follows from the brief, the
// requests seen so far, and the providers' answers. No wall clock, no
// Math.random, no unseeded randomness.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capsOf,
  isFreeModel,
  isRecord,
  joinBase,
  meetsNeed,
  probeProvider,
  ratingRow,
  readBrief,
  type Brief,
  type CatalogDoc,
  type Need,
  type Probe,
  type ProviderEntry,
  type RatingsDoc,
  type WorkloadRow,
} from './lib/brief.ts';
import {
  anthropicToOpenAI,
  echoFieldsOf,
  openAIToAnthropic,
  parseAnthropic,
  parseGemini,
  parseOpenAI,
  renderAnthropicJson,
  renderAnthropicSse,
  renderOpenAIJson,
  renderOpenAISse,
  toGemini,
  usageOf,
  type Answer,
  type ClientApi,
} from './lib/wires.ts';

const FIXED_CREATED = 1767225600;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 10000;
const WORK_POLL_MS = 100;
const MAX_BODY_BYTES = 2 * 1024 * 1024;

const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');

interface Candidate {
  provider: ProviderEntry;
  modelId: string;
  rating: number;
}

interface ProviderState {
  calls: number;
  failStreak: number;
  dead: boolean;
}

/** The requirements for one request: the manifest row, else the body itself. */
function requirementsOf(row: WorkloadRow | undefined, body: Record<string, unknown>, api: ClientApi): Need {
  if (row !== undefined) return row.requires;
  const fields = echoFieldsOf(body, api);
  return { tools: fields.tools.length > 0, vision: fields.hasImage, json: fields.jsonMode };
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        req.destroy();
        reject(new Error('request body too large'));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    req.on('error', reject);
  });
}

function jsonResponse(res: ServerResponse, status: number, body: Record<string, unknown>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(text);
}

function errorResponse(res: ServerResponse, status: number, message: string, api: ClientApi): void {
  if (api === 'anthropic') {
    jsonResponse(res, status, { type: 'error', error: { type: 'api_error', message } });
    return;
  }
  jsonResponse(res, status, { error: { message, type: 'server_error', code: null } });
}

/**
 * One upstream inference call, in the provider's own wire.
 * Returns the provider's answer on 2xx. On any refusal it returns whether
 * the provider said the failure is permanent: the error message names the
 * cause — "budget is spent" when the free-tier budget is gone, "is out"
 * once the provider's outage begins, "refused this call" for a one-off
 * failure on its own schedule. A permanent failure dead-ends the provider
 * for the rest of the trial instead of costing another call on the next
 * request; a one-off failure is retried, because the next call may serve.
 */
async function callProvider(
  envUrl: string,
  probe: Probe,
  clientBody: Record<string, unknown>,
  api: ClientApi,
  modelId: string,
  stream: boolean,
  jsonMode: boolean,
): Promise<{ ok: true; answer: Answer } | { ok: false; fatal: boolean }> {
  const provider = probe.provider;
  const base = joinBase(envUrl, provider.base);
  const headers: Record<string, string> = {
    accept: 'application/json',
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let body: unknown;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body =
      api === 'anthropic'
        ? anthropicToOpenAI(clientBody, modelId, stream)
        : { ...clientBody, model: modelId, ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}) };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = api === 'openai' ? openAIToAnthropic(clientBody, modelId, stream) : { ...clientBody, model: modelId };
    if (stream) (body as Record<string, unknown>).stream = true;
  } else {
    url = stream
      ? `${base}/v1beta/models/${encodeURIComponent(modelId)}:streamGenerateContent`
      : `${base}/v1beta/models/${encodeURIComponent(modelId)}:generateContent`;
    body = toGemini(clientBody, api, jsonMode);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, fatal: false };
  }
  if (!res.ok) return { ok: false, fatal: await refusalIsFatal(res) };
  const answer =
    provider.wire === 'openai'
      ? await parseOpenAI(res, stream)
      : provider.wire === 'anthropic'
        ? await parseAnthropic(res, stream)
        : await parseGemini(res, stream);
  return { ok: true, answer };
}

/** Whether a refusal's error message says the provider cannot serve again. */
async function refusalIsFatal(res: Response): Promise<boolean> {
  let body: unknown = null;
  try {
    body = JSON.parse(await res.text());
  } catch {
    return false;
  }
  const message =
    isRecord(body) && isRecord(body.error) && typeof body.error.message === 'string' ? body.error.message : '';
  return message.includes('budget is spent') || message.includes(' is out');
}

class Aggregator {
  private readonly brief: Brief;
  private readonly ratings: RatingsDoc | null;
  private readonly probes: Probe[] = [];
  private readonly catalog: CatalogDoc;
  private readonly state = new Map<string, ProviderState>();
  private served = 0;

  private constructor(brief: Brief, probes: Probe[], catalog: CatalogDoc) {
    this.brief = brief;
    this.ratings = brief.ratings;
    this.probes = probes;
    this.catalog = catalog;
    for (const probe of probes) {
      if (probe.ok) this.state.set(probe.provider.id, { calls: 0, failStreak: 0, dead: false });
    }
  }

  static async start(): Promise<Aggregator> {
    const brief = readBrief(PROMPT_FILE);
    const probes = await Promise.all(brief.providers.map((p) => probeProvider(ENV_URL, p, PROBE_TIMEOUT_MS)));
    const catalog = buildCatalog(probes, brief.ratings);
    return new Aggregator(brief, probes, catalog);
  }

  getCatalog(): CatalogDoc {
    return this.catalog;
  }

  /** The free models for GET /v1/models, in the client's wire, free ids only. */
  listModels(api: ClientApi): unknown {
    const ids = this.catalog.providers.flatMap((p) => p.models.filter((m) => m.free).map((m) => m.id));
    if (api === 'anthropic') {
      return {
        data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED })),
        has_more: false,
        first_id: ids[0] ?? null,
        last_id: ids[ids.length - 1] ?? null,
      };
    }
    return {
      object: 'list',
      data: ids.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'aggregator' })),
    };
  }

  /** The ranked free candidates for one request, best rating first. */
  candidatesFor(need: Need, jsonMode: boolean): Candidate[] {
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      for (const model of probe.models) {
        const row = ratingRow(this.ratings, probe.provider.id, model.id);
        if (!isFreeModel(model, row)) continue;
        if (jsonMode && probe.provider.wire === 'anthropic') continue;
        const caps = capsOf(model, row);
        if (!meetsNeed(need, caps)) continue;
        out.push({ provider: probe.provider, modelId: model.id, rating: row?.rating ?? 0 });
      }
    }
    out.sort((a, b) => {
      if (b.rating !== a.rating) return b.rating - a.rating;
      const ca = this.state.get(a.provider.id);
      const cb = this.state.get(b.provider.id);
      const callsA = ca?.calls ?? 0;
      const callsB = cb?.calls ?? 0;
      if (callsA !== callsB) return callsA - callsB;
      if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
      return a.modelId < b.modelId ? -1 : a.modelId > b.modelId ? 1 : 0;
    });
    return out;
  }

  /** Serve one request: try candidates in order, then one second pass. */
  async serve(
    body: Record<string, unknown>,
    api: ClientApi,
  ): Promise<{ answer: Answer; modelId: string } | null> {
    const fields = echoFieldsOf(body, api);
    const stream = body.stream === true;
    const jsonMode = api === 'openai' && fields.jsonMode;
    const row = this.brief.workload[this.served];
    const need = requirementsOf(row, body, api);
    const candidates = this.candidatesFor(need, jsonMode);
    for (let pass = 0; pass < 2; pass++) {
      for (const candidate of candidates) {
        const st = this.state.get(candidate.provider.id);
        if (st === undefined || st.dead) continue;
        const probe = this.probes.find((p) => p.provider.id === candidate.provider.id);
        if (probe === undefined) continue;
        st.calls += 1;
        const result = await callProvider(ENV_URL, probe, body, api, candidate.modelId, stream, jsonMode);
        if (result.ok) {
          st.failStreak = 0;
          const answer = result.answer;
          const usage = answer.usage ?? usageOf(fields, answer.text);
          return { answer: { text: answer.text, usage, stopFamily: answer.stopFamily }, modelId: candidate.modelId };
        }
        st.failStreak += 1;
        // A permanent refusal dead-ends the provider at once; a one-off
        // failure needs a second consecutive failure before it does.
        if (result.fatal || st.failStreak >= 2) st.dead = true;
      }
    }
    return null;
  }

  async handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? '/', 'http://localhost');
    if (req.method === 'GET' && url.pathname === '/v1/models') {
      const api: ClientApi = req.headers['x-api-key'] !== undefined ? 'anthropic' : 'openai';
      jsonResponse(res, 200, this.listModels(api) as Record<string, unknown>);
      return;
    }
    if (req.method === 'POST' && (url.pathname === '/v1/chat/completions' || url.pathname === '/v1/messages')) {
      const api: ClientApi = url.pathname === '/v1/messages' ? 'anthropic' : 'openai';
      let text: string;
      try {
        text = await readBody(req);
      } catch (err) {
        errorResponse(res, 400, (err as Error).message, api);
        return;
      }
      let body: unknown;
      try {
        body = JSON.parse(text);
      } catch {
        errorResponse(res, 400, 'request body is not valid JSON', api);
        return;
      }
      if (typeof body !== 'object' || body === null || Array.isArray(body)) {
        errorResponse(res, 400, 'request body must be a JSON object', api);
        return;
      }
      const stream = (body as Record<string, unknown>).stream === true;
      const served = await this.serve(body as Record<string, unknown>, api);
      this.served += 1;
      if (served === null) {
        errorResponse(res, 503, 'no free provider could serve this request', api);
        return;
      }
      const { answer, modelId } = served;
      if (api === 'openai') {
        if (stream) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
          res.end(renderOpenAISse(answer, modelId));
        } else {
          jsonResponse(res, 200, renderOpenAIJson(answer, modelId));
        }
        return;
      }
      if (stream) {
        res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' });
        res.end(renderAnthropicSse(answer, modelId));
      } else {
        jsonResponse(res, 200, renderAnthropicJson(answer, modelId));
      }
      return;
    }
    jsonResponse(res, 404, { error: { message: `unknown route ${req.method ?? ''} ${url.pathname}`, type: 'not_found', code: null } });
  }
}

async function register(envUrl: string, port: number, catalog: CatalogDoc): Promise<void> {
  const res = await fetch(`${envUrl}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`register answered HTTP ${res.status}`);
  const body: unknown = await res.json();
  if (!isRecord(body) || body.ok !== true) throw new Error(`register rejected: ${JSON.stringify(body)}`);
}

async function waitForWork(envUrl: string): Promise<void> {
  for (;;) {
    const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error(`work poll answered HTTP ${res.status}`);
    const body: unknown = await res.json();
    if (isRecord(body) && body.done === true) return;
    await new Promise((resolve) => setTimeout(resolve, WORK_POLL_MS));
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const aggregator = await Aggregator.start();
  if (args.includes('--list')) {
    process.stdout.write(`${JSON.stringify(aggregator.getCatalog())}\n`);
    return;
  }
  const server = createServer((req, res) => {
    aggregator.handle(req, res).catch((err: Error) => {
      if (!res.headersSent) jsonResponse(res, 500, { error: { message: err.message, type: 'server_error', code: null } });
      else res.destroy();
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  const port = typeof address === 'object' && address !== null ? address.port : 0;
  if (ENV_URL !== '') {
    await register(ENV_URL, port, aggregator.getCatalog());
    await waitForWork(ENV_URL);
  }
  server.close();
}

main().catch((err: Error) => {
  process.stderr.write(`${err.stack ?? err.message}\n`);
  process.exit(1);
});
