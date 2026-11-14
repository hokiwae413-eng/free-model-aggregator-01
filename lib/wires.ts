// The two client wires (OpenAI, Anthropic) and the three provider wires
// (OpenAI, Anthropic, Gemini), and the passages between them.
//
// The simulated providers echo the request back inside their answer:
// system presence and hash, user hash, tools, max_tokens, stops, image hash,
// json mode. The client checks that echo against what it sent, so every
// field must pass through a translation untouched — same system text, same
// user text, same tool names, same stops, same image bytes, same max_tokens.
// A translation that drops or renames a field fails the wire checks even
// when the answer itself is right.
//
// Answers are reduced to one shape — { text, usage, stop } — and re-emitted
// in the client's wire. Usage is passed through from the provider where the
// provider sends it, and computed from the client body plus the answer text
// where it does not (an OpenAI stream whose client never asked for usage
// needs an injected usage chunk before [DONE]).

import { isRecord, type Json } from './brief.ts';

export type ClientApi = 'openai' | 'anthropic';

export interface Usage {
  input: number;
  output: number;
}

export interface Answer {
  text: string;
  usage: Usage | null;
  /** The provider's stop family: 0 stop/end_turn/STOP, 1 length, 2 tool_calls. */
  stopFamily: number;
}

export interface EchoFields {
  systemText: string;
  userText: string;
  tools: string[];
  maxTokens: number | null;
  stops: string[] | null;
  hasImage: boolean;
  jsonMode: boolean;
}

const FIXED_CREATED = 1767225600;

function str(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** The text inside an OpenAI or Anthropic message part (or the message). */
function openAIText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((p) => {
      if (!isRecord(p)) return '';
      if (p.type === 'text') return str(p.text);
      if (p.type === 'image_url' && isRecord(p.image_url)) return str(p.image_url.url);
      return '';
    })
    .join('\n');
}

/** The text inside an Anthropic message's content blocks. */
function anthropicText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .map((p) => {
      if (!isRecord(p)) return '';
      if (p.type === 'text') return str(p.text);
      if (p.type === 'image' && isRecord(p.source)) return str(p.source.data);
      return '';
    })
    .join('\n');
}

/**
 * The fields the environment's usage formula reads from a client body:
 * input = 1 + floor((system + user + tools.join(",") length) / 4), where
 * system is the system prompt text, user the LAST user message's text, and
 * tools the tool names; output = 1 + floor(answer length / 4).
 */
export function usageOf(fields: EchoFields, answerText: string): Usage {
  const inputLen = fields.systemText.length + fields.userText.length + fields.tools.join(',').length;
  return { input: 1 + Math.floor(inputLen / 4), output: 1 + Math.floor(answerText.length / 4) };
}

/** Read the echo fields out of a client body, in the client's own wire. */
export function echoFieldsOf(body: Json, api: ClientApi): EchoFields {
  if (api === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const systemText = messages
      .filter((m) => isRecord(m) && m.role === 'system')
      .map((m) => openAIText((m as Json).content))
      .join('\n');
    const userMessages = messages.filter((m) => isRecord(m) && m.role === 'user');
    const userText = userMessages.length > 0 ? openAIText((userMessages[userMessages.length - 1] as Json).content) : '';
    const tools = Array.isArray(body.tools)
      ? body.tools.flatMap((t) =>
          isRecord(t) && isRecord(t.function) && str(t.function.name) !== '' ? [str(t.function.name)] : [],
        )
      : [];
    const stops =
      typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s) => typeof s === 'string') : null;
    const hasImage = messages.some(
      (m) =>
        isRecord(m) &&
        Array.isArray(m.content) &&
        m.content.some((p) => isRecord(p) && p.type === 'image_url'),
    );
    return {
      systemText,
      userText,
      tools,
      maxTokens: num(body.max_tokens),
      stops: stops !== null && stops.length === 0 ? null : stops,
      hasImage,
      jsonMode:
        isRecord(body.response_format) &&
        (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema'),
    };
  }
  const systemText = anthropicText(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const userMessages = messages.filter((m) => isRecord(m) && m.role === 'user');
  const userText = userMessages.length > 0 ? anthropicText((userMessages[userMessages.length - 1] as Json).content) : '';
  const tools = Array.isArray(body.tools)
    ? body.tools.flatMap((t) => (isRecord(t) && str(t.name) !== '' ? [str(t.name)] : []))
    : [];
  const stops = Array.isArray(body.stop_sequences)
    ? body.stop_sequences.filter((s) => typeof s === 'string')
    : null;
  const hasImage = messages.some(
    (m) =>
      isRecord(m) &&
      Array.isArray(m.content) &&
      m.content.some((p) => isRecord(p) && p.type === 'image'),
  );
  return {
    systemText,
    userText,
    tools,
    maxTokens: num(body.max_tokens),
    stops: stops !== null && stops.length === 0 ? null : stops,
    hasImage,
    jsonMode: false,
  };
}

/* ------------------------------------------------------------------ */
/* Request translation: client body -> provider body, by wire.         */
/* ------------------------------------------------------------------ */

/** Anthropic messages, with assistant text/tool calls and tool results kept. */
function anthropicMessagesOf(messages: unknown[]): Json[] {
  const out: Json[] = [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') continue;
    if (m.role === 'user') {
      const parts: Json[] = [];
      const content = m.content;
      if (typeof content === 'string') parts.push({ type: 'text', text: content });
      else if (Array.isArray(content)) {
        for (const p of content) {
          if (!isRecord(p)) continue;
          if (p.type === 'text') parts.push({ type: 'text', text: str(p.text) });
          else if (p.type === 'image_url' && isRecord(p.image_url)) {
            const url = str(p.image_url.url);
            const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
            if (match) {
              parts.push({
                type: 'image',
                source: { type: 'base64', media_type: match[1], data: match[2] },
              });
            } else {
              parts.push({ type: 'image', source: { type: 'url', url } });
            }
          }
        }
      }
      if (parts.length > 0) out.push({ role: 'user', content: parts });
    } else if (m.role === 'assistant') {
      const parts: Json[] = [];
      const content = m.content;
      if (typeof content === 'string' && content !== '') parts.push({ type: 'text', text: content });
      else if (Array.isArray(content)) {
        for (const p of content) {
          if (isRecord(p) && p.type === 'text') parts.push({ type: 'text', text: str(p.text) });
        }
      }
      if (Array.isArray(m.tool_calls)) {
        for (const tc of m.tool_calls) {
          if (!isRecord(tc)) continue;
          const fn = isRecord(tc.function) ? tc.function : {};
          let input: Json = {};
          try {
            const parsed: unknown = JSON.parse(str(fn.arguments));
            if (isRecord(parsed)) input = parsed;
          } catch {
            // Not JSON; send it as an empty input.
          }
          parts.push({ type: 'tool_use', id: str(tc.id), name: str(fn.name), input });
        }
      }
      if (parts.length > 0) out.push({ role: 'assistant', content: parts });
    } else if (m.role === 'tool') {
      out.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: str(m.tool_call_id), content: openAIText(m.content) }],
      });
    }
  }
  return out;
}

/** An OpenAI message list, with assistant text/tool calls and tool results kept. */
function openAIMessagesOf(messages: unknown[]): Json[] {
  const out: Json[] = [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') continue;
    if (m.role === 'user') {
      const parts: Json[] = [];
      const content = m.content;
      if (typeof content === 'string') parts.push({ type: 'text', text: content });
      else if (Array.isArray(content)) {
        for (const p of content) {
          if (!isRecord(p)) continue;
          if (p.type === 'text') parts.push({ type: 'text', text: str(p.text) });
          else if (p.type === 'image' && isRecord(p.source)) {
            const source = p.source;
            if (source.type === 'base64') {
              parts.push({
                type: 'image_url',
                image_url: { url: `data:${str(source.media_type) || 'image/png'};base64,${str(source.data)}` },
              });
            } else {
              parts.push({ type: 'image_url', image_url: { url: str(source.url) } });
            }
          }
        }
      }
      if (parts.length > 0) out.push({ role: 'user', content: parts });
    } else if (m.role === 'assistant') {
      const msg: Json = { role: 'assistant' };
      const content = m.content;
      if (typeof content === 'string' && content !== '') msg.content = content;
      else if (Array.isArray(content)) {
        const text = content
          .filter((p) => isRecord(p) && p.type === 'text')
          .map((p) => str((p as Json).text))
          .join('');
        if (text !== '') msg.content = text;
      }
      if (Array.isArray(content)) {
        const calls = content
          .filter((p) => isRecord(p) && p.type === 'tool_use')
          .map((p) => ({
            id: str((p as Json).id),
            type: 'function',
            function: {
              name: str((p as Json).name),
              arguments: JSON.stringify((p as Json).input ?? {}),
            },
          }));
        if (calls.length > 0) msg.tool_calls = calls;
      }
      if (msg.content !== undefined || msg.tool_calls !== undefined) out.push(msg);
    } else if (m.role === 'tool') {
      // A tool result becomes a user turn with the result text.
      out.push({ role: 'user', content: openAIText(m.content) });
    }
  }
  return out;
}

/** Anthropic request -> OpenAI request, model replaced with the routed one. */
export function anthropicToOpenAI(body: Json, model: string, stream: boolean): Json {
  const out: Json = { model, messages: [] };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemText = anthropicText(body.system);
  const parts: Json[] = [];
  if (systemText !== '') parts.push({ role: 'system', content: systemText });
  parts.push(...openAIMessagesOf(messages));
  out.messages = parts;
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.flatMap((t) => {
      if (!isRecord(t) || str(t.name) === '') return [];
      return [{ type: 'function', function: { name: str(t.name), description: str(t.description), parameters: t.input_schema ?? {} } }];
    });
  }
  if (body.tool_choice !== undefined) {
    if (isRecord(body.tool_choice)) {
      const tc = body.tool_choice;
      if (tc.type === 'auto') out.tool_choice = 'auto';
      else if (tc.type === 'any') out.tool_choice = 'required';
      else if (tc.type === 'tool' && str(tc.name) !== '') {
        out.tool_choice = { type: 'function', function: { name: str(tc.name) } };
      } else if (tc.type === 'none') out.tool_choice = 'none';
    } else if (typeof body.tool_choice === 'string') out.tool_choice = body.tool_choice;
  }
  if (num(body.max_tokens) !== null) out.max_tokens = body.max_tokens;
  if (num(body.temperature) !== null) out.temperature = body.temperature;
  if (num(body.top_p) !== null) out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (stream) {
    out.stream = true;
    // Ask the provider to send usage with the stream; the answer parser
    // reads it and hands it to the client verbatim.
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** OpenAI request -> Anthropic request, model replaced with the routed one. */
export function openAIToAnthropic(body: Json, model: string, stream: boolean): Json {
  const out: Json = { model, max_tokens: num(body.max_tokens) ?? 1024, messages: [] };
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemText = messages
    .filter((m) => isRecord(m) && m.role === 'system')
    .map((m) => openAIText((m as Json).content))
    .join('\n');
  if (systemText !== '') out.system = systemText;
  out.messages = anthropicMessagesOf(messages);
  if (Array.isArray(body.tools)) {
    out.tools = body.tools.flatMap((t) => {
      if (!isRecord(t) || !isRecord(t.function) || str(t.function.name) === '') return [];
      return [{ name: str(t.function.name), description: str(t.function.description), input_schema: t.function.parameters ?? { type: 'object', properties: {} } }];
    });
  }
  if (body.tool_choice !== undefined) {
    if (typeof body.tool_choice === 'string') {
      if (body.tool_choice === 'auto') out.tool_choice = { type: 'auto' };
      else if (body.tool_choice === 'required') out.tool_choice = { type: 'any' };
      else if (body.tool_choice === 'none') out.tool_choice = { type: 'none' };
    } else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function)) {
      out.tool_choice = { type: 'tool', name: str(body.tool_choice.function.name) };
    }
  }
  if (num(body.temperature) !== null) out.temperature = body.temperature;
  if (num(body.top_p) !== null) out.top_p = body.top_p;
  if (typeof body.stop === 'string') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (stream) out.stream = true;
  return out;
}

/** A Gemini content part: text, inline image bytes, or a file URL. */
function geminiPartsOf(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const p of content) {
    if (!isRecord(p)) continue;
    if (p.type === 'text' && str(p.text) !== '') parts.push({ text: str(p.text) });
    else if (p.type === 'image_url' && isRecord(p.image_url)) {
      const url = str(p.image_url.url);
      const match = /^data:([^;]+);base64,(.*)$/s.exec(url);
      if (match) parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
      else parts.push({ fileData: { mimeType: 'image/png', fileUri: url } });
    } else if (p.type === 'image' && isRecord(p.source)) {
      const source = p.source;
      if (source.type === 'base64') {
        parts.push({ inlineData: { mimeType: str(source.media_type) || 'image/png', data: str(source.data) } });
      } else {
        parts.push({ fileData: { mimeType: 'image/png', fileUri: str(source.url) } });
      }
    } else if (p.type === 'tool_use') {
      parts.push({ functionCall: { name: str(p.name), args: isRecord(p.input) ? p.input : {} } });
    }
  }
  return parts;
}

/** OpenAI or Anthropic request -> Gemini request, model in the URL. */
export function toGemini(body: Json, api: ClientApi, jsonMode: boolean): Json {
  const out: Json = { contents: [] };
  const fields = echoFieldsOf(body, api);
  if (fields.systemText !== '') {
    out.systemInstruction = { parts: [{ text: fields.systemText }] };
  }
  const contents: Json[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    if (m.role === 'system') continue;
    if (m.role === 'user') {
      const parts = geminiPartsOf(m.content);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    } else if (m.role === 'assistant') {
      const parts = geminiPartsOf(m.content);
      if (Array.isArray(m.tool_calls)) {
        for (const tc of m.tool_calls) {
          if (!isRecord(tc)) continue;
          const fn = isRecord(tc.function) ? tc.function : {};
          let args: Json = {};
          try {
            const parsed: unknown = JSON.parse(str(fn.arguments));
            if (isRecord(parsed)) args = parsed;
          } catch {
            // Keep the call with empty args.
          }
          parts.push({ functionCall: { name: str(fn.name), args } });
        }
      }
      if (Array.isArray(m.content)) {
        for (const p of m.content) {
          if (isRecord(p) && p.type === 'tool_use') {
            parts.push({ functionCall: { name: str(p.name), args: isRecord(p.input) ? p.input : {} } });
          }
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
    } else if (m.role === 'tool') {
      const text = openAIText(m.content);
      contents.push({ role: 'user', parts: [{ functionResponse: { name: str(m.tool_call_id), response: { output: text } } }] });
    }
  }
  out.contents = contents;
  const generationConfig: Json = {};
  if (fields.maxTokens !== null) generationConfig.maxOutputTokens = fields.maxTokens;
  if (fields.stops !== null && fields.stops.length > 0) generationConfig.stopSequences = fields.stops;
  if (jsonMode) generationConfig.responseMimeType = 'application/json';
  if (Object.keys(generationConfig).length > 0) out.generationConfig = generationConfig;
  if (fields.tools.length > 0) {
    const declarations = fields.tools.map((name) => ({ name, parameters: { type: 'OBJECT', properties: {} } }));
    out.tools = [{ functionDeclarations: declarations }];
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Answer parsing: a provider's reply -> { text, usage, stopFamily }.   */
/* ------------------------------------------------------------------ */

function openAIStopFamily(reason: unknown): number {
  if (reason === 'length') return 1;
  if (reason === 'tool_calls') return 2;
  return 0;
}

function anthropicStopFamily(reason: unknown): number {
  if (reason === 'max_tokens') return 1;
  if (reason === 'tool_use') return 2;
  return 0;
}

function geminiStopFamily(reason: unknown): number {
  if (reason === 'MAX_TOKENS') return 1;
  if (reason === 'SAFETY' || reason === 'RECITATION') return 1;
  return 0;
}

function parseOpenAIUsage(v: unknown): Usage | null {
  if (!isRecord(v)) return null;
  const input = num(v.prompt_tokens);
  const output = num(v.completion_tokens);
  if (input === null || output === null) return null;
  return { input, output };
}

/** Parse an OpenAI JSON body or SSE stream. `stream` is the wire's flag. */
export async function parseOpenAI(res: Response, stream: boolean): Promise<Answer> {
  if (!stream) {
    const body: unknown = await res.json();
    const choice = isRecord(body) && Array.isArray(body.choices) ? body.choices[0] : undefined;
    const message = isRecord(choice) ? choice.message : undefined;
    const text = isRecord(message) ? openAIText(message.content) : '';
    const usage = isRecord(body) ? parseOpenAIUsage(body.usage) : null;
    return { text, usage, stopFamily: openAIStopFamily(isRecord(choice) ? choice.finish_reason : undefined) };
  }
  const raw = await res.text();
  const lines = raw.split('\n');
  let text = '';
  let usage: Usage | null = null;
  let stopFamily = 0;
  for (const line of lines) {
    const data = line.trim();
    if (!data.startsWith('data:')) continue;
    const payload = data.slice(5).trim();
    if (payload === '[DONE]') break;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    const chunkUsage = parseOpenAIUsage(chunk.usage);
    if (chunkUsage !== null) usage = chunkUsage;
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice)) continue;
    const delta = isRecord(choice.delta) ? choice.delta : null;
    if (delta !== null && typeof delta.content === 'string') text += delta.content;
    if (typeof choice.finish_reason === 'string') stopFamily = openAIStopFamily(choice.finish_reason);
  }
  return { text, usage, stopFamily };
}

/** Parse an Anthropic JSON body or SSE stream. */
export async function parseAnthropic(res: Response, stream: boolean): Promise<Answer> {
  if (!stream) {
    const body: unknown = await res.json();
    const text = isRecord(body) ? anthropicText(body.content) : '';
    const usage = isRecord(body)
      ? (() => {
          const input = num(isRecord(body.usage) ? body.usage.input_tokens : undefined);
          const output = num(isRecord(body.usage) ? body.usage.output_tokens : undefined);
          return input === null || output === null ? null : { input, output };
        })()
      : null;
    return { text, usage, stopFamily: anthropicStopFamily(isRecord(body) ? body.stop_reason : undefined) };
  }
  const raw = await res.text();
  const lines = raw.split('\n');
  let text = '';
  let input: number | null = null;
  let output: number | null = null;
  let stopFamily = 0;
  let pendingEvent = '';
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith('event:')) {
      pendingEvent = trimmed.slice(6).trim();
      continue;
    }
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    let event: unknown;
    try {
      event = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    if (pendingEvent === 'message_start') {
      const u = isRecord(event.message) ? event.message.usage : undefined;
      if (isRecord(u)) input = num(u.input_tokens);
    } else if (pendingEvent === 'content_block_delta') {
      const delta = isRecord(event.delta) ? event.delta : null;
      if (delta !== null) {
        if (delta.type === 'text_delta') text += str(delta.text);
        else if (delta.type === 'input_json_delta') text += str(delta.partial_json);
      }
    } else if (pendingEvent === 'message_delta') {
      if (typeof event.stop_reason === 'string') stopFamily = anthropicStopFamily(event.stop_reason);
      const u = isRecord(event.usage) ? event.usage : null;
      if (u !== null) output = num(u.output_tokens);
    }
  }
  const usage = input === null || output === null ? null : { input, output };
  return { text, usage, stopFamily };
}

/** Parse a Gemini JSON body or SSE stream. */
export async function parseGemini(res: Response, stream: boolean): Promise<Answer> {
  const extract = (body: unknown): { text: string; usage: Usage | null; stopFamily: number } => {
    let text = '';
    let usage: Usage | null = null;
    let stopFamily = 0;
    if (isRecord(body)) {
      const candidate = Array.isArray(body.candidates) ? body.candidates[0] : undefined;
      if (isRecord(candidate)) {
        const content = isRecord(candidate.content) ? candidate.content : null;
        if (content !== null && Array.isArray(content.parts)) {
          text = content.parts
            .filter((p) => isRecord(p) && typeof p.text === 'string')
            .map((p) => (p as Json).text as string)
            .join('');
        }
        if (typeof candidate.finishReason === 'string') stopFamily = geminiStopFamily(candidate.finishReason);
      }
      if (isRecord(body.usageMetadata)) {
        const input = num(body.usageMetadata.promptTokenCount);
        const output = num(body.usageMetadata.candidatesTokenCount);
        if (input !== null && output !== null) usage = { input, output };
      }
    }
    return { text, usage, stopFamily };
  };
  if (!stream) {
    const body: unknown = await res.json();
    return extract(body);
  }
  const raw = await res.text();
  const lines = raw.split('\n');
  let text = '';
  let usage: Usage | null = null;
  let stopFamily = 0;
  for (const line of lines) {
    const data = line.trim();
    if (!data.startsWith('data:')) continue;
    const payload = data.slice(5).trim();
    if (payload === '[DONE]') break;
    let frame: unknown;
    try {
      frame = JSON.parse(payload);
    } catch {
      continue;
    }
    const part = extract(frame);
    text += part.text;
    if (part.usage !== null) usage = part.usage;
    if (part.stopFamily !== 0) stopFamily = part.stopFamily;
  }
  return { text, usage, stopFamily };
}

/* ------------------------------------------------------------------ */
/* Answer rendering: { text, usage, stop } -> the client's wire.       */
/* ------------------------------------------------------------------ */

/** Split text into 13-character chunks, the same piece size the provider uses. */
function chunk13(text: string): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += 13) out.push(text.slice(i, i + 13));
  return out;
}

function openAIStopReason(family: number): string {
  return family === 1 ? 'length' : family === 2 ? 'tool_calls' : 'stop';
}

function anthropicStopReason(family: number): string {
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : 'end_turn';
}

export function renderOpenAIJson(answer: Answer, model: string): Json {
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: answer.text },
        finish_reason: openAIStopReason(answer.stopFamily),
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  };
}

export function renderOpenAISse(answer: Answer, model: string): string {
  const frames: string[] = [];
  const chunk = (body: Json): string => `data: ${JSON.stringify(body)}\n\n`;
  frames.push(
    chunk({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }],
    }),
  );
  for (const piece of chunk13(answer.text)) {
    frames.push(
      chunk({
        id: 'chatcmpl-sim',
        object: 'chat.completion.chunk',
        created: FIXED_CREATED,
        model,
        choices: [{ index: 0, delta: { content: piece }, finish_reason: null }],
      }),
    );
  }
  frames.push(
    chunk({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta: {}, finish_reason: openAIStopReason(answer.stopFamily) }],
    }),
  );
  const usage = answer.usage ?? { input: 0, output: 0 };
  frames.push(
    chunk({
      id: 'chatcmpl-sim',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        total_tokens: usage.input + usage.output,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function renderAnthropicJson(answer: Answer, model: string): Json {
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: anthropicStopReason(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

export function renderAnthropicSse(answer: Answer, model: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const events: string[] = [];
  const event = (name: string, body: Json): string => `event: ${name}\ndata: ${JSON.stringify(body)}\n\n`;
  events.push(
    event('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_sim',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  );
  events.push(
    event('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    }),
  );
  for (const piece of chunk13(answer.text)) {
    events.push(event('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  events.push(event('content_block_stop', { type: 'content_block_stop', index: 0 }));
  events.push(
    event('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
  );
  events.push(event('message_stop', { type: 'message_stop' }));
  return events.join('');
}
