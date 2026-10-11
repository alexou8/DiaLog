/**
 * Anthropic provider, implemented with plain `fetch` — no `@anthropic-ai/sdk`
 * dependency (per project constraints: no new npm packages).
 *
 * Structured JSON comes from the Messages API's structured outputs:
 * `output_config.format = { type: 'json_schema', schema }`, and the answer is
 * the JSON text block of the response.
 *
 * Why not the older pattern (one tool plus a forced `tool_choice`, with a
 * `temperature`), review H1, 2026-10: the default model, `claude-sonnet-5`,
 * rejects a non-default `temperature` with a 400, and the current models
 * (`claude-opus-5-5`, `claude-sonnet-5-5`) also reject forced `tool_choice`
 * with a 400. Every request failed, and the pipeline silently answered from
 * the local engine. So the request now sends:
 *  - no `temperature` (removed on current models);
 *  - no `tools` / `tool_choice` (structured outputs replace the forced tool);
 *  - no `thinking` (thinking is always on and adaptive for `claude-opus-5-5`;
 *    `effort` is the control), with headroom added to `max_tokens` because
 *    thinking tokens count against it;
 *  - the server-side refusal fallback on the models that support it, so a
 *    safety-classifier false positive on a health question is retried on a
 *    fallback model rather than becoming an error.
 *
 * Never logs request or response bodies (they contain health data) — only
 * status codes, durations and the API's error type.
 */
import {
  AIProviderError,
  type AIProvider,
  type CompletionRequest,
  type CompletionResult,
} from '../provider';
import { postJson, safeErrorType } from './http';
import { adaptSchema, ANTHROPIC_UNSUPPORTED_KEYWORDS } from './schema-compat';

const ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const ANTHROPIC_VERSION = '2023-06-01';
const REQUEST_TIMEOUT_MS = 30_000;

/** Current default model. Override with ANTHROPIC_MODEL. */
export const DEFAULT_ANTHROPIC_MODEL = 'claude-opus-5-5';

/**
 * Thinking counts against `max_tokens`. The structured answers themselves are
 * small, so this headroom is what keeps a long think from truncating the JSON.
 */
const THINKING_HEADROOM_TOKENS = 4096;

/**
 * `low`: the tasks are short reads of a small evidence bundle, the action
 * waits on the reply under a 30s timeout, and the guardrails, not model
 * depth, are what keep the output safe. `claude-opus-5-5` defaults to
 * `medium`, so it is set explicitly.
 */
const EFFORT = 'low';

/**
 * Models that accept `output_config.effort` at `low`. Older ones (Sonnet 4.5,
 * Haiku 4.5) reject the field with a 400, and an ANTHROPIC_MODEL override to
 * one of them made every call fail and silently answer from the local engine.
 * An unlisted id (including a dated snapshot) gets no `effort` and runs at its
 * own default: slower at worst, never a 400. Kept separate from the fallback
 * list below because effort is accepted far more widely than `fallbacks`;
 * gating it on that list would quietly drop `low` on e.g. Opus 4.8.
 */
const EFFORT_MODELS = new Set([
  'claude-fable-5-1',
  'claude-fable-5',
  'claude-opus-5-5',
  'claude-opus-5',
  'claude-opus-4-8',
  'claude-opus-4-7',
  'claude-opus-4-6',
  'claude-opus-4-5',
  'claude-sonnet-5-5',
  'claude-sonnet-5',
  'claude-sonnet-4-6',
  'claude-haiku-5-5',
]);

/** Models that accept `fallbacks: "default"` (beta) on the Claude API. */
const SERVER_SIDE_FALLBACK_MODELS = new Set([
  'claude-fable-5-1',
  'claude-opus-5-5',
  'claude-opus-5',
  'claude-sonnet-5-5',
]);
const SERVER_SIDE_FALLBACK_BETA = 'server-side-fallback-2026-07-01';

export class AnthropicProvider implements AIProvider {
  readonly id = 'anthropic';
  readonly name = 'Anthropic Claude';
  readonly isExternal = true;

  private apiKey(): string | undefined {
    return process.env.ANTHROPIC_API_KEY;
  }

  private model(): string {
    return process.env.ANTHROPIC_MODEL || DEFAULT_ANTHROPIC_MODEL;
  }

  available(): boolean {
    return !!this.apiKey();
  }

  /** The exact request body sent to the Messages API. Exported shape is asserted in tests. */
  buildRequestBody(req: CompletionRequest): Record<string, unknown> {
    const model = this.model();
    return {
      model,
      max_tokens: req.maxTokens + THINKING_HEADROOM_TOKENS,
      system: req.system,
      messages: req.messages.map((m) => ({ role: m.role, content: m.content })),
      output_config: {
        ...(EFFORT_MODELS.has(model) ? { effort: EFFORT } : {}),
        format: {
          type: 'json_schema',
          schema: adaptSchema(req.responseSchema, {
            strip: ANTHROPIC_UNSUPPORTED_KEYWORDS,
            typeArraysToAnyOf: true,
          }),
        },
      },
      ...(SERVER_SIDE_FALLBACK_MODELS.has(model) ? { fallbacks: 'default' } : {}),
    };
  }

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const apiKey = this.apiKey();
    if (!apiKey) {
      throw new AIProviderError(this.id, 'unavailable', 'ANTHROPIC_API_KEY is not configured');
    }

    const headers: Record<string, string> = {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': ANTHROPIC_VERSION,
    };
    if (SERVER_SIDE_FALLBACK_MODELS.has(this.model())) {
      headers['anthropic-beta'] = SERVER_SIDE_FALLBACK_BETA;
    }

    const response = await postJson({
      providerId: this.id,
      displayName: 'Anthropic',
      url: ANTHROPIC_API_URL,
      headers,
      payload: this.buildRequestBody(req),
      timeoutMs: REQUEST_TIMEOUT_MS,
    });

    if (!response.ok) {
      const errorType = safeErrorType(response.body);
      if (errorType) console.error(`[ai:anthropic] errorType=${errorType}`);
      throw new AIProviderError(
        this.id,
        'http_error',
        `Anthropic API returned status ${response.status}`,
        response.status,
      );
    }
    if (!response.parsed) {
      throw new AIProviderError(this.id, 'malformed_json', 'Anthropic response was not valid JSON');
    }

    const message = response.body as AnthropicMessageResponse;
    if (message.stop_reason === 'refusal') {
      throw new AIProviderError(this.id, 'refused', 'Anthropic declined the request');
    }
    if (message.stop_reason === 'max_tokens') {
      throw new AIProviderError(this.id, 'malformed_json', 'Anthropic response was truncated');
    }

    const text = lastTextBlock(message);
    if (text === undefined) {
      throw new AIProviderError(
        this.id,
        'malformed_json',
        'Anthropic response did not contain a structured text block',
      );
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new AIProviderError(
        this.id,
        'malformed_json',
        'Anthropic structured output was not valid JSON',
      );
    }
    return { json: parsed, providerId: this.id, raw: text };
  }
}

interface AnthropicContentBlock {
  type: string;
  text?: string;
}

interface AnthropicMessageResponse {
  content?: AnthropicContentBlock[];
  stop_reason?: string | null;
}

/**
 * The structured JSON is the response's text block. Thinking blocks come
 * first, and after a server-side fallback the turn may also hold a `fallback`
 * marker, so the last text block is the one the serving model produced.
 */
function lastTextBlock(message: AnthropicMessageResponse): string | undefined {
  if (!Array.isArray(message.content)) return undefined;
  for (let i = message.content.length - 1; i >= 0; i -= 1) {
    const block = message.content[i];
    if (block?.type === 'text' && typeof block.text === 'string') return block.text;
  }
  return undefined;
}
