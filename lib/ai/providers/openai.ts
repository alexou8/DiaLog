/**
 * OpenAI provider, implemented with plain `fetch` — no `openai` npm
 * dependency (per project constraints: no new npm packages).
 *
 * Structured JSON output uses Chat Completions `response_format: {
 * type: 'json_schema', json_schema: { ...strict } }`, the documented way to
 * constrain OpenAI output to a JSON Schema. Strict mode rejects string-length
 * keywords, so `schema-compat.ts` strips them (zod still enforces them).
 * `max_completion_tokens` replaces the deprecated `max_tokens` (review L3).
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
import { adaptSchema, OPENAI_UNSUPPORTED_KEYWORDS } from './schema-compat';

const OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions';
const REQUEST_TIMEOUT_MS = 30_000;

/**
 * Reasoning models (gpt-5 family, o-series) reject `temperature` with a 400,
 * so an OPENAI_MODEL override to one made every call fail and silently answer
 * from the local engine. They get no `temperature` and run at their default.
 */
export function isReasoningModel(model: string): boolean {
  return /^(gpt-5|o1|o3|o4)/.test(model.trim().toLowerCase());
}

export class OpenAIProvider implements AIProvider {
  readonly id = 'openai';
  readonly name = 'OpenAI';
  readonly isExternal = true;

  private apiKey(): string | undefined {
    return process.env.OPENAI_API_KEY;
  }

  private model(): string {
    return process.env.OPENAI_MODEL || 'gpt-4o-mini';
  }

  available(): boolean {
    return !!this.apiKey();
  }

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const apiKey = this.apiKey();
    if (!apiKey) {
      throw new AIProviderError(this.id, 'unavailable', 'OPENAI_API_KEY is not configured');
    }

    const response = await postJson({
      providerId: this.id,
      displayName: 'OpenAI',
      url: OPENAI_API_URL,
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${apiKey}`,
      },
      payload: {
        model: this.model(),
        ...(isReasoningModel(this.model()) ? {} : { temperature: req.temperature }),
        max_completion_tokens: req.maxTokens,
        messages: [{ role: 'system', content: req.system }, ...req.messages],
        response_format: {
          type: 'json_schema',
          json_schema: {
            name: 'structured_output',
            strict: true,
            schema: adaptSchema(req.responseSchema, { strip: OPENAI_UNSUPPORTED_KEYWORDS }),
          },
        },
      },
      timeoutMs: REQUEST_TIMEOUT_MS,
    });

    if (!response.ok) {
      const errorType = safeErrorType(response.body);
      if (errorType) console.error(`[ai:openai] errorType=${errorType}`);
      throw new AIProviderError(
        this.id,
        'http_error',
        `OpenAI API returned status ${response.status}`,
        response.status,
      );
    }
    if (!response.parsed) {
      throw new AIProviderError(this.id, 'malformed_json', 'OpenAI response was not valid JSON');
    }

    const message = firstMessage(response.body);
    if (typeof message?.refusal === 'string' && message.refusal.length > 0) {
      throw new AIProviderError(this.id, 'refused', 'OpenAI declined the request');
    }
    const text = message?.content ?? undefined;
    if (text === undefined) {
      throw new AIProviderError(
        this.id,
        'malformed_json',
        'OpenAI response did not contain message content',
      );
    }

    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new AIProviderError(
        this.id,
        'malformed_json',
        'OpenAI message content was not valid JSON',
      );
    }

    return { json: parsed, providerId: this.id, raw: text };
  }
}

interface OpenAIMessage {
  content?: string | null;
  refusal?: string | null;
}

interface OpenAIChatResponse {
  choices?: { message?: OpenAIMessage }[];
}

function firstMessage(body: unknown): OpenAIMessage | undefined {
  if (!body || typeof body !== 'object') return undefined;
  return (body as OpenAIChatResponse).choices?.[0]?.message;
}
