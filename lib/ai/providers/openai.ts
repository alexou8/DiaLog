/**
 * OpenAI provider, implemented with plain `fetch` — no `openai` npm
 * dependency (per project constraints: no new npm packages).
 *
 * Structured JSON output uses Chat Completions `response_format: {
 * type: 'json_schema', json_schema: { ...strict } }`, the documented way to
 * constrain OpenAI output to a JSON Schema.
 *
 * Never logs request or response bodies (they contain health data) — only
 * status codes and durations.
 */
import {
  AIProviderError,
  type AIProvider,
  type CompletionRequest,
  type CompletionResult,
} from '../provider';

const OPENAI_API_URL = 'https://api.openai.com/v1/chat/completions';
const REQUEST_TIMEOUT_MS = 30_000;

export class OpenAIProvider implements AIProvider {
  readonly id = 'openai';
  readonly name = 'OpenAI';
  readonly isExternal = true;

  private apiKey(): string | undefined {
    return process.env.OPENAI_API_KEY;
  }

  private model(): string {
    return process.env.OPENAI_MODEL ?? 'gpt-4o-mini';
  }

  available(): boolean {
    return !!this.apiKey();
  }

  async complete(req: CompletionRequest): Promise<CompletionResult> {
    const apiKey = this.apiKey();
    if (!apiKey) {
      throw new AIProviderError(this.id, 'unavailable', 'OPENAI_API_KEY is not configured');
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const startedAt = Date.now();

    let response: Response;
    try {
      response = await fetch(OPENAI_API_URL, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: this.model(),
          temperature: req.temperature,
          max_tokens: req.maxTokens,
          messages: [{ role: 'system', content: req.system }, ...req.messages],
          response_format: {
            type: 'json_schema',
            json_schema: {
              name: 'structured_output',
              strict: true,
              schema: req.responseSchema,
            },
          },
        }),
        signal: controller.signal,
      });
    } catch (err) {
      const durationMs = Date.now() - startedAt;
      if (err instanceof Error && err.name === 'AbortError') {
        console.error(`[ai:openai] request timed out after ${durationMs}ms`);
        throw new AIProviderError(this.id, 'timeout', 'OpenAI request timed out');
      }
      console.error(`[ai:openai] request failed after ${durationMs}ms`);
      throw new AIProviderError(this.id, 'unknown', 'OpenAI request failed');
    } finally {
      clearTimeout(timeout);
    }

    const durationMs = Date.now() - startedAt;
    console.info(`[ai:openai] status=${response.status} durationMs=${durationMs}`);

    if (!response.ok) {
      throw new AIProviderError(
        this.id,
        'http_error',
        `OpenAI API returned status ${response.status}`,
        response.status,
      );
    }

    let body: unknown;
    try {
      body = await response.json();
    } catch {
      throw new AIProviderError(this.id, 'malformed_json', 'OpenAI response was not valid JSON');
    }

    const text = extractMessageContent(body);
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

interface OpenAIChoice {
  message?: { content?: string | null };
}

interface OpenAIChatResponse {
  choices?: OpenAIChoice[];
}

function extractMessageContent(body: unknown): string | undefined {
  if (!body || typeof body !== 'object') return undefined;
  const choices = (body as OpenAIChatResponse).choices;
  const content = choices?.[0]?.message?.content;
  return content ?? undefined;
}
