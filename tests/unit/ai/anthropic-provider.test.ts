import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { AnthropicProvider, DEFAULT_ANTHROPIC_MODEL } from '@/lib/ai/providers/anthropic';
import { AssistantAnswerJsonSchema } from '@/lib/ai/schemas';

const ORIGINAL_FETCH = global.fetch;
const ORIGINAL_KEY = process.env.ANTHROPIC_API_KEY;
const ORIGINAL_MODEL = process.env.ANTHROPIC_MODEL;

beforeEach(() => {
  process.env.ANTHROPIC_API_KEY = 'sk-ant-test-key';
  delete process.env.ANTHROPIC_MODEL;
});

afterEach(() => {
  global.fetch = ORIGINAL_FETCH;
  if (ORIGINAL_KEY === undefined) delete process.env.ANTHROPIC_API_KEY;
  else process.env.ANTHROPIC_API_KEY = ORIGINAL_KEY;
  if (ORIGINAL_MODEL === undefined) delete process.env.ANTHROPIC_MODEL;
  else process.env.ANTHROPIC_MODEL = ORIGINAL_MODEL;
  vi.restoreAllMocks();
  vi.useRealTimers();
});

const baseReq = {
  system: 'system prompt',
  messages: [{ role: 'user' as const, content: 'hello' }],
  responseSchema: { type: 'object', additionalProperties: false, properties: {}, required: [] },
  maxTokens: 100,
  temperature: 0.2,
};

/** A Messages API response carrying `payload` as the structured-output text block. */
function messageResponse(payload: unknown, extra: Record<string, unknown> = {}): Response {
  return new Response(
    JSON.stringify({
      stop_reason: 'end_turn',
      content: [
        { type: 'thinking', thinking: '', signature: 'sig' },
        { type: 'text', text: JSON.stringify(payload) },
      ],
      ...extra,
    }),
    { status: 200 },
  );
}

function sentRequest(fetchMock: ReturnType<typeof vi.fn>): {
  url: unknown;
  headers: Record<string, string>;
  body: Record<string, unknown>;
} {
  const [url, init] = fetchMock.mock.calls[0]!;
  const request = init as RequestInit;
  return {
    url,
    headers: request.headers as Record<string, string>,
    body: JSON.parse(request.body as string),
  };
}

describe('AnthropicProvider', () => {
  it('available() is false without an API key', () => {
    delete process.env.ANTHROPIC_API_KEY;
    const provider = new AnthropicProvider();
    expect(provider.available()).toBe(false);
  });

  it('available() is true with an API key', () => {
    const provider = new AnthropicProvider();
    expect(provider.available()).toBe(true);
  });

  it('defaults to the current recommended model', () => {
    expect(DEFAULT_ANTHROPIC_MODEL).toBe('claude-opus-5-5');
  });

  it('sends exactly the request body current models accept', async () => {
    const fetchMock = vi.fn().mockResolvedValue(messageResponse({ ok: true }));
    global.fetch = fetchMock as unknown as typeof fetch;

    await new AnthropicProvider().complete({
      ...baseReq,
      responseSchema: AssistantAnswerJsonSchema,
    });

    const { url, headers, body } = sentRequest(fetchMock);
    expect(url).toBe('https://api.anthropic.com/v1/messages');
    expect(headers['anthropic-version']).toBe('2023-06-01');
    expect(headers['anthropic-beta']).toBe('server-side-fallback-2026-07-01');
    expect(Object.keys(body).sort()).toEqual(
      ['fallbacks', 'max_tokens', 'messages', 'model', 'output_config', 'system'].sort(),
    );
    expect(body).toMatchObject({
      model: 'claude-opus-5-5',
      system: 'system prompt',
      messages: [{ role: 'user', content: 'hello' }],
      fallbacks: 'default',
      output_config: { effort: 'low', format: { type: 'json_schema' } },
    });
    // Thinking is always on for this model and counts against max_tokens.
    expect(body.max_tokens).toBeGreaterThan(baseReq.maxTokens);
    // Rejected with a 400 by current models: never sent.
    expect(body).not.toHaveProperty('temperature');
    expect(body).not.toHaveProperty('tools');
    expect(body).not.toHaveProperty('tool_choice');
    expect(body).not.toHaveProperty('thinking');
    // Structured outputs reject length constraints; they are stripped here and enforced by zod.
    expect(JSON.stringify(body.output_config)).not.toMatch(/maxLength|minLength|maxItems/);
  });

  it('omits the fallback beta for a model that does not support it', async () => {
    process.env.ANTHROPIC_MODEL = 'claude-haiku-5-5';
    const fetchMock = vi.fn().mockResolvedValue(messageResponse({ ok: true }));
    global.fetch = fetchMock as unknown as typeof fetch;

    await new AnthropicProvider().complete(baseReq);

    const { headers, body } = sentRequest(fetchMock);
    expect(body.model).toBe('claude-haiku-5-5');
    expect(body).not.toHaveProperty('fallbacks');
    expect(headers).not.toHaveProperty('anthropic-beta');
  });

  it('omits effort for a model that rejects it, keeping structured output', async () => {
    // Sonnet 4.5 / Haiku 4.5 answer a request carrying `effort` with a 400.
    for (const model of ['claude-sonnet-4-5', 'claude-haiku-4-5']) {
      process.env.ANTHROPIC_MODEL = model;
      const fetchMock = vi.fn().mockResolvedValue(messageResponse({ ok: true }));
      global.fetch = fetchMock as unknown as typeof fetch;

      await new AnthropicProvider().complete(baseReq);

      const { body } = sentRequest(fetchMock);
      expect(body.model).toBe(model);
      const outputConfig = body.output_config as Record<string, unknown>;
      expect(outputConfig).not.toHaveProperty('effort');
      expect(outputConfig).toMatchObject({ format: { type: 'json_schema' } });
      expect(body).not.toHaveProperty('fallbacks');
    }
  });

  it('keeps effort on a model that accepts it but has no server-side fallback', async () => {
    process.env.ANTHROPIC_MODEL = 'claude-opus-4-8';
    const fetchMock = vi.fn().mockResolvedValue(messageResponse({ ok: true }));
    global.fetch = fetchMock as unknown as typeof fetch;

    await new AnthropicProvider().complete(baseReq);

    const { body } = sentRequest(fetchMock);
    expect(body.output_config).toMatchObject({ effort: 'low' });
    expect(body).not.toHaveProperty('fallbacks');
  });

  it('extracts structured JSON from the text block', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(
        messageResponse({ shortAnswer: 'hi', ok: true }),
      ) as unknown as typeof fetch;
    const result = await new AnthropicProvider().complete(baseReq);
    expect(result.json).toEqual({ shortAnswer: 'hi', ok: true });
    expect(result.providerId).toBe('anthropic');
  });

  it('throws AIProviderError with kind http_error on a non-200 response', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(new Response('', { status: 500 })) as unknown as typeof fetch;
    const provider = new AnthropicProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({
      name: 'AIProviderError',
      kind: 'http_error',
      status: 500,
    });
  });

  it('logs the API error type, never the error message, on a 400', async () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    vi.spyOn(console, 'info').mockImplementation(() => {});
    global.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          type: 'error',
          error: { type: 'invalid_request_error', message: 'echoes hello' },
        }),
        { status: 400 },
      ),
    ) as unknown as typeof fetch;
    await expect(new AnthropicProvider().complete(baseReq)).rejects.toMatchObject({
      kind: 'http_error',
      status: 400,
    });
    const logged = errorSpy.mock.calls.flat().join(' ');
    expect(logged).toContain('errorType=invalid_request_error');
    expect(logged).not.toContain('echoes hello');
  });

  it('throws AIProviderError with kind malformed_json when body is not valid JSON', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(new Response('not json', { status: 200 })) as unknown as typeof fetch;
    const provider = new AnthropicProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({
      kind: 'malformed_json',
    });
  });

  it('throws AIProviderError with kind malformed_json when there is no text block', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ stop_reason: 'end_turn', content: [] }), { status: 200 }),
      );
    await expect(new AnthropicProvider().complete(baseReq)).rejects.toMatchObject({
      kind: 'malformed_json',
    });
  });

  it('throws malformed_json on a truncated (max_tokens) response', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(
        messageResponse({ partial: true }, { stop_reason: 'max_tokens' }),
      ) as unknown as typeof fetch;
    await expect(new AnthropicProvider().complete(baseReq)).rejects.toMatchObject({
      kind: 'malformed_json',
    });
  });

  it('throws kind refused when the model declines', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(
        messageResponse({}, { stop_reason: 'refusal' }),
      ) as unknown as typeof fetch;
    await expect(new AnthropicProvider().complete(baseReq)).rejects.toMatchObject({
      kind: 'refused',
    });
  });

  it('throws AIProviderError with kind timeout when the request aborts', async () => {
    vi.useFakeTimers();
    global.fetch = vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          const err = new Error('aborted');
          err.name = 'AbortError';
          reject(err);
        });
      });
    }) as unknown as typeof fetch;

    const provider = new AnthropicProvider();
    const promise = provider.complete(baseReq);
    const assertion = expect(promise).rejects.toMatchObject({ kind: 'timeout' });
    await vi.advanceTimersByTimeAsync(30_001);
    await assertion;
  });

  it('times out when headers arrive but the body stalls', async () => {
    vi.useFakeTimers();
    global.fetch = vi.fn().mockImplementation((_url: string, init?: RequestInit) => {
      const stalledBody = new ReadableStream({
        start(controller) {
          init?.signal?.addEventListener('abort', () => {
            const err = new Error('aborted');
            err.name = 'AbortError';
            controller.error(err);
          });
        },
      });
      return Promise.resolve(new Response(stalledBody, { status: 200 }));
    }) as unknown as typeof fetch;

    const promise = new AnthropicProvider().complete(baseReq);
    const assertion = expect(promise).rejects.toMatchObject({ kind: 'timeout' });
    await vi.advanceTimersByTimeAsync(30_001);
    await assertion;
  });

  it('throws AIProviderError with kind unavailable when no API key is configured', async () => {
    delete process.env.ANTHROPIC_API_KEY;
    const provider = new AnthropicProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({ kind: 'unavailable' });
  });

  it('never logs request or response bodies', async () => {
    const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    global.fetch = vi
      .fn()
      .mockResolvedValue(
        messageResponse({ secretHealthValue: 'do-not-log-me' }),
      ) as unknown as typeof fetch;

    const provider = new AnthropicProvider();
    await provider.complete(baseReq);

    const allLoggedText = [...infoSpy.mock.calls, ...errorSpy.mock.calls].flat().join(' ');
    expect(allLoggedText).not.toContain('secretHealthValue');
    expect(allLoggedText).not.toContain('do-not-log-me');
    expect(allLoggedText).not.toContain('hello'); // the request message content
  });
});
