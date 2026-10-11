import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OpenAIProvider, isReasoningModel } from '@/lib/ai/providers/openai';

const ORIGINAL_FETCH = global.fetch;
const ORIGINAL_KEY = process.env.OPENAI_API_KEY;
const ORIGINAL_MODEL = process.env.OPENAI_MODEL;

beforeEach(() => {
  process.env.OPENAI_API_KEY = 'sk-oa-test-key';
});

afterEach(() => {
  global.fetch = ORIGINAL_FETCH;
  if (ORIGINAL_KEY === undefined) delete process.env.OPENAI_API_KEY;
  else process.env.OPENAI_API_KEY = ORIGINAL_KEY;
  if (ORIGINAL_MODEL === undefined) delete process.env.OPENAI_MODEL;
  else process.env.OPENAI_MODEL = ORIGINAL_MODEL;
  vi.restoreAllMocks();
});

const baseReq = {
  system: 'system prompt',
  messages: [{ role: 'user' as const, content: 'hello' }],
  responseSchema: { type: 'object', properties: {} },
  maxTokens: 100,
};

describe('OpenAIProvider', () => {
  it('available() is false without an API key', () => {
    delete process.env.OPENAI_API_KEY;
    const provider = new OpenAIProvider();
    expect(provider.available()).toBe(false);
  });

  it('extracts and parses structured JSON from message content', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [{ message: { content: JSON.stringify({ shortAnswer: 'hi', ok: true }) } }],
        }),
        { status: 200 },
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const provider = new OpenAIProvider();
    const result = await provider.complete(baseReq);
    expect(result.json).toEqual({ shortAnswer: 'hi', ok: true });
    expect(result.providerId).toBe('openai');
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe('https://api.openai.com/v1/chat/completions');
    const sentBody = JSON.parse((init as RequestInit).body as string);
    expect(sentBody.response_format.type).toBe('json_schema');
    expect(sentBody.response_format.json_schema.strict).toBe(true);
    // `max_tokens` is deprecated in favour of `max_completion_tokens`.
    expect(sentBody.max_completion_tokens).toBe(100);
    expect(sentBody).not.toHaveProperty('max_tokens');
  });

  it('omits temperature for reasoning models, which reject it with a 400', async () => {
    const reply = () =>
      new Response(JSON.stringify({ choices: [{ message: { content: '{}' } }] }), { status: 200 });
    for (const [model, sendsTemperature] of [
      ['gpt-5-mini', false],
      ['o3', false],
      ['o4-mini', false],
      ['o1-preview', false],
      ['gpt-4o-mini', true],
    ] as const) {
      process.env.OPENAI_MODEL = model;
      const fetchMock = vi.fn().mockResolvedValue(reply());
      global.fetch = fetchMock as unknown as typeof fetch;
      await new OpenAIProvider().complete({ ...baseReq, temperature: 0.2 });
      const sentBody = JSON.parse((fetchMock.mock.calls[0]![1] as RequestInit).body as string);
      expect(sentBody.model).toBe(model);
      if (sendsTemperature) expect(sentBody.temperature).toBe(0.2);
      else expect(sentBody).not.toHaveProperty('temperature');
    }
    expect(isReasoningModel('GPT-5')).toBe(true);
    expect(isReasoningModel('gpt-4.1')).toBe(false);
  });

  it('throws AIProviderError with kind http_error on a non-200 response', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(new Response('', { status: 429 })) as unknown as typeof fetch;
    const provider = new OpenAIProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({
      kind: 'http_error',
      status: 429,
    });
  });

  it('throws AIProviderError with kind malformed_json when message content is not JSON', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ choices: [{ message: { content: 'not json' } }] }), {
        status: 200,
      }),
    );
    const provider = new OpenAIProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({ kind: 'malformed_json' });
  });

  it('throws AIProviderError with kind malformed_json when there is no message content', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify({ choices: [] }), { status: 200 }));
    const provider = new OpenAIProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({ kind: 'malformed_json' });
  });

  it('throws kind refused when the model declines', async () => {
    global.fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ choices: [{ message: { content: null, refusal: 'no' } }] }), {
        status: 200,
      }),
    );
    await expect(new OpenAIProvider().complete(baseReq)).rejects.toMatchObject({
      kind: 'refused',
    });
  });

  it('times out when headers arrive but the body stalls', async () => {
    vi.useFakeTimers();
    try {
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

      const promise = new OpenAIProvider().complete(baseReq);
      const assertion = expect(promise).rejects.toMatchObject({ kind: 'timeout' });
      await vi.advanceTimersByTimeAsync(30_001);
      await assertion;
    } finally {
      vi.useRealTimers();
    }
  });

  it('never logs request or response bodies', async () => {
    const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    global.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          choices: [
            { message: { content: JSON.stringify({ secretHealthValue: 'do-not-log-me' }) } },
          ],
        }),
        { status: 200 },
      ),
    ) as unknown as typeof fetch;

    const provider = new OpenAIProvider();
    await provider.complete(baseReq);

    const allLoggedText = [...infoSpy.mock.calls, ...errorSpy.mock.calls].flat().join(' ');
    expect(allLoggedText).not.toContain('secretHealthValue');
    expect(allLoggedText).not.toContain('do-not-log-me');
  });
});
