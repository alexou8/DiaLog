import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OpenAIProvider } from '@/lib/ai/providers/openai';

const ORIGINAL_FETCH = global.fetch;
const ORIGINAL_KEY = process.env.OPENAI_API_KEY;

beforeEach(() => {
  process.env.OPENAI_API_KEY = 'sk-oa-test-key';
});

afterEach(() => {
  global.fetch = ORIGINAL_FETCH;
  if (ORIGINAL_KEY === undefined) delete process.env.OPENAI_API_KEY;
  else process.env.OPENAI_API_KEY = ORIGINAL_KEY;
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
  });

  it('throws AIProviderError with kind http_error on a non-200 response', async () => {
    global.fetch = vi.fn().mockResolvedValue(new Response('', { status: 429 })) as unknown as typeof fetch;
    const provider = new OpenAIProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({ kind: 'http_error', status: 429 });
  });

  it('throws AIProviderError with kind malformed_json when message content is not JSON', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(new Response(JSON.stringify({ choices: [{ message: { content: 'not json' } }] }), { status: 200 }));
    const provider = new OpenAIProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({ kind: 'malformed_json' });
  });

  it('throws AIProviderError with kind malformed_json when there is no message content', async () => {
    global.fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ choices: [] }), { status: 200 }));
    const provider = new OpenAIProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({ kind: 'malformed_json' });
  });

  it('never logs request or response bodies', async () => {
    const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    global.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({ choices: [{ message: { content: JSON.stringify({ secretHealthValue: 'do-not-log-me' }) } }] }),
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
