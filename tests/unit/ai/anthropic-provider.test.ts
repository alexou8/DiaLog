import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { AnthropicProvider } from '@/lib/ai/providers/anthropic';
import { AIProviderError } from '@/lib/ai/provider';

const ORIGINAL_FETCH = global.fetch;
const ORIGINAL_KEY = process.env.ANTHROPIC_API_KEY;

beforeEach(() => {
  process.env.ANTHROPIC_API_KEY = 'sk-ant-test-key';
});

afterEach(() => {
  global.fetch = ORIGINAL_FETCH;
  if (ORIGINAL_KEY === undefined) delete process.env.ANTHROPIC_API_KEY;
  else process.env.ANTHROPIC_API_KEY = ORIGINAL_KEY;
  vi.restoreAllMocks();
});

const baseReq = {
  system: 'system prompt',
  messages: [{ role: 'user' as const, content: 'hello' }],
  responseSchema: { type: 'object', properties: {} },
  maxTokens: 100,
};

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

  it('extracts structured JSON from a forced tool_use response', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          content: [
            {
              type: 'tool_use',
              name: 'emit_structured_output',
              input: { shortAnswer: 'hi', ok: true },
            },
          ],
        }),
        { status: 200 },
      ),
    );
    global.fetch = fetchMock as unknown as typeof fetch;

    const provider = new AnthropicProvider();
    const result = await provider.complete(baseReq);
    expect(result.json).toEqual({ shortAnswer: 'hi', ok: true });
    expect(result.providerId).toBe('anthropic');
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0]!;
    expect(url).toBe('https://api.anthropic.com/v1/messages');
    expect((init as RequestInit).method).toBe('POST');
    const sentBody = JSON.parse((init as RequestInit).body as string);
    expect(sentBody.tool_choice).toEqual({ type: 'tool', name: 'emit_structured_output' });
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

  it('throws AIProviderError with kind malformed_json when body is not valid JSON', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(new Response('not json', { status: 200 })) as unknown as typeof fetch;
    const provider = new AnthropicProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({
      kind: 'malformed_json',
    });
  });

  it('throws AIProviderError with kind malformed_json when no matching tool_use block is present', async () => {
    global.fetch = vi
      .fn()
      .mockResolvedValue(
        new Response(JSON.stringify({ content: [{ type: 'text', text: 'oops' }] }), {
          status: 200,
        }),
      );
    const provider = new AnthropicProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({
      kind: 'malformed_json',
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
    vi.useRealTimers();
  });

  it('throws AIProviderError with kind unavailable when no API key is configured', async () => {
    delete process.env.ANTHROPIC_API_KEY;
    const provider = new AnthropicProvider();
    await expect(provider.complete(baseReq)).rejects.toMatchObject({ kind: 'unavailable' });
  });

  it('never logs request or response bodies', async () => {
    const infoSpy = vi.spyOn(console, 'info').mockImplementation(() => {});
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    global.fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          content: [
            {
              type: 'tool_use',
              name: 'emit_structured_output',
              input: { secretHealthValue: 'do-not-log-me' },
            },
          ],
        }),
        {
          status: 200,
        },
      ),
    ) as unknown as typeof fetch;

    const provider = new AnthropicProvider();
    await provider.complete(baseReq);

    const allLoggedText = [...infoSpy.mock.calls, ...errorSpy.mock.calls].flat().join(' ');
    expect(allLoggedText).not.toContain('secretHealthValue');
    expect(allLoggedText).not.toContain('do-not-log-me');
    expect(allLoggedText).not.toContain('hello'); // the request message content
  });
});
