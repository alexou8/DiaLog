import { afterEach, describe, expect, it, vi } from 'vitest';
import { ConsoleEmailProvider, takeCapturedEmails } from '@/lib/email/providers/console';
import { HttpEmailProvider } from '@/lib/email/providers/http';
import { EmailProviderError, getEmailProvider } from '@/lib/email/provider';

const message = {
  to: 'person@example.com',
  url: 'https://dialog.example/api/auth/password/reset?token=test-bearer',
};
afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  takeCapturedEmails();
});

describe('email transport boundaries', () => {
  it('captures test mail in memory without printing', async () => {
    const log = vi.spyOn(console, 'log');
    await new ConsoleEmailProvider().sendRecovery(message);
    expect(log).not.toHaveBeenCalled();
    expect(takeCapturedEmails()).toEqual([message]);
    expect(takeCapturedEmails()).toEqual([]);
  });
  it('prints only in explicitly development mode', async () => {
    vi.stubEnv('NODE_ENV', 'development');
    const log = vi.spyOn(console, 'log').mockImplementation(() => {});
    await new ConsoleEmailProvider().sendRecovery(message);
    expect(log).toHaveBeenCalledExactlyOnceWith(message.url);
  });
  it.each(['production', ''])(
    'refuses console delivery and capture access outside development/test',
    async (mode) => {
      vi.stubEnv('NODE_ENV', mode);
      const log = vi.spyOn(console, 'log');
      const provider = new ConsoleEmailProvider();
      expect(provider.available()).toBe(false);
      await expect(provider.sendRecovery(message)).rejects.toMatchObject({ kind: 'unavailable' });
      expect(() => takeCapturedEmails()).toThrow(EmailProviderError);
      expect(log).not.toHaveBeenCalled();
    },
  );
  it('resolves explicit selection before environment and defaults safely', () => {
    vi.stubEnv('EMAIL_PROVIDER', 'http');
    expect(getEmailProvider('console').id).toBe('console');
    expect(getEmailProvider().id).toBe('http');
    vi.stubEnv('EMAIL_PROVIDER', undefined);
    expect(getEmailProvider().id).toBe('console');
    expect(() => getEmailProvider('typo')).toThrow(EmailProviderError);
  });
  it('sends authenticated mail with TLS, a deadline and redirects disabled', async () => {
    vi.stubEnv('EMAIL_HTTP_URL', 'https://relay.example/mail');
    vi.stubEnv('EMAIL_HTTP_TOKEN', 'test-transport-key');
    vi.stubEnv('EMAIL_FROM', 'recovery@dialog.example');
    const fetcher = vi.fn().mockResolvedValue(new Response(null, { status: 202 }));
    vi.stubGlobal('fetch', fetcher);
    const provider = new HttpEmailProvider();
    expect(provider.available()).toBe(true);
    await provider.sendRecovery(message);
    expect(fetcher).toHaveBeenCalledWith(
      new URL('https://relay.example/mail'),
      expect.objectContaining({
        method: 'POST',
        redirect: 'error',
        signal: expect.any(AbortSignal),
        headers: { authorization: 'Bearer test-transport-key', 'content-type': 'application/json' },
      }),
    );
    expect(JSON.parse(fetcher.mock.calls[0]![1].body)).toMatchObject({
      to: message.to,
      text: expect.stringContaining(message.url),
    });
    fetcher.mockRejectedValue(new Error(message.url));
    await expect(provider.sendRecovery(message)).rejects.toEqual(
      new EmailProviderError('delivery_failed'),
    );
  });
  it('rejects missing settings and non-TLS remote relays', () => {
    vi.stubEnv('EMAIL_HTTP_URL', 'http://relay.example/mail');
    vi.stubEnv('EMAIL_HTTP_TOKEN', 'test-key');
    vi.stubEnv('EMAIL_FROM', 'recovery@dialog.example');
    expect(new HttpEmailProvider().available()).toBe(false);
    vi.stubEnv('EMAIL_HTTP_URL', 'https://relay.example/mail');
    vi.stubEnv('EMAIL_HTTP_TOKEN', '');
    expect(new HttpEmailProvider().available()).toBe(false);
  });
});
