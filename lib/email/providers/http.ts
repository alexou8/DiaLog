import { EmailProviderError, type EmailProvider, type RecoveryEmail } from '../provider';

function config() {
  try {
    const url = new URL(process.env.EMAIL_HTTP_URL ?? '');
    const key = process.env.EMAIL_HTTP_TOKEN;
    const from = process.env.EMAIL_FROM;
    // Loopback HTTP supports a local mail relay and the isolated e2e receiver.
    // Every remote transport must use TLS. Never follow credential-bearing redirects.
    const loopback = ['127.0.0.1', '[::1]', 'localhost'].includes(url.hostname);
    if (
      (url.protocol !== 'https:' && !(loopback && url.protocol === 'http:')) ||
      url.username ||
      url.password ||
      !key ||
      !from
    )
      return null;
    return { url, key, from };
  } catch {
    return null;
  }
}

/** Authenticated JSON mail-relay transport; the relay owns vendor-specific delivery. */
export class HttpEmailProvider implements EmailProvider {
  readonly id = 'http';
  available(): boolean {
    return config() !== null;
  }
  async sendRecovery(message: RecoveryEmail): Promise<void> {
    const settings = config();
    if (!settings) throw new EmailProviderError('unavailable');
    try {
      const response = await fetch(settings.url, {
        method: 'POST',
        redirect: 'error',
        signal: AbortSignal.timeout(10_000),
        headers: { authorization: `Bearer ${settings.key}`, 'content-type': 'application/json' },
        body: JSON.stringify({
          from: settings.from,
          to: message.to,
          subject: 'Reset your DiaLog password',
          text: `Reset your password using this link within 30 minutes:\n${message.url}\n\nIf you did not request this, ignore this email.`,
        }),
      });
      await response.body?.cancel();
      if (!response.ok) throw new EmailProviderError('delivery_failed');
    } catch {
      // Fetch errors and response bodies may echo the bearer URL or API key.
      throw new EmailProviderError('delivery_failed');
    }
  }
}
