import { ConsoleEmailProvider } from './providers/console';
import { HttpEmailProvider } from './providers/http';

export interface RecoveryEmail {
  to: string;
  url: string;
}

export interface EmailProvider {
  id: string;
  available(): boolean;
  sendRecovery(message: RecoveryEmail): Promise<void>;
}

/** Fixed messages only: transport errors can contain credentials or mail bodies. */
export class EmailProviderError extends Error {
  constructor(readonly kind: 'unavailable' | 'delivery_failed') {
    super(kind === 'unavailable' ? 'Email delivery is not configured.' : 'Email delivery failed.');
    this.name = 'EmailProviderError';
  }
}

/** Same explicit → environment → safe default resolution as the AI seam. */
export function getEmailProvider(preferred?: string): EmailProvider {
  const id = preferred ?? process.env.EMAIL_PROVIDER ?? 'console';
  // Unknown settings fail closed rather than pretending a message was sent.
  if (id === 'http') return new HttpEmailProvider();
  if (id === 'console') return new ConsoleEmailProvider();
  throw new EmailProviderError('unavailable');
}
