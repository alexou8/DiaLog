import { EmailProviderError, type EmailProvider, type RecoveryEmail } from '../provider';

const captured: RecoveryEmail[] = [];

/** Test-process memory only; never an application HTTP endpoint or persistent log. */
export function takeCapturedEmails(): RecoveryEmail[] {
  if (process.env.NODE_ENV !== 'test') throw new EmailProviderError('unavailable');
  return captured.splice(0);
}

export class ConsoleEmailProvider implements EmailProvider {
  readonly id = 'console';
  available(): boolean {
    return process.env.NODE_ENV === 'development' || process.env.NODE_ENV === 'test';
  }
  async sendRecovery(message: RecoveryEmail): Promise<void> {
    // The developer's local terminal is the sole authorised logging exception.
    // An explicit development check prevents production misconfiguration from
    // leaking bearer links, and refusal prevents silently dropping recovery mail.
    if (process.env.NODE_ENV === 'development') {
      console.log(message.url);
      return;
    }
    if (process.env.NODE_ENV === 'test') {
      if (captured.length >= 100) captured.shift();
      captured.push({ ...message });
      return;
    }
    throw new EmailProviderError('unavailable');
  }
}
