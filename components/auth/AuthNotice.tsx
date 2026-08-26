import { Icon } from '@/components/ui';
import { oauthMessage } from '@/lib/auth/oauth/link';

/**
 * Explains why a Google sign-in did not go through. Rendered on the server
 * from the `?error=` code the callback redirected with, so the reason survives
 * the redirect and is announced to screen readers on arrival.
 */
export function AuthNotice({ code }: { code?: string | string[] }) {
  const message = oauthMessage(Array.isArray(code) ? code[0] : code);
  if (!message) return null;
  return (
    <p
      role="alert"
      className="mt-6 rounded-xl border border-critical/40 bg-critical-soft p-3 text-sm font-medium text-critical"
    >
      <Icon name="caution" className="shrink-0" />
      {message}
    </p>
  );
}
