import { Callout } from '@/components/ui';
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
    <div className="mt-6">
      <Callout tone="critical" icon="caution" role="alert">
        {message}
      </Callout>
    </div>
  );
}
