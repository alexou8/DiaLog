import clsx from 'clsx';

/** Google's mark, inlined so the button works offline and needs no tracking. */
function GoogleMark() {
  return (
    <svg aria-hidden="true" viewBox="0 0 18 18" className="h-5 w-5 shrink-0">
      <path
        fill="#4285F4"
        d="M17.64 9.2c0-.64-.06-1.25-.16-1.84H9v3.48h4.84a4.14 4.14 0 0 1-1.8 2.72v2.26h2.92c1.7-1.57 2.68-3.88 2.68-6.62Z"
      />
      <path
        fill="#34A853"
        d="M9 18c2.43 0 4.47-.8 5.96-2.18l-2.92-2.26c-.8.54-1.84.86-3.04.86-2.34 0-4.32-1.58-5.03-3.7H.96v2.33A9 9 0 0 0 9 18Z"
      />
      <path
        fill="#FBBC05"
        d="M3.97 10.72a5.4 5.4 0 0 1 0-3.44V4.95H.96a9 9 0 0 0 0 8.1l3.01-2.33Z"
      />
      <path
        fill="#EA4335"
        d="M9 3.58c1.32 0 2.5.45 3.44 1.35l2.58-2.58C13.46.89 11.43 0 9 0A9 9 0 0 0 .96 4.95l3.01 2.33C4.68 5.16 6.66 3.58 9 3.58Z"
      />
    </svg>
  );
}

/**
 * Starts the Google flow. A plain link rather than a form: the start route only
 * mints a single-use attempt cookie and redirects, and the callback is what
 * actually validates the round trip.
 */
export function GoogleButton({
  mode = 'signin',
  next,
  label,
  className,
}: {
  mode?: 'signin' | 'link';
  next?: string;
  label: string;
  className?: string;
}) {
  const params = new URLSearchParams();
  if (mode === 'link') params.set('mode', 'link');
  if (next) params.set('next', next);
  const query = params.toString();

  return (
    <a
      href={`/api/auth/google/start${query ? `?${query}` : ''}`}
      className={clsx(
        'dl-target inline-flex items-center justify-center gap-3 rounded-xl border-2 border-line-strong bg-surface px-5 py-2.5 text-base font-semibold text-ink transition-colors hover:border-brand hover:text-brand-ink',
        className,
      )}
    >
      <GoogleMark />
      {label}
    </a>
  );
}

/** "or" rule between the password form and the provider button. */
export function AuthDivider({ label = 'or' }: { label?: string }) {
  return (
    <div className="my-6 flex items-center gap-4">
      <span className="h-px flex-1 bg-line" aria-hidden="true" />
      <span className="text-sm text-ink-muted">{label}</span>
      <span className="h-px flex-1 bg-line" aria-hidden="true" />
    </div>
  );
}
