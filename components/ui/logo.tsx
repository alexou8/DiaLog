/**
 * DiaLog brand marks.
 *
 * The mark is inlined as JSX rather than loaded from `/logo-mark.svg` through
 * next/image for two reasons. It paints with the first HTML response instead
 * of after a second request — the same reason `globals.css` refuses to
 * download a webfont — and it scales cleanly to the 16/24/32/48px sizes the
 * brand guide requires, which a fixed-resolution PNG did not.
 *
 * `/logo-mark.svg`, `/logo.svg`, `/logo-wordmark.svg` and
 * `/logo-monochrome.svg` remain in `public/` as the source-of-truth vector
 * assets for anything outside the app (README, press, social cards). Keep the
 * geometry below identical to `public/logo-mark.svg` if either changes.
 *
 * The literal `#155E69` here is deliberate and is not a design-token bypass.
 * It is the fixed brand colour, and it has to match the baked-in PNG and ICO
 * favicons pixel for pixel; a token that moved with the theme would leave the
 * in-app mark and the browser-tab icon disagreeing.
 */
import { cn } from '@/lib/utils';

const BRAND = '#155E69';

type LogoMarkProps = {
  /** Rendered size in px. The mark is drawn on a 128 grid and scales cleanly. */
  size?: number;
  className?: string;
  /**
   * Accessible name. Omit wherever the word "DiaLog" already sits next to the
   * mark — a header link that reads "DiaLog DiaLog" helps nobody. Pass a name
   * only when the mark is the sole brand identifier.
   */
  title?: string;
};

export function LogoMark({ size = 32, className, title }: LogoMarkProps) {
  return (
    <svg
      viewBox="0 0 128 128"
      width={size}
      height={size}
      className={cn('shrink-0', className)}
      role={title ? 'img' : undefined}
      aria-label={title}
      aria-hidden={title ? undefined : true}
      focusable="false"
    >
      <rect width="128" height="128" rx="34" fill={BRAND} />
      {/* Dialogue bubble: conversation and plain-language explanation. */}
      <path
        d="M36 39c0-6.627 5.373-12 12-12h32c6.627 0 12 5.373 12 12v33c0 6.627-5.373 12-12 12H67L49 98V84H48c-6.627 0-12-5.373-12-12V39z"
        fill="none"
        stroke="#fff"
        strokeWidth="7"
        strokeLinejoin="round"
      />
      {/* Glucose trend line: the evidence the explanation is built from. */}
      <path
        d="M46 65h10l8-13 9 24 9-17 8 8h9"
        fill="none"
        stroke="#fff"
        strokeWidth="6.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/**
 * The full lockup — mark, wordmark and descriptor — for large public brand
 * surfaces.
 *
 * The wordmark is real text rather than the `<text>` elements inside
 * `public/logo.svg`. An `<img>`-loaded SVG cannot see the page's theme, so the
 * shipped file's `#172126` wordmark would go all but invisible against the
 * dark background; and its descriptor would be resampled rather than laid out,
 * which is exactly the "descriptor too small to read" case the brand guide
 * rules out. Live text inherits the theme, the type scale, and the user's
 * large-text preference.
 */
export function Logo({ className }: { className?: string }) {
  return (
    <span className={cn('inline-flex items-center gap-3', className)}>
      <LogoMark size={56} />
      <span className="flex flex-col">
        <span className="text-3xl font-bold leading-none tracking-tight sm:text-4xl">
          Dia<span className="text-brand-ink">Log</span>
        </span>
        <span className="mt-1.5 text-meta font-semibold uppercase tracking-[0.16em] text-ink-muted">
          Glucose · Metabolic health · Insight
        </span>
      </span>
    </span>
  );
}
