/**
 * The landing page's one piece of product imagery.
 *
 * It is drawn rather than screenshotted on purpose. A screenshot of the real
 * dashboard goes stale the first time the dashboard changes, and the previous
 * README already carries a gallery of real screenshots for people who want to
 * see the actual UI. What the hero needs is the *idea*: a trace, the in-range
 * band it is being read against, and a plain-language sentence carrying the
 * evidence grade that made it safe to say. That idea is stable.
 *
 * Colours come from the theme tokens rather than literals so the illustration
 * inverts with the rest of the page — the deep teal brand becomes a light teal
 * in dark mode, and a hardcoded #155E69 here would fail contrast there.
 */
export function HeroVisual() {
  return (
    <div className="rounded-[var(--radius-card)] border border-line bg-surface p-5 shadow-[0_1px_2px_rgb(0_0_0/0.04)]">
      <div className="flex items-baseline justify-between gap-3">
        <h2 className="text-base font-semibold tracking-tight">Last 14 days</h2>
        <span className="dl-numeric text-meta text-ink-muted">84 readings</span>
      </div>

      {/* Decorative: every fact this picture carries is also written out in the
          text below it and in the hero copy, so announcing it again would only
          make the page longer to listen to. */}
      <svg
        viewBox="0 0 320 132"
        className="mt-4 w-full"
        role="presentation"
        aria-hidden="true"
        focusable="false"
      >
        {/* The in-range band. Range is shown as a region rather than as two
            threshold lines because "in range" is the thing being judged. */}
        {/* Tinted from the brand colour rather than using --color-brand-soft
            directly: the dark theme's brand-soft is close enough to the dark
            brand stroke that the trace disappeared into the band. A mix
            against the surface stays subtle in both themes. */}
        <rect
          x="0"
          y="46"
          width="320"
          height="46"
          rx="4"
          fill="color-mix(in oklab, var(--color-brand) 14%, var(--color-surface))"
        />
        <line x1="0" y1="46" x2="320" y2="46" stroke="var(--color-line)" strokeWidth="1" />
        <line x1="0" y1="92" x2="320" y2="92" stroke="var(--color-line)" strokeWidth="1" />

        <polyline
          points="8,84 42,70 76,78 110,52 144,60 178,34 212,56 246,44 280,62 312,54"
          fill="none"
          stroke="var(--color-brand)"
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <circle cx="312" cy="54" r="4.5" fill="var(--color-brand)" />

        {/* One excursion above the band, called out with a shape as well as a
            colour — nothing in this product encodes meaning in colour alone. */}
        <circle
          cx="178"
          cy="34"
          r="4.5"
          fill="var(--color-surface)"
          stroke="var(--color-notice)"
          strokeWidth="2.5"
        />
      </svg>

      <div className="mt-4 border-t border-line pt-4">
        <p className="font-medium">Readings after a walk sat lower than on days without one.</p>
        <p className="dl-meta mt-1 text-ink-muted">
          Based on 21 days of your data &middot; moderate evidence
        </p>
      </div>
    </div>
  );
}
