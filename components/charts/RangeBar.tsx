import { ChartFrame } from './primitives';

export interface RangeSlice {
  id: string;
  label: string;
  count: number;
  tone: 'critical' | 'caution' | 'positive' | 'notice';
  /** Redundant non-colour cue. */
  icon: string;
}

const TONE_FILL: Record<RangeSlice['tone'], string> = {
  critical: 'var(--color-critical)',
  caution: 'var(--color-caution)',
  positive: 'var(--color-positive)',
  notice: 'var(--color-notice)',
};

/**
 * Proportion of readings in each band. Deliberately a single stacked bar with
 * an explicit legend rather than a pie chart: proportions are easier to read
 * and the legend carries the labels and icons that make it colour-independent.
 */
export function RangeBar({ slices, total }: { slices: RangeSlice[]; total: number }) {
  const pct = (n: number) => (total === 0 ? 0 : (n / total) * 100);
  const summary =
    total === 0
      ? 'No readings yet in this period.'
      : slices
          .filter((s) => s.count > 0)
          .map((s) => `${Math.round(pct(s.count))}% ${s.label.toLowerCase()}`)
          .join(', ') + `, out of ${total} readings.`;

  let offset = 0;
  return (
    <ChartFrame
      title="Where your readings fell"
      summary={summary}
      table={{
        caption: 'Number and share of readings in each range band.',
        head: ['Band', 'Readings', 'Share'],
        rows: slices.map((s) => [s.label, s.count, `${pct(s.count).toFixed(0)}%`]),
      }}
      footer={
        <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-sm">
          {slices
            .filter((s) => s.count > 0)
            .map((s) => (
              <li key={s.id} className="flex items-center gap-2">
                <span
                  aria-hidden="true"
                  className="inline-block h-3 w-3 rounded-sm"
                  style={{ background: TONE_FILL[s.tone] }}
                />
                <span aria-hidden="true">{s.icon}</span>
                <span>
                  {s.label}: <strong className="tabular-nums">{Math.round(pct(s.count))}%</strong>{' '}
                  <span className="text-ink-muted">({s.count})</span>
                </span>
              </li>
            ))}
        </ul>
      }
    >
      <svg viewBox="0 0 720 44" className="h-auto w-full" preserveAspectRatio="none">
        {total === 0 ? (
          <rect x="0" y="8" width="720" height="28" rx="8" fill="var(--color-surface-sunken)" />
        ) : (
          slices.map((s) => {
            const w = (pct(s.count) / 100) * 720;
            const el = (
              <rect key={s.id} x={offset} y={8} width={Math.max(0, w)} height={28} fill={TONE_FILL[s.tone]}>
                <title>{`${s.label}: ${s.count} readings (${pct(s.count).toFixed(0)}%)`}</title>
              </rect>
            );
            offset += w;
            return el;
          })
        )}
      </svg>
    </ChartFrame>
  );
}
