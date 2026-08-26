import { ChartFrame, linearScale, ticks } from './primitives';

export interface Bar {
  label: string;
  value: number | null;
  /** Optional count shown in the table alternative to convey sample size. */
  n?: number;
}

const W = 720;
const H = 230;
const PAD = { top: 14, right: 12, bottom: 42, left: 48 };

/**
 * Categorical comparison (by hour, weekday, meal type, …). Bars below the
 * minimum sample size are drawn hatched and labelled "not enough data" rather
 * than being silently omitted.
 */
export function BarChart({
  bars,
  valueLabel,
  title,
  summary,
  minSample = 0,
  format = (v: number) => v.toFixed(1),
}: {
  bars: Bar[];
  valueLabel: string;
  title: string;
  summary: string;
  minSample?: number;
  format?: (value: number) => string;
}) {
  const values = bars.map((b) => b.value).filter((v): v is number => v != null);
  const maxV = values.length ? Math.max(...values) * 1.15 : 1;
  const y = linearScale([0, maxV], [H - PAD.bottom, PAD.top]);
  const slot = (W - PAD.left - PAD.right) / Math.max(1, bars.length);
  const barW = Math.min(46, slot * 0.66);

  return (
    <ChartFrame
      title={title}
      summary={summary}
      table={{
        caption: `${title}: ${valueLabel} for each group, with the number of records behind it.`,
        head: ['Group', valueLabel, 'Records'],
        rows: bars.map((b) => [
          b.label,
          b.value == null ? 'Not enough data' : format(b.value),
          b.n ?? 'No data',
        ]),
      }}
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" preserveAspectRatio="xMidYMid meet">
        <defs>
          <pattern
            id="dl-hatch"
            width="6"
            height="6"
            patternTransform="rotate(45)"
            patternUnits="userSpaceOnUse"
          >
            <line x1="0" y1="0" x2="0" y2="6" stroke="var(--color-line-strong)" strokeWidth="3" />
          </pattern>
        </defs>
        {ticks(0, maxV, 4).map((t) => (
          <g key={t}>
            <line x1={PAD.left} x2={W - PAD.right} y1={y(t)} y2={y(t)} stroke="var(--color-line)" />
            <text
              x={PAD.left - 8}
              y={y(t) + 4}
              fontSize="12"
              textAnchor="end"
              fill="var(--color-ink-muted)"
            >
              {format(t)}
            </text>
          </g>
        ))}
        {bars.map((b, i) => {
          const cx = PAD.left + slot * i + slot / 2;
          const insufficient = b.value == null || (b.n != null && b.n < minSample);
          const top = b.value == null ? y(0) : y(b.value);
          return (
            <g key={b.label}>
              {b.value != null ? (
                <rect
                  x={cx - barW / 2}
                  y={top}
                  width={barW}
                  height={Math.max(1, H - PAD.bottom - top)}
                  rx={4}
                  fill={insufficient ? 'url(#dl-hatch)' : 'var(--color-brand)'}
                >
                  <title>{`${b.label}: ${format(b.value)} ${valueLabel}${b.n != null ? ` (${b.n} records)` : ''}`}</title>
                </rect>
              ) : (
                <text
                  x={cx}
                  y={H - PAD.bottom - 6}
                  fontSize="11"
                  textAnchor="middle"
                  fill="var(--color-ink-muted)"
                >
                  n/a
                </text>
              )}
              <text
                x={cx}
                y={H - PAD.bottom + 18}
                fontSize="12"
                textAnchor="middle"
                fill="var(--color-ink-muted)"
              >
                {b.label}
              </text>
            </g>
          );
        })}
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={H - PAD.bottom}
          y2={H - PAD.bottom}
          stroke="var(--color-line-strong)"
        />
      </svg>
    </ChartFrame>
  );
}
