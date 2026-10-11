import { maxOf } from '@/lib/domain/units';
import { AxisLabel, ChartFrame, linearScale, pct, ticks } from './primitives';

export interface Bar {
  label: string;
  value: number | null;
  /** Optional count shown in the table alternative to convey sample size. */
  n?: number;
}

const W = 720;
const H = 230;
// Labels are HTML beside and below the SVG (see AxisLabel), so the plot runs
// almost edge to edge.
const PAD = { top: 14, right: 6, bottom: 4, left: 6 };

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
  // maxOf rather than Math.max(...values), which throws on very long arrays.
  const maxV = (maxOf(values) ?? 0) * 1.15 || 1;
  const y = linearScale([0, maxV], [H - PAD.bottom, PAD.top]);
  const slot = (W - PAD.left - PAD.right) / Math.max(1, bars.length);
  const barW = Math.min(46, slot * 0.66);
  const yTicks = ticks(0, maxV, 4);
  const anyInsufficient = bars.some((b) => b.value != null && b.n != null && b.n < minSample);

  return (
    <ChartFrame
      title={title}
      summary={summary}
      footer={
        anyInsufficient ? (
          <p className="mt-2 flex items-center gap-2 text-sm text-ink-muted">
            <svg viewBox="0 0 16 16" width="1em" height="1em" aria-hidden="true" focusable="false">
              <rect
                x="1"
                y="1"
                width="14"
                height="14"
                rx="2"
                fill="url(#dl-hatch)"
                stroke="var(--color-ink-muted)"
                strokeWidth="2"
              />
            </svg>
            Striped bars rest on fewer than {minSample} records, so treat them as a rough guide.
          </p>
        ) : undefined
      }
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
      <div className="grid grid-cols-[auto_minmax(0,1fr)] text-xs text-ink-muted">
        <div className="relative w-[3.5em]">
          {yTicks.map((t) => (
            <AxisLabel key={t} x="calc(100% - 0.5em)" y={pct(y(t), H)} align="end">
              <span className="dl-numeric">{format(t)}</span>
            </AxisLabel>
          ))}
        </div>

        <div className="relative">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="block h-auto w-full"
            preserveAspectRatio="xMidYMid meet"
          >
            <defs>
              {/* Stripes and outline in ink-muted, 7:1 against the surface.
                  They were line-strong at 2:1 with no outline, so a "not
                  enough data" bar was close to invisible. */}
              <pattern
                id="dl-hatch"
                width="8"
                height="8"
                patternTransform="rotate(45)"
                patternUnits="userSpaceOnUse"
              >
                <line x1="0" y1="0" x2="0" y2="8" stroke="var(--color-ink-muted)" strokeWidth="2" />
              </pattern>
            </defs>
            {yTicks.map((t) => (
              <line
                key={t}
                x1={PAD.left}
                x2={W - PAD.right}
                y1={y(t)}
                y2={y(t)}
                stroke="var(--color-line)"
              />
            ))}
            {bars.map((b, i) => {
              if (b.value == null) return null;
              const cx = PAD.left + slot * i + slot / 2;
              const insufficient = b.n != null && b.n < minSample;
              const top = y(b.value);
              return (
                <rect
                  key={b.label}
                  x={cx - barW / 2}
                  y={top}
                  width={barW}
                  height={Math.max(1, H - PAD.bottom - top)}
                  rx={4}
                  fill={insufficient ? 'url(#dl-hatch)' : 'var(--color-brand)'}
                  stroke={insufficient ? 'var(--color-ink-muted)' : undefined}
                  strokeWidth={insufficient ? 2 : undefined}
                >
                  <title>{`${b.label}: ${format(b.value)} ${valueLabel}${b.n != null ? ` (${b.n} records)` : ''}`}</title>
                </rect>
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
          {bars.map((b, i) =>
            b.value == null ? (
              <AxisLabel
                key={b.label}
                x={pct(PAD.left + slot * i + slot / 2, W)}
                y={`calc(${pct(H - PAD.bottom, H)} - 0.4em)`}
                align="center"
                vAlign="middle"
                className="-translate-y-full"
              >
                No data
              </AxisLabel>
            ) : null,
          )}
        </div>

        <div />
        {/* Category labels, one per bar slot, wrapping within the slot rather
            than overlapping the next one on a narrow screen. */}
        <ul className="flex pt-1.5" aria-hidden="true">
          {bars.map((b) => (
            <li
              key={b.label}
              className="min-w-0 flex-1 px-0.5 text-center leading-tight break-words"
            >
              {b.label}
            </li>
          ))}
        </ul>
      </div>
    </ChartFrame>
  );
}
