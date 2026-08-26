import type { GlucoseUnit } from '@prisma/client';
import { classifyGlucose, type TargetRange } from '@/lib/domain/thresholds';
import { formatGlucose, fromMgdl, unitLabel } from '@/lib/domain/units';
import { ChartFrame, linearScale, ticks } from './primitives';

export interface TimelinePoint {
  takenAt: Date;
  valueMgdl: number;
  label?: string;
}

export interface TimelineMarker {
  at: Date;
  kind: 'meal' | 'exercise';
  label: string;
}

const W = 720;
const H = 260;
const PAD = { top: 16, right: 16, bottom: 34, left: 48 };

/**
 * Glucose over time with the user's target range drawn as a labelled band.
 * Points outside the band are drawn as triangles (above) or diamonds (below)
 * so the status is legible without colour.
 */
export function GlucoseTimeline({
  points,
  markers = [],
  unit,
  range,
  locale = 'en-CA',
  timeZone,
}: {
  points: TimelinePoint[];
  markers?: TimelineMarker[];
  unit: GlucoseUnit;
  range: TargetRange;
  locale?: string;
  timeZone: string;
}) {
  const sorted = [...points].sort((a, b) => a.takenAt.getTime() - b.takenAt.getTime());
  const values = sorted.map((p) => fromMgdl(p.valueMgdl, unit));
  const lowU = fromMgdl(range.lowMgdl, unit);
  const highU = fromMgdl(range.highMgdl, unit);

  const minV = Math.min(...values, lowU) * 0.9;
  const maxV = Math.max(...values, highU) * 1.08;
  const t0 = sorted[0]?.takenAt.getTime() ?? 0;
  const t1 = sorted[sorted.length - 1]?.takenAt.getTime() ?? t0 + 1;

  const x = linearScale([t0, t1 === t0 ? t0 + 1 : t1], [PAD.left, W - PAD.right]);
  const y = linearScale([minV, maxV], [H - PAD.bottom, PAD.top]);

  const path = sorted
    .map(
      (p, i) =>
        `${i === 0 ? 'M' : 'L'}${x(p.takenAt.getTime()).toFixed(1)},${y(fromMgdl(p.valueMgdl, unit)).toFixed(1)}`,
    )
    .join(' ');

  const timeFmt = new Intl.DateTimeFormat(locale, { timeZone, month: 'short', day: 'numeric' });
  const stampFmt = new Intl.DateTimeFormat(locale, {
    timeZone,
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });

  const inRange = sorted.filter(
    (p) => classifyGlucose(p.valueMgdl, range).id === 'in-range',
  ).length;
  const summary =
    sorted.length === 0
      ? 'No readings in this period.'
      : `${sorted.length} readings between ${stampFmt.format(sorted[0]!.takenAt)} and ${stampFmt.format(
          sorted[sorted.length - 1]!.takenAt,
        )}. ${inRange} of ${sorted.length} were inside your target range of ${formatGlucose(
          range.lowMgdl,
          unit,
          locale,
        )} to ${formatGlucose(range.highMgdl, unit, locale)} ${unitLabel(unit)}.`;

  return (
    <ChartFrame
      title="Glucose readings over time"
      summary={summary}
      table={{
        caption: 'Every glucose reading shown in the chart, with its date, value and status.',
        head: ['When', `Reading (${unitLabel(unit)})`, 'Status'],
        rows: sorted.map((p) => [
          stampFmt.format(p.takenAt),
          formatGlucose(p.valueMgdl, unit, locale),
          classifyGlucose(p.valueMgdl, range).label,
        ]),
      }}
    >
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" preserveAspectRatio="xMidYMid meet">
        {/* Target range band */}
        <rect
          x={PAD.left}
          y={y(highU)}
          width={W - PAD.left - PAD.right}
          height={Math.max(0, y(lowU) - y(highU))}
          fill="var(--color-positive-soft)"
          stroke="var(--color-positive)"
          strokeDasharray="4 4"
          strokeOpacity={0.5}
        />
        <text x={PAD.left + 6} y={y(highU) + 14} fontSize="11" fill="var(--color-positive)">
          Your target range
        </text>

        {/* Y axis */}
        {ticks(minV, maxV, 4).map((t) => (
          <g key={t}>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={y(t)}
              y2={y(t)}
              stroke="var(--color-line)"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 8}
              y={y(t) + 4}
              fontSize="12"
              textAnchor="end"
              fill="var(--color-ink-muted)"
            >
              {t.toFixed(unit === 'MMOLL' ? 1 : 0)}
            </text>
          </g>
        ))}

        {/* Context markers */}
        {markers.map((m, i) => (
          <g key={i}>
            <line
              x1={x(m.at.getTime())}
              x2={x(m.at.getTime())}
              y1={PAD.top}
              y2={H - PAD.bottom}
              stroke="var(--color-line-strong)"
              strokeWidth={1}
              strokeDasharray="2 3"
            />
            <text
              x={x(m.at.getTime())}
              y={PAD.top - 4}
              fontSize="11"
              textAnchor="middle"
              fill="var(--color-ink-muted)"
            >
              {m.kind === 'meal' ? 'M' : 'A'}
            </text>
          </g>
        ))}

        {sorted.length > 1 ? (
          <path
            d={path}
            fill="none"
            stroke="var(--color-brand)"
            strokeWidth={2}
            strokeLinejoin="round"
          />
        ) : null}

        {sorted.map((p, i) => {
          const band = classifyGlucose(p.valueMgdl, range);
          const cx = x(p.takenAt.getTime());
          const cy = y(fromMgdl(p.valueMgdl, unit));
          const title = `${stampFmt.format(p.takenAt)}: ${formatGlucose(p.valueMgdl, unit, locale)} ${unitLabel(unit)}, ${band.label}`;
          if (band.id === 'in-range') {
            return (
              <circle key={i} cx={cx} cy={cy} r={4.5} fill="var(--color-brand)">
                <title>{title}</title>
              </circle>
            );
          }
          const isHigh = band.id === 'above-range' || band.id === 'very-high';
          const colour =
            band.tone === 'critical'
              ? 'var(--color-critical)'
              : isHigh
                ? 'var(--color-notice)'
                : 'var(--color-caution)';
          const shape = isHigh
            ? `${cx},${cy - 6} ${cx - 5.5},${cy + 4} ${cx + 5.5},${cy + 4}`
            : `${cx},${cy + 6} ${cx - 5.5},${cy - 4} ${cx + 5.5},${cy - 4}`;
          return (
            <polygon key={i} points={shape} fill={colour}>
              <title>{title}</title>
            </polygon>
          );
        })}

        {/* X axis */}
        <line
          x1={PAD.left}
          x2={W - PAD.right}
          y1={H - PAD.bottom}
          y2={H - PAD.bottom}
          stroke="var(--color-line-strong)"
        />
        {sorted.length > 0
          ? [sorted[0]!, sorted[Math.floor(sorted.length / 2)]!, sorted[sorted.length - 1]!].map(
              (p, i) => (
                <text
                  key={i}
                  x={x(p.takenAt.getTime())}
                  y={H - PAD.bottom + 18}
                  fontSize="12"
                  textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}
                  fill="var(--color-ink-muted)"
                >
                  {timeFmt.format(p.takenAt)}
                </text>
              ),
            )
          : null}
      </svg>
    </ChartFrame>
  );
}
