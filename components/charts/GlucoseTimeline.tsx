import type { GlucoseUnit } from '@prisma/client';
import {
  allBands,
  classifyGlucose,
  type BandId,
  type GlucoseBand,
  type TargetRange,
} from '@/lib/domain/thresholds';
import { formatGlucose, fromMgdl, maxOf, minOf, toMgdl, unitLabel } from '@/lib/domain/units';
import { AxisLabel, ChartFrame, linearScale, pct, ticks } from './primitives';

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
// Small left padding: the y-axis labels are HTML in their own column beside
// the SVG (see AxisLabel). The top padding leaves room for marker letters.
const PAD = { top: 28, right: 12, bottom: 10, left: 6 };

const BAND_FILL: Record<GlucoseBand['tone'], string> = {
  critical: 'var(--color-critical)',
  caution: 'var(--color-caution)',
  notice: 'var(--color-notice)',
  positive: 'var(--color-brand)',
};

/**
 * One reading's mark. Every band has its own shape, so the status reads
 * without colour:
 *
 *   in range        ●  circle
 *   above / below   ▲ / ▼  triangle pointing the way the reading left the range
 *   well above/below        the same triangle, larger, inside a ring
 *
 * "Well below" and "below" used to be the same down-triangle, told apart only
 * by critical vs caution fills that measure 1.23:1 against each other in the
 * light theme and 1.06:1 in dark. The most safety-relevant state was in effect
 * colour-only. The ring is the extra, colour-independent cue.
 */
function PointMark({
  band,
  cx,
  cy,
  title,
}: {
  band: BandId;
  cx: number;
  cy: number;
  title?: string;
}) {
  const fill = BAND_FILL[classifyTone(band)];
  const tip = title ? <title>{title}</title> : null;
  if (band === 'in-range') {
    return (
      <circle cx={cx} cy={cy} r={4.5} fill={fill}>
        {tip}
      </circle>
    );
  }
  const up = band === 'above-range' || band === 'very-high';
  const severe = band === 'very-high' || band === 'very-low';
  const s = severe ? 1.25 : 1;
  const shape = up
    ? `${cx},${cy - 6 * s} ${cx - 5.5 * s},${cy + 4 * s} ${cx + 5.5 * s},${cy + 4 * s}`
    : `${cx},${cy + 6 * s} ${cx - 5.5 * s},${cy - 4 * s} ${cx + 5.5 * s},${cy - 4 * s}`;
  return (
    <g>
      {tip}
      {severe ? <circle cx={cx} cy={cy} r={11} fill="none" stroke={fill} strokeWidth={2} /> : null}
      <polygon points={shape} fill={fill} />
    </g>
  );
}

function classifyTone(band: BandId): GlucoseBand['tone'] {
  return allBands().find((b) => b.id === band)?.tone ?? 'positive';
}

/** The shape key, drawn with the same marks as the chart. */
function ShapeLegend({ present, markers }: { present: Set<BandId>; markers: boolean }) {
  const bands = allBands().filter((band) => present.has(band.id));
  if (bands.length === 0 && !markers) return null;
  return (
    <ul className="mt-3 flex flex-wrap gap-x-5 gap-y-2 text-sm" aria-label="Chart key">
      {bands.map((band) => (
        <li key={band.id} className="flex items-center gap-2">
          <svg
            viewBox="0 0 28 28"
            width="1.4em"
            height="1.4em"
            aria-hidden="true"
            focusable="false"
          >
            <PointMark band={band.id} cx={14} cy={14} />
          </svg>
          {band.label}
        </li>
      ))}
      {markers ? (
        <>
          <li className="flex items-center gap-2">
            <span aria-hidden="true" className="w-[1.4em] text-center font-semibold">
              M
            </span>
            Meal logged
          </li>
          <li className="flex items-center gap-2">
            <span aria-hidden="true" className="w-[1.4em] text-center font-semibold">
              A
            </span>
            Activity logged
          </li>
        </>
      ) : null}
    </ul>
  );
}

/**
 * Glucose over time with the user's target range drawn as a labelled band.
 * Every status has its own point shape (see PointMark), with a key below.
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

  // Loop-based, not Math.min(...values): spreading a long CGM history into
  // arguments throws a RangeError past roughly 120k points.
  const minV = Math.min(minOf(values) ?? lowU, lowU) * 0.9;
  const maxV = Math.max(maxOf(values) ?? highU, highU) * 1.08;
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

  const classified = sorted.map((p) => classifyGlucose(p.valueMgdl, range));
  const present = new Set(classified.map((band) => band.id));
  const inRange = classified.filter((band) => band.id === 'in-range').length;
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

  // Tick values are in the display unit; formatted through formatGlucose (via
  // mg/dL, the only unit it accepts) so "5,6" in fr-CA matches the table and
  // the stats beside the chart instead of a bare toFixed "5.6".
  const yTicks = ticks(minV, maxV, 4);
  // First, middle and last reading, each labelled once: with one or two
  // readings those positions coincide, and the same date used to be drawn two
  // or three times on top of itself. The first label hangs right from its
  // point and the last hangs left, unless there is only one.
  const last = sorted.length - 1;
  const xLabels = [
    ...new Set(sorted.length > 0 ? [0, Math.floor(sorted.length / 2), last] : []),
  ].map((index) => ({
    index,
    point: sorted[index]!,
    align: (sorted.length === 1
      ? 'center'
      : index === 0
        ? 'start'
        : index === last
          ? 'end'
          : 'center') as 'start' | 'center' | 'end',
  }));

  return (
    <ChartFrame
      title="Glucose readings over time"
      summary={summary}
      table={{
        caption: 'Every glucose reading shown in the chart, with its date, value and status.',
        head: ['When', `Reading (${unitLabel(unit)})`, 'Status'],
        rows: sorted.map((p, i) => [
          stampFmt.format(p.takenAt),
          formatGlucose(p.valueMgdl, unit, locale),
          classified[i]!.label,
        ]),
      }}
      footer={<ShapeLegend present={present} markers={markers.length > 0} />}
    >
      <div className="grid grid-cols-[auto_minmax(0,1fr)] text-xs text-ink-muted">
        {/* Y axis labels, aligned to the SVG's own height by the grid row. */}
        <div className="relative w-[3.5em]">
          {yTicks.map((t) => (
            <AxisLabel key={t} x="calc(100% - 0.5em)" y={pct(y(t), H)} align="end">
              <span className="dl-numeric">{formatGlucose(toMgdl(t, unit), unit, locale)}</span>
            </AxisLabel>
          ))}
        </div>

        <div className="relative">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="block h-auto w-full"
            preserveAspectRatio="xMidYMid meet"
          >
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

            {yTicks.map((t) => (
              <line
                key={t}
                x1={PAD.left}
                x2={W - PAD.right}
                y1={y(t)}
                y2={y(t)}
                stroke="var(--color-line)"
                strokeWidth={1}
              />
            ))}

            {/* Context markers */}
            {markers.map((m, i) => (
              <line
                key={i}
                x1={x(m.at.getTime())}
                x2={x(m.at.getTime())}
                y1={PAD.top}
                y2={H - PAD.bottom}
                stroke="var(--color-line-strong)"
                strokeWidth={1}
                strokeDasharray="2 3"
              />
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

            {sorted.map((p, i) => (
              <PointMark
                key={i}
                band={classified[i]!.id}
                cx={x(p.takenAt.getTime())}
                cy={y(fromMgdl(p.valueMgdl, unit))}
                title={`${stampFmt.format(p.takenAt)}: ${formatGlucose(p.valueMgdl, unit, locale)} ${unitLabel(unit)}, ${classified[i]!.label}`}
              />
            ))}

            {/* X axis */}
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={H - PAD.bottom}
              y2={H - PAD.bottom}
              stroke="var(--color-line-strong)"
            />
          </svg>

          <AxisLabel
            x="0.5em"
            y={`calc(${pct(y(highU), H)} + 0.4em)`}
            align="start"
            vAlign="top"
            className="font-medium text-positive"
          >
            Your target range
          </AxisLabel>
          {markers.map((m, i) => (
            <AxisLabel
              key={i}
              x={pct(x(m.at.getTime()), W)}
              y="0.15em"
              vAlign="top"
              className="font-semibold"
            >
              {m.kind === 'meal' ? 'M' : 'A'}
            </AxisLabel>
          ))}
        </div>

        <div />
        <div className="relative h-[2em]">
          {xLabels.map(({ index, point, align }) => (
            <AxisLabel
              key={index}
              x={pct(x(point.takenAt.getTime()), W)}
              y="0.5em"
              vAlign="top"
              align={align}
            >
              {timeFmt.format(point.takenAt)}
            </AxisLabel>
          ))}
        </div>
      </div>
    </ChartFrame>
  );
}
