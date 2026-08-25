import type { GlucoseContext, GlucoseUnit } from '@prisma/client';
import { classifyGlucose, GLUCOSE_CONTEXT_LABELS, type TargetRange } from '@/lib/domain/thresholds';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import { Badge } from '@/components/ui';

const BAND_ICON = { alert: '!', down: '▼', check: '✓', up: '▲' } as const;

/**
 * One reading, shown with value, unit, plain-language status, icon and time.
 * Four redundant cues, so the status survives greyscale, colour blindness and
 * a screen reader.
 */
export function GlucoseReadingRow({
  valueMgdl,
  takenAt,
  context,
  note,
  unit,
  range,
  locale,
  timeZone,
  action,
}: {
  valueMgdl: number;
  takenAt: Date;
  context: GlucoseContext;
  note?: string | null;
  unit: GlucoseUnit;
  range: TargetRange;
  locale: string;
  timeZone: string;
  action?: React.ReactNode;
}) {
  const band = classifyGlucose(valueMgdl, range);
  const when = new Intl.DateTimeFormat(locale, {
    timeZone,
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(takenAt);

  return (
    <li className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 last:border-0">
      <div className="min-w-0">
        <p className="flex items-baseline gap-2">
          <span className="text-2xl font-bold tabular-nums">
            {formatGlucose(valueMgdl, unit, locale)}
          </span>
          <span className="text-sm text-ink-muted">{unitLabel(unit)}</span>
        </p>
        <p className="text-sm text-ink-muted">
          {when}
          {context !== 'UNKNOWN' ? ` · ${GLUCOSE_CONTEXT_LABELS[context]}` : ''}
        </p>
        {note ? <p className="mt-1 text-sm">{note}</p> : null}
      </div>
      <div className="flex items-center gap-2">
        <Badge tone={band.tone} icon={BAND_ICON[band.icon]}>
          {band.label}
        </Badge>
        {action}
      </div>
    </li>
  );
}
