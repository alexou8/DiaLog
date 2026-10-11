import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { prisma } from '@/lib/db/prisma';
import { analyzeUser, defaultWindow } from '@/lib/services/analytics-service';
import { allBands, classifyGlucose } from '@/lib/domain/thresholds';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import { TIME_OF_DAY_LABELS } from '@/lib/domain/time';
import {
  ButtonLink,
  Callout,
  Card,
  CardHeader,
  EmptyState,
  PageHeader,
  Stat,
  WhyThis,
} from '@/components/ui';
import { GlucoseTimeline, type TimelineMarker } from '@/components/charts/GlucoseTimeline';
import { BarChart } from '@/components/charts/BarChart';
import { RangeBar } from '@/components/charts/RangeBar';
import { GlucoseReadingRow } from '@/components/GlucoseReadingRow';

export const metadata: Metadata = { title: 'Glucose' };

// No loading.tsx here or at app/app/loading.tsx, on purpose. On Next 15 a
// loading boundary that stays mounted across a navigation inside its section
// (the `?days=` links below, "Add a reading", or the redirect back here after
// saving one) left that navigation suspended for good: the URL never changed
// and the form sat on "Saving…" with the reading already stored. A boundary
// one level down, around a segment that the navigation replaces outright
// (meals, health, activity, settings), does not have the problem.
// dashboard-and-insights.spec.ts and logging.spec.ts guard both paths.
export const dynamic = 'force-dynamic';

/**
 * Most points the timeline draws. Its table alternative lists exactly the
 * points drawn (no more), and the page says when that is fewer than the
 * period holds; every reading is in History.
 */
const CHART_MAX_POINTS = 500;
const LIST_SIZE = 100;

const WEEKDAY_SHORT = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

export default async function GlucosePage({
  searchParams,
}: {
  searchParams: Promise<{ added?: string; days?: string }>;
}) {
  const params = await searchParams;
  const user = await requireOnboardedUser();
  const { profile } = user;
  const { locale, timezone, glucoseUnit: unit } = profile;
  const range = { lowMgdl: profile.targetLowMgdl, highMgdl: profile.targetHighMgdl };

  const days = params.days === '7' || params.days === '90' ? Number(params.days) : 30;
  const window = defaultWindow(days);

  const [{ result, input }, newest] = await Promise.all([
    analyzeUser(user.id, profile, window),
    // Only the list needs a query of its own: it shows notes, which the
    // analysis input deliberately leaves out.
    prisma.glucoseReading.findMany({
      where: { userId: user.id, takenAt: { gte: window.from } },
      orderBy: { takenAt: 'desc' },
      take: LIST_SIZE,
      select: { id: true, takenAt: true, valueMgdl: true, context: true, note: true },
    }),
  ]);

  const summary = result.summary;
  // Everything below is built from the analysis input, i.e. every reading in
  // the window, so the stats, the band bar and the chart describe one set of
  // data. This page used to cap its own query at 400 rows and compute the band
  // bar and "N readings" from that, while the stats above came from the full
  // window: on a CGM history the two disagreed on one screen.
  const readings = input.glucose;
  // Newest N, not oldest N: the chart is for what happened recently.
  const chartReadings = readings.slice(-CHART_MAX_POINTS);
  const timeline = chartReadings.map((r) => ({ takenAt: r.takenAt, valueMgdl: r.valueMgdl }));
  const chartIsTruncated = readings.length > chartReadings.length;
  const chartStart = chartReadings[0]?.takenAt;
  const markers: TimelineMarker[] = chartStart
    ? [
        ...input.meals
          .filter((m) => m.takenAt >= chartStart)
          .map((m) => ({ at: m.takenAt, kind: 'meal' as const, label: m.description })),
        ...input.exercise
          .filter((e) => e.takenAt >= chartStart)
          .map((e) => ({ at: e.takenAt, kind: 'exercise' as const, label: e.activity })),
      ]
    : [];
  const dateFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    month: 'short',
    day: 'numeric',
  });

  // One pass over the window rather than one filter per band.
  const countsByBand = new Map<string, number>();
  for (const reading of readings) {
    const id = classifyGlucose(reading.valueMgdl, range).id;
    countsByBand.set(id, (countsByBand.get(id) ?? 0) + 1);
  }
  const bandCounts = allBands().map((band) => ({
    id: band.id,
    label: band.label,
    tone: band.tone,
    icon: band.icon === 'check' ? ('ok' as const) : (band.icon as 'up' | 'down' | 'alert'),
    count: countsByBand.get(band.id) ?? 0,
  }));

  return (
    <div className="space-y-8">
      <PageHeader
        title="Glucose"
        description={`Your readings for the last ${days} days, in ${unitLabel(unit)}.`}
        action={<ButtonLink href="/app/glucose/new">Add a reading</ButtonLink>}
      />

      {params.added ? (
        <div role="status">
          <Callout tone="positive" icon="ok" title="Reading saved">
            Your reading has been added to your history.
          </Callout>
        </div>
      ) : null}

      <nav aria-label="Time period">
        <ul className="flex gap-2">
          {[7, 30, 90].map((option) => (
            <li key={option}>
              <ButtonLink
                href={`/app/glucose?days=${option}`}
                variant={option === days ? 'primary' : 'secondary'}
                className="px-4 py-2 text-sm"
                aria-current={option === days ? 'page' : undefined}
              >
                {option} days
              </ButtonLink>
            </li>
          ))}
        </ul>
      </nav>

      {summary.count === 0 ? (
        <EmptyState
          title="No readings in this period"
          icon="glucose"
          action={
            <>
              <ButtonLink href="/app/glucose/new">Add a reading</ButtonLink>
              <ButtonLink href="/app/import" variant="secondary">
                Import a file
              </ButtonLink>
            </>
          }
        >
          <p>
            Add a reading by hand, or import the export file from your meter&apos;s software. Either
            way your trends will start appearing here.
          </p>
        </EmptyState>
      ) : (
        <>
          <section aria-labelledby="numbers">
            <Card>
              <CardHeader id="numbers" title="The numbers" />
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <Stat
                  label="Average"
                  value={
                    summary.averageMgdl == null
                      ? 'No data'
                      : formatGlucose(summary.averageMgdl, unit, locale)
                  }
                  unit={unitLabel(unit)}
                />
                <Stat
                  label="Middle value (median)"
                  value={
                    summary.medianMgdl == null
                      ? 'No data'
                      : formatGlucose(summary.medianMgdl, unit, locale)
                  }
                  unit={unitLabel(unit)}
                />
                <Stat
                  label="In your target range"
                  value={
                    summary.percentInRange == null
                      ? 'No data'
                      : `${Math.round(summary.percentInRange)}%`
                  }
                  hint="Share of readings"
                />
                <Stat
                  label="Readings per day"
                  value={
                    summary.readingsPerDay == null ? 'No data' : summary.readingsPerDay.toFixed(1)
                  }
                  hint={`${summary.daysWithReadings} days with readings`}
                />
              </div>
              <div className="mt-5">
                <RangeBar slices={bandCounts} total={readings.length} />
              </div>
            </Card>
          </section>

          <section aria-labelledby="timeline">
            <Card>
              <CardHeader
                id="timeline"
                title="Over time"
                description="Dotted lines mark the meals and activity you logged."
              />
              {chartIsTruncated && chartStart ? (
                // The stats and the band bar above cover the whole period; the
                // chart cannot draw every reading, so say which part it shows.
                <p className="mb-3 text-sm text-ink-muted">
                  The chart shows your latest {timeline.length} of {readings.length} readings, from{' '}
                  {dateFmt.format(chartStart)} onwards. The figures above cover all {days} days.
                </p>
              ) : null}
              <GlucoseTimeline
                points={timeline}
                markers={markers}
                unit={unit}
                range={range}
                locale={locale}
                timeZone={timezone}
              />
            </Card>
          </section>

          <section aria-labelledby="patterns">
            <Card>
              <CardHeader
                id="patterns"
                title="Your daily and weekly pattern"
                description="Averages by time of day and by day of the week. Groups with very few readings are marked rather than hidden."
              />
              <div className="space-y-8">
                <BarChart
                  title="Average reading by time of day"
                  summary={describeHourly(summary.byHourOfDay, unit, locale)}
                  valueLabel={`Average (${unitLabel(unit)})`}
                  minSample={3}
                  format={(v) => formatGlucose(v, unit, locale)}
                  bars={groupByBucket(summary.byHourOfDay)}
                />
                <BarChart
                  title="Average reading by day of the week"
                  summary="How your average reading compares across the days of the week."
                  valueLabel={`Average (${unitLabel(unit)})`}
                  minSample={3}
                  format={(v) => formatGlucose(v, unit, locale)}
                  bars={summary.byWeekday.map((day) => ({
                    label: WEEKDAY_SHORT[day.weekday] ?? String(day.weekday),
                    value: day.averageMgdl,
                    n: day.count,
                  }))}
                />
              </div>
              <WhyThis label="Why are some bars marked as not enough data?">
                <p>
                  A bar drawn with diagonal hatching is based on fewer than three readings. An
                  average of one or two readings tells you about those readings, not about that time
                  of day, so DiaLog shows it as unreliable rather than dropping it silently.
                </p>
              </WhyThis>
            </Card>
          </section>

          <section aria-labelledby="all-readings">
            <Card>
              <CardHeader
                id="all-readings"
                title="All readings in this period"
                description={
                  summary.count > newest.length
                    ? `${summary.count} readings in this period. Showing the newest ${newest.length}.`
                    : `${summary.count} readings, newest first.`
                }
              />
              <ul>
                {newest.map((reading) => (
                  <GlucoseReadingRow
                    key={reading.id}
                    valueMgdl={reading.valueMgdl}
                    takenAt={reading.takenAt}
                    context={reading.context}
                    note={reading.note}
                    unit={unit}
                    range={range}
                    locale={locale}
                    timeZone={timezone}
                  />
                ))}
              </ul>
              {summary.count > newest.length ? (
                <p className="mt-4 text-sm text-ink-muted">
                  The full list, with deleting, is in{' '}
                  <a href="/app/history?type=glucose" className="underline underline-offset-4">
                    History
                  </a>
                  .
                </p>
              ) : null}
            </Card>
          </section>
        </>
      )}
    </div>
  );
}

/** Collapse 24 hourly buckets into four readable time-of-day groups. */
function groupByBucket(hours: { hour: number; averageMgdl: number | null; count: number }[]) {
  const buckets: Record<string, { sum: number; n: number }> = {};
  for (const hour of hours) {
    if (hour.averageMgdl == null || hour.count === 0) continue;
    const key =
      hour.hour < 6
        ? 'overnight'
        : hour.hour < 12
          ? 'morning'
          : hour.hour < 18
            ? 'afternoon'
            : 'evening';
    const bucket = (buckets[key] ??= { sum: 0, n: 0 });
    bucket.sum += hour.averageMgdl * hour.count;
    bucket.n += hour.count;
  }
  return (['overnight', 'morning', 'afternoon', 'evening'] as const).map((key) => {
    const bucket = buckets[key];
    return {
      label: TIME_OF_DAY_LABELS[key],
      value: bucket && bucket.n > 0 ? bucket.sum / bucket.n : null,
      n: bucket?.n ?? 0,
    };
  });
}

function describeHourly(
  hours: { hour: number; averageMgdl: number | null; count: number }[],
  unit: Parameters<typeof formatGlucose>[1],
  locale: string,
): string {
  const bars = groupByBucket(hours).filter((b) => b.value != null);
  if (bars.length === 0) return 'Not enough readings yet to compare times of day.';
  const highest = bars.reduce((a, b) => ((b.value ?? 0) > (a.value ?? 0) ? b : a));
  return `Your highest average is in the ${highest.label.toLowerCase()}, at ${formatGlucose(highest.value!, unit, locale)} ${unitLabel(unit)} across ${highest.n} readings.`;
}
