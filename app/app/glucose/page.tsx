import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { prisma } from '@/lib/db/prisma';
import { analyzeUser, defaultWindow } from '@/lib/services/analytics-service';
import { allBands, classifyGlucose } from '@/lib/domain/thresholds';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import { TIME_OF_DAY_LABELS } from '@/lib/domain/time';
import { ButtonLink, Callout, Card, CardHeader, EmptyState, PageHeader, Stat, WhyThis } from '@/components/ui';
import { GlucoseTimeline, type TimelineMarker } from '@/components/charts/GlucoseTimeline';
import { BarChart } from '@/components/charts/BarChart';
import { RangeBar } from '@/components/charts/RangeBar';
import { GlucoseReadingRow } from '@/components/GlucoseReadingRow';

export const metadata: Metadata = { title: 'Glucose' };
export const dynamic = 'force-dynamic';

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

  const [{ result }, readings, meals, exercise] = await Promise.all([
    analyzeUser(user.id, profile, window),
    prisma.glucoseReading.findMany({
      where: { userId: user.id, takenAt: { gte: window.from } },
      orderBy: { takenAt: 'desc' },
      take: 400,
      select: { id: true, takenAt: true, valueMgdl: true, context: true, note: true },
    }),
    prisma.meal.findMany({
      where: { userId: user.id, takenAt: { gte: window.from } },
      orderBy: { takenAt: 'asc' },
      take: 200,
      select: { takenAt: true, description: true },
    }),
    prisma.exerciseSession.findMany({
      where: { userId: user.id, takenAt: { gte: window.from } },
      orderBy: { takenAt: 'asc' },
      take: 200,
      select: { takenAt: true, activity: true },
    }),
  ]);

  const summary = result.summary;
  const timeline = [...readings].reverse().map((r) => ({ takenAt: r.takenAt, valueMgdl: r.valueMgdl }));
  const markers: TimelineMarker[] = [
    ...meals.map((m) => ({ at: m.takenAt, kind: 'meal' as const, label: m.description })),
    ...exercise.map((e) => ({ at: e.takenAt, kind: 'exercise' as const, label: e.activity })),
  ];

  const bandCounts = allBands().map((band) => ({
    id: band.id,
    label: band.label,
    tone: band.tone,
    icon: band.icon === 'check' ? '✓' : band.icon === 'up' ? '▲' : band.icon === 'down' ? '▼' : '!',
    count: readings.filter((r) => classifyGlucose(r.valueMgdl, range).id === band.id).length,
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
          <Callout tone="positive" icon="✓" title="Reading saved">
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

      {readings.length === 0 ? (
        <EmptyState
          title="No readings in this period"
          icon="💧"
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
                  value={summary.averageMgdl == null ? '—' : formatGlucose(summary.averageMgdl, unit, locale)}
                  unit={unitLabel(unit)}
                />
                <Stat
                  label="Middle value (median)"
                  value={summary.medianMgdl == null ? '—' : formatGlucose(summary.medianMgdl, unit, locale)}
                  unit={unitLabel(unit)}
                />
                <Stat
                  label="In your target range"
                  value={summary.percentInRange == null ? '—' : `${Math.round(summary.percentInRange)}%`}
                  hint="Share of readings"
                />
                <Stat
                  label="Readings per day"
                  value={summary.readingsPerDay == null ? '—' : summary.readingsPerDay.toFixed(1)}
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
                description="Dotted lines mark meals (🍽) and activity (🚶) you logged."
              />
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
                  A bar drawn with diagonal hatching is based on fewer than three readings. An average
                  of one or two readings tells you about those readings, not about that time of day,
                  so DiaLog shows it as unreliable rather than dropping it silently.
                </p>
              </WhyThis>
            </Card>
          </section>

          <section aria-labelledby="all-readings">
            <Card>
              <CardHeader
                id="all-readings"
                title="All readings in this period"
                description={`${readings.length} readings, newest first.`}
              />
              <ul>
                {readings.slice(0, 100).map((reading) => (
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
              {readings.length > 100 ? (
                <p className="mt-4 text-sm text-ink-muted">
                  Showing the 100 most recent. The full list, with editing and deleting, is in{' '}
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
    const key = hour.hour < 6 ? 'overnight' : hour.hour < 12 ? 'morning' : hour.hour < 18 ? 'afternoon' : 'evening';
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
