import { cache, Suspense } from 'react';
import type { Metadata } from 'next';
import Link from 'next/link';
import { requireOnboardedUser, type CurrentUser } from '@/lib/auth/current-user';
import {
  analyzeUser,
  defaultWindow,
  insightsFor,
  type AnalyticsWindow,
} from '@/lib/services/analytics-service';
import { prisma } from '@/lib/db/prisma';
import { classifyGlucose } from '@/lib/domain/thresholds';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import { dayKeyInZone, startOfDayInZone } from '@/lib/domain/time';
import {
  Badge,
  ButtonLink,
  Card,
  CardHeader,
  EmptyState,
  Icon,
  Skeleton,
  Stat,
  WhyThis,
} from '@/components/ui';
import { GlucoseTimeline } from '@/components/charts/GlucoseTimeline';
import { InsightCardView } from '@/components/InsightCardView';
import { GlucoseReadingRow } from '@/components/GlucoseReadingRow';

export const metadata: Metadata = { title: 'Home' };
export const dynamic = 'force-dynamic';

/** Band name to icon-registry name. Status also carries a text label. */
const BAND_ICON = { alert: 'alert', down: 'down', check: 'ok', up: 'up' } as const;

/**
 * Most points the dashboard chart draws. The chart ships a real table of every
 * point as well, so this also bounds the size of that table in the page.
 */
const CHART_MAX_POINTS = 500;

/** A reading older than this is no longer "how you are doing now". */
const STALE_AFTER_MS = 24 * 60 * 60 * 1000;

/**
 * The cheapest question the dashboard asks: has this account ever recorded
 * anything? It used to be eight `COUNT(*)` queries, each scanning every row a
 * long-time CGM user owns, only to compare the sum with zero. An existence
 * check needs one index probe per table, and because glucose is by far the
 * common case it is asked alone first: a normal request pays for one query,
 * and only an account with no readings fans out to the rest. Hydration,
 * symptom and note entries are included so an account that has logged only
 * those is not told it has nothing.
 */
async function hasAnyRecords(userId: string): Promise<boolean> {
  const probe = { where: { userId }, select: { id: true } } as const;
  if (await prisma.glucoseReading.findFirst(probe)) return true;
  const others = await Promise.all([
    prisma.meal.findFirst(probe),
    prisma.exerciseSession.findFirst(probe),
    prisma.sleepSession.findFirst(probe),
    prisma.medicationEvent.findFirst(probe),
    prisma.weightMeasurement.findFirst(probe),
    prisma.bloodPressureMeasurement.findFirst(probe),
    prisma.moodEntry.findFirst(probe),
    prisma.hydrationEvent.findFirst(probe),
    prisma.symptomEntry.findFirst(probe),
    prisma.noteEntry.findFirst(probe),
  ]);
  return others.some((row) => row !== null);
}

/** Shared by the "most recent" and "latest readings" sections: one query per request. */
const loadRecentReadings = cache((userId: string) =>
  prisma.glucoseReading.findMany({
    where: { userId },
    orderBy: { takenAt: 'desc' },
    take: 5,
    select: { id: true, takenAt: true, valueMgdl: true, context: true, note: true },
  }),
);

/**
 * "today at 2:05 p.m." / "yesterday at 9:14 a.m." / "3 days ago (Mon, Oct 7,
 * 9:14 a.m.)". A bare clock time made a reading from last week look like one
 * from this morning, and the day is what tells you whether to trust the
 * number. Days are compared as calendar days in the user's zone, not as
 * 24-hour spans, so 11 p.m. last night is still "yesterday" at 1 a.m.
 */
function makeWhenFormatter(locale: string, timeZone: string) {
  const time = new Intl.DateTimeFormat(locale, { timeZone, hour: 'numeric', minute: '2-digit' });
  const dateTime = new Intl.DateTimeFormat(locale, {
    timeZone,
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
  return (when: Date, now: Date): string => {
    const days = Math.round(
      (Date.parse(`${dayKeyInZone(now, timeZone)}T00:00:00Z`) -
        Date.parse(`${dayKeyInZone(when, timeZone)}T00:00:00Z`)) /
        86_400_000,
    );
    // Zero or negative: a device clock slightly ahead of ours is still "today".
    if (days <= 0) return `today at ${time.format(when)}`;
    if (days === 1) return `yesterday at ${time.format(when)}`;
    return `${days} days ago (${dateTime.format(when)})`;
  };
}

export default async function HomePage() {
  const user = await requireOnboardedUser();
  const { profile } = user;
  const { locale, timezone } = profile;

  if (!(await hasAnyRecords(user.id))) {
    return <FirstRun name={profile.displayName} />;
  }

  // One window for every analytics section, so they all hit the same cached
  // analysis (see `analyzeUser`) instead of each stamping its own "now".
  const window = defaultWindow(30);

  const greeting = profile.displayName ? `Hello, ${profile.displayName}` : 'Hello';
  const dayFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });

  // Each section below fetches for itself and streams in behind its own
  // Suspense boundary. The page used to await the analytics run (seconds on a
  // CGM-sized history) before sending a byte, so the latest reading, the one
  // thing people open the app to check, waited on the slowest part.
  return (
    <div className="space-y-8">
      <header className="border-b border-line pb-5">
        <p className="dl-meta uppercase tracking-wide">{dayFmt.format(new Date())}</p>
        <h1 className="mt-1 text-2xl font-semibold tracking-tight sm:text-[1.75rem]">{greeting}</h1>
      </header>

      <Suspense fallback={<TodaySkeleton />}>
        <TodaySection user={user} />
      </Suspense>

      <Suspense fallback={<SectionSkeleton chart />}>
        <PeriodSection user={user} window={window} />
      </Suspense>

      <Suspense fallback={<SectionSkeleton />}>
        <InsightsSection user={user} window={window} />
      </Suspense>

      <Suspense fallback={<SectionSkeleton />}>
        <RecentReadingsSection user={user} />
      </Suspense>
    </div>
  );
}

/* ------------------------------------------------------------------ Today */

async function TodaySection({ user }: { user: CurrentUser }) {
  const { profile } = user;
  const { locale, timezone, glucoseUnit: unit } = profile;
  const range = { lowMgdl: profile.targetLowMgdl, highMgdl: profile.targetHighMgdl };
  const now = new Date();

  const [recentReadings, readingsToday, recentMeals, recentExercise] = await Promise.all([
    loadRecentReadings(user.id),
    // Bounded by the user's own midnight. This used to filter the five most
    // recent rows, so a day with eight readings was reported as five.
    prisma.glucoseReading.count({
      where: { userId: user.id, takenAt: { gte: startOfDayInZone(now, timezone), lte: now } },
    }),
    prisma.meal.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 3,
      select: { id: true, takenAt: true, description: true, mealType: true, carbsG: true },
    }),
    prisma.exerciseSession.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 3,
      select: { id: true, takenAt: true, activity: true, durationMin: true },
    }),
  ]);

  const latest = recentReadings[0];
  const when = makeWhenFormatter(locale, timezone);
  // Classified once: the same band drives the value, the badge and the
  // explanation below it.
  const latestBand = latest ? classifyGlucose(latest.valueMgdl, range) : null;
  const latestIsStale = latest ? now.getTime() - latest.takenAt.getTime() > STALE_AFTER_MS : false;

  return (
    <section aria-labelledby="today-heading">
      <div>
        <CardHeader
          id="today-heading"
          title="How you're doing today"
          description={
            readingsToday === 0
              ? 'No readings logged yet today.'
              : `${readingsToday} ${readingsToday === 1 ? 'reading' : 'readings'} logged today.`
          }
          action={<ButtonLink href="/app/glucose/new">Add a reading</ButtonLink>}
        />

        {latest && latestBand ? (
          /* The most recent value is the one thing someone opens the app to
               check, so it gets the only surface on this screen and the
               largest type in the product. Status is carried by the value, a
               badge with an icon, and a sentence, never by colour alone. */
          <div className="rounded-[var(--radius-card)] border border-line bg-surface p-5 sm:p-6">
            <p className="dl-meta font-medium uppercase tracking-wide">Your most recent reading</p>
            <p className="mt-2 flex flex-wrap items-baseline gap-x-3 gap-y-1">
              <span className="dl-numeric text-5xl font-semibold leading-none sm:text-6xl">
                {formatGlucose(latest.valueMgdl, unit, locale)}
              </span>
              <span className="text-lg font-medium text-ink-muted">{unitLabel(unit)}</span>
              <span className="dl-meta">{when(latest.takenAt, now)}</span>
            </p>
            {latestIsStale ? (
              // Text and an icon, not a colour change: a stale value shown as
              // if it were current is the misleading case in a glucose app.
              <p className="dl-measure mt-3 flex items-start gap-2 text-sm font-medium">
                <Icon name="caution" className="mt-0.5 shrink-0" />
                <span>
                  This reading is more than a day old, so it may not show how you are doing now.
                </span>
              </p>
            ) : null}
            <p className="mt-4">
              <Badge tone={latestBand.tone} icon={BAND_ICON[latestBand.icon]}>
                {latestBand.label}
              </Badge>
            </p>
            <p className="dl-measure mt-3 text-sm text-ink-muted">
              {latestBand.description} Your target range is{' '}
              {formatGlucose(range.lowMgdl, unit, locale)} to{' '}
              {formatGlucose(range.highMgdl, unit, locale)} {unitLabel(unit)}, which you can change
              in Settings.
            </p>
            {latestBand.safetyMessage ? (
              <p className="dl-measure mt-4 rounded-[var(--radius-control)] border border-line bg-surface-sunken p-3 text-sm">
                {latestBand.safetyMessage}
              </p>
            ) : null}
          </div>
        ) : (
          <p className="text-ink-muted">Add your first reading to see it here.</p>
        )}

        <div className="mt-6 grid gap-6 sm:grid-cols-2">
          <div>
            <h3 className="text-sm font-semibold">Recent meals</h3>
            {recentMeals.length === 0 ? (
              <p className="mt-1 text-sm">
                Nothing logged.{' '}
                <Link href="/app/meals/new" className="underline underline-offset-4">
                  Log a meal
                </Link>
              </p>
            ) : (
              <ul className="mt-1 space-y-1 text-sm">
                {recentMeals.map((meal) => (
                  <li key={meal.id}>
                    {meal.description}
                    <span className="text-ink-muted">
                      {' '}
                      · {when(meal.takenAt, now)}
                      {meal.carbsG != null ? ` · about ${Math.round(meal.carbsG)} g carbs` : ''}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <h3 className="text-sm font-semibold">Recent activity</h3>
            {recentExercise.length === 0 ? (
              <p className="mt-1 text-sm">
                Nothing logged.{' '}
                <Link href="/app/activity/new" className="underline underline-offset-4">
                  Log activity
                </Link>
              </p>
            ) : (
              <ul className="mt-1 space-y-1 text-sm">
                {recentExercise.map((session) => (
                  <li key={session.id}>
                    {session.activity}
                    <span className="text-ink-muted">
                      {' '}
                      · {session.durationMin} min · {when(session.takenAt, now)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------ Last 30 days */

async function PeriodSection({ user, window }: { user: CurrentUser; window: AnalyticsWindow }) {
  const { profile } = user;
  const { locale, timezone, glucoseUnit: unit } = profile;
  const range = { lowMgdl: profile.targetLowMgdl, highMgdl: profile.targetHighMgdl };

  const { result, input } = await analyzeUser(user.id, profile, window);
  const summary = result.summary;

  // The analysis already loaded every reading in the window, oldest first, so
  // the chart takes the newest N from it rather than querying again. The old
  // query took the *oldest* 500 (`orderBy asc, take 500`): for a CGM history
  // the chart ended around day two of "your last 30 days" and never showed
  // today.
  const timelinePoints = input.glucose
    .slice(-CHART_MAX_POINTS)
    .map((r) => ({ takenAt: r.takenAt, valueMgdl: r.valueMgdl }));
  const chartIsTruncated = input.glucose.length > timelinePoints.length;
  const chartStart = timelinePoints[0]?.takenAt;
  const dateFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    month: 'short',
    day: 'numeric',
  });

  return (
    <section aria-labelledby="period-heading" className="dl-rule pt-8">
      <div>
        <CardHeader
          id="period-heading"
          title="Your last 30 days"
          description={
            summary.count === 0
              ? 'No readings in this period yet.'
              : `${summary.count} readings across ${summary.daysWithReadings} days.`
          }
          action={
            <ButtonLink href="/app/glucose" variant="secondary">
              See all glucose
            </ButtonLink>
          }
        />

        {summary.count === 0 ? (
          <EmptyState title="Nothing to summarise yet" icon="chart">
            <p>
              Once you have a few readings in this period, this is where the averages and the trend
              will appear.
            </p>
          </EmptyState>
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-3">
              <Stat
                label="Average reading"
                value={
                  summary.averageMgdl == null
                    ? 'No data'
                    : formatGlucose(summary.averageMgdl, unit, locale)
                }
                unit={unitLabel(unit)}
                hint={`Across ${summary.count} readings`}
              />
              <Stat
                label="Readings in your target range"
                value={
                  summary.percentInRange == null
                    ? 'No data'
                    : `${Math.round(summary.percentInRange)}%`
                }
                hint="Share of readings, not time"
                tone={
                  summary.percentInRange != null && summary.percentInRange >= 70
                    ? 'positive'
                    : 'neutral'
                }
              />
              <Stat
                label="Variability"
                value={summary.cv == null ? 'No data' : `${Math.round(summary.cv * 100)}%`}
                hint="How spread out your readings are"
              />
            </div>

            <WhyThis label="What do these numbers mean?">
              <p>
                <strong>Average</strong> is the plain mean of every reading you logged in the
                period.
              </p>
              <p className="mt-2">
                <strong>Readings in your target range</strong> is the share of your individual
                readings that fell inside the range you set. Because these are separate readings
                rather than continuous monitoring, this is not the same as the &ldquo;time in
                range&rdquo; figure a CGM reports, and it is affected by when you happen to test.
              </p>
              <p className="mt-2">
                <strong>Variability</strong> is the coefficient of variation: the spread of your
                readings relative to their average. Lower generally means more consistent days.
              </p>
            </WhyThis>

            {timelinePoints.length > 1 ? (
              <div className="mt-6">
                {chartIsTruncated && chartStart ? (
                  // The numbers above cover the whole period; the chart cannot
                  // draw every reading, so say which part it is showing.
                  <p className="mb-3 text-sm text-ink-muted">
                    The chart shows your latest {timelinePoints.length} of {input.glucose.length}{' '}
                    readings, from {dateFmt.format(chartStart)} onwards. The figures above cover the
                    whole 30 days.
                  </p>
                ) : null}
                <GlucoseTimeline
                  points={timelinePoints}
                  unit={unit}
                  range={range}
                  locale={locale}
                  timeZone={timezone}
                />
              </div>
            ) : null}
          </>
        )}
      </div>
    </section>
  );
}

/* --------------------------------------------------------------- Insights */

async function InsightsSection({ user, window }: { user: CurrentUser; window: AnalyticsWindow }) {
  const { result } = await analyzeUser(user.id, user.profile, window);
  const insights = insightsFor(result).slice(0, 2);

  return (
    <section aria-labelledby="insights-heading" className="dl-rule pt-8">
      <div className="mb-4 flex items-end justify-between gap-3">
        <h2 id="insights-heading" className="text-lg font-semibold tracking-tight sm:text-xl">
          What your data shows
        </h2>
        <Link href="/app/insights" className="dl-target font-semibold underline underline-offset-4">
          See all observations
        </Link>
      </div>
      {insights.length === 0 ? (
        <EmptyState title="No observations yet" icon="search">
          <p>
            DiaLog needs a bit more of your history before it can say anything meaningful. Keep
            logging readings and the observations will start here.
          </p>
        </EmptyState>
      ) : (
        <ul className="space-y-4">
          {insights.map((insight) => (
            <InsightCardView key={insight.kind + insight.title} insight={insight} />
          ))}
        </ul>
      )}
    </section>
  );
}

/* ---------------------------------------------------------- Recent entries */

async function RecentReadingsSection({ user }: { user: CurrentUser }) {
  const { profile } = user;
  const recentReadings = await loadRecentReadings(user.id);
  const range = { lowMgdl: profile.targetLowMgdl, highMgdl: profile.targetHighMgdl };
  // An account with only meals or sleep logged has nothing to list; a heading
  // over an empty list would just be noise.
  if (recentReadings.length === 0) return null;

  return (
    <section aria-labelledby="recent-heading" className="dl-rule pt-8">
      <div>
        <CardHeader id="recent-heading" title="Your latest readings" level={2} />
        <ul>
          {recentReadings.map((reading) => (
            <GlucoseReadingRow
              key={reading.id}
              valueMgdl={reading.valueMgdl}
              takenAt={reading.takenAt}
              context={reading.context}
              note={reading.note}
              unit={profile.glucoseUnit}
              range={range}
              locale={profile.locale}
              timeZone={profile.timezone}
            />
          ))}
        </ul>
      </div>
    </section>
  );
}

/* --------------------------------------------------------------- Skeletons */

function TodaySkeleton() {
  return (
    <section aria-labelledby="today-loading">
      {/* The skeleton blocks are aria-hidden, so without this a screen-reader
          user would hear nothing at all while the page streams in. */}
      <p id="today-loading" role="status" className="sr-only">
        Loading your latest reading
      </p>
      <Skeleton className="mb-4 h-7 w-56" />
      <Skeleton className="h-52 w-full" />
    </section>
  );
}

function SectionSkeleton({ chart = false }: { chart?: boolean }) {
  return (
    <div className="dl-rule pt-8" aria-hidden="true">
      <Skeleton className="mb-4 h-7 w-48" />
      <Skeleton className="h-24 w-full" />
      {chart ? <Skeleton className="mt-6 h-64 w-full" /> : null}
    </div>
  );
}

/** Shown when the account has no records at all. Teaches the next step. */
function FirstRun({ name }: { name: string | null }) {
  return (
    <div className="space-y-6">
      <header className="border-b border-line pb-5">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-[1.75rem]">
          {name ? `Welcome, ${name}` : 'Welcome to DiaLog'}
        </h1>
        <p className="dl-measure mt-2 text-ink-muted">
          There is nothing here yet, which is expected. Pick whichever of these is easiest for you
          right now. You can always do the others later.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <h2 className="text-lg font-semibold tracking-tight">Add one reading</h2>
          <p className="mt-2 text-ink-muted">The fastest start. Value, time, done.</p>
          <ButtonLink href="/app/glucose/new" className="mt-4">
            Add a reading
          </ButtonLink>
        </Card>
        <Card>
          <h2 className="text-lg font-semibold tracking-tight">Import a file</h2>
          <p className="mt-2 text-ink-muted">
            Bring in the export from your meter software, a spreadsheet, or your phone&apos;s health
            app.
          </p>
          <ButtonLink href="/app/import" variant="secondary" className="mt-4">
            Import data
          </ButtonLink>
        </Card>
        <Card>
          <h2 className="text-lg font-semibold tracking-tight">Describe your day</h2>
          <p className="mt-2 text-ink-muted">
            Type what you ate and did in ordinary words, then check what DiaLog suggests before
            saving.
          </p>
          <ButtonLink href="/app/quick-log" variant="secondary" className="mt-4">
            Try quick logging
          </ButtonLink>
        </Card>
      </div>

      <Card>
        <h2 className="text-lg font-semibold tracking-tight">What happens next</h2>
        <p className="dl-measure mt-2 text-ink-muted">
          As readings build up, DiaLog starts comparing them against your own history and shows what
          it finds, with the number of records behind each observation. Until there is enough, it
          will say so rather than guess.
        </p>
      </Card>
    </div>
  );
}
