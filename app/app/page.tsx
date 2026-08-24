import type { Metadata } from 'next';
import Link from 'next/link';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { analyzeUser, defaultWindow, insightsFor } from '@/lib/services/analytics-service';
import { prisma } from '@/lib/db/prisma';
import { recordCounts } from '@/lib/db/health-records';
import { classifyGlucose } from '@/lib/domain/thresholds';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import { dayKeyInZone } from '@/lib/domain/time';
import { Badge, ButtonLink, Card, CardHeader, EmptyState, Stat, WhyThis } from '@/components/ui';
import { GlucoseTimeline } from '@/components/charts/GlucoseTimeline';
import { InsightCardView } from '@/components/InsightCardView';
import { GlucoseReadingRow } from '@/components/GlucoseReadingRow';

export const metadata: Metadata = { title: 'Home' };
export const dynamic = 'force-dynamic';

const BAND_ICON = { alert: '!', down: '▼', check: '✓', up: '▲' } as const;

export default async function HomePage() {
  const user = await requireOnboardedUser();
  const { profile } = user;
  const { locale, timezone, glucoseUnit: unit } = profile;
  const range = { lowMgdl: profile.targetLowMgdl, highMgdl: profile.targetHighMgdl };

  const counts = await recordCounts(user.id);
  const totalRecords = Object.values(counts).reduce((a, b) => a + b, 0);

  if (totalRecords === 0) {
    return <FirstRun name={profile.displayName} />;
  }

  const window = defaultWindow(30);
  const [{ result }, recentReadings, recentMeals, recentExercise] = await Promise.all([
    analyzeUser(user.id, profile, window),
    prisma.glucoseReading.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 5,
      select: { id: true, takenAt: true, valueMgdl: true, context: true, note: true },
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

  const insights = insightsFor(result).slice(0, 2);
  const latest = recentReadings[0];
  const today = dayKeyInZone(new Date(), timezone);
  const todaysReadings = recentReadings.filter((r) => dayKeyInZone(r.takenAt, timezone) === today);

  const greeting = profile.displayName ? `Hello, ${profile.displayName}` : 'Hello';
  const timeFmt = new Intl.DateTimeFormat(locale, { timeZone: timezone, hour: 'numeric', minute: '2-digit' });
  const dayFmt = new Intl.DateTimeFormat(locale, { timeZone: timezone, weekday: 'long', month: 'long', day: 'numeric' });

  const summary = result.summary;
  const timelinePoints = (
    await prisma.glucoseReading.findMany({
      where: { userId: user.id, takenAt: { gte: window.from } },
      orderBy: { takenAt: 'asc' },
      select: { takenAt: true, valueMgdl: true },
      take: 500,
    })
  ).map((r) => ({ takenAt: r.takenAt, valueMgdl: r.valueMgdl }));

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">{greeting}</h1>
        <p className="mt-1 text-ink-muted">{dayFmt.format(new Date())}</p>
      </header>

      {/* ---------------------------------------------------------- Today */}
      <section aria-labelledby="today-heading">
        <Card>
          <CardHeader
            id="today-heading"
            title="How you're doing today"
            description={
              todaysReadings.length === 0
                ? 'No readings logged yet today.'
                : `${todaysReadings.length} ${todaysReadings.length === 1 ? 'reading' : 'readings'} logged today.`
            }
            action={<ButtonLink href="/app/glucose/new">Add a reading</ButtonLink>}
          />

          {latest ? (
            <div className="rounded-xl border border-line bg-surface-sunken p-5">
              <p className="text-sm font-medium text-ink-muted">Your most recent reading</p>
              <p className="mt-1 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="text-4xl font-bold tabular-nums sm:text-5xl">
                  {formatGlucose(latest.valueMgdl, unit, locale)}
                </span>
                <span className="text-lg text-ink-muted">{unitLabel(unit)}</span>
                <span className="text-base text-ink-muted">at {timeFmt.format(latest.takenAt)}</span>
              </p>
              <p className="mt-3">
                <Badge
                  tone={classifyGlucose(latest.valueMgdl, range).tone}
                  icon={BAND_ICON[classifyGlucose(latest.valueMgdl, range).icon]}
                >
                  {classifyGlucose(latest.valueMgdl, range).label}
                </Badge>
              </p>
              <p className="mt-3 text-sm text-ink-muted">
                {classifyGlucose(latest.valueMgdl, range).description} Your target range is{' '}
                {formatGlucose(range.lowMgdl, unit, locale)}–{formatGlucose(range.highMgdl, unit, locale)}{' '}
                {unitLabel(unit)}, which you can change in Settings.
              </p>
              {classifyGlucose(latest.valueMgdl, range).safetyMessage ? (
                <p className="mt-3 rounded-lg border border-line bg-surface p-3 text-sm">
                  {classifyGlucose(latest.valueMgdl, range).safetyMessage}
                </p>
              ) : null}
            </div>
          ) : (
            <p className="text-ink-muted">Add your first reading to see it here.</p>
          )}

          <div className="mt-5 grid gap-4 sm:grid-cols-2">
            <div>
              <h3 className="text-sm font-semibold text-ink-muted">Recent meals</h3>
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
                        · {timeFmt.format(meal.takenAt)}
                        {meal.carbsG != null ? ` · about ${Math.round(meal.carbsG)} g carbs` : ''}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div>
              <h3 className="text-sm font-semibold text-ink-muted">Recent activity</h3>
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
                        · {session.durationMin} min · {timeFmt.format(session.takenAt)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </Card>
      </section>

      {/* ------------------------------------------------------- This week */}
      <section aria-labelledby="period-heading">
        <Card>
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
            <EmptyState title="Nothing to summarise yet" icon="📈">
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
                  value={summary.averageMgdl == null ? '—' : formatGlucose(summary.averageMgdl, unit, locale)}
                  unit={unitLabel(unit)}
                  hint={`Across ${summary.count} readings`}
                />
                <Stat
                  label="Readings in your target range"
                  value={summary.percentInRange == null ? '—' : `${Math.round(summary.percentInRange)}%`}
                  hint="Share of readings, not time"
                  tone={summary.percentInRange != null && summary.percentInRange >= 70 ? 'positive' : 'neutral'}
                />
                <Stat
                  label="Variability"
                  value={summary.cv == null ? '—' : `${Math.round(summary.cv * 100)}%`}
                  hint="How spread out your readings are"
                />
              </div>

              <WhyThis label="What do these numbers mean?">
                <p>
                  <strong>Average</strong> is the plain mean of every reading you logged in the period.
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
        </Card>
      </section>

      {/* --------------------------------------------------------- Insights */}
      <section aria-labelledby="insights-heading">
        <div className="mb-4 flex items-end justify-between gap-3">
          <h2 id="insights-heading" className="text-xl font-semibold sm:text-2xl">
            What your data shows
          </h2>
          <Link href="/app/insights" className="dl-target font-semibold underline underline-offset-4">
            See all observations
          </Link>
        </div>
        {insights.length === 0 ? (
          <EmptyState title="No observations yet" icon="🔎">
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

      {/* ---------------------------------------------------- Recent entries */}
      <section aria-labelledby="recent-heading">
        <Card>
          <CardHeader id="recent-heading" title="Your latest readings" level={2} />
          <ul>
            {recentReadings.map((reading) => (
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
        </Card>
      </section>
    </div>
  );
}

/** Shown when the account has no records at all. Teaches the next step. */
function FirstRun({ name }: { name: string | null }) {
  return (
    <div className="space-y-6">
      <header>
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
          {name ? `Welcome, ${name}` : 'Welcome to DiaLog'}
        </h1>
        <p className="mt-2 max-w-prose text-ink-muted">
          There is nothing here yet — that is expected. Pick whichever of these is easiest for you
          right now. You can always do the others later.
        </p>
      </header>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <h2 className="text-lg font-semibold">Add one reading</h2>
          <p className="mt-2 text-ink-muted">The fastest start. Value, time, done.</p>
          <ButtonLink href="/app/glucose/new" className="mt-4">
            Add a reading
          </ButtonLink>
        </Card>
        <Card>
          <h2 className="text-lg font-semibold">Import a file</h2>
          <p className="mt-2 text-ink-muted">
            Bring in the export from your meter software, a spreadsheet, or your phone&apos;s health app.
          </p>
          <ButtonLink href="/app/import" variant="secondary" className="mt-4">
            Import data
          </ButtonLink>
        </Card>
        <Card>
          <h2 className="text-lg font-semibold">Describe your day</h2>
          <p className="mt-2 text-ink-muted">
            Type what you ate and did in ordinary words, then check what DiaLog suggests before saving.
          </p>
          <ButtonLink href="/app/quick-log" variant="secondary" className="mt-4">
            Try quick logging
          </ButtonLink>
        </Card>
      </div>

      <Card>
        <h2 className="text-lg font-semibold">What happens next</h2>
        <p className="mt-2 max-w-prose text-ink-muted">
          As readings build up, DiaLog starts comparing them against your own history and shows what
          it finds — with the number of records behind each observation. Until there is enough, it
          will say so rather than guess.
        </p>
      </Card>
    </div>
  );
}
