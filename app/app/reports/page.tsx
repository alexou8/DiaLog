import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { analyzeUser, defaultWindow, insightsFor } from '@/lib/services/analytics-service';
import { prisma } from '@/lib/db/prisma';
import { EVIDENCE_LABELS } from '@/lib/domain/evidence';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import {
  ButtonLink,
  Callout,
  Card,
  CardHeader,
  EmptyState,
  MedicalDisclaimer,
  PageHeader,
} from '@/components/ui';

export const metadata: Metadata = { title: 'Reports' };
export const dynamic = 'force-dynamic';

export default async function ReportsPage({
  searchParams,
}: {
  searchParams: Promise<{ period?: string }>;
}) {
  const params = await searchParams;
  const user = await requireOnboardedUser();
  const { profile } = user;
  const days = params.period === 'month' ? 30 : 7;
  const window = defaultWindow(days);

  const { result } = await analyzeUser(user.id, profile, window);
  const insights = insightsFor(result);
  const unit = profile.glucoseUnit;

  const [mealCount, carbStats, activity, sleep] = await Promise.all([
    prisma.meal.count({ where: { userId: user.id, takenAt: { gte: window.from } } }),
    prisma.meal.aggregate({
      where: { userId: user.id, takenAt: { gte: window.from }, carbsG: { not: null } },
      _avg: { carbsG: true },
      _count: { carbsG: true },
    }),
    prisma.exerciseSession.aggregate({
      where: { userId: user.id, takenAt: { gte: window.from } },
      _sum: { durationMin: true },
      _count: { _all: true },
    }),
    prisma.sleepSession.aggregate({
      where: { userId: user.id, takenAt: { gte: window.from } },
      _avg: { durationMin: true },
      _count: { _all: true },
    }),
  ]);

  const dateFmt = new Intl.DateTimeFormat(profile.locale, {
    timeZone: profile.timezone,
    dateStyle: 'long',
  });

  const summary = result.summary;
  const strongest = insights.filter((insight) => insight.evidenceLevel !== 'INSUFFICIENT');
  const clinicianQuestions = buildClinicianQuestions(result, insights.length);

  return (
    <div className="space-y-8">
      <PageHeader
        title={days === 7 ? 'Your week' : 'Your month'}
        description={`${dateFmt.format(window.from)} to ${dateFmt.format(window.to)}. Written to be understandable without medical training, and to be useful to bring to an appointment.`}
        action={
          <ButtonLink
            href={days === 7 ? '/app/reports?period=month' : '/app/reports?period=week'}
            variant="secondary"
          >
            {days === 7 ? 'Show the month' : 'Show the week'}
          </ButtonLink>
        }
      />

      {summary.count === 0 ? (
        <EmptyState
          title="No readings in this period"
          icon="📄"
          action={<ButtonLink href="/app/glucose/new">Add a reading</ButtonLink>}
        >
          <p>A report needs readings to report on. Add a few and this page will fill itself in.</p>
        </EmptyState>
      ) : (
        <>
          <section aria-labelledby="report-glucose">
            <Card>
              <CardHeader id="report-glucose" title="Glucose" />
              <dl className="grid gap-4 sm:grid-cols-2">
                <div>
                  <dt className="text-sm font-medium text-ink-muted">Average reading</dt>
                  <dd className="text-xl font-bold tabular-nums">
                    {summary.averageMgdl == null
                      ? '—'
                      : formatGlucose(summary.averageMgdl, unit, profile.locale)}{' '}
                    <span className="text-base font-normal">{unitLabel(unit)}</span>
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-ink-muted">
                    Readings inside your target range
                  </dt>
                  <dd className="text-xl font-bold tabular-nums">
                    {summary.percentInRange == null
                      ? '—'
                      : `${Math.round(summary.percentInRange)}%`}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-ink-muted">
                    How much your readings varied
                  </dt>
                  <dd className="text-xl font-bold tabular-nums">
                    {summary.cv == null ? '—' : `${Math.round(summary.cv * 100)}%`}
                  </dd>
                </div>
                <div>
                  <dt className="text-sm font-medium text-ink-muted">Readings logged</dt>
                  <dd className="text-xl font-bold tabular-nums">
                    {summary.count} over {summary.daysWithReadings} days
                  </dd>
                </div>
              </dl>
              <p className="mt-4 text-ink-muted">{describeTrend(result.trend.classification)}</p>
            </Card>
          </section>

          <section aria-labelledby="report-lifestyle">
            <Card>
              <CardHeader id="report-lifestyle" title="Food, movement and sleep" />
              <dl className="grid gap-4 sm:grid-cols-3">
                <div>
                  <dt className="text-sm font-medium text-ink-muted">Meals logged</dt>
                  <dd className="text-xl font-bold tabular-nums">{mealCount}</dd>
                  <p className="text-sm text-ink-muted">
                    {carbStats._count.carbsG > 0 && carbStats._avg.carbsG != null
                      ? `About ${Math.round(carbStats._avg.carbsG)} g of carbohydrate on average, across the ${carbStats._count.carbsG} meals where you recorded it.`
                      : 'No carbohydrate amounts recorded, so carbohydrate patterns could not be looked at.'}
                  </p>
                </div>
                <div>
                  <dt className="text-sm font-medium text-ink-muted">Activity</dt>
                  <dd className="text-xl font-bold tabular-nums">
                    {activity._sum.durationMin ?? 0} min
                  </dd>
                  <p className="text-sm text-ink-muted">
                    Across {activity._count._all} logged{' '}
                    {activity._count._all === 1 ? 'session' : 'sessions'}.
                  </p>
                </div>
                <div>
                  <dt className="text-sm font-medium text-ink-muted">Sleep</dt>
                  <dd className="text-xl font-bold tabular-nums">
                    {sleep._avg.durationMin == null
                      ? '—'
                      : `${(sleep._avg.durationMin / 60).toFixed(1)} h`}
                  </dd>
                  <p className="text-sm text-ink-muted">
                    {sleep._count._all === 0
                      ? 'No sleep recorded this period.'
                      : `Average across ${sleep._count._all} nights.`}
                  </p>
                </div>
              </dl>
            </Card>
          </section>

          <section aria-labelledby="report-changed">
            <Card>
              <CardHeader
                id="report-changed"
                title="What stood out"
                description="Only observations with enough of your data behind them are listed here."
              />
              {strongest.length === 0 ? (
                <p className="text-ink-muted">
                  Nothing reached the point where DiaLog would call it a pattern this period. That
                  is a finding in itself — it usually means either a steady stretch, or not enough
                  logged days to compare.
                </p>
              ) : (
                <ul className="space-y-4">
                  {strongest.slice(0, 5).map((insight) => (
                    <li key={insight.kind + insight.title}>
                      <h3 className="font-semibold">{insight.title}</h3>
                      <p className="text-ink-muted">{insight.summary}</p>
                      <p className="mt-1 text-sm text-ink-muted">
                        {EVIDENCE_LABELS[insight.evidenceLevel].label} · based on{' '}
                        {insight.sampleSize} records
                      </p>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          </section>

          <section aria-labelledby="report-questions">
            <Card>
              <CardHeader
                id="report-questions"
                title="Questions you might raise with your healthcare professional"
                description="These are conversation starters drawn from your data, not recommendations."
              />
              <ul className="list-disc space-y-2 pl-5 text-ink-muted">
                {clinicianQuestions.map((question, index) => (
                  <li key={index}>{question}</li>
                ))}
              </ul>
              <div className="mt-5">
                <Callout tone="info" icon="ⓘ" title="Taking this to an appointment">
                  You can download your full history from Settings as a file, so your healthcare
                  professional can see the underlying readings rather than only this summary.
                </Callout>
              </div>
            </Card>
          </section>
        </>
      )}

      <MedicalDisclaimer />
    </div>
  );
}

function describeTrend(classification: string): string {
  switch (classification) {
    case 'improving':
      return 'Across this period your readings drifted downwards. A single period is not a trend on its own, so this is worth watching rather than concluding from.';
    case 'rising':
      return 'Across this period your readings drifted upwards. Many ordinary things can do that — illness, a change in routine, different timing of tests — so it is a prompt to look, not a verdict.';
    case 'more-variable':
      return 'Your readings were more spread out than earlier in the period, which usually means the days differed more from each other than usual.';
    case 'stable':
      return 'Your readings held broadly steady across this period.';
    default:
      return 'There were not enough days with readings in this period to say anything about a trend.';
  }
}

/** Questions are derived from what the data does and does not support. */
function buildClinicianQuestions(
  result: {
    summary: { percentInRange: number | null; cv: number | null };
    dataQuality: { skippedAnalyses: unknown[] };
  },
  insightCount: number,
): string[] {
  const questions: string[] = [];
  const { percentInRange, cv } = result.summary;

  if (percentInRange != null && percentInRange < 60) {
    questions.push(
      'A good share of my readings sat outside the range I have been aiming for. Is that range still the right one for me?',
    );
  }
  if (cv != null && cv > 0.36) {
    questions.push(
      'My readings varied quite a lot from one to the next this period. Is that variation something we should look into?',
    );
  }
  if (result.dataQuality.skippedAnalyses.length > 0) {
    questions.push(
      'There are things my data could not answer because I do not test often enough. When and how often would be most useful for you to see?',
    );
  }
  if (insightCount > 0) {
    questions.push(
      'Here are the patterns my tracking picked up. Do any of them match what you would expect?',
    );
  }
  questions.push(
    'Is there anything you would like me to start recording that I am not recording now?',
  );
  return questions;
}
