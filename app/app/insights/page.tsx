import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { analyzeUser, defaultWindow, insightsFor } from '@/lib/services/analytics-service';
import { EVIDENCE_LABELS, EVIDENCE_THRESHOLDS } from '@/lib/domain/evidence';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import {
  ButtonLink,
  Callout,
  Card,
  CardHeader,
  EmptyState,
  PageHeader,
  WhyThis,
} from '@/components/ui';
import { InsightCardView } from '@/components/InsightCardView';

export const metadata: Metadata = { title: 'Insights' };
export const dynamic = 'force-dynamic';

export default async function InsightsPage() {
  const user = await requireOnboardedUser();
  const { profile } = user;
  const window = defaultWindow(30);
  const { result } = await analyzeUser(user.id, profile, window);
  const insights = insightsFor(result);

  const unusual = result.anomalies;
  // A one- or two-day group is not a recurring pattern; the insight builder
  // applies the same floor.
  const visiblePatterns = (result.dayPatterns ?? []).filter((pattern) => pattern.size >= 3);
  const importance = result.featureImportance;

  return (
    <div className="space-y-8">
      <PageHeader
        title="Insights"
        description="What DiaLog can and cannot tell from your last 30 days — with the evidence behind each observation."
      />

      {insights.length === 0 ? (
        <EmptyState
          title="Nothing to report yet"
          icon="🔎"
          action={<ButtonLink href="/app/glucose/new">Add a reading</ButtonLink>}
        >
          <p>
            Observations appear once there is enough of your own history to compare against. That is
            usually a couple of weeks of regular readings.
          </p>
        </EmptyState>
      ) : (
        <section aria-labelledby="observations">
          <h2 id="observations" className="sr-only">
            Observations
          </h2>
          <ul className="space-y-4">
            {insights.map((insight) => (
              <InsightCardView key={insight.kind + insight.title} insight={insight} />
            ))}
          </ul>
        </section>
      )}

      {unusual.length > 0 ? (
        <section aria-labelledby="unusual">
          <Card>
            <CardHeader
              id="unusual"
              title="Readings that stand out from your own pattern"
              description="These are unusual compared with your typical reading at that time of day. Unusual does not mean wrong or dangerous — it means worth a look."
            />
            <ul className="space-y-2">
              {unusual.slice(0, 8).map((flag) => (
                <li key={flag.readingId} className="border-b border-line pb-2 last:border-0">
                  <p className="font-semibold tabular-nums">
                    {formatGlucose(flag.valueMgdl, profile.glucoseUnit, profile.locale)}{' '}
                    {unitLabel(profile.glucoseUnit)}
                    <span className="ml-2 font-normal text-ink-muted">
                      {new Intl.DateTimeFormat(profile.locale, {
                        timeZone: profile.timezone,
                        weekday: 'short',
                        month: 'short',
                        day: 'numeric',
                        hour: 'numeric',
                        minute: '2-digit',
                      }).format(flag.takenAt)}
                    </span>
                  </p>
                  <p className="text-sm text-ink-muted">
                    Your typical {flag.bucket} reading is around{' '}
                    {formatGlucose(flag.baselineMedianMgdl, profile.glucoseUnit, profile.locale)}{' '}
                    {unitLabel(profile.glucoseUnit)}, based on {flag.baselineSize} readings.
                  </p>
                </li>
              ))}
            </ul>
            <WhyThis>
              <p>
                DiaLog compares each reading with the middle value of your other readings at the
                same time of day, using a measure of spread that is not thrown off by one extreme
                value. A reading is flagged when it sits far outside that spread. The comparison is
                always against your own history, never against a population average.
              </p>
            </WhyThis>
          </Card>
        </section>
      ) : null}

      {visiblePatterns.length > 0 ? (
        <section aria-labelledby="day-patterns">
          <Card>
            <CardHeader
              id="day-patterns"
              title="Kinds of days you tend to have"
              description="Your days grouped by how they look across glucose, food, movement and sleep."
            />
            <ul className="grid gap-4 sm:grid-cols-2">
              {visiblePatterns.map((pattern, index) => (
                <li key={index} className="rounded-xl border border-line bg-surface-sunken p-4">
                  <h3 className="font-semibold">{pattern.label}</h3>
                  <p className="mt-1 text-sm text-ink-muted">
                    {pattern.size} {pattern.size === 1 ? 'day' : 'days'} in the last 30
                  </p>
                  <p className="mt-2 text-sm">
                    Days like these: {pattern.dayKeys.slice(0, 3).join(', ')}
                    {pattern.dayKeys.length > 3 ? ` and ${pattern.dayKeys.length - 3} more` : ''}.
                  </p>
                </li>
              ))}
            </ul>
            <WhyThis>
              <p>
                This grouping is produced by a clustering algorithm looking only at your own days.
                It describes what your days have in common — it does not say one kind of day causes
                another, and it cannot see anything you did not log.
              </p>
            </WhyThis>
          </Card>
        </section>
      ) : null}

      {importance ? (
        <section aria-labelledby="importance">
          <Card>
            <CardHeader
              id="importance"
              title="What moves together with your readings"
              description="Ranked by how strongly each logged factor is associated with your post-meal readings."
            />
            <ol className="space-y-2">
              {importance.coefficients.slice(0, 6).map((coefficient) => (
                <li
                  key={coefficient.feature}
                  className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-b border-line pb-2 last:border-0"
                >
                  <span>{coefficient.label}</span>
                  <span className="text-sm tabular-nums text-ink-muted">
                    goes with {coefficient.standardizedCoefficient >= 0 ? 'higher' : 'lower'}{' '}
                    readings · strength {Math.abs(coefficient.standardizedCoefficient).toFixed(2)}
                  </span>
                </li>
              ))}
            </ol>
            <Callout tone="notice" icon="⚠" title="Association is not cause">
              {importance.warning} Based on {importance.sampleSize} readings after meals.
            </Callout>
          </Card>
        </section>
      ) : null}

      <section aria-labelledby="quality">
        <Card>
          <CardHeader
            id="quality"
            title="What DiaLog cannot tell yet"
            description="Being honest about the limits of your current data is part of the job."
          />
          <dl className="space-y-3">
            <div>
              <dt className="font-semibold">Days with readings</dt>
              <dd className="text-ink-muted">
                {result.dataQuality.coverageDays} of the last {result.dataQuality.periodDays} days.
              </dd>
            </div>
            <div>
              <dt className="font-semibold">Records logged in this period</dt>
              <dd className="text-ink-muted">
                {result.dataQuality.counts.glucose} readings, {result.dataQuality.counts.meals}{' '}
                meals, {result.dataQuality.counts.exercise} activity sessions,{' '}
                {result.dataQuality.counts.sleep} sleep records.
              </dd>
            </div>
          </dl>

          {result.dataQuality.skippedAnalyses.length > 0 ? (
            <>
              <h3 className="mt-5 font-semibold">Comparisons that were skipped</h3>
              <ul className="mt-2 space-y-2">
                {result.dataQuality.skippedAnalyses.map((skipped) => (
                  <li
                    key={skipped.analysis}
                    className="rounded-lg border border-line bg-surface-sunken p-3"
                  >
                    <p className="font-medium">{skipped.analysis}</p>
                    <p className="text-sm text-ink-muted">{skipped.reason}</p>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="mt-4 text-ink-muted">
              Every comparison DiaLog knows how to run had enough data this period.
            </p>
          )}

          <WhyThis label="How much data does each kind of comparison need?">
            <ul className="list-disc space-y-1 pl-5">
              {Object.entries(EVIDENCE_THRESHOLDS).map(([name, thresholds]) => (
                <li key={name}>
                  <strong className="capitalize">{name}</strong>: {thresholds.early}+ records for an
                  early signal, {thresholds.emerging}+ for an emerging pattern,{' '}
                  {thresholds.consistent}+ for a consistent one.
                </li>
              ))}
            </ul>
            <p className="mt-3">
              {EVIDENCE_LABELS.INSUFFICIENT.description} These thresholds are set once, in one
              place, and applied the same way to every observation.
            </p>
          </WhyThis>
        </Card>
      </section>
    </div>
  );
}
