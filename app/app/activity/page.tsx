import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { prisma } from '@/lib/db/prisma';
import { dayKeyInZone, daysAgo, weekdayInZone } from '@/lib/domain/time';
import { ButtonLink, Card, CardHeader, EmptyState, Stat } from '@/components/ui';
import { BarChart, type Bar } from '@/components/charts/BarChart';
import { DeleteRecordButton } from './DeleteRecordButton';

export const metadata: Metadata = { title: 'Activity' };
export const dynamic = 'force-dynamic';

const WEEKDAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const INTENSITY_LABELS: Record<string, string> = {
  LIGHT: 'Light',
  MODERATE: 'Moderate',
  VIGOROUS: 'Vigorous',
};

export default async function ActivityPage() {
  const user = await requireOnboardedUser();
  const { locale, timezone } = user.profile;

  const sessions = await prisma.exerciseSession.findMany({
    where: { userId: user.id },
    orderBy: { takenAt: 'desc' },
    take: 100,
    select: {
      id: true,
      takenAt: true,
      activity: true,
      durationMin: true,
      intensity: true,
      distanceKm: true,
      steps: true,
      note: true,
    },
  });

  if (sessions.length === 0) {
    return (
      <div className="space-y-6">
        <header>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Activity</h1>
        </header>
        <EmptyState
          title="No activity logged yet"
          icon="🚶"
          action={<ButtonLink href="/app/activity/new">Log your first session</ButtonLink>}
        >
          <p>
            A walk, a swim, housework — anything that gets you moving counts. Just the activity and
            how long is enough to start.
          </p>
        </EmptyState>
      </div>
    );
  }

  const dayFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });
  const timeFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    hour: 'numeric',
    minute: '2-digit',
  });

  const groups = new Map<string, typeof sessions>();
  for (const s of sessions) {
    const key = dayKeyInZone(s.takenAt, timezone);
    const bucket = groups.get(key);
    if (bucket) bucket.push(s);
    else groups.set(key, [s]);
  }

  const weekStart = daysAgo(6);
  const thisWeek = sessions.filter((s) => s.takenAt >= weekStart);
  const totalMinutesThisWeek = thisWeek.reduce((sum, s) => sum + s.durationMin, 0);

  const minutesByWeekday = new Array<number>(7).fill(0);
  for (const s of thisWeek) {
    const idx = weekdayInZone(s.takenAt, timezone);
    if (idx >= 0) minutesByWeekday[idx] = (minutesByWeekday[idx] ?? 0) + s.durationMin;
  }
  const bars: Bar[] = WEEKDAY_LABELS.map((label, i) => ({
    label,
    value: minutesByWeekday[i] ?? 0,
  }));

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Activity</h1>
          <p className="mt-1 text-ink-muted">Your recent sessions, grouped by day.</p>
        </div>
        <ButtonLink href="/app/activity/new">
          <span aria-hidden="true">＋</span> Log activity
        </ButtonLink>
      </header>

      <section aria-labelledby="activity-week-heading">
        <Card>
          <CardHeader id="activity-week-heading" title="This week" level={2} />
          <div className="grid gap-3 sm:grid-cols-2">
            <Stat
              label="Total minutes this week"
              value={totalMinutesThisWeek}
              unit="min"
              hint={`${thisWeek.length} ${thisWeek.length === 1 ? 'session' : 'sessions'}`}
            />
          </div>
          <div className="mt-6">
            <BarChart
              bars={bars}
              valueLabel="Minutes"
              title="Minutes by weekday"
              summary="Total minutes of activity logged for each day of the last seven days."
              minSample={0}
              format={(v) => `${Math.round(v)}`}
            />
          </div>
        </Card>
      </section>

      <section aria-labelledby="activity-list-heading">
        <h2 id="activity-list-heading" className="sr-only">
          Activity log
        </h2>
        <div className="space-y-6">
          {[...groups.entries()].map(([dayKey, daySessions]) => (
            <Card key={dayKey}>
              <h3 className="mb-3 text-base font-semibold text-ink-muted">
                {dayFmt.format(daySessions[0]?.takenAt ?? new Date(`${dayKey}T00:00:00`))}
              </h3>
              <ul>
                {daySessions.map((s) => (
                  <li
                    key={s.id}
                    className="flex flex-wrap items-start justify-between gap-3 border-b border-line py-3 last:border-0"
                  >
                    <div className="min-w-0">
                      <p className="text-base font-semibold">{s.activity}</p>
                      <p className="text-sm text-ink-muted">
                        {s.durationMin} min · {INTENSITY_LABELS[s.intensity] ?? s.intensity} ·{' '}
                        {timeFmt.format(s.takenAt)}
                        {s.distanceKm != null ? ` · ${s.distanceKm} km` : ''}
                        {s.steps != null ? ` · ${s.steps} steps` : ''}
                      </p>
                      {s.note ? <p className="mt-1 text-sm">{s.note}</p> : null}
                    </div>
                    <DeleteRecordButton type="exercise" id={s.id} label="this activity" />
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>
      </section>
    </div>
  );
}
