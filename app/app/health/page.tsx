import type { Metadata } from 'next';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { prisma } from '@/lib/db/prisma';
import { Badge, ButtonLink, Callout, Card, CardHeader, EmptyState, Icon } from '@/components/ui';
import { DeleteRecordButton } from './DeleteRecordButton';

export const metadata: Metadata = { title: 'Health' };
export const dynamic = 'force-dynamic';

const ADDED_MESSAGES: Record<string, string> = {
  sleep: 'Sleep saved.',
  medication: 'Medication event saved.',
  weight: 'Weight saved.',
  bloodPressure: 'Blood pressure saved.',
  mood: 'Mood saved.',
  hydration: 'Drink saved.',
  symptom: 'Note saved.',
};

export default async function HealthPage({
  searchParams,
}: {
  searchParams: Promise<{ added?: string }>;
}) {
  const user = await requireOnboardedUser();
  const { locale, timezone } = user.profile;
  const { added } = await searchParams;
  const addedMessage = added ? (ADDED_MESSAGES[added] ?? null) : null;

  const [sleep, medication, weight, bloodPressure, mood, hydration, symptoms] = await Promise.all([
    prisma.sleepSession.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 5,
      select: { id: true, takenAt: true, endedAt: true, durationMin: true, quality: true },
    }),
    prisma.medicationEvent.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 5,
      select: { id: true, takenAt: true, name: true, dose: true, route: true },
    }),
    prisma.weightMeasurement.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 5,
      select: { id: true, takenAt: true, weightKg: true },
    }),
    prisma.bloodPressureMeasurement.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 5,
      select: { id: true, takenAt: true, systolic: true, diastolic: true, pulse: true },
    }),
    prisma.moodEntry.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 5,
      select: { id: true, takenAt: true, mood: true, stress: true },
    }),
    prisma.hydrationEvent.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 5,
      select: { id: true, takenAt: true, volumeMl: true },
    }),
    prisma.symptomEntry.findMany({
      where: { userId: user.id },
      orderBy: { takenAt: 'desc' },
      take: 5,
      select: { id: true, takenAt: true, symptom: true, severity: true, note: true },
    }),
  ]);

  const dtFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Health</h1>
        <p className="mt-1 text-ink-muted">Everything else you track, in one place.</p>
      </header>

      <div aria-live="polite">
        {addedMessage ? (
          <p className="rounded-xl border border-positive/40 bg-positive-soft p-3 text-sm font-medium text-positive">
            <Icon name="ok" className="shrink-0" />
            {addedMessage}
          </p>
        ) : null}
      </div>

      {/* --------------------------------------------------------- Sleep */}
      <section aria-labelledby="sleep-heading">
        <Card>
          <CardHeader
            id="sleep-heading"
            title="Sleep"
            action={
              <ButtonLink href="/app/health/sleep/new" variant="secondary">
                Add
              </ButtonLink>
            }
          />
          {sleep.length === 0 ? (
            <EmptyState title="No sleep logged yet" icon="sleep">
              <p>Log your bedtime and wake time to start seeing patterns.</p>
            </EmptyState>
          ) : (
            <ul>
              {sleep.map((s) => (
                <li
                  key={s.id}
                  className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 last:border-0"
                >
                  <div>
                    <p className="font-semibold">
                      {Math.floor(s.durationMin / 60)}h {s.durationMin % 60}m
                    </p>
                    <p className="text-sm text-ink-muted">
                      {dtFmt.format(s.takenAt)} – {dtFmt.format(s.endedAt)}
                      {s.quality != null ? ` · Quality ${s.quality}/5` : ''}
                    </p>
                  </div>
                  <DeleteRecordButton type="sleep" id={s.id} label="this sleep entry" />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      {/* ----------------------------------------------------- Medication */}
      <section aria-labelledby="medication-heading">
        <Card>
          <CardHeader
            id="medication-heading"
            title="Medication"
            action={
              <ButtonLink href="/app/health/medication/new" variant="secondary">
                Add
              </ButtonLink>
            }
          />
          <Callout tone="info" icon="info">
            DiaLog records medication events so you can see your own timing next to your readings.
            It never calculates or suggests doses.
          </Callout>
          {medication.length === 0 ? (
            <EmptyState title="No medication events yet" icon="medication">
              <p>Record when you take something to see it alongside your readings.</p>
            </EmptyState>
          ) : (
            <ul className="mt-4">
              {medication.map((m) => (
                <li
                  key={m.id}
                  className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 last:border-0"
                >
                  <div>
                    <p className="font-semibold">{m.name}</p>
                    <p className="text-sm text-ink-muted">
                      {dtFmt.format(m.takenAt)}
                      {m.dose ? ` · ${m.dose}` : ''}
                      {m.route ? ` · ${m.route}` : ''}
                    </p>
                  </div>
                  <DeleteRecordButton type="medication" id={m.id} label="this medication event" />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      {/* --------------------------------------------------------- Weight */}
      <section aria-labelledby="weight-heading">
        <Card>
          <CardHeader
            id="weight-heading"
            title="Weight"
            action={
              <ButtonLink href="/app/health/weight/new" variant="secondary">
                Add
              </ButtonLink>
            }
          />
          {weight.length === 0 ? (
            <EmptyState title="No weight logged yet" icon="weight">
              <p>Log your weight to track it over time.</p>
            </EmptyState>
          ) : (
            <ul>
              {weight.map((w) => (
                <li
                  key={w.id}
                  className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 last:border-0"
                >
                  <div>
                    <p className="font-semibold">{formatWeight(w.weightKg, locale)} kg</p>
                    <p className="text-sm text-ink-muted">{dtFmt.format(w.takenAt)}</p>
                  </div>
                  <DeleteRecordButton type="weight" id={w.id} label="this weight entry" />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      {/* -------------------------------------------------- Blood pressure */}
      <section aria-labelledby="bp-heading">
        <Card>
          <CardHeader
            id="bp-heading"
            title="Blood pressure"
            action={
              <ButtonLink href="/app/health/blood-pressure/new" variant="secondary">
                Add
              </ButtonLink>
            }
          />
          {bloodPressure.length === 0 ? (
            <EmptyState title="No blood pressure logged yet" icon="bloodPressure">
              <p>Log a reading to start tracking your blood pressure over time.</p>
            </EmptyState>
          ) : (
            <ul>
              {bloodPressure.map((b) => (
                <li
                  key={b.id}
                  className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 last:border-0"
                >
                  <div>
                    <p className="font-semibold">
                      {b.systolic}/{b.diastolic}{' '}
                      <span className="font-normal text-ink-muted text-sm">mmHg</span>
                    </p>
                    <p className="text-sm text-ink-muted">
                      {dtFmt.format(b.takenAt)}
                      {b.pulse != null ? ` · Pulse ${b.pulse} bpm` : ''}
                    </p>
                  </div>
                  <DeleteRecordButton
                    type="bloodPressure"
                    id={b.id}
                    label="this blood pressure reading"
                  />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      {/* -------------------------------------------------- Mood / stress */}
      <section aria-labelledby="mood-heading">
        <Card>
          <CardHeader
            id="mood-heading"
            title="Mood & stress"
            action={
              <ButtonLink href="/app/health/mood/new" variant="secondary">
                Add
              </ButtonLink>
            }
          />
          {mood.length === 0 ? (
            <EmptyState title="No mood entries yet" icon="mood">
              <p>A quick check-in helps you see how you have been feeling over time.</p>
            </EmptyState>
          ) : (
            <ul>
              {mood.map((m) => (
                <li
                  key={m.id}
                  className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 last:border-0"
                >
                  <div>
                    <p className="flex flex-wrap items-center gap-2">
                      <Badge tone="neutral">Mood {m.mood}/5</Badge>
                      {m.stress != null ? <Badge tone="neutral">Stress {m.stress}/5</Badge> : null}
                    </p>
                    <p className="mt-1 text-sm text-ink-muted">{dtFmt.format(m.takenAt)}</p>
                  </div>
                  <DeleteRecordButton type="mood" id={m.id} label="this mood entry" />
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      {/* ------------------------------------------------------- Hydration */}
      <section aria-labelledby="hydration-heading">
        <Card>
          <CardHeader
            id="hydration-heading"
            title="Drinks"
            action={
              <ButtonLink href="/app/health/hydration/new" variant="secondary">
                Add
              </ButtonLink>
            }
          />
          {hydration.length === 0 ? (
            <EmptyState title="No drinks logged yet" icon="\u{1F4A7}">
              <p>
                A rough note of what you drink adds context to your readings. It is entirely
                optional.
              </p>
            </EmptyState>
          ) : (
            <ul>
              {hydration.map((entry) => (
                <li
                  key={entry.id}
                  className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 last:border-0"
                >
                  <div>
                    <p className="font-semibold tabular-nums">{entry.volumeMl} mL</p>
                    <p className="mt-1 text-sm text-ink-muted">{dtFmt.format(entry.takenAt)}</p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>

      {/* -------------------------------------------------------- Symptoms */}
      <section aria-labelledby="symptom-heading">
        <Card>
          <CardHeader
            id="symptom-heading"
            title="How you've been feeling"
            description="Notes in your own words. DiaLog records them, it does not interpret them."
            action={
              <ButtonLink href="/app/health/symptom/new" variant="secondary">
                Add
              </ButtonLink>
            }
          />
          {symptoms.length === 0 ? (
            <EmptyState title="Nothing noted yet" icon="\u{1F4DD}">
              <p>
                If you notice something (feeling tired, shaky, unusually thirsty), noting it here
                gives you something concrete to mention at your next appointment.
              </p>
            </EmptyState>
          ) : (
            <ul>
              {symptoms.map((entry) => (
                <li
                  key={entry.id}
                  className="flex flex-wrap items-center justify-between gap-3 border-b border-line py-3 last:border-0"
                >
                  <div>
                    <p className="font-semibold">{entry.symptom}</p>
                    <p className="mt-1 text-sm text-ink-muted">
                      {dtFmt.format(entry.takenAt)}
                      {entry.severity != null ? ` \u00B7 strength ${entry.severity}/5` : ''}
                    </p>
                    {entry.note ? <p className="mt-1 text-sm">{entry.note}</p> : null}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </section>
    </div>
  );
}

function formatWeight(kg: number, locale: string): string {
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(kg);
}
