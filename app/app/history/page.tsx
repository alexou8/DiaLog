import type { Metadata } from 'next';
import Link from 'next/link';
import type { DataSource } from '@prisma/client';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { prisma } from '@/lib/db/prisma';
import { pageGlucose, RECORD_TYPES, isRecordType, type RecordType } from '@/lib/db/health-records';
import { classifyGlucose } from '@/lib/domain/thresholds';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import { dayKeyInZone } from '@/lib/domain/time';
import { ButtonLink, Card, EmptyState, PageHeader } from '@/components/ui';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { DeleteRecordButton } from '@/components/records/DeleteRecordButton';

export const metadata: Metadata = { title: 'History' };
export const dynamic = 'force-dynamic';

const TAB_LABELS: Record<RecordType, string> = {
  glucose: 'Glucose',
  meal: 'Meals',
  exercise: 'Activity',
  sleep: 'Sleep',
  medication: 'Medication',
  weight: 'Weight',
  bloodPressure: 'Blood pressure',
  mood: 'Mood',
};

const PAGE_SIZE = 20;

interface Row {
  id: string;
  /** What the row shows: the value plus any status wording. */
  summary: string;
  /** What kind of record this is, for the delete control's accessible name. */
  noun: string;
  /** The bare value or description, for the delete control's accessible name. */
  detail: string;
  when: Date;
  provenance: string;
}

export default async function HistoryPage({
  searchParams,
}: {
  searchParams: Promise<{ type?: string; cursor?: string }>;
}) {
  const user = await requireOnboardedUser();
  const { locale, timezone, glucoseUnit, targetLowMgdl, targetHighMgdl } = user.profile;
  const params = await searchParams;
  const type: RecordType = isRecordType(params.type ?? '')
    ? (params.type as RecordType)
    : 'glucose';
  const cursor = params.cursor;

  // Full date and time: this names the record in its delete control, where
  // the day heading the row sits under is not available to a screen reader
  // navigating by buttons.
  const dtFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
  const timeFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    hour: 'numeric',
    minute: '2-digit',
  });
  const dayFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    weekday: 'long',
    month: 'long',
    day: 'numeric',
  });
  const dayWithYearFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });
  const thisYear = dayKeyInZone(new Date(), timezone).slice(0, 4);

  const { rows, nextCursor } = await loadPage({
    type,
    userId: user.id,
    cursor,
    locale,
    glucoseUnit,
    range: { lowMgdl: targetLowMgdl, highMgdl: targetHighMgdl },
  });

  // One heading per local day, so a long list is navigable by heading and the
  // rows themselves need only the time. Rows arrive newest first, so insertion
  // order is already day order. A day can continue onto the next page of
  // results; its heading then simply repeats there.
  const days = new Map<string, Row[]>();
  for (const row of rows) {
    const key = dayKeyInZone(row.when, timezone);
    const bucket = days.get(key);
    if (bucket) bucket.push(row);
    else days.set(key, [row]);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="History"
        description="Browse and manage everything you have logged."
        action={
          <ButtonLink href="/app/settings" variant="secondary">
            Export your data
          </ButtonLink>
        }
      />

      {/* This section deliberately has no loading.tsx. With one, Next 15
          kept the boundary mounted across a change of `?type=` and the
          navigation never committed: the filter did nothing. The same applies
          to any section whose links stay inside it; see the comment in
          app/app/glucose/page.tsx. Guarded by dashboard-and-insights.spec.ts. */}
      <nav aria-label="Record type" className="-mx-1 overflow-x-auto pb-1">
        <ul className="flex min-w-max gap-1 px-1">
          {RECORD_TYPES.map((t) => {
            const active = t === type;
            return (
              <li key={t}>
                <Link
                  href={`/app/history?type=${t}`}
                  aria-current={active ? 'page' : undefined}
                  // Links, not a shadcn Tabs widget: the record type lives in
                  // the URL so a filter can be shared, bookmarked and paged
                  // through on the server. `aria-current="page"` is the right
                  // announcement for that; `role="tab"` would be a lie. They
                  // borrow the button styling so they still read as one system.
                  className={cn(
                    buttonVariants({ variant: active ? 'primary' : 'secondary', size: 'sm' }),
                    'dl-target',
                  )}
                >
                  {TAB_LABELS[t]}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <Card>
        {rows.length === 0 ? (
          <EmptyState title={`No ${TAB_LABELS[type].toLowerCase()} records yet`} icon="history">
            <p>
              Once you log some, they will show up here with the option to review or delete them.
            </p>
          </EmptyState>
        ) : (
          <>
            <div className="space-y-6">
              {[...days.entries()].map(([dayKey, dayRows]) => {
                const first = dayRows[0]!.when;
                // The year is only worth saying when it is not this one.
                const heading = (dayKey.startsWith(thisYear) ? dayFmt : dayWithYearFmt).format(
                  first,
                );
                return (
                  <section key={dayKey} aria-labelledby={`day-${dayKey}`}>
                    <h2 id={`day-${dayKey}`} className="text-base font-semibold text-ink-muted">
                      {heading}
                    </h2>
                    <ul>
                      {dayRows.map((row) => (
                        <li
                          key={row.id}
                          className="flex flex-wrap items-start justify-between gap-3 border-b border-line py-3 last:border-0"
                        >
                          <div className="min-w-0">
                            <p className="text-base font-semibold">{row.summary}</p>
                            <p className="text-sm text-ink-muted">{timeFmt.format(row.when)}</p>
                            <p className="break-words text-sm text-ink-muted">{row.provenance}</p>
                          </div>
                          <DeleteRecordButton
                            type={type}
                            id={row.id}
                            noun={row.noun}
                            when={dtFmt.format(row.when)}
                            detail={row.detail}
                          />
                        </li>
                      ))}
                    </ul>
                  </section>
                );
              })}
            </div>
            {nextCursor ? (
              <div className="mt-4 flex justify-center">
                <ButtonLink
                  href={`/app/history?type=${type}&cursor=${nextCursor}`}
                  variant="secondary"
                >
                  Show older records
                </ButtonLink>
              </div>
            ) : null}
          </>
        )}
      </Card>
    </div>
  );
}

async function loadPage(args: {
  type: RecordType;
  userId: string;
  cursor?: string;
  locale: string;
  glucoseUnit: 'MGDL' | 'MMOLL';
  range: { lowMgdl: number; highMgdl: number };
}): Promise<{ rows: Row[]; nextCursor?: string }> {
  const { type, userId, cursor, locale, glucoseUnit, range } = args;
  // The same paging for every type. Selecting columns (rather than reading
  // whole rows) keeps each record's raw import payload out of the page.
  const paging = (after: object) => ({
    where: { userId, ...after },
    orderBy: [{ takenAt: 'desc' as const }, { id: 'desc' as const }],
    take: PAGE_SIZE + 1,
  });
  const batch = { select: { filename: true, connectorName: true } };

  switch (type) {
    case 'glucose': {
      const { rows, nextCursor } = await pageGlucose({ userId, take: PAGE_SIZE, cursor });
      return {
        rows: rows.map((r) => {
          const value = `${formatGlucose(r.valueMgdl, glucoseUnit, locale)} ${unitLabel(glucoseUnit)}`;
          return {
            id: r.id,
            summary: `${value} · ${classifyGlucose(r.valueMgdl, range).label}`,
            noun: 'glucose reading',
            detail: value,
            when: r.takenAt,
            provenance: provenanceFrom(r.source, r.importBatch),
          };
        }),
        nextCursor,
      };
    }
    case 'meal': {
      const after = await afterCursor((args) => prisma.meal.findFirst(args), userId, cursor);
      const found = await prisma.meal.findMany({
        ...paging(after),
        select: {
          id: true,
          takenAt: true,
          description: true,
          carbsG: true,
          source: true,
          importBatch: batch,
        },
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((m) => ({
          id: m.id,
          summary: `${m.description}${m.carbsG != null ? ` · ${Math.round(m.carbsG)} g carbs` : ''}`,
          noun: 'meal',
          detail: m.description,
          when: m.takenAt,
          provenance: provenanceFrom(m.source, m.importBatch),
        })),
        nextCursor,
      };
    }
    case 'exercise': {
      const after = await afterCursor(
        (args) => prisma.exerciseSession.findFirst(args),
        userId,
        cursor,
      );
      const found = await prisma.exerciseSession.findMany({
        ...paging(after),
        select: {
          id: true,
          takenAt: true,
          activity: true,
          durationMin: true,
          source: true,
          importBatch: batch,
        },
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((s) => ({
          id: s.id,
          summary: `${s.activity} · ${s.durationMin} min`,
          noun: 'activity',
          detail: `${s.activity}, ${s.durationMin} min`,
          when: s.takenAt,
          provenance: provenanceFrom(s.source, s.importBatch),
        })),
        nextCursor,
      };
    }
    case 'sleep': {
      const after = await afterCursor(
        (args) => prisma.sleepSession.findFirst(args),
        userId,
        cursor,
      );
      const found = await prisma.sleepSession.findMany({
        ...paging(after),
        select: { id: true, takenAt: true, durationMin: true, source: true, importBatch: batch },
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((s) => {
          const asleep = `${Math.floor(s.durationMin / 60)}h ${s.durationMin % 60}m`;
          return {
            id: s.id,
            summary: `${asleep} asleep`,
            noun: 'sleep entry',
            detail: asleep,
            when: s.takenAt,
            provenance: provenanceFrom(s.source, s.importBatch),
          };
        }),
        nextCursor,
      };
    }
    case 'medication': {
      const after = await afterCursor(
        (args) => prisma.medicationEvent.findFirst(args),
        userId,
        cursor,
      );
      const found = await prisma.medicationEvent.findMany({
        ...paging(after),
        select: {
          id: true,
          takenAt: true,
          name: true,
          dose: true,
          source: true,
          importBatch: batch,
        },
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((m) => ({
          id: m.id,
          summary: `${m.name}${m.dose ? ` · ${m.dose}` : ''}`,
          noun: 'medication event',
          detail: m.name,
          when: m.takenAt,
          provenance: provenanceFrom(m.source, m.importBatch),
        })),
        nextCursor,
      };
    }
    case 'weight': {
      const after = await afterCursor(
        (args) => prisma.weightMeasurement.findFirst(args),
        userId,
        cursor,
      );
      const found = await prisma.weightMeasurement.findMany({
        ...paging(after),
        select: { id: true, takenAt: true, weightKg: true, source: true, importBatch: batch },
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((w) => {
          const weight = `${new Intl.NumberFormat(locale, { minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(w.weightKg)} kg`;
          return {
            id: w.id,
            summary: weight,
            noun: 'weight entry',
            detail: weight,
            when: w.takenAt,
            provenance: provenanceFrom(w.source, w.importBatch),
          };
        }),
        nextCursor,
      };
    }
    case 'bloodPressure': {
      // No import batch relation on this model: provenance comes from `source`.
      const after = await afterCursor(
        (args) => prisma.bloodPressureMeasurement.findFirst(args),
        userId,
        cursor,
      );
      const found = await prisma.bloodPressureMeasurement.findMany({
        ...paging(after),
        select: { id: true, takenAt: true, systolic: true, diastolic: true, source: true },
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((b) => ({
          id: b.id,
          summary: `${b.systolic}/${b.diastolic} mmHg`,
          noun: 'blood pressure reading',
          detail: `${b.systolic}/${b.diastolic} mmHg`,
          when: b.takenAt,
          provenance: provenanceFrom(b.source),
        })),
        nextCursor,
      };
    }
    case 'mood': {
      const after = await afterCursor((args) => prisma.moodEntry.findFirst(args), userId, cursor);
      const found = await prisma.moodEntry.findMany({
        ...paging(after),
        select: { id: true, takenAt: true, mood: true, stress: true, source: true },
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((m) => ({
          id: m.id,
          summary: `Mood ${m.mood}/5${m.stress != null ? ` · Stress ${m.stress}/5` : ''}`,
          noun: 'mood entry',
          detail: `Mood ${m.mood}/5`,
          when: m.takenAt,
          provenance: provenanceFrom(m.source),
        })),
        nextCursor,
      };
    }
  }
}

/**
 * Where-clause fragment for "strictly after this record" in (takenAt desc, id
 * desc) order. The cursor arrives from the URL, so it is looked up scoped to
 * the owner (`where: { id, userId }`) and never handed to Prisma as a bare
 * `cursor: { id }`: a record id belonging to another account would otherwise
 * position the page from that account's row. An unknown or foreign cursor
 * simply yields no condition, and the list restarts from the newest record.
 * Same pattern as `pageGlucose`.
 */
async function afterCursor(
  lookup: (args: {
    where: { id: string; userId: string };
    select: { id: true; takenAt: true };
  }) => Promise<{ id: string; takenAt: Date } | null>,
  userId: string,
  cursor?: string,
): Promise<object> {
  if (!cursor) return {};
  const row = await lookup({ where: { id: cursor, userId }, select: { id: true, takenAt: true } });
  if (!row) return {};
  return {
    OR: [{ takenAt: { lt: row.takenAt } }, { takenAt: row.takenAt, id: { lt: row.id } }],
  };
}

function splitPage<T extends { id: string }>(rows: T[]): { page: T[]; nextCursor?: string } {
  const hasMore = rows.length > PAGE_SIZE;
  const page = hasMore ? rows.slice(0, PAGE_SIZE) : rows;
  return { page, nextCursor: hasMore ? page[page.length - 1]?.id : undefined };
}

/**
 * Where a record really came from. Every row used to say "Added by you" unless
 * it was a glucose reading or a meal with an import batch attached, so an
 * imported sleep or weight entry was credited to the user. `source` is
 * recorded on every model, so it backs up the batch: a record whose batch was
 * later deleted (the foreign key sets it null) is still known to be imported.
 */
function provenanceFrom(
  source: DataSource,
  importBatch?: { filename: string; connectorName: string } | null,
): string {
  if (importBatch) return `Imported from ${importBatch.connectorName} (${importBatch.filename})`;
  switch (source) {
    case 'IMPORT':
      return 'Imported from a file';
    case 'DEVICE':
      return 'Recorded by a connected device';
    case 'AI_ASSISTED':
      return 'Suggested by the assistant, saved by you';
    case 'SEED':
      return 'Sample data';
    case 'MANUAL':
      return 'Added by you';
  }
}
