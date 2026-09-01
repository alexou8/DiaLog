import type { Metadata } from 'next';
import Link from 'next/link';
import { requireOnboardedUser } from '@/lib/auth/current-user';
import { prisma } from '@/lib/db/prisma';
import { pageGlucose, RECORD_TYPES, isRecordType, type RecordType } from '@/lib/db/health-records';
import { classifyGlucose } from '@/lib/domain/thresholds';
import { formatGlucose, unitLabel } from '@/lib/domain/units';
import { ButtonLink, Card, EmptyState, PageHeader } from '@/components/ui';
import { buttonVariants } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { DeleteRecordButton } from './DeleteRecordButton';

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
  summary: string;
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

  const dtFmt = new Intl.DateTimeFormat(locale, {
    timeZone: timezone,
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });

  const { rows, nextCursor } = await loadPage({
    type,
    userId: user.id,
    cursor,
    locale,
    timezone,
    glucoseUnit,
    range: { lowMgdl: targetLowMgdl, highMgdl: targetHighMgdl },
    dtFmt,
  });

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
            <ul>
              {rows.map((row) => (
                <li
                  key={row.id}
                  className="flex flex-wrap items-start justify-between gap-3 border-b border-line py-3 last:border-0"
                >
                  <div className="min-w-0">
                    <p className="text-base font-semibold">{row.summary}</p>
                    <p className="text-sm text-ink-muted">{dtFmt.format(row.when)}</p>
                    <p className="text-sm text-ink-muted">{row.provenance}</p>
                  </div>
                  <DeleteRecordButton type={type} id={row.id} label="this record" />
                </li>
              ))}
            </ul>
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
  timezone: string;
  glucoseUnit: 'MGDL' | 'MMOLL';
  range: { lowMgdl: number; highMgdl: number };
  dtFmt: Intl.DateTimeFormat;
}): Promise<{ rows: Row[]; nextCursor?: string }> {
  const { type, userId, cursor, locale, glucoseUnit, range } = args;

  switch (type) {
    case 'glucose': {
      const { rows, nextCursor } = await pageGlucose({ userId, take: PAGE_SIZE, cursor });
      return {
        rows: rows.map((r) => ({
          id: r.id,
          summary: `${formatGlucose(r.valueMgdl, glucoseUnit, locale)} ${unitLabel(glucoseUnit)} · ${classifyGlucose(r.valueMgdl, range).label}`,
          when: r.takenAt,
          provenance: provenanceFrom(r.importBatch),
        })),
        nextCursor,
      };
    }
    case 'meal': {
      const found = await prisma.meal.findMany({
        where: { userId },
        orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
        take: PAGE_SIZE + 1,
        ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
        include: { importBatch: { select: { filename: true, connectorName: true } } },
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((m) => ({
          id: m.id,
          summary: `${m.description}${m.carbsG != null ? ` · ${Math.round(m.carbsG)} g carbs` : ''}`,
          when: m.takenAt,
          provenance: provenanceFrom(m.importBatch),
        })),
        nextCursor,
      };
    }
    case 'exercise': {
      const found = await prisma.exerciseSession.findMany({
        where: { userId },
        orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
        take: PAGE_SIZE + 1,
        ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((s) => ({
          id: s.id,
          summary: `${s.activity} · ${s.durationMin} min`,
          when: s.takenAt,
          provenance: 'Added by you',
        })),
        nextCursor,
      };
    }
    case 'sleep': {
      const found = await prisma.sleepSession.findMany({
        where: { userId },
        orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
        take: PAGE_SIZE + 1,
        ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((s) => ({
          id: s.id,
          summary: `${Math.floor(s.durationMin / 60)}h ${s.durationMin % 60}m asleep`,
          when: s.takenAt,
          provenance: 'Added by you',
        })),
        nextCursor,
      };
    }
    case 'medication': {
      const found = await prisma.medicationEvent.findMany({
        where: { userId },
        orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
        take: PAGE_SIZE + 1,
        ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((m) => ({
          id: m.id,
          summary: `${m.name}${m.dose ? ` · ${m.dose}` : ''}`,
          when: m.takenAt,
          provenance: 'Added by you',
        })),
        nextCursor,
      };
    }
    case 'weight': {
      const found = await prisma.weightMeasurement.findMany({
        where: { userId },
        orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
        take: PAGE_SIZE + 1,
        ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((w) => ({
          id: w.id,
          summary: `${new Intl.NumberFormat(locale, { minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(w.weightKg)} kg`,
          when: w.takenAt,
          provenance: 'Added by you',
        })),
        nextCursor,
      };
    }
    case 'bloodPressure': {
      const found = await prisma.bloodPressureMeasurement.findMany({
        where: { userId },
        orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
        take: PAGE_SIZE + 1,
        ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((b) => ({
          id: b.id,
          summary: `${b.systolic}/${b.diastolic} mmHg`,
          when: b.takenAt,
          provenance: 'Added by you',
        })),
        nextCursor,
      };
    }
    case 'mood': {
      const found = await prisma.moodEntry.findMany({
        where: { userId },
        orderBy: [{ takenAt: 'desc' }, { id: 'desc' }],
        take: PAGE_SIZE + 1,
        ...(cursor ? { cursor: { id: cursor }, skip: 1 } : {}),
      });
      const { page, nextCursor } = splitPage(found);
      return {
        rows: page.map((m) => ({
          id: m.id,
          summary: `Mood ${m.mood}/5${m.stress != null ? ` · Stress ${m.stress}/5` : ''}`,
          when: m.takenAt,
          provenance: 'Added by you',
        })),
        nextCursor,
      };
    }
  }
}

function splitPage<T extends { id: string }>(rows: T[]): { page: T[]; nextCursor?: string } {
  const hasMore = rows.length > PAGE_SIZE;
  const page = hasMore ? rows.slice(0, PAGE_SIZE) : rows;
  return { page, nextCursor: hasMore ? page[page.length - 1]?.id : undefined };
}

function provenanceFrom(
  importBatch: { filename: string; connectorName: string } | null | undefined,
): string {
  if (!importBatch) return 'Added by you';
  return `Imported from ${importBatch.filename} via ${importBatch.connectorName}`;
}
