/**
 * A streamed export used to do its first database read only after the route
 * had committed a 200, so a missing profile or a database outage became a
 * 200 with an aborted body. `openJsonExport` / `openCsvExport` read the
 * profile and the first page up front so those failures throw before any
 * response exists. The Prisma client is mocked: this pins the ordering, and
 * tests/integration/export-complete.test.ts covers the real queries.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

const db = vi.hoisted(() => {
  const empty = () => ({ findMany: vi.fn(async () => [] as unknown[]) });
  return {
    profile: { findUniqueOrThrow: vi.fn() },
    glucoseReading: empty(),
    meal: empty(),
    exerciseSession: empty(),
    sleepSession: empty(),
    medicationEvent: empty(),
    weightMeasurement: empty(),
    bloodPressureMeasurement: empty(),
    hydrationEvent: empty(),
    symptomEntry: empty(),
    moodEntry: empty(),
    noteEntry: empty(),
    device: empty(),
    importBatch: empty(),
  };
});
vi.mock('@/lib/db/prisma', () => ({ prisma: db }));

import { openCsvExport, openJsonExport } from '@/lib/services/export-service';

const PROFILE = {
  displayName: null,
  glucoseUnit: 'MMOLL',
  locale: 'en-CA',
  timezone: 'America/Toronto',
  condition: null,
  targetLowMgdl: 70,
  targetHighMgdl: 180,
  goals: [],
  detailLevel: 'STANDARD',
  createdAt: new Date('2026-01-01T00:00:00Z'),
};

const READING = {
  id: 'r1',
  takenAt: new Date('2026-01-02T08:00:00Z'),
  valueMgdl: 101,
  context: 'FASTING',
  note: null,
  source: 'MANUAL',
  deviceId: null,
  importBatchId: null,
  externalId: null,
};

async function collect(chunks: AsyncIterable<string>): Promise<string> {
  let out = '';
  for await (const chunk of chunks) out += chunk;
  return out;
}

beforeEach(() => {
  vi.clearAllMocks();
  db.profile.findUniqueOrThrow.mockResolvedValue(PROFILE);
  db.glucoseReading.findMany.mockResolvedValue([]);
});

describe('openJsonExport', () => {
  it('throws before returning a stream when the profile is missing', async () => {
    db.profile.findUniqueOrThrow.mockRejectedValue(new Error('No Profile found'));
    await expect(openJsonExport('user-1')).rejects.toThrow();
  });

  it('throws before returning a stream when the first page cannot be read', async () => {
    db.glucoseReading.findMany.mockRejectedValue(new Error('connection refused'));
    await expect(openJsonExport('user-1')).rejects.toThrow();
  });

  it('streams the prefetched first page without reading it twice', async () => {
    db.glucoseReading.findMany.mockResolvedValue([READING]);
    const stream = await openJsonExport('user-1');
    expect(db.glucoseReading.findMany).toHaveBeenCalledTimes(1);
    const parsed = JSON.parse(await collect(stream));
    expect(parsed.records.glucose.map((r: { id: string }) => r.id)).toEqual(['r1']);
    expect(parsed.meta.glucose).toEqual({ truncated: false, count: 1 });
    // One page shorter than the page size is the last page: no second query.
    expect(db.glucoseReading.findMany).toHaveBeenCalledTimes(1);
  });
});

describe('openCsvExport', () => {
  it('throws before returning a stream when the first page cannot be read', async () => {
    db.glucoseReading.findMany.mockRejectedValue(new Error('connection refused'));
    await expect(openCsvExport('user-1', 'glucose')).rejects.toThrow();
  });

  it('streams the header and the prefetched page', async () => {
    db.glucoseReading.findMany.mockResolvedValue([READING]);
    const csv = await collect(await openCsvExport('user-1', 'glucose'));
    const lines = csv.trimEnd().split('\r\n');
    expect(lines).toHaveLength(2);
    expect(lines[1]!.startsWith('r1,')).toBe(true);
    expect(db.glucoseReading.findMany).toHaveBeenCalledTimes(1);
  });
});
