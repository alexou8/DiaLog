import { afterAll, describe, expect, it } from 'vitest';
import {
  analyzeUser,
  insightsFor,
  toEvidenceBundle,
  type AnalyticsWindow,
} from '@/lib/services/analytics-service';
import { prisma, createTestUser, deleteTestUser } from './test-helpers';

const createdUserIds: string[] = [];

afterAll(async () => {
  for (const id of createdUserIds) await deleteTestUser(id);
});

/**
 * Recursively walks a value looking for the privacy-invariant violation:
 * an array whose elements look like raw health records (they carry a
 * `valueMgdl` or `takenAt` field). `toEvidenceBundle` must never expose one.
 */
function findRawRecordArray(value: unknown, pathLabel = 'root'): string | null {
  if (Array.isArray(value)) {
    for (const item of value) {
      if (item !== null && typeof item === 'object' && !Array.isArray(item)) {
        const keys = Object.keys(item as Record<string, unknown>);
        if (keys.includes('valueMgdl') || keys.includes('takenAt')) {
          return pathLabel;
        }
      }
    }
    for (let i = 0; i < value.length; i++) {
      const hit = findRawRecordArray(value[i], `${pathLabel}[${i}]`);
      if (hit) return hit;
    }
    return null;
  }
  if (value !== null && typeof value === 'object') {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      const hit = findRawRecordArray(v, `${pathLabel}.${k}`);
      if (hit) return hit;
    }
  }
  return null;
}

describe('analytics pipeline on a known synthetic history', () => {
  it('finds the expected post-dinner-activity association with a sensible sample size and evidence level', async () => {
    const { user, profile } = await createTestUser('analytics-known');
    createdUserIds.push(user.id);

    const days = 20;
    for (let day = 1; day <= days; day++) {
      const dinnerTime = new Date(Date.UTC(2026, 1, day, 18, 0, 0)); // Feb 2026
      const active = day % 2 === 0;
      await prisma.meal.create({
        data: {
          userId: user.id,
          takenAt: dinnerTime,
          mealType: 'DINNER',
          description: 'Dinner',
          carbsG: 50,
          dedupeKey: `known-dinner-${day}`,
        },
      });
      const postReading = new Date(dinnerTime.getTime() + 90 * 60_000);
      await prisma.glucoseReading.create({
        data: {
          userId: user.id,
          takenAt: postReading,
          valueMgdl: active ? 120 : 170,
          context: 'AFTER_MEAL',
          dedupeKey: `known-post-${day}`,
        },
      });
      if (active) {
        await prisma.exerciseSession.create({
          data: {
            userId: user.id,
            takenAt: new Date(dinnerTime.getTime() + 30 * 60_000),
            activity: 'walk',
            durationMin: 20,
            dedupeKey: `known-walk-${day}`,
          },
        });
      }
    }

    const window: AnalyticsWindow = {
      from: new Date(Date.UTC(2026, 1, 1)),
      to: new Date(Date.UTC(2026, 1, 28)),
    };
    const { result } = await analyzeUser(user.id, profile, window);

    const finding = result.findings.find((f) => f.kind === 'post-dinner-activity');
    expect(finding).toBeDefined();
    expect(finding?.sampleSize).toBe(days);
    expect(finding?.evidenceLevel).not.toBe('INSUFFICIENT');
    expect(finding?.metrics.avgWithActivityMgdl as number).toBeLessThan(
      finding?.metrics.avgWithoutActivityMgdl as number,
    );

    const cards = insightsFor(result);
    const card = cards.find((c) => c.kind === 'post-dinner-activity');
    expect(card).toBeDefined();
    expect(card?.evidenceLevel).toBe(finding?.evidenceLevel);
  });

  it('a user with only 3 readings produces INSUFFICIENT evidence and skipped analyses, not confident claims', async () => {
    const { user, profile } = await createTestUser('analytics-sparse');
    createdUserIds.push(user.id);

    const base = new Date(Date.UTC(2026, 2, 1, 8, 0, 0));
    for (let i = 0; i < 3; i++) {
      await prisma.glucoseReading.create({
        data: {
          userId: user.id,
          takenAt: new Date(base.getTime() + i * 86_400_000),
          valueMgdl: 110,
          dedupeKey: `sparse-${i}`,
        },
      });
    }

    const window: AnalyticsWindow = {
      from: new Date(Date.UTC(2026, 1, 1)),
      to: new Date(Date.UTC(2026, 2, 28)),
    };
    const { result } = await analyzeUser(user.id, profile, window);

    // No association findings should have fired off 3 readings — every association
    // requires at least 8-10 by EVIDENCE_THRESHOLDS.
    expect(result.findings).toHaveLength(0);
    expect(result.trend.classification).toBe('not-enough-data');
    expect(result.dayPatterns).toBeNull();
    expect(result.dataQuality.skippedAnalyses.length).toBeGreaterThan(0);

    const cards = insightsFor(result);
    expect(
      cards.every((c) => c.evidenceLevel !== 'CONSISTENT' && c.evidenceLevel !== 'EMERGING'),
    ).toBe(true);
  });
});

describe('toEvidenceBundle privacy invariant', () => {
  it('contains no raw record arrays (nothing shaped like a list of readings)', async () => {
    const { user, profile } = await createTestUser('analytics-bundle-privacy');
    createdUserIds.push(user.id);

    // Enough varied data to exercise every section of the bundle.
    for (let day = 1; day <= 15; day++) {
      const t = new Date(Date.UTC(2026, 3, day, 8, 0, 0));
      await prisma.glucoseReading.create({
        data: { userId: user.id, takenAt: t, valueMgdl: 100 + day, dedupeKey: `bundle-${day}` },
      });
    }

    const window: AnalyticsWindow = {
      from: new Date(Date.UTC(2026, 3, 1)),
      to: new Date(Date.UTC(2026, 3, 30)),
    };
    const { result } = await analyzeUser(user.id, profile, window);
    const bundle = toEvidenceBundle(result, profile, window);

    const violation = findRawRecordArray(bundle);
    expect(violation).toBeNull();

    // Sanity: the bundle is still informative (not just empty).
    expect(bundle.summary.readingCount).toBe(15);
    expect(Array.isArray(bundle.findings)).toBe(true);
  });
});
