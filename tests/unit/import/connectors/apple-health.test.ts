import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseXmlText } from '@/lib/import/parse';
import { appleHealthConnector } from '@/lib/import/connectors/apple-health';
import type { ExerciseRecord, GlucoseRecord, MealRecord, ParsedFile, SleepRecord, WeightRecord } from '@/lib/import/types';

const NOW = new Date('2026-08-24T12:00:00Z');

function loadFixture(): ParsedFile {
  const text = readFileSync(path.join(process.cwd(), 'tests/fixtures/import/apple-health-export.xml'), 'utf8');
  return { filename: 'export.xml', mimeType: 'application/xml', text, xml: parseXmlText(text) };
}

describe('appleHealthConnector', () => {
  it('detects a <HealthData> export', () => {
    expect(appleHealthConnector.detect(loadFixture())).toBeGreaterThan(0);
  });

  it('does not detect an unrelated XML document', () => {
    const file: ParsedFile = { filename: 'x.xml', mimeType: 'application/xml', xml: { foo: 'bar' } };
    expect(appleHealthConnector.detect(file)).toBe(0);
  });

  it('parses a blood glucose record, converting the Apple date format', async () => {
    const result = await appleHealthConnector.parse(loadFixture(), { now: NOW });
    const glucose = result.records.find((r): r is GlucoseRecord => r.kind === 'glucose');
    expect(glucose?.valueMgdl).toBe(110);
    // "2026-01-10 08:00:00 -0500" -> 13:00 UTC
    expect(glucose?.takenAt.toISOString()).toBe('2026-01-10T13:00:00.000Z');
  });

  it('parses dietary carbohydrates into a meal record', async () => {
    const result = await appleHealthConnector.parse(loadFixture(), { now: NOW });
    const meal = result.records.find((r): r is MealRecord => r.kind === 'meal');
    expect(meal?.carbsG).toBe(45);
  });

  it('parses body mass, converting lb to kg', async () => {
    const result = await appleHealthConnector.parse(loadFixture(), { now: NOW });
    const weight = result.records.find((r): r is WeightRecord => r.kind === 'weight');
    expect(weight?.weightKg).toBeCloseTo(180 * 0.45359237, 3);
  });

  it('parses a sleep analysis record into a sleep session with computed duration', async () => {
    const result = await appleHealthConnector.parse(loadFixture(), { now: NOW });
    const sleep = result.records.find((r): r is SleepRecord => r.kind === 'sleep');
    expect(sleep?.durationMin).toBe(7 * 60 + 30);
  });

  it('parses a <Workout> element into an exercise record', async () => {
    const result = await appleHealthConnector.parse(loadFixture(), { now: NOW });
    const exercise = result.records.find((r): r is ExerciseRecord => r.kind === 'exercise');
    expect(exercise?.activity).toBe('Running');
    expect(exercise?.durationMin).toBe(30);
    expect(exercise?.distanceKm).toBe(5);
  });

  it('ignores irrelevant record types (e.g. step count) without producing an issue', async () => {
    const result = await appleHealthConnector.parse(loadFixture(), { now: NOW });
    // Step count is present in the fixture but is not one of the types this connector imports.
    expect(result.records.some((r) => r.kind === 'exercise' && r.activity === 'Running')).toBe(true);
    expect(result.issues).toHaveLength(0);
  });
});
